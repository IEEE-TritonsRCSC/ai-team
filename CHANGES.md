# RoboCup AI Training — Codebase Changes Log

All changes made to fix training issues, improve the environment, and implement curriculum learning for the TD3+HER trainer.

---

## 1. Added `approach_ball` Action

**File:** [ai_interface/utils/basic_commands.py](ai_interface/utils/basic_commands.py)

**Why:** The `goto()` function includes obstacle avoidance that treats the ball as an obstacle (with ~0.215m clearance). This caused the robot to route *around* the ball instead of reaching it, creating a local maximum where the robot circled the ball indefinitely without making contact. The team decision (per Lukas, RoboCup Chair, Discord) was to keep `goto()` as a simple navigation primitive and add a separate `approach_ball()` function with no obstacle avoidance so the robot can make direct contact.

**Change:** Added after `goto()`:

```python
def approach_ball(self_pose, game_state, kickable_dist: float = 1.0, speed: float = 100.0) -> str:
    """Dash directly toward the ball with no obstacle avoidance.
    Returns "done" when within kickable_dist.
    """
    if game_state is None or getattr(game_state, "ball_pos", None) is None:
        return "turn 0"
    ball_pos = game_state.ball_pos
    dx = float(ball_pos[0]) - float(self_pose[0])
    dy = float(ball_pos[1]) - float(self_pose[1])
    dist = np.sqrt(dx * dx + dy * dy)
    if dist <= kickable_dist:
        return "done"
    angle = np.arctan2(dy, dx) - float(self_pose[2])
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    speed = min(speed, max(dist * (1 / PLAYER_DECAY - 1) / dt, 20))
    return f"dash {speed:.2f} {angle:.4f}"
```

**Key detail:** Speed uses the same deceleration formula as `goto()` — `min(speed, max(dist * (1/PLAYER_DECAY - 1) / dt, 20))` — so the robot slows down as it approaches rather than slamming into the ball at full speed.

---

## 2. Expanded Action Space from 8D to 9D

**File:** [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)

**Why:** Added the new `approach_ball` action as a learnable discrete choice alongside the existing actions. The model learns *when* to use `approach_ball` vs `goto` vs `kick` etc. via reward signals.

**Changes:**

```python
# Line 19 — added import
from ai_interface.utils.basic_commands import goto, approach_ball

# Line 82 — updated comment
# Action design per robot (9D): [goto_logit, approach_ball_logit, turn_logit,
#   kick_logit, start_dribble_logit, stop_dribble_logit, goto_x, goto_y, turn_theta]

# Line 83 — expanded dim
self.action_dim_per_robot = 9  # was 8

# Lines ~503-513 — re-indexed all action dims, added approach_ball_logit
goto_logit          = float(action_arr[base + 0])
approach_ball_logit = float(action_arr[base + 1])  # NEW
turn_logit          = float(action_arr[base + 2])
kick_logit          = float(action_arr[base + 3])
start_dribble_logit = float(action_arr[base + 4])
stop_dribble_logit  = float(action_arr[base + 5])
goto_x_raw          = float(action_arr[base + 6])
goto_y_raw          = float(action_arr[base + 7])
turn_theta_raw      = float(action_arr[base + 8])
logits = np.array([goto_logit, approach_ball_logit, turn_logit,
                   kick_logit, start_dribble_logit, stop_dribble_logit],
                  dtype=np.float32)

# Line ~521 — added to action types list
action_types = ["goto", "approach_ball", "turn", "kick", "start_dribble", "stop_dribble"]

# Lines ~562-577 — new elif branch in _action_to_commands()
elif action_type == "approach_ball":
    if pose is None or game_state is None:
        command = "turn 0"
    else:
        self_pose = np.array([
            float(pose[0]), float(pose[1]),
            float(np.deg2rad(pose[2])),
        ], dtype=np.float32)
        command = approach_ball(
            self_pose=self_pose,
            game_state=game_state,
            kickable_dist=self.kickable_dist,
        )
        if command == "done":
            command = "turn 0"
```

---

## 3. Fixed Double Simulator Cycle Waste Per RL Step

**File:** [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)

**Why:** Each call to `_get_game_state()` blocks until the next UDP packet arrives from the simulator, consuming one 100ms simulator cycle. The original `step()` made **two** such calls per step:
1. `current_game_state` — before sending commands (robot idles for a cycle)
2. `next_game_state` — after sending commands

This meant every RL step consumed 2 simulator cycles. 200 RL steps = 400 cycles = 40 real seconds. The simulator showed 50k cycles while train logs showed only 25k RL steps, confirming 2× waste.

**Fix:** Cache `next_game_state` and reuse it as `current_game_state` at the start of the next step:

```python
# In __init__:
self._cached_game_state = None

# In reset():
game_state = self._get_game_state(retries=max(...), sleep_s=...)
self._cached_game_state = game_state  # cache for first step

# In step():
current_game_state = self._cached_game_state
if current_game_state is None:
    current_game_state = self._get_game_state(...)

# ... send commands ...

next_game_state = self._get_game_state(...)
self._cached_game_state = next_game_state  # cache for next step
```

**Result:** Each RL step now consumes exactly 1 simulator cycle. 200 RL steps = 200 cycles = 20 real seconds. The robot acts on every single frame.

---

## 4. Added Episode Action Distribution Logging

**File:** [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)

**Why:** Training logs showed only reward and episode length. No visibility into which actions the model was actually selecting (goto vs approach_ball vs kick etc.), making it impossible to diagnose convergence issues.

**Change:** Added a summary log at episode end using `collections.Counter`:

```python
if terminated or truncated:
    total = len(self.episode_actions)
    if total > 0:
        from collections import Counter
        counts = Counter(self.episode_actions)
        dist = "  ".join(
            f"{k}={100*v//total}%" for k, v in sorted(counts.items())
        )
        self.logger.info(
            "Episode %d action distribution (%d steps): %s",
            self.episode_num, total, dist,
        )
    end_reason = term_reason if terminated else "max_steps"
    self.logger.info(
        "Episode %d ended — reason=%s  steps=%d  total_reward=%.2f",
        self.episode_num, end_reason, self.current_step, self.total_rewards,
    )
```

**Example output:**
```
Episode 48 action distribution (200 steps): approach_ball=100%
Episode 48 ended — reason=max_steps  steps=200  total_reward=48.86
```

---

## 5. Fixed Terminal Conditions (`terminated` Always `False`)

**File:** [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)

**Why:** The original code had `terminated = False` hardcoded. Episodes never ended on goal, ball out of bounds, or robot out of bounds — only on `max_steps`. This is wrong for RL: terminal states must not have value bootstrapped from the next state. Scoring a goal gave +70 reward but the episode continued for many wasteful steps, and the TD3 critic underestimated how good goal-scoring actions were.

**Change:** Added `_check_terminal()` method:

```python
_GOAL_HALF_HEIGHT: float = 5.0  # SSL Div B goal width=1000mm → 10 sim units → half=5.0

def _check_terminal(self, game_state) -> tuple[bool, str]:
    if game_state is None:
        return False, ""
    ball_pos = getattr(game_state, "ball_pos", None)
    if ball_pos is None:
        return False, ""
    bx, by = float(ball_pos[0]), float(ball_pos[1])
    if bx >= FIELD_X[1] and abs(by) < self._GOAL_HALF_HEIGHT:
        return True, "goal_scored"
    if abs(by) > FIELD_Y[1] or abs(bx) > FIELD_X[1]:
        return True, "ball_out_of_bounds"
    robot_poses = getattr(game_state, "robot_poses", {})
    for team_robots in robot_poses.values():
        for robot in team_robots:
            for unum, pose in robot.items():
                if int(unum) in self.robot_ids:
                    if abs(float(pose[0])) > FIELD_X[1] or abs(float(pose[1])) > FIELD_Y[1]:
                        return True, "robot_out_of_bounds"
    return False, ""
```

And updated `step()`:

```python
terminated, term_reason = self._check_terminal(next_game_state)
truncated = (not terminated) and self.current_step >= self.max_steps
if terminated:
    self._cached_game_state = None  # clear stale state
```

---

## 6. Fixed `goal_half_height` in Reward Function

**File:** [ai_interface/envs/reward.py](ai_interface/envs/reward.py)

**Why:** The reward function used 3.66 as the goal half-height for `goal_scored` detection. This missed ~27% of actual goals scored near the posts. Verified against rcssserver source: `GOAL_WIDTH = 10.0` → half = 5.0 simulator units.

**Change:**

```python
# Before
goal_half_height: float = 3.66

# After
goal_half_height: float = 5.0  # SSL Div B goal width=1000mm → 10 sim units → half=5.0
```

Note: The same correction was applied to `_GOAL_HALF_HEIGHT` in `JAL_env.py` (see change 5 above).

---

## 7. Config Hyperparameter Updates

**File:** [configs/td3_jal_her_config.json](configs/td3_jal_her_config.json)

**Why:** Analysis of training logs revealed the model was stuck in a local maximum (`approach_ball=100%`) due to:
- `learning_starts=1000` — only ~5 random episodes before policy converges; not enough diverse kick/goal transitions in replay buffer
- `action_noise_std=0.05` — too small to flip the dominant logit once approach_ball converges (logit gap grows to ~7; probability of flipping with σ=0.05 ≈ 0%)
- `max_steps=200` — with the step-caching fix, 200 steps = only 20 real seconds (was 40 before the fix)

**Changes:**

```json
"max_steps": 300,
"model_params": {
    "action_noise_std": 0.2,
    "learning_starts": 5000
}
```

**Reasoning:**
- `max_steps` 200→300: restores equivalent real game time (~30 sec) to what it was before the step-caching fix
- `action_noise_std` 0.05→0.2: 4× more noise to sustain action type exploration; 0.3 was considered but rejected because at 0.3 the `goto_x`/`goto_y` dims jitter by ±13.5m, breaking navigation
- `learning_starts` 1000→5000: ~17 full episodes of random exploration before policy starts converging, ensuring kick+goal transitions exist in the replay buffer when learning starts

---

## 8. Fixed HER Buffer Crash on Model Resume

**File:** [ai_interface/trainers/td3_jal_her_trainer.py](ai_interface/trainers/td3_jal_her_trainer.py)

**Why:** When loading a checkpoint, SB3 restores `num_timesteps` (e.g., 10,000). Since `learning_starts=5000 < 10000`, SB3 immediately tries to sample the replay buffer. But the replay buffer is **not saved** in TD3 checkpoints, so the buffer is empty. HER requires at least one complete episode before sampling, causing:

```
RuntimeError: Unable to sample before the end of the first episode.
```

**Fix:** Reset step counters after loading so `learning_starts` kicks in fresh:

```python
if self.config.get("load_model"):
    load_path = self.config["load_model"]
    self.model = TD3.load(load_path, env=env, device=str(self.device))
    # Reset so learning_starts applies fresh — HER buffer is empty after load
    self.model.num_timesteps = 0
    self.model._episode_num = 0
    self.logger.info("TD3+HER model loaded (step counters reset for fresh buffer collection)")
    return
```

---

## 9. Curriculum Ball Positioning (Fix for Local Maximum)

**Files:** [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py), [networking/networker.py](networking/networker.py), [ai_interface/envs/reward.py](ai_interface/envs/reward.py)

**Why:** After 110k+ steps of training, the model converged to `approach_ball=100%` and scored zero goals. Root cause:

1. **TD3 is a deterministic policy.** Once the actor pushes approach_ball logit to ~5.0 and kick logit to ~-2.0, the 7-point gap cannot be bridged by 0.2 noise (probability ≈ 0%). No hyperparameter tuning fixes this — it is architectural.

2. **The approach reward is too dense.** `near_ball_bonus × 250 steps of parking ≈ 50 reward`. The model never needed to kick.

3. **The goal reward is too sparse.** With ball at centre and random kick exploration, the probability of accidentally scoring is near zero. The model never sees `goal_reward=+70` in its replay buffer, so it can't learn the kick→goal connection.

**Solution:** Start each episode with the ball placed near the opponent goal. Random kicks (occasionally selected via exploration noise) are very likely to score when the ball is 5m from goal. The model sees `goal_reward=+70` early, its Q-network learns that kick has high value, and the actor begins increasing kick's logit.

### `networking/networker.py` — `reset_sim()` accepts optional ball position:

```python
def reset_sim(self, ball_pos=None):
    # ... existing player reset code ...

    bx, by = (0.0, 0.0) if ball_pos is None else (float(ball_pos[0]), float(ball_pos[1]))
    monitor_sock.sendto(f"(move (ball) {bx} {by})\0".encode(), monitor_addr)
    monitor_sock.recvfrom(16)

    self.game_watcher.restart_game()
```

### `ai_interface/envs/JAL_env.py` — Curriculum schedule:

```python
def _get_curriculum_ball_pos(self) -> tuple[float, float]:
    ep = self.episode_num
    if ep < 50:
        return (40.0, 0.0)   # 5m from goal — very high score chance
    elif ep < 120:
        return (30.0, 0.0)   # 15m from goal
    elif ep < 220:
        return (15.0, 0.0)   # 30m from goal
    else:
        return (0.0, 0.0)    # full task — centre kickoff
```

### `ai_interface/envs/JAL_env.py` — `reset()` passes curriculum position:

```python
ball_pos = self._get_curriculum_ball_pos()
self.networker.reset_sim(ball_pos=ball_pos)
if ball_pos != (0.0, 0.0):
    self.logger.info("Episode %d curriculum ball start: (%.1f, %.1f)",
                     self.episode_num + 1, ball_pos[0], ball_pos[1])
```

### `ai_interface/envs/reward.py` — Reduced parking bonus:

```python
# Before
near_ball_bonus: float = 0.2

# After
near_ball_bonus: float = 0.05
```

Reducing `near_ball_bonus` 0.2 → 0.05 drops the "park at ball" reward ceiling from ~69 to ~28, making kicking relatively more attractive. Combined with curriculum (ball near goal → kicks score), this breaks the local maximum without requiring a completely different reward landscape.

---

## 10. Reward Function: Ball-Speed Bonus and Tuning to Break Parking Local-Max

**Files:** [ai_interface/envs/reward.py](ai_interface/envs/reward.py), [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)

**Why:** After change 9 the model still converged to "park at ball" — `has_ball_bonus = 0.8/step` made standing on the ball worth ~32 reward per episode (40 contact steps), more than the ~1.2-per-step cap on `goal_progress`. We needed three changes working together: shrink the parking reward, let strong kicks earn proportionally, and add a direct signal for "ball is moving fast" (i.e. a kick connected). We also wanted to stop the model from farming `approach` reward by oscillating near the ball — once it has the ball, approach reward should turn off.

**Changes:**

```python
# reward.py — RewardConfig dataclass
goal_progress_clip: float = 2.0        # was 0.4 — let strong kicks count
has_ball_bonus:    float = 0.1         # was 0.8 — parking is no longer dominant
# new fields for ball-speed bonus:
ball_speed_bonus_weight: float = 2.0
ball_speed_threshold:    float = 0.2
ball_speed_clip:         float = 2.0
```

```python
# reward.py — RewardInputs gains an optional prev_ball_pos so we can compute
# ball displacement per step (a proxy for kick connection).
@dataclass(frozen=True)
class RewardInputs:
    ...
    prev_ball_pos: Optional[Tuple[float, float]] = None

# RewardIntermediates gains ball_speed
ball_speed: Optional[float]
```

```python
# reward.py — calculate_reward_intermediates() — compute ball_speed
ball_speed = None
if inputs.prev_ball_pos is not None:
    pbx, pby = inputs.prev_ball_pos
    ball_speed = float(math.hypot(bx - pbx, by - pby))

# reward.py — calculate_reward() — apply the new term, gate approach on has_ball
if intermediates.approach is not None and not intermediates.has_ball:
    reward += intermediates.approach * config.approach_weight
...
if intermediates.ball_speed is not None and intermediates.ball_speed > config.ball_speed_threshold:
    reward += min(intermediates.ball_speed, config.ball_speed_clip) * config.ball_speed_bonus_weight
```

```python
# JAL_env.py — track prev_ball_pos at the env level (single value; ball is shared)
self.prev_reward_ball_pos: Optional[Tuple[float, float]] = None  # __init__
self.prev_reward_ball_pos = None                                 # reset()

# JAL_env.py — pass through to build_reward_inputs, update after the per-robot loop
def build_reward_inputs(..., prev_ball_pos: Optional[Tuple[float, float]] = None):
    return RewardInputs(..., prev_ball_pos=prev_ball_pos)

if current_game_state.ball_pos is not None:
    self.prev_reward_ball_pos = (float(current_game_state.ball_pos[0]),
                                 float(current_game_state.ball_pos[1]))
```

**Effect:**
- Parking is no longer the dominant strategy (per-step parking reward dropped from ~0.88 to ~0.18).
- A clean kick that moves the ball 2 units toward goal now earns `goal_progress×3.0 + ball_speed×2.0 ≈ 6+4 = 10` in a single step, versus ~0.18 for parking.
- The model gets a positive signal when ball velocity > 0.2 units/step regardless of where the ball ends up — directly rewarding "the kick connected".

---

## 11. Curriculum Stage Transition: Shut Down Old Networker Cleanly

**File:** [ai_interface/trainers/td3_jal_her_trainer.py](ai_interface/trainers/td3_jal_her_trainer.py)

**Why:** When stage1 completed (1 robot) and stage2 began (2 robots), the trainer built a new `Networker` without shutting down the previous one. The stale Commander still held a UDP connection registered as `TritonBots #1` on rcssserver; the new stage tried to re-register `#1` and the server rejected the duplicate, raising an exception. Worse, that exception escaped to `train.py`'s `except Exception as e: print(...)` block — printed to stdout only, never to the log file — so stage failures appeared as a silent jump from "stage1 complete" to "Training session completed".

**Changes:**

```python
# setup_environment — release the previous networker before creating the new one
if self.networker is not None:
    try:
        self.networker.shutdown()
    except Exception as exc:
        self.logger.warning("Error shutting down previous networker: %s", exc)
    self.networker = None
```

```python
# Wrap stage body so any exception is logged to the file, not just stdout
def _run_stage(self, stage_name, stage_config):
    ...
    try:
        self._run_stage_inner(stage_name, num_robots, robot_ids, timesteps, stage_config=stage_config)
    except Exception:
        self.logger.exception("Stage %s failed — see traceback above", stage_name)
        raise

def _run_stage_inner(self, ...):
    # body that used to be _run_stage
```

**Effect:** Curriculum stage transitions release their rcssserver registrations cleanly. Any future stage failure is captured in `train_log.log` with a full traceback.

---

## 12. Episode-Start `playmode` Logging (Time-Over Tripwire)

**File:** [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)

**Why:** rcssserver freezes physics when it enters `PM_TimeOver` (a real rule — see `stadium.cpp` `incMovableObjects()` is only called when not `PM_TimeOver`). If a long-running server hit its `half_time` budget, our env would keep stepping but the ball would never move and the rewards would be meaningless — a silent training corruption. Added a one-line per-episode log as a tripwire.

**Change in `reset()`:**

```python
game_state = self._get_game_state(...)
self._cached_game_state = game_state

playmode = getattr(game_state, "playmode", None) if game_state is not None else None
self.logger.info("Episode %d playmode=%s", self.episode_num, playmode)
if playmode == "time_over":
    self.logger.warning(
        "Episode %d started in time_over — physics is frozen; "
        "rewards will be meaningless until the rcssserver session is restarted",
        self.episode_num,
    )
```

**Status:** In practice `playmode` is always `None` in this repo because the `client_data` channel that would populate it isn't wired through `Networker`. The check still fires correctly if it ever does become `"time_over"`, but for now it's a passive guard.

---

## 13. Action Masking via `disabled_actions`

**File:** [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)

**Why:** To narrow Stage 1 to "pure kick training", we need to prevent the policy from selecting `goto`, `approach_ball`, `start_dribble`, or `stop_dribble`. Without masking, the model could still produce high logits for those actions and the argmax would pick them. Solution: inject `-1e9` into disabled actions' logits before the softmax/argmax, making them mathematically unselectable.

**Changes:**

```python
# __init__ — accept the list from config
disabled_actions: Optional[List[str]] = None,
...
self.disabled_actions: List[str] = list(disabled_actions) if disabled_actions else []
```

```python
# _action_to_commands() — mask logits in place
logits = np.array([goto_logit, approach_ball_logit, turn_logit,
                   kick_logit, start_dribble_logit, stop_dribble_logit],
                  dtype=np.float32)
action_types = ["goto", "approach_ball", "turn", "kick",
                "start_dribble", "stop_dribble"]

if self.disabled_actions:
    for j, name in enumerate(action_types):
        if name in self.disabled_actions:
            logits[j] = -1e9

logits_shifted = logits - np.max(logits)
exp_logits = np.exp(logits_shifted)
probs = exp_logits / np.sum(exp_logits)
action_idx = int(np.argmax(probs))
```

**Effect:** With Stage 1's config (`disabled_actions: ["goto", "approach_ball", "start_dribble", "stop_dribble"]`), the policy can only ever output `kick` or `turn`. The network still has 9 output dimensions (action_dim unchanged); the parameter dims (`goto_x`, `goto_y`, `turn_theta`) still receive gradients but the logits for the masked actions are ignored.

---

## 14. Pre-Position the Robot at the Ball (`spawn_robot_at_ball`)

**Files:** [networking/networker.py](networking/networker.py), [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)

**Why:** With `goto`/`approach_ball` disabled, the robot can't move under its own power — it can only `kick` or `turn`. To still let it interact with the ball, we spawn it directly behind the ball every episode. This isolates the kick skill: the model doesn't have to learn navigation simultaneously.

**Changes:**

```python
# networking/networker.py — reset_sim() accepts a per-episode override
def reset_sim(self, ball_pos=None, player_poses_override=None):
    ...
    poses_to_apply = (player_poses_override
                      if player_poses_override is not None
                      else self.commander.desired_init_poses)
    for obj_name, pose in poses_to_apply:
        cmd = f"(move {obj_name} {pose[0]} {pose[1]} {pose[2]})\0".encode()
        ...
```

```python
# JAL_env.py — __init__ accepts the toggle and offset
spawn_robot_at_ball: bool = False,
spawn_offset_behind_ball: float = 1.0,
...
self.spawn_robot_at_ball = bool(spawn_robot_at_ball)
self.spawn_offset_behind_ball = float(spawn_offset_behind_ball)
```

```python
# JAL_env.py — reset() builds the override and passes it to the networker
ball_pos = self._get_curriculum_ball_pos()
player_poses_override = None
if self.spawn_robot_at_ball and self.networker is not None:
    commander = getattr(self.networker, "commander", None)
    default_poses = getattr(commander, "desired_init_poses", None) if commander else None
    if default_poses:
        first_obj_name = default_poses[0][0]
        robot_pose = (
            float(ball_pos[0]) - float(self.spawn_offset_behind_ball),
            float(ball_pos[1]),
            0.0,  # facing +x (opponent goal)
        )
        # Override only the first robot; keep other players (incl. TeamB) at defaults
        player_poses_override = [(first_obj_name, robot_pose)] + list(default_poses[1:])

self.networker.reset_sim(ball_pos=ball_pos, player_poses_override=player_poses_override)
```

**Effect:** Every episode, the trained robot lands at `(ball_x - offset, ball_y, θ=0)` facing the opponent goal, with the ball directly in front of it. The rcssserver's overlap-resolution rule then pushes the robot and ball to a non-overlapping distance — but they remain within kickable range (see change 22 for the kickable distance constants).

---

## 15. Random Ball X Curriculum (`random_ball_x`)

**File:** [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)

**Why:** The hardcoded episode-based curriculum (ball at x=34 for ep 0–49, x=25 for 50–119, etc.) memorizes a single ball position per phase, so the model would only ever practice kicking from those exact distances. For Stage 1, we want a *range* of kick distances every episode, randomized — but always inside the kickable layout (5 ≤ x ≤ 30, y = 0).

**Change in `_get_curriculum_ball_pos()`:**

```python
if self.random_ball_x:
    x_min, x_max = self.random_ball_x_range
    return (float(np.random.uniform(x_min, x_max)), 0.0)
# ...else fall through to the episode-phase curriculum unchanged
```

`__init__` accepts `random_ball_x: bool = False` and `random_ball_x_range: Tuple[float, float] = (5.0, 30.0)`. With Stage 1's config (`random_ball_x: true`, `random_ball_x_range: [5.0, 30.0]`), each episode draws a ball position uniformly from x in [5, 30].

**Why x in [5, 30]:** Below 5 the kick is trivial (very close to centre); above 30 the ball ends up inside the right penalty area, which triggers `BallStuckRef` (see changes 19, 21).

---

## 16. Per-Stage Reward Config Overrides (`reward_config_overrides`)

**Files:** [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py), [ai_interface/envs/JAL_her_env.py](ai_interface/envs/JAL_her_env.py)

**Why:** Stage 1 needs a different reward profile than later stages — specifically, with the robot spawned at the ball, `has_ball_bonus` and `near_ball_bonus` would reward sitting still from step 0 (the opposite of what we want). Other stages should keep those bonuses to encourage possession. We needed per-stage reward weight overrides without forking the reward function.

**Changes:**

```python
# JAL_env.py — __init__ accepts the override dict, builds a RewardConfig
reward_config_overrides: Optional[Dict[str, float]] = None,
...
if reward_config_overrides:
    known_fields = set(RewardConfig.__dataclass_fields__.keys())
    unknown = [k for k in reward_config_overrides if k not in known_fields]
    if unknown:
        raise ValueError(f"Unknown reward_config_overrides keys: {unknown}")
    self.reward_config = RewardConfig(**{
        k: v for k, v in reward_config_overrides.items() if k in known_fields
    })
else:
    self.reward_config = RewardConfig()
```

```python
# JAL_env.py — _calculate_reward() now passes self.reward_config to evaluate_reward
reward_result = evaluate_reward(reward_inputs, config=self.reward_config)
```

`JALHEREnv` inherits this automatically through `**kwargs`. The HER-specific reward terms in `JAL_her_env.py` add on top of the parent reward unchanged.

**Stage 1 overrides used (see change 18):**

```json
"reward_config_overrides": {
  "has_ball_bonus":          0.0,    // no parking reward
  "near_ball_bonus":         0.0,    // no proximity reward
  "approach_weight":         0.0,    // robot is already at ball
  "step_bonus":              0.0,    // force action — no idle reward
  "goal_progress_weight":    3.0,    // dominant shaping signal
  "goal_progress_clip":      2.0,    // let strong kicks earn proportionally
  "ball_speed_bonus_weight": 2.0,    // direct kick-connection reward
  "ball_speed_threshold":    0.2,
  "ball_speed_clip":         2.0,
  "goal_reward":             70.0    // terminal goal bonus unchanged
}
```

---

## 17. Trainer Plumbs Per-Stage Config to the Env

**File:** [ai_interface/trainers/td3_jal_her_trainer.py](ai_interface/trainers/td3_jal_her_trainer.py)

**Why:** The trainer's `setup_environment()` only read top-level config keys, so the new per-stage knobs (changes 13–16) couldn't be set per stage. Added a `_stage_or_top` helper that prefers per-stage values and falls back to top-level config, and threaded `stage_config` through to `setup_environment`.

**Changes:**

```python
def setup_environment(self, num_robots, robot_ids, stage_config=None):
    ...
    stage_config = stage_config or {}
    def _stage_or_top(key, default):
        if key in stage_config:
            return stage_config[key]
        return self.config.get(key, default)

    disabled_actions          = _stage_or_top("disabled_actions", [])
    spawn_robot_at_ball       = _stage_or_top("spawn_robot_at_ball", False)
    spawn_offset_behind_ball  = _stage_or_top("spawn_offset_behind_ball", 1.0)
    random_ball_x             = _stage_or_top("random_ball_x", False)
    random_ball_x_range       = _stage_or_top("random_ball_x_range", [5.0, 30.0])
    reward_config_overrides   = _stage_or_top("reward_config_overrides", None)

    self.env = JALHEREnv(
        ..., # existing kwargs
        disabled_actions=list(disabled_actions) if disabled_actions else [],
        spawn_robot_at_ball=bool(spawn_robot_at_ball),
        spawn_offset_behind_ball=float(spawn_offset_behind_ball),
        random_ball_x=bool(random_ball_x),
        random_ball_x_range=tuple(random_ball_x_range),
        reward_config_overrides=dict(reward_config_overrides) if reward_config_overrides else None,
    )

# _run_stage / _run_stage_inner forward stage_config
self._run_stage_inner(stage_name, num_robots, robot_ids, timesteps, stage_config=stage_config)
self.env = self.setup_environment(num_robots, robot_ids, stage_config=stage_config)
```

**Effect:** Each curriculum stage can specify its own action mask, spawn behavior, ball randomization, and reward weights. Stages without these keys fall back to defaults (regular env behavior).

---

## 18. Stage 1 Narrowed: Pure Kick Training Config

**File:** [configs/td3_jal_her_config.json](configs/td3_jal_her_config.json)

**Why:** Two previous 200k-step runs both collapsed to "approach ball + body-contact score at x=34" without ever learning to kick. To break the local maximum we narrowed Stage 1 down to a single skill: kick a stationary ball into the goal, with the robot pre-positioned in kickable range. Approach, dribble, and goal-aligned kicking are deferred to later stages.

**Stage 1 config:**

```json
"stage1": {
  "description": "Pure kick training — robot spawns at ball, only kick + turn actions, random ball x.",
  "num_robots": 1,
  "timesteps": 100000,
  "robot_ids": [1],
  "disabled_actions": ["goto", "approach_ball", "start_dribble", "stop_dribble"],
  "spawn_robot_at_ball": true,
  "spawn_offset_behind_ball": 0.5,
  "random_ball_x": true,
  "random_ball_x_range": [5.0, 30.0],
  "reward_config_overrides": { ... }  // see change 16
}
```

**Other config changes:**

- `timesteps: 100000` (was 200000) — narrowed task should be learnable in less time, early termination on goal shortens effective episodes further.
- `action_noise_std: 0.2` (already set, but now load-bearing) — the previous 0.05 was too small to keep alternative actions in play once one logit dominated.

`stage2` and `stage3` were left as placeholders with no overrides; they'll need their own narrowing when revisited.

---

## 19. Right-Penalty-Area Termination (`ball_in_penalty_off_target`)

**File:** [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)

**Why:** rcssserver has a `BallStuckRef` rule: if the ball stays effectively still inside a penalty area for `drop_ball_time` (default 100) cycles, the server fires `awardDropBall()` which teleports the ball to the corner of the penalty area, also disturbing player positions. With the model frequently kicking the ball into the right penalty area but off-target (missing the goal mouth, |y| > 5), `BallStuckRef` was firing mid-episode and corrupting the rest of the trajectory.

Adding an explicit termination condition for this case does two useful things at once: it avoids the bug, and it gives the model a small negative reward shaping signal so it learns to *aim* at the goal mouth, not just "kick forward".

**Changes:**

```python
# JAL_env.py — class constant
_RIGHT_PENALTY_AREA_X: float = 35.0

# _check_terminal — new condition (added before the position-jump backstop)
if bx >= self._RIGHT_PENALTY_AREA_X and abs(by) > self._GOAL_HALF_HEIGHT:
    return True, "ball_in_penalty_off_target"

# step() — apply the -5 penalty when this reason fires
if terminated and term_reason == "ball_in_penalty_off_target":
    reward -= 5.0
    self.total_rewards -= 5.0
```

**Why -5:** Small enough not to dominate the +70 goal reward (so a kick attempt is still net positive if the aim was *close* to the goal mouth — partial credit via `goal_progress`), large enough to discourage spamming straight-ahead kicks from off-centre starts.

---

## 20. Position-Jump Teleport Detection (Backstop)

**File:** [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)

**Why:** Server-side teleports (`BallStuckRef`, dead-ball repositioning, etc.) can move the ball or robot mid-episode in ways our env doesn't natively expect. Continuing to learn from those transitions poisons the replay buffer with unphysical state changes. Added a generic backstop: if the ball or robot moves further in one step than physics allows, end the episode with a teleport reason.

**Subtle bugs found and fixed while implementing this:**

1. **Cross-team unum collision:** the initial implementation iterated all teams in `robot_poses.values()` and matched by `unum` only. Both TritonBots and TeamB have a player #1, so the check was comparing our robot's pose to TeamB's static (20, 10) pose and flagging "teleport" every single step. Fixed by filtering to `robot_poses.get(self.team_name, [])`.

2. **Update-order ordering:** `prev_reward_ball_pos` and `prev_robot_pose_by_id` are both written by `_calculate_reward` / `_game_state_to_obs`, which run *before* `_check_terminal` in `step()`. So by the time the teleport check ran, those "prev" fields had already been overwritten with the current state — every comparison was 0. Fixed by passing `prev_game_state` (the cached state from the *start* of this step) into `_check_terminal` explicitly.

**Final changes:**

```python
# JAL_env.py — thresholds
_BALL_TELEPORT_THRESHOLD:  float = 5.0   # ball can't move >5 units in one step physically
_ROBOT_TELEPORT_THRESHOLD: float = 1.0   # robot can't move at all when locomotion is disabled

# _check_terminal signature now takes both states
def _check_terminal(self, game_state, prev_game_state=None):
    ...
    # Ball jump
    prev_ball_pos = getattr(prev_game_state, "ball_pos", None) if prev_game_state else None
    if prev_ball_pos is not None:
        pbx, pby = float(prev_ball_pos[0]), float(prev_ball_pos[1])
        ball_jump = math.hypot(bx - pbx, by - pby)
        if ball_jump > self._BALL_TELEPORT_THRESHOLD:
            self.logger.warning("Ball teleport detected: prev=(%.2f, %.2f) curr=(%.2f, %.2f) jump=%.2f",
                                pbx, pby, bx, by, ball_jump)
            return True, "ball_teleport"

    # Robot jump — only when locomotion actions are masked (robot shouldn't be able to move)
    team_robots      = getattr(game_state,     "robot_poses", {}).get(self.team_name, [])
    prev_team_robots = (getattr(prev_game_state, "robot_poses", {}).get(self.team_name, [])
                        if prev_game_state else [])
    prev_pose_by_unum: Dict[int, Tuple[float, float]] = {}
    for entry in prev_team_robots:
        for unum, pose in entry.items():
            prev_pose_by_unum[int(unum)] = (float(pose[0]), float(pose[1]))

    for robot in team_robots:
        for unum, pose in robot.items():
            if int(unum) in self.robot_ids:
                rx, ry = float(pose[0]), float(pose[1])
                if abs(rx) > FIELD_X[1] or abs(ry) > FIELD_Y[1]:
                    return True, "robot_out_of_bounds"
                if self.disabled_actions and not self._can_robot_move():
                    prev = prev_pose_by_unum.get(int(unum))
                    if prev is not None:
                        jump = math.hypot(rx - prev[0], ry - prev[1])
                        if jump > self._ROBOT_TELEPORT_THRESHOLD:
                            self.logger.warning("Robot %s teleport detected: ...", unum)
                            return True, "robot_teleport"
    return False, ""

# step() — wire in prev_game_state
terminated, term_reason = self._check_terminal(next_game_state, prev_game_state=current_game_state)
```

```python
# Helper to gate robot-teleport detection: it only makes sense when the action
# mask removes all locomotion actions, since otherwise the robot can move legitimately.
def _can_robot_move(self) -> bool:
    locomotion_actions = {"goto", "approach_ball"}
    return any(a not in self.disabled_actions for a in locomotion_actions)
```

---

## 21. rcssserver: Disable `BallStuckRef` During Training

**Files:** `~/.rcssserver/server.conf` (line 43), [update_server_conf.sh](update_server_conf.sh) (line 46)

**Why:** During Stage 1, the robot is at-ball but its early-random kicks often don't move the ball (or move it weakly). After 100 cycles of "ball effectively stuck", rcssserver fires `BallStuckRef`, teleporting the robot ~5 units backward and changing the playmode. This corrupts the replay buffer with unphysical transitions. Change 19 catches the symptom (the position teleport), but the cleanest fix is to disable the rule itself for the duration of training.

**Changes:**

```
# ~/.rcssserver/server.conf, line 43
server::drop_ball_time = 99999    # was 100
```

```bash
# update_server_conf.sh — also added so future runs of the script preserve the override
replace_server_param "drop_ball_time" "99999"
```

**Important:** This is a **temporary** training override. The 100-cycle drop ball rule is real RoboCup game flow; leaving it disabled in competition would let the agent exploit dead-ball situations that wouldn't exist in real matches. A persistent reminder is recorded in `~/.claude/projects/.../memory/project_server_config_overrides.md` with a revert checklist for when Stage 1 succeeds. The user must restart `rcssserver` for the conf change to take effect — `server.conf` is only read at startup.

---

## 22. Fixed `kickable_dist` Formula (Off-By-/2 Bug)

**File:** [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)

**Why:** The previous formula `KICKABLE_MARGIN + BALL_SIZE / 2 + PLAYER_SIZE / 2 = 0.6575` was wrong — it treated the constants as diameters. rcssserver treats them as **radii**, and the kickable check is the plain sum (no division by 2). With the wrong formula, `has_ball_now` was False on every step (because the rcssserver overlap-resolution rule pushes the robot to ~1.115 units from the ball, well above 0.6575), so every `kick` action fell through to `turn 0` and no kicks were ever sent to the server. A 900-step training run produced **zero** `kick` commands in the rcssserver text log.

**Source-code proof (downloaded rcssserver 19.0.0):**

- `robocup_downloads/rcssserver-19.0.0/src/serverparam.cpp:1474`:
  ```cpp
  M_kickable_area = M_player_size + M_kickable_margin + M_ball_size;
  ```
  Plain sum. No `/2`.

- `robocup_downloads/rcssserver-19.0.0/src/object.h:397`:
  ```cpp
  double M_size; //! object's radiuos value
  ```
  (typo for "radius") — `M_size` is the field shared by both player and ball, used directly with `M_kickable_area` as a radius.

**Empirical confirmation (diagnostic at step 1 showed):**

- Robot spawned at `(ball_x - 0.5, 0, 0)` → server pushed it to `dist = 1.1150` from the ball.
- `1.1150 = PLAYER_SIZE + BALL_SIZE = 0.9 + 0.215` — exactly the sum of the two **full** values, confirming they are radii (not diameters).

**Change:**

```python
# Before — divided by 2 incorrectly:
self.kickable_dist = KICKABLE_MARGIN + BALL_SIZE / 2 + PLAYER_SIZE / 2  # = 0.6575

# After:
self.kickable_dist = KICKABLE_MARGIN + BALL_SIZE + PLAYER_SIZE          # = 1.215
```

**Empirical result after the fix:**

| Metric                  | Old formula (`/2`) | New formula     |
| ----------------------- | ------------------ | --------------- |
| Kicks sent to server    | 0 in 900 steps     | many per ep.    |
| Goals scored            | 0                  | ~22 of 29 eps   |
| Avg episode reward      | -33                | +207            |
| Avg episode length      | 300 (max_steps)    | ~17 (early term)|

**Latent issue still open:** `KICKABLE_MARGIN` in Python is 0.1, but rcssserver's runtime value is 0.7 (because `update_server_conf.sh` can't run — see below). After the formula fix:
- Python `kickable_dist` = 0.1 + 0.215 + 0.9 = **1.215**
- rcssserver `kickable_dist` = 0.7 + 0.215 + 0.9 = **1.815**

Python is *stricter* than the server, which is fine for now (any kick our env decides to send, the server will accept), but the two values should be reconciled before later stages.

---

## 23. `update_server_conf.sh` — Latent Issue Identified (Not Fixed)

**File:** [update_server_conf.sh](update_server_conf.sh)

**Issue:** The script copies `~/.rcssserver/backup_server.conf` over `~/.rcssserver/server.conf` before applying its overrides, and exits with an error if the backup doesn't exist. Currently the backup file is missing, so the script has never been able to run. As a result, rcssserver is running with stock defaults for `kick_rand`, `ball_rand`, `coach_w_referee`, `kickable_margin`, `text_log_dir`, etc. — *not* the project's intended overrides.

**Why this matters now:** the kickable_margin mismatch (Python 0.1 vs server 0.7, see change 22) is one downstream symptom.

**Fix path** (not done yet): `cp ~/.rcssserver/server.conf ~/.rcssserver/backup_server.conf` once to seed the backup, then run `update_server_conf.sh`. Should be done before Stage 2 work begins.

---

## Training Configuration — Final State (Stage 1)

**File:** [configs/td3_jal_her_config.json](configs/td3_jal_her_config.json)

```json
{
  "max_steps": 300,
  "load_model": null,
  "curriculum": {
    "stage1": {
      "num_robots": 1,
      "timesteps": 100000,
      "robot_ids": [1],
      "disabled_actions": ["goto", "approach_ball", "start_dribble", "stop_dribble"],
      "spawn_robot_at_ball": true,
      "spawn_offset_behind_ball": 0.5,
      "random_ball_x": true,
      "random_ball_x_range": [5.0, 30.0],
      "reward_config_overrides": {
        "has_ball_bonus": 0.0,
        "near_ball_bonus": 0.0,
        "approach_weight": 0.0,
        "step_bonus": 0.0,
        "goal_progress_weight": 3.0,
        "goal_progress_clip": 2.0,
        "ball_speed_bonus_weight": 2.0,
        "ball_speed_threshold": 0.2,
        "ball_speed_clip": 2.0,
        "goal_reward": 70.0
      }
    }
  },
  "model_params": {
    "action_noise_std": 0.2,
    "learning_starts": 5000
  }
}
```

**Server-side:** `~/.rcssserver/server.conf` has `drop_ball_time = 99999` (revert to 100 before competition).

`load_model: null` — training runs from scratch. Old checkpoints have different action masks and spawn behavior; they would not transfer.

---

## What to Watch For in Logs

- **`reason=goal_scored`** in episode termination logs — the 387-episode run before changes 1–9 had zero. With changes 1–22 in place, the latest run scored on ~22 of 29 random-exploration episodes (76%) before TD3 even started learning.
- **`reason=ball_in_penalty_off_target`** — early in training, expect this for kicks aimed slightly off the goal mouth. Frequency should drop as the policy learns to aim.
- **Reward climbing past 200 average** — indicates kicks are routinely connecting and scoring.
- **Action distribution shows only `kick` and `turn`** — confirms `disabled_actions` is taking effect.
- **`(kick ...)` lines in `text_logs/<latest>.rcl`** — server-side proof that kicks are being sent. Grep with `grep -c "kick" text_logs/<latest>.rcl`.
- **No `reason=robot_teleport` or `reason=ball_teleport`** under normal operation — these should only fire if the server-side referee rules teleport something unexpectedly.
- **`Episode N curriculum ball start: (X.X, 0.0)`** — confirms the random ball x is being drawn from `[5.0, 30.0]` per episode.

---

## Planned / Future Changes (NOT yet implemented)

### F1. Parallel learning for TD3+HER (deferred — revisit at Stage 2)

**Status:** Not built. Deliberately deferred. Do **not** implement for Stage 1.

**Why deferred:** `--num-envs` is silently ignored by the `td3_jal_her` trainer — it always builds a single `JALHEREnv` connected to `_sim_endpoint_for_env(0)` and trains on one env. We considered adding real parallelism to speed up Stage 1, but measured the bottleneck first:

- Pre-learning (embedded sim only): **~4,500 steps/s**
- Post-learning, MPS: ~76 steps/s; post-learning, **CPU: ~246 steps/s** (see F-note below)

So once gradient updates are active, the **sim is only ~5% of wall-clock** — training is gradient-bound, not collection-bound. Parallel envs only speed up data collection, so the best-case wall-clock saving is ~5%, for a large amount of new code. Wrong lever for off-policy + a fast in-process sim. (Parallelism is wired for the on-policy PPO trainers, where collection is a big serial chunk, which is why it helps *those*.)

**Triggers to actually build it (any one):**
1. **Collection-bound regime** — e.g. reverting to UDP `sim-only` (each step waits ~10–100 ms real-time), or Stage 2+ multi-robot sims heavy enough that per-step sim cost grows.
2. **Experience diversity for stability** (a *quality* reason, not speed) — multiple envs decorrelate the replay buffer, which helps on harder tasks (opponents, sparser rewards).
3. **Move to a CUDA box** — cheap gradient updates shift the bottleneck back to collection.

**Before building: re-measure the collect-vs-gradient split in the new regime.** Only parallelize if collection has become a meaningful fraction.

**Design constraints when implemented:**
- **Must use `SubprocVecEnv`, not `DummyVecEnv`.** The embedded sim keeps **process-global state** — `ServerParam` is a singleton and the C++ engine has static vars (e.g. `s_half_time_count` in `referee.cpp`). Two embedded sims in one process would corrupt each other. One embedded sim **per process** only.
- Each subprocess builds its own `Networker` + embedded sim on its own endpoint.
- `HerReplayBuffer` must be constructed with `n_envs>1` (verify the local `_sample_goals` clamp patch still holds for vectorized envs).
- Retune `gradient_steps` to ~`n_envs` so sample-efficiency-per-transition is preserved (otherwise N envs → 1/N updates per transition → slower convergence).
- Wire `num_envs` through `td3_jal_her_trainer.setup_environment` (currently hardcoded to env 0).

### F2. Device default baked to CPU (DONE for the td3_jal_her config; note for other configs)

CPU beat MPS by ~3.2× for this small net (256×256, batch 64) — MPS kernel-launch/transfer overhead dominates. `"device": "cpu"` is now set in `configs/td3_jal_her_config.json`. **This is mac-mini/Apple-Silicon-specific** — on a CUDA box, remove the key (or set `"cuda"`) so the GPU is used. The `--device {cpu,cuda,mps}` flag overrides the config per-run for re-benchmarking.

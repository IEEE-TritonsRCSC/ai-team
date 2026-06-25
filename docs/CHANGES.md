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

---

## 24. Stage 1.2 TD3+HER Kick-Collapse Fixes (2026-06-02)

**Context:** Stage 1.2 repeatedly collapsed to `kick=100%`. Bad kicks were reward-positive because forward ball progress was paid even when the projected ball path missed the goal. Later, after the reward fix worked under exploration, deterministic inference exposed a separate TD3 decoder tie: the actor saturated both `turn` and `kick` logits at `1.0`, and `np.argmax` always selected the earlier `turn` slot.

### Reward math fix

**Files:**
- [ai_interface/envs/reward.py](ai_interface/envs/reward.py)
- [ai_interface/envs/JAL_her_env.py](ai_interface/envs/JAL_her_env.py)

**Changes:**
- Added `aim_quality_from_prediction(predicted_y_at_goal_line, goal_half_height, target_y=0.0)`.
- For fast moving balls, positive `goal_progress` is multiplied by projected aim quality.
- Negative `goal_progress` remains ungated, so moving away from goal is still penalized.
- Slow/stationary progress remains unchanged for approach-style stages.
- Ball-speed reward is also aim-gated so a hard off-target kick cannot earn speed reward.
- HER live progress shaping uses the same projected aim gate; HER relabeled sparse reward is unchanged.

**EV effect:**
- Before: a bad kick could earn up to `goal_progress_clip 2.0 * goal_progress_weight 3.0 = +6.0/step`, plus HER live progress up to `0.3 * 2.0 = +0.6/step`, while `bad_aim_kick_penalty` was only up to `-1.0`.
- After: off-target positive progress has `aim_quality=0`, so positive progress and speed/HER shaping are zero. Bad kick EV is no longer positive.
- On-target kicks keep the strong success terms: `goal_reward=150`, `kick_aim_bonus_weight=20`, plus gated movement shaping.

### Curriculum kick gate

**File:** [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)

**Changes:**
- Added stage/env knobs:
  - `kick_requires_aim`
  - `kick_min_aim_quality`
  - `spawn_theta_relative_to_goal`
  - `spawn_theta_min_abs_deg`
- If the actor requests `kick` while holding the ball but projected aim quality is below the threshold, the env blocks the kick, executes the actor's own `turn_theta`, records an invalid action, and applies `invalid_action_penalty`.
- The bad-kick gate is curriculum-specific. It is intended for at-ball turn/kick stages where blocking strategically bad kicks is acceptable.

**Stage 1.2 config values:**
- `kick_requires_aim=true`
- `kick_min_aim_quality=0.05`
- `invalid_action_penalty=0.5`
- `spawn_theta_relative_to_goal=true`
- `spawn_theta_min_abs_deg=25.0`
- `random_spawn_theta_range_deg=[-55.0, 55.0]`

### Deterministic TD3 turn/kick tie-break

**Files:**
- [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py)
- [ai_interface/trainers/td3_jal_her_trainer.py](ai_interface/trainers/td3_jal_her_trainer.py)
- [infer.py](infer.py)

**Changes:**
- Added stage/env knobs:
  - `kick_tie_break_when_aimed`
  - `kick_tie_break_epsilon`
- When `turn` and `kick` are both enabled, the robot has the ball, and projected aim quality is at least `kick_min_aim_quality`, the decoder selects `kick` if `kick_logit >= turn_logit - epsilon`.
- If aim is not ready, the decoder leaves the action as `turn`.
- Added diagnostics for raw action, requested action, executed action, tie-break firing, projected aim quality, projected y-at-goal-line, invalid status, and logits.
- `td3_jal_her_trainer.py` and `infer.py` now pass the new stage knobs into `JALHEREnv`.

**Why:** TD3 emits continuous primitive logits and the env decodes them with argmax. In deterministic infer the trained actor output `turn=1.0` and `kick=1.0`; legacy `np.argmax` chose `turn` forever because turn appears before kick. The tie-break makes deterministic deployment match the intended learned behavior without adding eval noise.

### Warm-start action noise fix

**File:** [ai_interface/trainers/td3_jal_her_trainer.py](ai_interface/trainers/td3_jal_her_trainer.py)

**Change:** build action noise from the current config before loading a checkpoint, and reattach it to the loaded TD3 model. Previously `TD3.load(...)` returned before the config noise was applied, so warm-started runs silently reused whatever noise object was pickled inside the checkpoint.

**Active noise values:**
- `action_noise_logit_std=0.4`
- `action_noise_param_std=0.1`
- `target_policy_noise=0.2`
- `target_noise_clip=0.5`

**Reasoning:** slot logits need enough collection noise to keep `turn`/`kick` sampled, while the continuous turn parameter needs low noise so aimed kicks are precise. `param_sigma=0.1` corresponds to about `0.1*pi = 18°`.

### Tests

**File:** [tests/test_td3_kick_collapse_fixes.py](tests/test_td3_kick_collapse_fixes.py)

Added focused tests for:
- Off-target fast positive progress gives `0` positive progress reward.
- On-target fast positive progress is scaled by aim quality.
- Negative progress is still penalized.
- Slow/stationary progress remains unchanged.
- Bad-aim kick is converted to turn and counted invalid.
- Aligned kick fires normally.
- Gate-disabled behavior preserves legacy kick execution.
- Deterministic turn/kick tie selects kick only when aim is good and config enables tie-break.

### Stage 1.2 result

**Training:** `train_logs/20260602_180617_572684`
- 34,800 / 35,000 steps.
- 547 episodes.
- Overall goal rate: 420/547 = 76.8%.
- Last 100 goal rate: 80.0%.
- Bad-aim kicks: 0.0%.
- Checkpoint: `models/20260602_180617_572684_td3_jal_her/stage1_2_turn_warmup_complete.zip`.

**Deterministic infer after tie-break:** `infer_logs/20260602_182934_stage1_2_turn_warmup_complete`
- Overall goal rate: 75.5%.
- Last 100 goal rate: 77.0%.
- Executed actions: `turn=5742`, `kick=258`.
- Invalid actions: 0.
- Bad-aim kicks: 0.0%.

---

## 25. Stage 1.3 Full-Turn Promotion and Results (2026-06-02)

**Context:** Stage 1.2 was accepted, so the next curriculum step was full heading coverage at the ball. Approach remained disabled by design; `approach_ball` should first be unlocked in Stage 1.4.

### Config promotion

**File:** [configs/td3_jal_her_config.json](configs/td3_jal_her_config.json)

**Changes:**
- Moved `stage1_3_full_turn` from `_pending_substages` into `curriculum`.
- Moved `stage1_2_turn_warmup` into `_completed_substages`.
- Set `load_model` to `models/20260602_180617_572684_td3_jal_her/stage1_2_turn_warmup_complete.zip`.
- Kept `approach_ball` disabled in Stage 1.3:
  - `disabled_actions=["goto", "approach_ball", "start_dribble", "stop_dribble"]`
- Expanded heading range:
  - `random_spawn_theta_range_deg=[-180.0, 180.0]`
  - `spawn_theta_relative_to_goal=true`
  - `spawn_theta_min_abs_deg=0.0`
- Carried forward the Stage 1.2 kick safety:
  - `kick_requires_aim=true`
  - `kick_min_aim_quality=0.05`
  - `kick_tie_break_when_aimed=true`
  - `kick_tie_break_epsilon=0.05`
- Used `alignment_weight=2.0` for a small dense turn gradient. Main success reward remains `goal_reward=150` plus `kick_aim_bonus_weight=20`.

### Reward math

Stage 1.3 keeps the Stage 1.2 reward safety:
- Off-target fast positive progress is gated to zero by projected aim quality.
- Good kicks retain `+150` goal reward and up to `+20` aim bonus.
- Alignment is bounded and delta-based. Moving from fully backwards to fully aligned can contribute about `2.0 * (1 - -1) = +4` total, which is useful as a breadcrumb but cannot dominate the kick/goal terms.
- `bad_aim_kick_penalty=1.0` remains a backup. With `kick_requires_aim=true`, it should rarely fire because strategically bad kick requests are blocked before a real kick command.

### Stage 1.3 result

**Training:** `train_logs/20260602_184554_470708`
- 149,992 / 150,000 steps.
- 5195 episodes.
- Overall goal rate: 4225/5195 = 81.3%.
- Last 100 goal rate: 85.0%.
- Outcomes: `goal_scored=4225`, `ball_in_penalty_off_target=760`, `ball_out_of_bounds=204`, `max_steps=6`.
- Aim quality: first 200 kicks 0.507 / 0.0% bad aim; last 200 kicks 0.550 / 0.0% bad aim.
- Checkpoint: `models/20260602_184554_470708_td3_jal_her/stage1_3_full_turn_complete.zip`.

**Deterministic infer:** `infer_logs/20260602_190133_stage1_3_full_turn_complete`
- 306 episodes, 6000 steps.
- Overall goal rate: 250/306 = 81.7%.
- Last 100 goal rate: 85.0%.
- Outcomes: `goal_scored=250`, `ball_in_penalty_off_target=49`, `ball_out_of_bounds=7`.
- Kicks: 307 total, about one per episode.
- Average aim quality: 0.521.
- Bad-aim kicks: 0.0%.
- Invalid actions / blocked bad aim: 0.

**Note:** `summary.json` reports `last_100.kicks=0`, but the per-episode infer log shows kicks in the last 100. The correct recomputed value is 100 kicks in the last 100 episodes with 0 bad-aim kicks.

### Next config risk

Stage 1.4 unlocks `approach_ball`, a primitive slot that was masked in Stage 1.2 and Stage 1.3. Because TD3 uses continuous logits plus argmax, the newly enabled approach slot may create another deterministic decoding issue. Before training Stage 1.4, carry forward the kick gate and tie-break safety, then watch deterministic infer for collapse into `approach_ball`, `turn`, or `kick`.

---

## 26. PPO JAL "Continuous-Turning" Bug — Inference Param Noise (2026-06-03)

**Files:** [infer.py](infer.py), [launch_infer.py](launch_infer.py)

**Symptom:** In deterministic inference of the Stage 1.9 PPO JAL model
(`models/ppo_jal_curriculum/stage1_9_complete.pt`), the robot approached the ball,
turned, and then — even when essentially facing the goal — **kept turning forever to
"find the right angle" instead of kicking**, most often when the ball was near the
goal's center line. Deterministic inference timed out (`max_steps`) on ~19% of episodes
(73.6% goal rate), versus ~1% timeouts during stochastic training (~87.7% goal). The
robot was not broken; the *inference mode* was.

### Root cause (confirmed from per-step trace data)

The PPO actor's turn parameter head (the Gaussian mean for `turn_theta`, scaled by
`× np.pi` in [JAL_env.py](ai_interface/envs/JAL_env.py) → a turn command in radians,
≈ ±19°/step after rcssserver inertia) learned a **saturated bang-bang policy, not a
proportional one.** Measured on the baseline deterministic run
(`infer_logs/20260603_165526_stage1_9_complete/step_trace.jsonl`):

- **79%** of turn commands pinned at `|turn_theta| > 3.0` (≈ ±π); only 13% near zero —
  a sharply bimodal distribution, not a smooth proportional controller.
- Turn sign matched the *correct* direction (toward goal-aim) only **30%** of the time.
- Even when already within 5° of the goal aim, mean `|turn_theta|` was still **2.87**
  (a proportional controller would be ≈0).

**Mechanism:** under stochastic training the Gaussian *exploration noise* on the param
supplied the fine, variable turn corrections — the sampled `turn_theta` occasionally
landed small, so the robot settled onto the aim line and kicked. Deterministic inference
uses the param **mean** (no noise), leaving a fixed ±19°/step bang-bang command that
overshoots the aim line and oscillates in a strict ±π 2-cycle, never settling to the
sub-degree precision a kick needs. This is the textbook "deterministic mean of a
high-variance policy" failure, and it fully explains the ~1% (train) vs ~19% (det. infer)
`max_steps` gap.

### Fix: restore the fine corrections at inference time (no retrain)

Two new flags on the PPO JAL inference path in `infer.py`:

```python
# infer.py — argparse
--ppo_stochastic              # sample primitive + params (deterministic=False),
                              #   exactly as during training. Faithful confirmation.
--ppo_param_noise_std FLOAT   # keep argmax primitive, add Gaussian noise of this std
                              #   to the deterministic param MEAN, then re-clamp to [-1,1].
                              #   DEFAULT = 0.3 (was 0.0).
```

```python
# infer.py — _run_ppo_jal inference loop
action, _transition = agent.sample_action(
    obs=obs, disabled_actions=disabled_actions,
    deterministic=not bool(args.ppo_stochastic),
)
if not bool(args.ppo_stochastic) and float(args.ppo_param_noise_std) > 0.0:
    params = np.asarray(action["params"], dtype=np.float32)
    params = params + np.random.normal(0.0, float(args.ppo_param_noise_std),
                                       size=params.shape).astype(np.float32)
    action["params"] = np.clip(params, -1.0, 1.0)
```

`launch_infer.py` forwards both flags (`--ppo_stochastic`, `--ppo_param_noise_std`);
when unset it inherits infer.py's `0.3` default.

**Why default 0.3 (param-noise) instead of full stochastic:** it keeps the *primitive*
choice deterministic (predictable: approach → turn → kick) while jittering only the turn
angle enough to break the bang-bang 2-cycle. The small noise lets the realized command
occasionally land small, so the robot converges and kicks. Pure deterministic mean
(`--ppo_param_noise_std 0`) is retained for debugging only.

### Validation (both 6000-step sim-embedded runs, Stage 1.9 model)

| Run | flag | max_steps timeouts | goal rate | turn steps | sign-correct | mean \|turn\| when aimed |
|---|---|---|---|---|---|---|
| baseline (`165526`) | det. mean (0.0) | **~19%** (10/53) | 73.6% | 2437 | 30% | 2.87 |
| A (`170514`) | `--ppo_stochastic` | **0%** (0/67) | 86.6% | 1336 | 65% | 2.66 |
| B (`170626`) | `--ppo_param_noise_std 0.3` | **1.4%** (1/72) | **94.4%** | 837 | 76% | 2.21 |

The bang-bang *magnitude* is still present (still 46–78% saturated), but the noise breaks
the loop: turn-step count collapses (the robot stops thrashing) and sign-correct rises to
65–76%. **B is the new default** — it beats the original deterministic baseline by +21pp
goal rate and reduces timeouts from ~19% to ~1.4%.

### Carry-over / retrain decision

**No Stage-1 retrain.** The defect is an inference-mode artifact, not a flaw in the
learned policy. Stage 2 warm-starts from `stage1_9_complete.pt` and **trains
stochastically** (samples actions) — the regime where the policy already works (~1%
timeouts) — so the bang-bang mean does not carry over into training behavior. The only
requirement is that PPO JAL inference is never run in pure-deterministic mode; baking the
`0.3` default into `infer.py` enforces this.

**Future hardening (optional, not required):** if a *pure-deterministic* PPO policy is
ever needed (e.g. for reproducible competition deployment), anneal the param-head
entropy/std harder late in training, or shrink the turn action scale (the `× np.pi`), so
the deterministic mean itself learns to taper toward zero as the robot aligns.

---

## 27. Stage 2g Ball-Action Deadlocks — Claimant-Scoped Runtime Masks and Recovery (2026-06-22)

**Change completed:** 2026-06-22 07:15:37 IST (+0530)

**Files changed:**

- `ai_interface/envs/JAL_env.py`
- `ai_interface/algorithms/ppo_jal.py`
- `ai_interface/trainers/ppo_jal_curriculum_trainer.py`
- `infer.py`
- `configs/ppo_jal_curriculum_config.json`
- `tests/test_ball_action_recovery.py` (new)
- `tests/test_ppo_jal_expandable.py`

### Problem

Two 3,000-step Stage-2g debug inference runs exposed three independent deterministic
deadlocks after the latest `dribble_to` transport changes:

1. **Out-of-range `dribble_to` no-op loop.** The policy selected `dribble_to` while
   1.34–5.03 m from the ball. `dribble_to()` correctly returned `done` because it is
   intentionally a possession-only primitive, but the environment converted `done`
   into `turn 0`. Deterministic argmax selected `dribble_to` again on every following
   step, so the robot stopped behind the ball.
2. **Completed `approach_ball` no-op loop.** At approximately 1.12 m from the ball,
   the proximity-based `has_ball`/kickable check made `approach_ball()` return `done`.
   The environment again emitted `turn 0`, and deterministic argmax kept selecting
   `approach_ball`. The robot had not necessarily caught/glued the ball; `has_ball`
   only means that it is within kickable distance.
3. **Stationary `turn` fixed point.** Three episodes per run selected `turn` for the
   rest of the 300-step episode while still 2.14–4.13 m from the ball. A zero-param-
   noise diagnostic proved that inference noise was not the root cause: the primitive
   remained `turn`, while its deterministic parameter mean shrank to approximately
   ±0.05–0.21 and produced almost no useful rotation or translation.

The simulator itself was not stale: simulator counts advanced every cycle. The
`frozen_state_stale_sim` outcomes were static-world detections caused by repeated
no-op commands.

An unconditional fallback from all robots to `approach_ball` was rejected because it
would make every controlled teammate chase the same ball in future multi-robot stages.

### Fix

#### 1. One sticky ball claimant per team

`JALTeamEnv._ball_claimant()` now chooses exactly one controlled robot that may execute
ball-seeking primitives. Priority is:

1. a robot with an already committed dribble macro;
2. a robot within kickable/possession range;
3. the nearest configured eligible robot.

The current claimant is retained while it remains within
`ball_claimant_switch_margin` of the nearest candidate, preventing rapid ownership
oscillation. `ball_claimant_robot_ids` allows goalkeepers or fixed support players to
be excluded. Non-claimants have `approach_ball`, `kick`, and `dribble_to` masked; they
retain non-ball actions such as `goto` and `turn`. A defensive stale-action guard holds
a non-claimant with `turn 0` instead of redirecting it toward the ball.

#### 2. Per-slot PPO runtime primitive mask

`JALTeamEnv.get_primitive_valid_mask()` returns an
`[a_max, num_primitives]` mask for the current simulator state:

- claimant outside kickable range: allow `approach_ball`, mask `kick`/`dribble_to`;
- claimant inside kickable range: mask completed `approach_ball`, allow
  `turn`/`kick`/`dribble_to`;
- non-claimants: mask all three ball-seeking primitives;
- stalled claimant: temporarily mask `turn`, forcing the categorical policy to choose
  another valid primitive (normally `approach_ball`).

`PPOJALAgent.sample_action()` applies this runtime mask before constructing the
Categorical distribution. The combined stage/reserved/runtime mask is stored in the
rollout transition and reused by the existing PPO update path, so training remains
on-policy rather than rewarding an executor-side action substitution.

Both single-environment and parallel PPO training loops, plus PPO inference, now pass
the environment's current runtime mask into `sample_action()`.

#### 3. Defensive executor recovery

The environment still validates the sampled action immediately before command
emission to cover callers without runtime masks and state changes between sampling and
execution:

- claimant `dribble_to` outside range → execute `approach_ball`;
- claimant `approach_ball` after reaching its margin → execute the policy's turn
  parameter instead of `turn 0`;
- non-claimant ball action → hold, never approach;
- claimant stationary `turn` for `turn_stall_limit` cycles while outside possession
  range → execute one `approach_ball` recovery step.

The turn watchdog is per robot and requires displacement no greater than
`turn_stall_displacement`. It resets after motion, a non-turn action, possession, or
claimant change. It is never applied to non-claimants because they may legitimately
turn or hold while maintaining team shape.

#### 4. Diagnostics and Stage-2g activation

Per-robot action info and `step_trace.jsonl` now include:

- requested and executed primitive;
- `fallback_reason`;
- `ball_claimant_id` / `is_ball_claimant`;
- robot-ball distance;
- turn-stall count.

Recovery defaults to disabled in the generic environment to avoid silently changing
legacy TD3 or earlier curriculum behavior. Stage 2g explicitly enables it with:

```json
"ball_action_recovery": true,
"ball_claimant_robot_ids": [1],
"ball_claimant_switch_margin": 0.75,
"turn_stall_limit": 12,
"turn_stall_displacement": 0.05
```

The current Stage-2g checkpoint remains a single-robot checkpoint. The ownership code
is multi-robot-safe, but a future multi-robot policy still requires training with the
additional agent slots active and a support/formation action available to
non-claimants.

### Code locations changed

| File | Lines after change | Change |
|---|---:|---|
| `ai_interface/envs/JAL_env.py` | 83–156 | Recovery configuration, claimant state, and per-robot turn-watchdog state. |
| `ai_interface/envs/JAL_env.py` | 316–427 | Sticky claimant selection and per-slot primitive-validity mask. |
| `ai_interface/envs/JAL_env.py` | 573–575 | Reset claimant and watchdog state at every episode reset. |
| `ai_interface/envs/JAL_env.py` | 1367–1370, 1463–1663 | Detect whether PPO already sampled with a runtime mask; resolve claimant, reject non-claimant ball actions, recover invalid dribble/approach selections, and apply stationary-turn watchdog. |
| `ai_interface/envs/JAL_env.py` | 1899–1903 | Export claimant, fallback, distance, and watchdog diagnostics. |
| `ai_interface/algorithms/ppo_jal.py` | 537–586, 615–619 | Accept, validate, combine, sample with, mark, and store the runtime primitive mask. |
| `ai_interface/trainers/ppo_jal_curriculum_trainer.py` | 228–238 | Forward recovery/claimant/watchdog stage configuration into the environment. |
| `ai_interface/trainers/ppo_jal_curriculum_trainer.py` | 609–611, 847–849 | Apply runtime masks in single and parallel rollout sampling. |
| `infer.py` | 683–687 | Mirror recovery configuration in PPO inference. |
| `infer.py` | 813–815 | Apply runtime primitive mask before deterministic/stochastic PPO sampling. |
| `infer.py` | 904–910 | Write requested/executed action, claimant, fallback, distance, and watchdog fields to the debug trace. |
| `configs/ppo_jal_curriculum_config.json` | 954–961 | Enable and configure claimant-scoped recovery for Stage 2g only. |
| `tests/test_ball_action_recovery.py` | 1–174 | Seven new single- and multi-robot recovery/ownership regression tests. |
| `tests/test_ppo_jal_expandable.py` | 153–171 | Verify runtime masks are enforced and stored by PPO sampling. |

### Validation

Using the `rcai` Python 3.11 environment:

- `tests/test_ball_action_recovery.py`: **7/7 passed**;
- `tests/test_ppo_jal_expandable.py`: **10/10 passed**;
- `tests/test_dribble_to.py`: **13/13 passed**;
- Python compilation passed for all modified Python files;
- `ppo_jal_curriculum_config.json` passed `json.tool` validation.

Total focused assertions passed: **30**. A fresh 3,000-step `--debug_infer` run is still
required to verify that `frozen_state_stale_sim` and turn-driven `max_steps` outcomes
disappear under live simulator timing.

---

## 2026-06-22 11:40:57 IST — Stage 2g physical-realism, accounting, and reward-system overhaul (changes 1–4)

This entry consolidates all four changes made after analysing the Stage 2g inference
videos, `infer_logs`, embedded simulator traces, and both rcssserver source trees.

### Change 1 — Enforce the physical robot's 20°/s angular-velocity limit

#### Problem

The command stack represented `turn` arguments as radians per second, but several
controllers requested `heading_error / dt`. With `dt=0.1`, this asked the simulator to
close an arbitrarily large heading error in one cycle. Turns approaching 180° were
therefore possible and did not represent the physical robot, whose measured maximum
angular velocity is 20°/s.

Applying a limit only in `dribble_to` would have left policy `turn`, kick-alignment,
goalie, and fallback commands unrestricted. Applying it only to simulator output would
also have left physical-robot multicast commands and debug traces inconsistent.

#### Fix

- Added the shared constant:

  ```python
  MAX_ANGULAR_VELOCITY = radians(20.0)  # 0.3490658504 rad/s
  ```

- Added `limit_turn_rate()`, which clamps every `turn` command to
  `[-0.3490658504, +0.3490658504] rad/s`.
- Applied the limiter to both simulator and physical-robot serialization.
- Applied the same limiter before JAL command logging so `step_trace.jsonl` records the
  command actually sent rather than the uncapped request.
- At the 100 ms simulator timestep, the maximum commanded rotation is now exactly
  `20°/s × 0.1 s = 2°` per cycle.
- Increased `dribble_to(max_align_steps)` from 3 to 90. Three capped cycles could align
  by only 6°; 90 cycles permit a full 180° correction at the physical limit while
  retaining a finite bound.

The first post-change inference confirmed a maximum command of exactly 20°/s. Observed
state deltas reached approximately 2.10° in a few cycles because rcssserver applies
player noise; no command exceeded the configured rate.

#### Code locations

| File | Lines after change | Change |
|---|---:|---|
| `networking/data_utils.py` | 14, 32–45 | Angular-rate constant and shared command limiter. |
| `networking/data_utils.py` | 240–280 | Clamp simulator and physical-robot turn output. |
| `ai_interface/envs/JAL_env.py` | 1906–1911 | Clamp before command storage/debug reporting. |
| `ai_interface/utils/basic_commands.py` | 360–378 | Expand bounded release-alignment budget to 90 cycles. |
| `tests/test_turn_rate_limit.py` | 1–39 | Positive, negative, below-limit, robot-output, and non-turn regression tests. |
| `tests/test_ball_action_recovery.py` | 85–110 | Update fallback expectations for the physical turn limit. |

### Change 2 — Remove catch-driven field-player pose correction

#### Problem

Successful stock rcssserver catches translate the player onto the ball, rotate its body
to the catch vector, and zero its velocity. That is appropriate for the simulator's
goalkeeper catch model but was also applied to field players using the custom catch-glue
dribble mechanism. The result looked like automatic alignment after a drop/catch and
would not occur on a real robot.

There are two independent simulator source trees in this workspace:

1. the external stock-server tree used by `launch_infer --env sim-only`;
2. the embedded-server tree compiled into the `rcssserver_embedded` Python extension.

Patching only the external tree did not affect `infer.py --env sim-embedded`.

#### Fix

In both active `Player::goalieCatch()` implementations, successful catch correction is
now conditional on `this->isGoalie()`:

- goalkeepers retain stock position, heading, and velocity correction;
- field players retain their existing position, heading, and velocity;
- `M_stadium.ballCaught(*this)` still runs for both, preserving ownership and catch-glue;
- catch geometry, probability, faults, release, and goalie behavior remain unchanged.

The embedded wheel was rebuilt and force-reinstalled into the `rcai` Python 3.11
environment. A native non-overlapping catch smoke test measured exactly `0.0 m` pose
change and `0.0°` heading change, then confirmed that subsequent dash movement still
transported the glued ball.

The following 3,000-step embedded inference independently confirmed across 30 catches:

- heading change at catch: exactly 0°;
- mean position change: 0.23 mm;
- maximum position change: 3.54 mm (ordinary residual motion, not a snap);
- no catch teleport or automatic body alignment.

#### Code locations

| File | Lines after change | Change |
|---|---:|---|
| `robocup_downloads/rcssserver-19.0.0/src/player.cpp` | 1687–1708 | Goalkeeper-only pose correction in the external server. |
| `robocup_downloads/rcssserver/src/player.cpp` | 1691–1715 | Equivalent goalkeeper-only correction in the embedded source tree. |
| `robocup_downloads/rcssserver/dist/rcssserver_embedded-0.1.0-cp311-cp311-macosx_26_0_arm64.whl` | binary artifact | Rebuilt embedded simulator wheel. |
| `docs/DRIBBLE_TO.md` | §11–§12 | Remove stale catch-teleport/forced-heading claims and document physical turning. |

### Change 3 — Correct Stage 2g reward geometry and PPO credit assignment

#### Problems found

The reward audit found several interacting mathematical errors:

1. **Delayed target credit:** `(Dx,Dy)` was sampled at dribble commitment, but target
   quality was paid only after catch verification. At the new 20°/s rate, a 90-step
   alignment leaves only `(γλ)^90 = (0.99×0.95)^90 ≈ 0.004` of GAE credit at the target
   action.
2. **Ignored-action credit:** while a macro was committed, PPO continued sampling new
   primitives and parameters even though the phase machine ignored them and used the
   latched target.
3. **Unreachable target valuation:** one legal carry moves roughly 0.85 m, but the
   target reward valued coordinates up to 10 m forward and 6 m lateral. In inference,
   hypothetical target deltas were about 6–11× larger than achieved deltas.
4. **One-sided reward:** `max(0, target_delta)` gave bad target parameters a flat zero
   signal instead of a gradient away from the bad region.
5. **Acquisition farming:** `dribble_active_bonus` fired whenever a target existed,
   including GRAB/SETTLE/VERIFY, while proximity bonuses could make waiting profitable.
6. **Wrong kick origin:** projected kick crossing used the robot pose even though the
   trajectory starts at the ball. Recent traces showed mean projection error of
   0.16–0.28 m and maxima near 0.9 m.
7. **Post discontinuity:** goalie-gap quality could be near one immediately inside a
   post and exactly zero at the post, encouraging unsafe post-seeking shots.
8. **Weak failure cost:** a failed kick could earn approximately +12 to +14 aim reward
   while paying only -1.5 for out-of-bounds and zero for a penalty-area off-target
   termination.
9. **Clipped-Gaussian likelihood mismatch:** PPO stored the Gaussian likelihood of an
   unclipped sample while the environment received `clip(sample,-1,1)`. Different
   latent samples could therefore cause the same action but receive different PPO
   probabilities.
10. **Diagnostic double counting:** achieved-gap, combo, and selected penalties were
    manually added to `total_rewards` and then included again when the complete step
    reward was accumulated. Training received the correct scalar, but episode logs did
    not.

#### Reward and geometry fixes

- Added `reachable_gap_delta()`. It projects the selected direction only as far as the
  legal 0.85 m carry endpoint and returns a signed gap change.
- A fresh, valid, in-range `dribble_to` commitment now receives the target-quality
  reward immediately:

  ```text
  target reward = weight × clip(Q(reachable endpoint) - Q(current ball), -1, 1)
  ```

- Removed the positive-only clamp. A target that worsens the reachable shot now gets a
  negative immediate parameter gradient.
- Retained achieved-gap reward at carry close so execution is still judged from the
  realized ball position.
- Added explicit `dribble_committed` telemetry and separate
  `Dribble target committed` / `Dribble carry opened` log events.
- Gated `dribble_active_bonus` on verified `is_dribbling`; acquisition and alignment no
  longer receive this bonus merely because a target exists.
- Kick projection now uses `(ball_x, ball_y)` as its ray origin consistently for kick
  gating, aim diagnostics, bad-aim handling, goalie-gap reward, and combo scaling.
- Added `post_safe_goalie_gap_quality()`: quality is unchanged through `|y|≤4` for the
  Stage 2g goal and tapers continuously to zero from `|y|=4` to the post at `|y|=5`.
- Added `kick_opposite_keeper_side`, corrected predicted-Y, and kick-gap diagnostics to
  debug traces.
- Added configurable off-target terminal penalty and corrected episode-total
  accumulation so each reward component is counted once.

#### PPO macro-credit fixes

- A committed dribble macro now masks all live primitives except:
  - `dribble_to`, interpreted as parameterless continuation;
  - `kick`, the permitted interrupt.
- `_mask_latched_dribble_params()` zeros the continuation transition's parameter mask
  and old parameter log-probability. Fresh `(Dx,Dy)` samples that the macro ignores can
  no longer receive PPO credit.
- The initial commitment remains the only transition whose `(Dx,Dy)` likelihood is
  trained against the immediate target-choice reward.
- Replaced hard clipping with a tanh-squashed Gaussian. PPO now stores and recomputes:

  ```text
  log π(a) = log Normal(z; μ,σ) - log(1 - tanh(z)² + ε)
  a = tanh(z)
  ```

  so bounded actions and likelihoods describe the same distribution.

#### Stage 2g reward values

```json
{
  "has_ball_bonus": 0.0,
  "near_ball_bonus": 0.02,
  "step_bonus": -0.05,
  "dribble_target_progress_weight": 0.5,
  "dribble_target_progress_clip": 0.3,
  "dribble_target_quality_weight": 4.0,
  "dribble_achieved_gap_weight": 4.0,
  "dribble_active_bonus": 0.01,
  "goal_post_safety_margin": 1.0,
  "ball_out_of_bounds_penalty": 5.0,
  "ball_in_penalty_off_target_penalty": 5.0
}
```

This also resolves the mismatch where the Stage 2g description specified target-quality
weight 4 while the live configuration used 2.

#### Code locations

| File | Lines after change | Change |
|---|---:|---|
| `ai_interface/envs/reward.py` | 89–142 | Post-safety and off-target configuration; verified-carry intermediate. |
| `ai_interface/envs/reward.py` | 290–306 | Continuous post-safe goalie-gap quality. |
| `ai_interface/envs/reward.py` | 389–410 | Signed one-segment reachable-gap calculation. |
| `ai_interface/envs/reward.py` | 563–570, 662–670 | Apply post safety to dense aim and gate active bonus on verified carry. |
| `ai_interface/envs/JAL_env.py` | 395–420 | Restrict a committed macro to continuation or kick. |
| `ai_interface/envs/JAL_env.py` | 770–840 | Ball-origin kick projection reward and post-safe goalie-gap bonus. |
| `ai_interface/envs/JAL_env.py` | 850–930 | Immediate signed reachable-target reward and achieved-gap accounting. |
| `ai_interface/envs/JAL_env.py` | 1020–1080 | Terminal penalties and single-count episode reward accumulation. |
| `ai_interface/envs/JAL_env.py` | 1535, 1799, 1951 | Fresh commitment detection and telemetry. |
| `ai_interface/envs/JAL_env.py` | 2085–2110 | Ball-origin kick projection helpers. |
| `ai_interface/trainers/ppo_jal_curriculum_trainer.py` | 61–73, 649, 890 | Remove ignored continuation-parameter credit in single/parallel training. |
| `ai_interface/algorithms/ppo_jal.py` | 597–626, 752–772 | Tanh-squashed sampling and Jacobian-corrected PPO likelihood/entropy. |
| `configs/ppo_jal_curriculum_config.json` | 1000–1035 | Stage 2g reward rebalance and failure penalties. |
| `infer.py` | 946–970 | Commitment, predicted-Y, gap-quality, and keeper-side trace fields. |
| `tests/test_stage2g_reward_fixes.py` | 1–82 | Geometry, post safety, signed endpoint, and carry-gating tests. |
| `tests/test_ppo_bounded_params.py` | 1–27 | Bounded-action and finite-likelihood regression test. |
| `tests/test_action_accounting.py` | 51–67 | Ensure latched continuation parameters receive zero credit. |
| `docs/DRIBBLE_TO.md` | §12 | Document Stage 2g reward and macro-credit semantics. |

### Change 4 — Correct requested/executed action accounting

#### Problem

Inference and trainer summaries iterated over the policy's full `a_max` output and
counted inactive padded slots as real robot actions. They also conflated the primitive
requested by the policy with the primitive executed by an active macro or fallback.
This produced totals up to five times the number of environment steps and misleading
dribble/turn percentages.

#### Fix

- Inference now counts only `info["action_info"]["per_robot"]`, which contains real
  controlled robots.
- Requested and executed counters are separate:
  - `requested_action_type`: categorical policy selection;
  - `action_type`: primitive actually executed after runtime masks/macros/fallbacks.
- Window summaries, final logs, and `summary.json` expose both distributions.
- PPO trainer diagnostics use `_accumulate_active_primitives()` and the
  `agent_active_mask`, excluding inactive `a_max` padding in both single- and
  multi-environment rollout loops.
- This changes reporting only; PPO losses and learned behavior are unaffected.

The first verification run produced exactly 3,000 requested and 3,000 executed actions
for 3,000 single-robot steps. Differences between categories were legitimate macro
behavior—for example, requested `turn` transitions becoming executed `dribble_to`
continuations—not accounting inflation.

#### Code locations

| File | Lines after change | Change |
|---|---:|---|
| `infer.py` | 54–71 | Accumulate requested/executed actions from real per-robot action info. |
| `infer.py` | 800–877 | Separate PPO inference counters and remove padded-slot counting. |
| `infer.py` | 990–1048 | Requested window logs and requested/executed summary JSON. |
| `ai_interface/trainers/ppo_jal_curriculum_trainer.py` | 39–58 | Active-slot-only primitive accumulator. |
| `ai_interface/trainers/ppo_jal_curriculum_trainer.py` | 653–655, 892–894 | Use active masks in single/parallel diagnostics. |
| `tests/test_action_accounting.py` | 1–67 | Requested/executed separation, inactive padding, multi-agent, and macro-param tests. |

### Consolidated validation

- Native embedded catch test: field-player pose delta `0.0 m`, heading delta `0.0°`,
  catch-glue transport preserved.
- External rcssserver build completed successfully after its catch change.
- Embedded CPython 3.11 wheel rebuilt and installed successfully.
- Post-change 3,000-step inference verified the 20°/s command cap, zero catch heading
  correction, and exact 3,000-action accounting.
- Reward/PPO focused suite: **36/36 tests passed**.
- Earlier catch/dribble/accounting focused regressions also passed.
- Python compilation passed for all modified Python files.
- `ppo_jal_curriculum_config.json` passed JSON validation.
- `git diff --check` passed.

An optional final 250-step embedded smoke run after the complete Change 3 reward rewrite
could not be launched because the execution approval service returned a 401 authentication
error. This was an orchestration failure before process creation, not a simulator or code
failure. The next Stage 2g retraining/inference run is therefore the remaining live
end-to-end validation. Existing Stage 2g checkpoints are not directly comparable because
the reward definition, turn dynamics, catch dynamics, macro action contract, and bounded
parameter distribution have all changed.

---

## 2026-06-23 11:32 IST — Stage 2h kick/dribble stabilization, fine-tune setup, and parallel inference comparison

### Problem

After applying the real-robot turn-rate constraint, Stage 2h exposed several coupled
failure modes:

1. A kick request could require many simulator cycles of alignment before the ball was
   fired. The policy could interrupt that alignment by sampling another primitive, so
   valid shooting opportunities were lost.
2. The model sometimes shot toward the keeper's occupied side or continued holding a
   stale shot target while the keeper moved during the slow alignment.
3. Dribble targets near the opponent penalty area could carry wide balls into the
   penalty region outside the goal mouth, causing `ball_in_penalty_off_target`.
4. Gap-quality dense reward peaked close to the keeper, so the reward gradient could
   pay the policy to over-dribble into the keeper even though the terminal
   `goalie_catch` penalty remained correct.
5. The existing inference launcher was not convenient for comparing several Stage 2h
   checkpoints under identical embedded-simulator conditions.
6. A full Stage 2h retrain from the Stage 2c checkpoint was performing poorly, so the
   known-good Stage 2h checkpoint needed to be preserved and fine-tuned safely without
   overwriting it.

### Fix

- Added a committed kick macro in `JAL_env.py`.
  - A real `kick` selection latches a target and keeps executing `kick` until
    `basic_commands.kick()` finishes internal alignment and fires.
  - Non-kick policy samples during the macro are recorded as requested primitives but
    executed as `kick_macro_continuation`.
  - The macro aborts if the robot loses ball eligibility or another claimant owns the
    ball.
  - The final command is still passed through `limit_turn_rate()`, so alignment uses
    turn-limited simulator commands rather than an instant heading correction.

- Added keeper-away kick targeting.
  - `kick_keeper_away_target_y` selects a deterministic in-mouth y target away from
    the keeper side.
  - A one-time retarget guard can flip the target if the keeper moves onto the selected
    side while the kick macro is still aligning.
  - Stage 2h uses `kick_keeper_away_target_y: 3.25`,
    `kick_keeper_retarget_max_count: 1`,
    `kick_keeper_retarget_min_gap_quality: 0.45`,
    `kick_keeper_retarget_same_side_y: 1.0`, and
    `kick_keeper_retarget_min_improvement: 0.05`.

- Added a penalty-area dribble guard.
  - When a decoded `dribble_to` target is near the opponent penalty-area x boundary,
    its lateral target is clamped to `|y| <= 3.5`.
  - This prevents wide, late dribbles from carrying the ball into the penalty area
    outside the goal mouth.
  - Debug traces expose `dribble_penalty_guard_applied`.

- Added keeper catch-zone suppression for dense reward.
  - `_keeper_zone_factor(point, goalie_pose)` ramps from `keeper_zone_floor` at the
    keeper pose to `1.0` at `keeper_zone_radius`.
  - Stage 2h uses `keeper_zone_radius: 4.0` and `keeper_zone_floor: 0.0`.
  - The factor is applied to the dense gap-scaled terms only:
    `kick_aim`, dribble→kick combo, dribble target quality, and achieved-gap reward.
  - Terminal `goal_reward: +70` and `goalie_catch: -70` remain unchanged.

- Expanded inference diagnostics.
  - `infer.py --debug_infer` now logs fired-kick probe events with fire origin,
    distance to keeper, predicted y at the goal line, aim quality, gap quality,
    keeper-zone factor, target side, macro state, retarget state, and terminal outcome.
  - `summary.json` includes shot-probe distance-bin summaries so catch/off-target rates
    can be tied to fire distance rather than inferred from video alone.

- Updated `launch_infer.py` to compare multiple checkpoints in parallel.
  - For `sim-only`, it starts one external server per model with separate port pairs.
  - For `sim-embedded`, it starts one independent `infer.py` subprocess per model; each
    subprocess owns its own embedded simulator instance.
  - At completion it reads each run's `summary.json` and prints a comparison table.

- Preserved the last known-good Stage 2h checkpoint before fine-tuning:
  - from
    `models/ppo_jal_expandable_wide/stage2h_newphys_pm10_complete.pt`
  - to
    `models/ppo_jal_expandable_wide/stage2h_newphys_pm10_complete_preserved_20260623_101555_IST.pt`
  - The copy was verified byte-for-byte.

- Converted the curriculum config to a conservative Stage 2h fine-tune setup.
  - Disabled the original `stage2h_newphys_pm10` entry by setting `timesteps: 0`.
  - Added active `stage2h_newphys_pm10_finetune` with a 50k-step budget.
  - `load_model` now points at the preserved Stage 2h checkpoint.
  - `save_path` now points at `models/ppo_jal_expandable_wide_finetune`.
  - PPO settings were made conservative for adaptation rather than relearning:
    `target_kl: 0.008`, `learning_rate_initial: 0.0001`,
    `learning_rate_final: 0.00003`, `ent_coef_initial: 0.005`,
    `ent_coef_final: 0.001`.

### Code locations

| File | Lines after change | Change |
|---|---:|---|
| `ai_interface/envs/JAL_env.py` | 300–313 | Fired-kick probe storage and committed kick-macro state. |
| `ai_interface/envs/JAL_env.py` | 861–908 | Apply keeper-zone factor to kick aim dense reward. |
| `ai_interface/envs/JAL_env.py` | 913–930 | Store per-kick shot probe diagnostics. |
| `ai_interface/envs/JAL_env.py` | 986–999 | Apply keeper-zone factor to dribble target-quality reward. |
| `ai_interface/envs/JAL_env.py` | 1030–1045 | Apply keeper-zone factor to achieved-gap reward. |
| `ai_interface/envs/JAL_env.py` | 1111–1114 | Apply keeper-zone factor to dribble→kick combo scaling. |
| `ai_interface/envs/JAL_env.py` | 1638–1646 | Per-step kick macro and retarget telemetry variables. |
| `ai_interface/envs/JAL_env.py` | 1735–1747 | Continue or abort an active committed kick macro. |
| `ai_interface/envs/JAL_env.py` | 1916–1935 | Preserve committed dribble ownership while allowing kick to pre-empt. |
| `ai_interface/envs/JAL_env.py` | 1936–2017 | Latch keeper-away target, retarget during alignment, call `basic_commands.kick()`. |
| `ai_interface/envs/JAL_env.py` | 2141–2179 | Apply turn-rate limit and expose macro/dribble diagnostics in `per_robot`. |
| `ai_interface/envs/JAL_env.py` | 2297–2303 | Reset committed kick macro state. |
| `ai_interface/envs/JAL_env.py` | 2336–2374 | Compute keeper-away target y. |
| `ai_interface/envs/JAL_env.py` | 2376–2441 | One-time keeper-aware kick retarget guard. |
| `ai_interface/envs/JAL_env.py` | 2443–2469 | Keeper catch-zone dense-reward suppression helper. |
| `ai_interface/envs/reward.py` | 117–125 | Penalty-area dribble guard config knobs. |
| `ai_interface/envs/reward.py` | 150–163 | Keeper-away kick target and retarget config knobs. |
| `ai_interface/envs/reward.py` | 197–208 | Keeper catch-zone suppression config knobs. |
| `infer.py` | 92–132 | Shot-probe distance-bin summarizer. |
| `infer.py` | 943–967 | Capture fired-kick probe events during inference. |
| `infer.py` | 986–1072 | Step-trace fields for kick macro, retarget, dribble guard, and keeper-zone diagnostics. |
| `launch_infer.py` | 1–18 | Document parallel external/embedded inference behavior. |
| `launch_infer.py` | 202–233 | Print final checkpoint comparison table from `summary.json`. |
| `launch_infer.py` | 267–300 | Launch independent embedded inference subprocesses without shared ports. |
| `launch_infer.py` | 376–414 | Forward model, stage, config, debug, noise, and port/backend args to each `infer.py`. |
| `configs/ppo_jal_curriculum_config.json` | 1041–1136 | Historical Stage 2h entry disabled, with final Stage 2h reward/mechanics retained. |
| `configs/ppo_jal_curriculum_config.json` | 1138–1233 | Active `stage2h_newphys_pm10_finetune` entry. |
| `configs/ppo_jal_curriculum_config.json` | 1247–1263 | Fine-tune save/load path and conservative PPO schedule. |
| `docs/TRAINING.md` | §36 | Logged the completed Stage 2h fine-tune run and outcome. |

### Validation and observed outcome

- Config validation passed with `python -m json.tool`.
- The preserved checkpoint copy was verified byte-for-byte.
- `git diff --check` passed for the updated config and docs after the fine-tune setup.
- The completed fine-tune training run was logged from
  `train_logs/20260623_105445_993837/train_log.log`:
  - 49,890 / 50,000 steps completed.
  - 280 total episodes.
  - 51.1% overall goal rate.
  - 54.0% goal rate over the last 100 episodes.
  - 23.2% goalie catches.
  - 22.1% out-of-bounds.
  - 3.6% penalty off-target.
  - No `max_steps` outcomes.

- Parallel embedded inference with `--ppo_param_noise_std 0.3` produced:

| Model | Episodes | Goals | Goal rate | Avg reward | Main failures |
|---|---:|---:|---:|---:|---|
| preserved Stage 2h baseline | 58 | 48 | 82.8% | 95.3 | 6 OOB, 4 off-target, 0 catches |
| fine-tune 10k | 58 | 44 | 75.9% | 86.0 | 5 catches, 6 OOB, 3 off-target |
| fine-tune 20k | 59 | 42 | 71.2% | 83.1 | 5 catches, 8 OOB, 4 off-target |
| fine-tune 30k | 47 | 33 | 70.2% | 82.5 | 7 catches, 6 OOB, 1 off-target |
| fine-tune 40k | 59 | 45 | 76.3% | 92.7 | 0 catches, 11 OOB, 3 off-target |
| fine-tune 50k | 57 | 39 | 68.4% | 78.7 | 8 catches, 7 OOB, 3 off-target |
| fine-tune complete | 64 | 36 | 56.2% | 67.3 | 9 catches, 14 OOB, 5 off-target |

Conclusion: the preserved Stage 2h baseline remains the best model and should not be
overwritten. The fine-tune checkpoints are useful diagnostics but should not be promoted.
The 40k checkpoint is the strongest fine-tune candidate, but it still underperforms the
preserved baseline due to out-of-bounds drift.

---

## 2026-06-24 12:30 IST — ±15 generalization of the preserved Stage 2H checkpoint via two no-retrain geometric fixes (kick aim gate + penalty-area entry guard)

### Problem

The preserved Stage 2H checkpoint
(`models/ppo_jal_expandable_wide/stage2h_newphys_pm10_complete_preserved_20260623_101555_IST.pt`)
scored 82.8% at the trained ±10 ball-y spawn but degraded to **64.3%** when the spawn was
widened to the full ±15 field width (8k-step embedded inference, `--ppo_param_noise_std 0.3`).
The shot-probe distance bins isolated two independent failure modes, both purely geometric:

1. **Wide-angle shots sail out of bounds.** All OOB losses came from a single fire-distance
   band, 6–8 units from the keeper: 13 shots, **38.5% goals, 8 OOB, mean aim 0.27** — versus
   0.44+ aim and 83–100% goals in every other band. From a wide ±15 ball at mid-range the
   policy sees an open goal (`gap_quality_at_fire` 0.78) but cannot align under the 20°/s turn
   cap, so it fires a low-aim shot that misses wide/long. The fire path had no aim gate.

2. **Wide dribbles cross the penalty line off-mouth.** In the gated ±15 run, **zero** of the
   `ball_in_penalty_off_target` episodes had a fired shot in the probe — they were all dribble
   carries. The existing penalty-area guard clamped only the target *y*, but a ball that is
   still wide (|y|>5) carried toward a y-clamped target inside the box still crosses
   x=`_RIGHT_PENALTY_AREA_X` (35) at a wide y MID-segment (e.g. (30,12)→(40,3.5) crosses x=35
   at y≈7.8), terminating off-target before the carry can pull it central.

### Fix

Both fixes are geometric, env-resident (so they apply to the existing checkpoint at inference
*and* would shape any future training), and require **no retraining**.

- **Kick aim gate** (already present in `JAL_env.py` but unplumbed/disabled). A fired kick is
  vetoed — held as a turn while the committed kick macro keeps aligning — whenever projected
  aim quality < `kick_min_aim_quality`. The knobs (`kick_requires_aim`, `kick_min_aim_quality`)
  existed on the env constructor but were not passed by the PPO inference build or the
  curriculum trainer, so the gate defaulted off. Added the config passthrough to both, and
  enabled `kick_requires_aim: true`, `kick_min_aim_quality: 0.30` on the Stage 2H stage.
  Threshold 0.30 chosen to veto the bad 6–8-unit band (mean aim 0.27) while sparing the good
  bands (0.36–0.51).

- **Penalty-area ENTRY guard** (rewrote the existing target-y-only guard in `JAL_env.py`).
  While the ball is still wide of the goal mouth (|y| > `_GOAL_HALF_HEIGHT`), the carry
  target's *x* is now capped to the penalty boundary (`_RIGHT_PENALTY_AREA_X -
  dribble_penalty_area_guard_margin`), so the carry pulls the ball toward centre OUTSIDE the
  box first; entry past the boundary resumes only once |y| is within the mouth. The existing
  y-clamp to `dribble_penalty_area_y_clip` is preserved for targets at/inside the boundary.
  No new config knobs — reuses the active `dribble_penalty_area_guard_margin: 1.0` and
  `dribble_penalty_area_y_clip: 3.5`.

### Code locations

| File | Lines after change | Change |
|---|---:|---|
| `ai_interface/envs/JAL_env.py` | 1878–1918 | Penalty-area ENTRY guard: cap carry target-x to the boundary while the ball is wide; preserve target-y clamp at/inside the boundary. |
| `infer.py` | 771–772 | Pass `kick_requires_aim` / `kick_min_aim_quality` from stage config into the PPO `JALTeamEnv` build. |
| `ai_interface/trainers/ppo_jal_curriculum_trainer.py` | 272–273 | Same passthrough in the trainer `_build_env` (train/infer consistency). |
| `configs/ppo_jal_curriculum_config.json` | `stage2h_newphys_pm10_finetune` | Enable `kick_requires_aim: true`, `kick_min_aim_quality: 0.30`. |

### Validation and observed outcome

Embedded inference on the preserved checkpoint, `--ppo_param_noise_std 0.3`:

| Spawn | Fixes | Steps | Eps | Goal rate | OOB | Off-target | Catch |
|---|---|---:|---:|---:|---:|---:|---:|
| ±10 | none (baseline) | — | 58 | 82.8% | ~6 | ~4 | 0 |
| ±15 | none | 8k | 42 | 64.3% | 9 | 6 | 0 |
| ±15 | + aim gate only | 8k | 44 | 68.2% | 3 | 10 | 1 |
| ±15 | + aim gate + entry guard | 8k | 38 | 97.4% | 0 | 1 | 0 |
| **±15** | **+ both (confirmation)** | **20k** | **85** | **89.4%** | **4** | **0** | **4** |
| **±10** | **+ both (confirmation)** | **10k** | **44** | **95.5%** | **1** | **1** | **0** |

- The aim gate moved the 6–8-unit band from 38.5% → **94.4%** goals (mean aim 0.27 → 0.44) and
  cut total OOB 9 → 3.
- The entry guard eliminated the off-target dribbles (10 → 0 in the 20k run).
- Both fixes hold at scale and neither regresses ±10 (82.8% → 95.5%).
- Net: the ±10-only ~83% checkpoint now generalizes to **~89% at full ±15 width with no
  training** — the Stage 2 widening target, achieved geometrically.
- Remaining ±15 failures are evenly split (4 OOB + 4 catches over 85 eps); no single dominant
  mode left.
- `JAL_env.py` compiles; `graphify update .` run after the edit.

---

## 2026-06-25 IST — Stage 3 v2 defender-scoring reward, SSL rule events, and dribble segment fix

### Problem

After upgrading the Stage 3 hardcoded defender from a passive blocker to an active threat defender,
the existing Stage 3 attacker checkpoint struggled to score. Embedded inference showed the policy
mostly kept selecting `dribble_to`/approach behavior and rarely fired useful shots. The old Stage 3
reward still had several mismatches:

1. **Defender-aware dribble reward was too narrow.** Stage 3 target quality discounted only the
   straight lane to goal centre. That teaches the attacker to open the centre lane, not to find any
   legal in-mouth shot lane around the defender.

2. **Stage 3 missed later Stage 2H real-physics fixes.** The old Stage 3 config still used older
   dribble incentives and did not include the proven Stage 2H keeper-zone, keeper-away, retarget,
   penalty-area, and achieved-gap settings.

3. **Dribble distance math used the wrong scale.** The SSL Division B field is 9m x 6m, while the
   embedded simulator uses 90 x 60 units. Therefore **1 real meter = 10 simulator units**. The
   previous `0.85` dribble segment limit was 8.5 cm, not a 15 cm safety margin below the SSL
   1 m excessive-dribbling limit.

4. **The simulator resolves contact but does not enforce SSL foul semantics.** Robot overlap/collision
   is physically separated by rcssserver, but attacker training had no explicit event/reward signal for
   crashing, pushing, no-progress deadlocks, excessive dribbling, or defense-area touches.

5. **Follow-up runtime bug.** The first implementation passed `dribble_segment_limit` into
   `dribble_to()` from `_action_to_commands()` but had only defined it in `step()`, causing:

   ```text
   NameError: name 'dribble_segment_limit' is not defined
   ```

### Fix

#### Reward quality

Added Stage 3 shot-quality helpers in `ai_interface/envs/reward.py`:

- `safe_goal_target_ys(...)`
- `best_defender_lane_quality(...)`
- updated `positional_shot_quality(...)`

The Stage 3 quality now checks the best clear defender lane to safe in-mouth targets:

```text
targets = {0, -kick_keeper_away_target_y, +kick_keeper_away_target_y}
Q3(point) = positional_gap_quality(point, goalie_y) * best_defender_lane_quality(point)
```

This lets the attacker get reward for opening either side of the defender, not only the centre lane.

#### Dribble reward and combo reward

Updated `ai_interface/envs/JAL_env.py` so Stage 3 uses combined keeper+defender shot quality:

- Dribble target quality now evaluates the reachable endpoint with `positional_shot_quality(...)`
  when `use_defender_lane_gate=true`.
- Achieved-gap reward now stores/compares the realized combined Stage 3 shot quality at carry open
  and carry close.
- Fired kicks now record `kick_defender_lane_clear`.
- Post-dribble kick combo now scales by:

```text
goalie_gap_quality * defender_lane_clear * keeper_zone_factor
```

So a dribble followed by a shot through the defender earns near-zero combo reward.

#### Corrected dribble segment unit conversion

Added `dribble_segment_limit` to `RewardConfig` and wired it into both:

- reward lookahead endpoint calculation
- the actual `dribble_to(..., segment_limit=...)` macro call

Stage 3 v2 sets:

```json
"dribble_segment_limit": 8.5
```

Math:

```text
Division B field: 9m x 6m
Embedded sim:     90 x 60 units
Scale:            10 units / real meter
SSL dribble max:  1m = 10 units
Chosen segment:   8.5 units = 0.85m, leaving 0.15m safety margin
```

#### SSL rule event tracker

Added `ai_interface/envs/ssl_rule_events.py`, an env-local rule detector with:

- `SSLRuleConfig`
- `SSLRuleEvent`
- `SSLRuleState`
- `SSLRuleTracker`
- pure helpers for crash, pushing, no-progress, and excessive-dribble detection

Events currently wired into `JAL_env.step()`:

| Event | Training consequence |
|---|---:|
| `attacker_crash` | one-shot `-8`, continue |
| `bot_crash_drawn` | one-shot `-4`, continue |
| `defender_crash` | `0`, continue |
| `attacker_push_foul` | `-20`, terminate |
| `defender_push_foul` | `0`, terminate/reset |
| `no_progress_forced_start` | terminal; `-10` only if attacker owned most contested frames |
| `attacker_excessive_dribble` | `-20`, terminate |
| `attacker_touched_ball_in_defense_area` | `-8`, continue |
| `defender_in_defense_area` | `0`, terminate/reset |

Rule math:

```text
Crash threshold:       1.5 m/s = 1.5 sim units/step
Drawn crash diff:      0.3 m/s = 0.3 sim units/step
Robot contact radius:  2 * PLAYER_SIZE = 1.8 units
Ball contact radius:   PLAYER_SIZE + BALL_SIZE = 1.115 units
No progress window:    100 steps = 10 seconds in Division B
```

No-progress was tightened to require a majority contested window before firing, rather than one
contested frame inside the window.

#### New active Stage 3 v2 curriculum entry

Updated `configs/ppo_jal_curriculum_config.json`:

- Added `stage3_defender_v2_finetune`
- Set old `stage3_defender` to `timesteps: 0` to preserve it as history
- Set old `stage2h_newphys_pm10_finetune` to `timesteps: 0`
- Set top-level load model to:

```text
models/ppo_jal_expandable_wide/stage3_defender_steps400000.pt
```

- Set save path to:

```text
models/ppo_jal_expandable_wide_stage3_v2
```

- Active stage now:

```text
stage3_defender_v2_finetune: 200000 timesteps
```

Key Stage 3 v2 settings:

- `disabled_actions: ["goto", "turn"]`
- `dribble_segment_limit: 8.5`
- `dribble_target_quality_weight: 5.0`
- `dribble_achieved_gap_weight: 5.0`
- `dribble_target_progress_weight: 0.3`
- `dribble_target_progress_clip: 0.2`
- `dribble_active_bonus: 0.0`
- `post_dribble_kick_bonus: 3.0`
- `post_dribble_kick_combo_window: 6`
- `goal_post_safety_margin: 1.0`
- `keeper_zone_radius: 4.0`
- `keeper_zone_floor: 0.0`
- `kick_keeper_away_target_y: 3.25`
- one retarget allowed via `kick_keeper_retarget_max_count: 1`
- PPO entropy start increased to `ent_coef_initial: 0.008`

#### Runtime `NameError` fix

Fixed the follow-up crash by adding the same local lookup inside `_action_to_commands()`:

```python
dribble_segment_limit = float(
    getattr(self.reward_config, "dribble_segment_limit", 0.85)
)
```

This keeps the reward-side lookup in `step()` and the command-side lookup in `_action_to_commands()`
separate and in scope.

### Code locations

| File | Change |
|---|---|
| `ai_interface/envs/reward.py` | Added `dribble_segment_limit`, safe goal target helpers, best defender lane quality, and defender-aware positional shot quality. |
| `ai_interface/envs/JAL_env.py` | Wired Stage 3 combined-quality dribble rewards, combo scaling, SSL rule events, and dribble segment limit into `dribble_to()`. |
| `ai_interface/envs/ssl_rule_events.py` | New SSL rule event tracker for crash, pushing, no-progress, excessive dribbling, and defense-area touches. |
| `configs/ppo_jal_curriculum_config.json` | Added active `stage3_defender_v2_finetune`, preserved old stages as disabled, updated load/save paths and PPO entropy. |
| `tests/test_stage3_reward_rules.py` | Added tests for Q3 lane quality, dribble unit math, crash threshold/drawn crash, pushing, and no-progress. |

### Validation

Targeted tests:

```bash
python -m pytest tests/test_defender.py tests/test_stage3_reward_rules.py
```

Result:

```text
19 passed
```

Compile / config validation:

```bash
python -m py_compile ai_interface/envs/JAL_env.py ai_interface/envs/reward.py ai_interface/envs/ssl_rule_events.py
python -m json.tool configs/ppo_jal_curriculum_config.json
```

Both passed.

Active curriculum check:

```text
active stages:
stage3_defender_v2_finetune 200000
load_model models/ppo_jal_expandable_wide/stage3_defender_steps400000.pt
save_path models/ppo_jal_expandable_wide_stage3_v2
ent 0.008 0.001
```

## 2026-06-26 IST — SSL pushing threshold and non-goalie defense-area filter

### Context

Post-training embedded inference for Stage 3 v2 showed a few `attacker_push_foul` terminations very
soon after catch/verify contact. The SSL rule defines pushing as sustained contact while exerting
force, but our detector used only `3` contact frames, which is `0.3s` at the embedded step rate.
The same review found that `defender_in_defense_area` checked every opponent, including the
scripted goalie, even though the SSL field-player defense-area restriction should not be applied to
the keeper.

### Changes

- Raised `SSLRuleConfig.pushing_contact_steps` from `3` to `8`.
- Added push-event details: attacker/opponent IDs, contact duration, movement projections, step
  vectors, positions, and threshold values.
- Added `opponent_goalie_ids` to `SSLRuleConfig`.
- Updated `defender_in_defense_area` to ignore configured opponent goalies and only flag non-goalie
  opponent robots touching the ball in their defense area.
- Wired opponent goalie IDs from `aux_team_policies` into `JALTeamEnv` through both PPO curriculum
  training and PPO inference.

### Validation

```bash
python -m py_compile ai_interface/envs/ssl_rule_events.py ai_interface/envs/JAL_env.py ai_interface/trainers/ppo_jal_curriculum_trainer.py infer.py
python -m pytest tests/test_defender.py tests/test_stage3_reward_rules.py
```

Result:

```text
21 passed
```

## 2026-06-26 IST — Physical front-reception kick gate

### Context

Stage 3 v2 inference exposed a simulator exploit: during contested dribbles, the attacker could
turn to its desired shot heading and issue `kick` while the ball was still only radially close, even
when the ball center was behind or outside the physical dribbler mouth. The embedded server's stock
`ballKickable()` check was distance-only, so `kick` could clear catch-glue and push the ball in the
robot heading without requiring realistic ball-mouth geometry.

### Changes

- Added `FRONT_RECEPTION_CENTER_ANGLE_DEG = 100.0` in `ai_interface/constants/player_constants.py`.
  This uses the CAD/mechanical ball-center reception angle from the shared image, not the wider
  outside-of-ball-radius angle.
- Added `ball_in_front_reception_cone()` in `ai_interface/utils/basic_commands.py`.
- Updated Python `kick()` command generation to return `failed` unless the ball center is within
  the front reception cone and radial kickable distance.
- Updated `JALTeamEnv._action_to_commands()` so kick execution requires both radial possession and
  front-cone reception. Direct blocked kicks become `turn 0`, set `fallback_reason =
  "kick_blocked_bad_reception"`, and expose `ball_in_reception_cone` /
  `kick_blocked_bad_reception` in action diagnostics. This solves the direct primitive exploit
  where an attacker could fire a shot with the ball behind or beside the robot.
- If a blocked kick happens during an active committed dribble macro, the env resumes one
  geometric `dribble_to()` step against the latched target instead of freezing on `turn 0`.
  Diagnostics report `fallback_reason = "kick_blocked_bad_reception_to_dribble"` and
  `action_type = "dribble_to"`. This solves the bad fallback where a physically invalid kick
  during a carry could cancel useful geometric dribble recovery and stall the attacker.
- Fixed a PPO inference crash where Stage 3 v2 disabled `goto`/`turn`, a kick macro was active,
  and the ball slipped outside the reception cone. The runtime mask now exposes a legal recovery
  primitive (`dribble_to` while still near the ball, `approach_ball` after real possession loss)
  instead of producing an all-zero primitive row. This solves the
  `ValueError: primitive_valid_mask disabled every primitive for a slot` crash in `infer.py`.
- Updated embedded simulator/server source:
  `robocup_downloads/rcssserver/src/player.cpp` now uses
  `ballKickableInFrontReceptionCone()` inside `Player::kick()`, so the physics layer rejects
  behind-ball and side-ball kicks even if command generation misses them.
- Updated the original simulator/server source:
  `robocup_downloads/rcssserver-19.0.0/src/player.cpp` and `src/player.h` now apply the same
  `ballKickableInFrontReceptionCone()` gate in `Player::kick()`. This keeps sim-only/original
  server behavior aligned with embedded inference so the policy cannot learn a kick that only
  works in one simulator.
- Updated inference logging to report `blocked_kicks` and include reception-cone diagnostics in
  action samples / step traces.

### Validation

```bash
python -m py_compile ai_interface/constants/player_constants.py ai_interface/utils/basic_commands.py ai_interface/envs/JAL_env.py infer.py
/opt/anaconda3/envs/rcai/bin/python tests/test_ball_action_recovery.py
/opt/anaconda3/envs/rcai/bin/python tests/test_ppo_jal_expandable.py
```

Result:

```text
PPO-JAL ball-action recovery tests: 15 passed
PPO-JAL expandable tests: 10 passed
```

Embedded wheel rebuild:

```bash
/opt/anaconda3/envs/rcai/bin/python -m pip install --no-build-isolation --force-reinstall robocup_downloads/rcssserver
```

Result:

```text
Successfully built rcssserver-embedded
Successfully installed rcssserver-embedded-0.1.0
```

Native embedded smoke:

```text
behind-ball kick speed: 0.0
front-mouth kick speed: 3.55
```

Original simulator rebuild:

```bash
cd robocup_downloads/rcssserver-19.0.0
make -j4
```

Result:

```text
Built robocup_downloads/rcssserver-19.0.0/src/.libs/rcssserver
Build completed successfully; only existing compiler warning noise was emitted.
```

## 2026-06-26 IST — Stage 3 cone-adaptation training guards

### Context

After the physical front-reception cone gate, Stage 3 needs one more fine-tune from the original
Stage 3 base checkpoint. The main learned failure was not lack of time: successful episodes usually
finished quickly, while failures reached `max_steps` because the attacker opened a carry, delayed
release/shot conversion, drifted into the opponent defense area, or oscillated between kick-alignment
toward goal and dribble reacquisition back toward the ball.

### Changes

- Added a short kick-reception recovery lockout in `JALTeamEnv`.
  When a kick is blocked because the ball is outside the front reception cone, the env now resets the
  kick macro, forces the committed dribble state back to `GRAB` while preserving the latched target,
  and exposes only dribble/approach recovery for `8` steps. This solves the kick-vs-dribble tug of
  war where kick turns toward goal and dribble turns back toward the ball every other step.
- Tightened committed-dribble interruption rules.
  `kick` is masked and runtime-redirected during `GRAB`, `SETTLE`, `VERIFY`, and `RELEASE`; kick
  interruption is allowed only during `CARRY` / `ALIGN_RELEASE` and only when the ball is inside the
  front reception cone. This prevents shooting before the ball is physically acquired in the mouth.
- Added recovery diagnostics:
  `kick_reception_recovery_steps`, `kick_reception_recovery_to_dribble`,
  `kick_blocked_during_dribble_acquisition_to_dribble`, and
  `kick_blocked_bad_reception_to_dribble`.
- Added repeated attacker defense-area-touch escalation.
  `SSLRuleConfig.defense_area_touch_terminal_count` terminates the episode after the configured
  count of attacker-caused touches in the opponent defense area. Stage 3 v2 sets this to `3`, so one
  accidental touch is recoverable (`-8`) but repeated box carries become terminal.
- Added carry urgency shaping.
  `RewardConfig.carry_urgency_grace_steps`, `carry_urgency_penalty_per_step`, and
  `carry_urgency_penalty_max` apply a capped total penalty after a verified carry stays open too
  long. Stage 3 v2 uses `25` grace steps, `0.03` per late step, and a `2.25` cap. Math: after the
  grace window, the cumulative penalty is `min(2.25, 0.03 * late_steps)`, so the maximum total
  penalty is `2.25`, not a runaway per-frame ramp.
- Updated the active Stage 3 v2 fine-tune config for the final cone-adaptation run:
  `timesteps=300000`, `post_dribble_kick_bonus=5.0`,
  `post_dribble_kick_combo_window=10`, carry urgency enabled, repeated defense-area touch terminal
  at `3`, and `max_steps` left at `400`.

### Validation

```bash
/opt/anaconda3/envs/rcai/bin/python tests/test_ball_action_recovery.py
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/envs/JAL_env.py ai_interface/envs/reward.py ai_interface/envs/ssl_rule_events.py
python -m pytest tests/test_stage3_reward_rules.py
/opt/anaconda3/envs/rcai/bin/python tests/test_ppo_jal_expandable.py
python -m json.tool configs/ppo_jal_curriculum_config.json >/tmp/ppo_jal_config_check.json
```

Result:

```text
PPO-JAL ball-action recovery tests: 19 passed
Stage 3 reward/rule tests: 11 passed
PPO-JAL expandable tests: 10 passed
Config JSON parse: passed
```

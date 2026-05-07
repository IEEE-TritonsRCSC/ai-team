# Implementing HER (Hindsight Experience Replay) with TD3 in Stable Baselines 3

## Hindsight Experience Replay (HER)

### The Core Problem

Training robots to score goals is *hard*. A robot might attempt thousands of actions — running, kicking, positioning — and **almost never score**. From a reinforcement learning perspective, it receives almost no positive reward signal. The agent is essentially stumbling in the dark.

---

### The Key Insight: Learn from Failure by Reframing It

HER asks a simple but powerful question:

> *"Even though you didn't achieve your **actual** goal, what goal **did** you accidentally achieve?"*

If a robot kicked the ball to the left corner instead of scoring, HER says: **"Pretend your goal WAS the left corner — and you succeeded!"** That experience is then stored and learned from as a *success*.

---

### How It Works (Step by Step)

1. **Robot attempts an episode** — tries to score, fails, ball ends up at position X
2. **Normal replay buffer** stores the experience as a *failure*
3. **HER also stores a modified version** — relabeling the goal as position X, making it a *success*
4. **Both** experiences are used to train the policy

The robot now has **rich, meaningful signal** from every single attempt, even total failures.

---

### Why It Matters for Robot Soccer

| Without HER | With HER |
|---|---|
| Reward only on goals scored | Reward on every meaningful outcome |
| Millions of wasted episodes | Every episode teaches *something* |
| Sparse, frustrating signal | Dense, efficient learning |
| Robots take forever to learn basics | Faster skill acquisition |

In a **multi-robot** team setting, HER becomes even more powerful because:

- A **defender** that failed to block a shot still reached *some* position — that's a reusable experience
- A **midfielder** that mis-passed still moved the ball *somewhere* — HER can mine that
- Each robot generates diverse failed trajectories, giving the team a **richer collective replay buffer**

---

### The Analogy

Think of it like a youth soccer coach reviewing game tape. Instead of only rewarding the one kid who scored, a good coach says:

> *"You didn't score, but you made a perfect run into space — let's learn from that."*

HER is that coach, systematically extracting lessons from **every** play, not just the highlights.

---

### Bottom Line

HER transforms the sparse-reward nightmare of goal-oriented tasks into a **dense learning signal** by retroactively reframing failures as successes toward different goals. For a robot soccer team — where goals are rare and exploration is expensive — it can be the difference between an agent that **never learns** and one that **masters the fundamentals** efficiently.

**HER fixes this by relabeling failed episodes as successes**, retroactively. After the episode ends, HER looks at where the ball actually went and asks: *"What if THAT was the goal we were trying for?"* It then re-tags those transitions with a success reward and stores them alongside the real transitions. This means the agent gets useful learning signal from almost every episode, even when it never scores.

This is especially valuable for our task because:
- Goals are sparse terminal events
- The ball can end up anywhere — every final ball position becomes a "virtual goal"
- Sample efficiency improves dramatically compared to plain TD3

---

## Prerequisites — What Needs to Be Done First

Before implementing HER, your teammates must have:

1. **Finalized the observation space** in `td3_jal_env.py` — specifically the dimension and contents of the observation vector
2. **Finalized the reward function** in `_calculate_reward()` — HER adds its own reward on top of (or instead of) part of this
3. **Finalized the algorithm/network architecture** in `td3_jal_trainer.py`

Do not start the HER implementation until those are stable. HER touches both files and changing obs dimensions or reward structure afterward will break the goal-conditioning logic.

---

## Overview of Changes Required

HER in SB3 requires two things:

| What | Where | Why |
|---|---|---|
| Convert env to `GoalEnv` with Dict observation | `td3_jal_env.py` | SB3's HER buffer only works with `{observation, achieved_goal, desired_goal}` obs dicts |
| Add `compute_reward()` method | `td3_jal_env.py` | HER calls this to relabel stored transitions — must be stateless and batch-compatible |
| Swap replay buffer to `HerReplayBuffer` | `td3_jal_trainer.py` | Activates the hindsight relabeling logic |
| Switch policy to `MultiInputPolicy` | `td3_jal_trainer.py` | Required when observation space is a Dict instead of a flat Box |

---

## Step 1 — Modify `td3_jal_env.py`

### 1a. Change the base class

```python
# Before
class TD3JALEnv(gym.Env):

# After
class TD3JALEnv(gym.Env):  # SB3's HER works with gym.Env as long as obs is a GoalEnv-style Dict
```

> **Note:** You do not strictly need to subclass `gym.GoalEnv`. SB3's `HerReplayBuffer` only requires that (a) the observation space is a `Dict` with the three required keys, and (b) the env has a `compute_reward` method. Keeping `gym.Env` avoids an extra abstract method requirement.

---

### 1b. Restructure the `observation_space`

SB3's HER buffer expects the observation space to be a `gymnasium.spaces.Dict` with **exactly these three keys**:

- `"observation"` — the regular state vector your policy sees (your existing flat obs)
- `"achieved_goal"` — what the agent actually achieved this step (we use ball position)
- `"desired_goal"` — what the agent is trying to achieve (the opponent goal centre)

```python
def __init__(self, ...):
    super().__init__()
    ...

    # Dimensions — update these to match whatever your teammates finalized
    obs_dim  = 10   # your existing flat observation vector length
    goal_dim = 2    # we represent goals as normalized (x, y) ball positions

    self.observation_space = spaces.Dict({
        "observation": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        ),
        "achieved_goal": spaces.Box(
            low=-1.0, high=1.0,
            shape=(goal_dim,),
            dtype=np.float32
        ),
        "desired_goal": spaces.Box(
            low=-1.0, high=1.0,
            shape=(goal_dim,),
            dtype=np.float32
        ),
    })

    # Fixed desired goal: opponent goal centre in normalized field coords
    # Field runs from -45 to +45 in x, so opponent goal at x=45 → normalized = 1.0
    # Goal is centred at y=0 → normalized = 0.0
    self._desired_goal = np.array([1.0, 0.0], dtype=np.float32)
```

**Why ball position for `achieved_goal`?**
The task is "get the ball into the goal." The ball's final position is what determines success or failure. Using the robot position would be wrong — the robot being near the goal doesn't mean it scored.

**Why normalize?**
The goal space bounds must match `achieved_goal` space bounds. Normalized coords `[-1, 1]` map cleanly to the field dimensions and keep values in a consistent range for the network.

---

### 1c. Add a helper to build the obs dict

Rather than duplicating the obs-building logic across `reset()` and `step()`, extract it into a private helper:

```python
def _get_obs_dict(self, game_state) -> dict:
    """Build the full HER-compatible observation dictionary."""

    # Your existing flat observation (call the method your teammates finalized)
    flat_obs = self._game_state_to_obs(game_state)  # shape: (obs_dim,)

    # Extract ball position for the achieved goal
    ball_x, ball_y = game_state.ball_pos

    # Normalize ball position to [-1, 1] using field half-lengths
    field_half_x = self.field_width  / 2   # 45.0
    field_half_y = self.field_height / 2   # 30.0

    achieved_goal = np.array([
        np.clip(ball_x / field_half_x, -1.0, 1.0),
        np.clip(ball_y / field_half_y, -1.0, 1.0),
    ], dtype=np.float32)

    return {
        "observation":   flat_obs,
        "achieved_goal": achieved_goal,
        "desired_goal":  self._desired_goal.copy(),
    }
```

---

### 1d. Update `reset()` to return a dict

```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.networker.reset_sim()

    # ... all your existing reset logic (step counter, history, etc.) ...

    game_state = self._get_game_state()
    obs = self._get_obs_dict(game_state)   # <-- return dict instead of flat array

    info = {
        "episode_num": self.episode_num,
        "step": self.current_step
    }
    return obs, info
```

---

### 1e. Update `step()` to return a dict

```python
def step(self, action):
    self.current_step += 1

    commands, action_info = self._action_to_commands(action)
    self._send_commands(commands)

    game_state = self._get_game_state()
    obs = self._get_obs_dict(game_state)   # <-- return dict instead of flat array

    terminated, truncated, termination_reason = self._check_termination(game_state)
    reward = self._calculate_reward(game_state, action_info, termination_reason=termination_reason)
    self.total_rewards += reward

    info = {
        "step": self.current_step,
        "action_type": action_info["action_type"],
        "total_reward": self.total_rewards,
        **action_info,
    }

    if terminated or truncated:
        info["termination_reason"] = termination_reason
        info["episode_reward"] = self.total_rewards

    return obs, reward, terminated, truncated, info
```

---

### 1f. Add `compute_reward()` — the most important HER method

This is the method HER calls offline to relabel stored transitions. It receives a **batch** of achieved and desired goals (numpy arrays) and must return a batch of rewards. It is called **after the episode**, not during it — so it cannot use `self.prev_ball_pos` or any other instance state.

```python
def compute_reward(
    self,
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    info: dict
) -> np.ndarray:
    """
    Compute reward for a batch of (achieved_goal, desired_goal) pairs.

    Called by HerReplayBuffer to relabel stored transitions with hindsight goals.
    Must be stateless — cannot reference self.prev_ball_pos or any episode state.

    Args:
        achieved_goal: shape (batch_size, 2) — normalized ball positions
        desired_goal:  shape (batch_size, 2) — normalized goal targets
        info:          dict (not used here, but required by the interface)

    Returns:
        rewards: shape (batch_size,) — sparse reward signal
    """
    # Opponent goal in normalized coords:
    #   x >= 0.977  →  ball_x >= 44.0m  (just past the goal line)
    #   |y| <= 0.122 →  |ball_y| <= 3.66m  (within the goal posts, half of 7.32m width)
    goal_half_norm = (7.32 / 2) / (self.field_height / 2)   # ≈ 0.122

    in_goal = (
        (achieved_goal[..., 0] >= 0.977) &
        (np.abs(achieved_goal[..., 1]) <= goal_half_norm)
    )

    # Sparse reward: 0.0 for success, -1.0 otherwise
    # This convention (success=0, failure=-1) is standard for HER in SB3
    return in_goal.astype(np.float32) - 1.0
```

**Why sparse rewards here?**
`compute_reward` is only called by HER for relabeled transitions — it does not replace your `_calculate_reward`. Your dense shaping (ball proximity, velocity progress) still runs in `step()` for real transitions. HER just needs this method to correctly signal whether a relabeled "virtual goal" was achieved.

**Why `{0, -1}` instead of `{+1, 0}`?**
SB3's HER documentation recommends this convention. A reward of `0` for success and `-1` for failure works better with the discount factor than `+1` / `0`, because the agent learns to minimize the number of `-1` steps before reaching `0`.

---

## Step 2 — Modify `td3_jal_trainer.py`

### 2a. Import `HerReplayBuffer`

```python
from stable_baselines3 import TD3
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
```

---

### 2b. Update `setup_model()` — three changes

```python
def setup_model(self, env: gym.Env):
    ...

    self.model = TD3(
        policy="MultiInputPolicy",        # CHANGED from "MlpPolicy"
                                          # Required for Dict observation spaces

        env=env,

        replay_buffer_class=HerReplayBuffer,   # ADDED — activates HER
        replay_buffer_kwargs={                  # ADDED — HER configuration
            "n_sampled_goal": 4,
            "goal_selection_strategy": "future",
        },

        # Everything below stays the same as before
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        policy_delay=policy_delay,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=str(self.device),
        tensorboard_log=str(self.run_dir / "tensorboard")
    )
```

---

## HER Configuration — What the Parameters Mean

### `goal_selection_strategy`

This controls how HER picks the "virtual goal" to relabel a transition with. There are three options:

| Strategy | How it picks the virtual goal | When to use |
|---|---|---|
| `"future"` | Randomly samples a state that comes **after** the current step in the same episode | **Default choice — use this** |
| `"episode"` | Randomly samples any state from the same episode | Slightly weaker than `future` |
| `"final"` | Always uses the very last state of the episode | Simple but less diverse |

**Use `"future"`**. It consistently performs best in the original HER paper and SB3 examples. It ensures the relabeled goal was actually reachable from the current state (because it happened later in the same episode).

### `n_sampled_goal`

How many hindsight relabelings to generate per real transition stored in the buffer.

- Setting this to `4` means: for every 1 real transition, HER also stores 4 relabeled versions with different virtual goals
- The buffer effectively gets `4×` more useful transitions for the same number of real steps
- `4` is the value used in the original HER paper and is a solid default
- You can increase to `8` if training is still slow to converge, at the cost of more memory

---

## What Happens Internally (How HER Actually Works)

Understanding this helps when debugging:

1. The agent runs an episode and collects transitions `(obs, action, reward, next_obs, done)` normally
2. Each transition's `obs` dict contains `achieved_goal` (where the ball actually is) and `desired_goal` (opponent goal centre)
3. At the end of the episode, `HerReplayBuffer` looks back at the episode
4. For each stored transition, it samples `n_sampled_goal=4` future states from that episode
5. It relabels each transition: replace `desired_goal` with the ball position from that future state
6. It calls `compute_reward(achieved_goal, new_desired_goal, info)` to get the reward for the relabeled transition
7. These relabeled transitions are stored in the buffer alongside the real ones
8. TD3 trains on batches that mix real and relabeled transitions

The result: even in episodes where the robot never scores, the buffer contains many "virtual successes" — transitions relabeled as if the robot was trying to move the ball to positions it actually reached.

---

## Quick Checklist Before Running

- [ ] `observation_space` is a `spaces.Dict` with keys `observation`, `achieved_goal`, `desired_goal`
- [ ] `reset()` returns a dict obs (not a flat array)
- [ ] `step()` returns a dict obs (not a flat array)
- [ ] `compute_reward(achieved_goal, desired_goal, info)` exists and works with batched numpy arrays
- [ ] `goal_selection_strategy` is `"future"`, `"episode"`, or `"final"` (string, not an enum)
- [ ] `policy="MultiInputPolicy"` in the TD3 constructor
- [ ] `replay_buffer_class=HerReplayBuffer` in the TD3 constructor
- [ ] `replay_buffer_kwargs` has both `n_sampled_goal` and `goal_selection_strategy`

---

## Common Mistakes

**Using `"MlpPolicy"` instead of `"MultiInputPolicy"`**
SB3 will crash with a cryptic error about observation space types. Any Dict obs requires `MultiInputPolicy`.

**Making `compute_reward` use instance state**
HER calls `compute_reward` in bulk after episodes, not step-by-step. Accessing `self.prev_ball_pos` or `self.current_step` here will produce wrong rewards or errors.

**Forgetting to normalize `achieved_goal`**
The `achieved_goal` values must stay within the bounds declared in `observation_space["achieved_goal"]`. If you declare `Box(-1, 1)` but return raw field coordinates (up to ±45), SB3 may clip or warn. Always normalize.

**Setting `buffer_size` too small**
With `n_sampled_goal=4`, the buffer stores up to `5×` as many transitions as real steps. The existing `buffer_size=100_000` is fine, but be aware memory usage is higher than plain TD3.

**Changing obs structure after HER is set up**
If teammates update the observation vector dimensions after you've added HER, you need to re-check `obs_dim`, `goal_dim`, the normalization in `_get_obs_dict`, and the threshold values in `compute_reward`.

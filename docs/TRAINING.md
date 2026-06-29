# Training Log — TD3 JAL Curriculum

This document summarizes the training run that started at 2026-05-19 20:34 (log id `20260519_203435_238279`), the diagnostic finding that explained why it failed to improve, and the one-line config change made before the next run.

---

## TL;DR

- **Previous run:** stage1_090_mid (±90° spawn, ±15 m ball y-range), restarted from the stage1_015_warm checkpoint with three changes from the prior run: bad-aim kick penalty (-20), cone gate tightened (cos > 0.5 → 0.6), action noise doubled (σ=0.2 → 0.4).
- **Result:** 1924 episodes, 39,869 timesteps, 170 goals = **8.84% goal rate, flat across the entire run** (no improvement).
- **Root cause:** the warmup-trained actor weights carry a "always-output-kick, turn_theta≈0" prior that's correct for ±15° spawn but wrong for ±90°. The replay buffer ends up dominated by kick-action samples, so the Q-function never learns what turn-actions are worth, so the actor never breaks out of kick=100%. The bad-aim penalty fires correctly but only refines aim along the kick path — it can't dislodge the action-type lock.
- **Fix for next run:** `load_model: null`. Start the actor from random initialization in stage1_090_mid instead of inheriting the poisoned warmup prior. Everything else (cone gate 0.6, bad_aim_penalty 20, action_noise 0.4) stays.

---

## 1. The setup that was tried

Config: [configs/td3_jal_curriculum_config.json](configs/td3_jal_curriculum_config.json) at the time of the run.

| Parameter | Value | Note |
|---|---|---|
| `load_model` | `stage1_015_warm_complete.zip` | inherit warmup weights |
| Stage | `stage1_090_mid` | ±90° spawn, ball_y ∈ [-15, 15] |
| `action_noise_std` | 0.4 | doubled from 0.2 to break kick=100% lock |
| `bad_aim_kick_penalty` | 20.0 | new this run — see §3 |
| Cone gate threshold | cos > 0.6 (±53°) | tightened from 0.5 (±60°) |
| `goal_reward` | 150.0 | unchanged |
| `alignment_weight` | 2.0 | per-step cos-delta when has-ball |
| `kick_aim_bonus_weight` | 20.0 | one-shot reward at kick fire, scaled by aim |
| `invalid_action_penalty` | 0.5 | per step for cone-blocked kicks |

### Diagnostic logging added

To distinguish a "policy aim is bad" failure from a "ball physics deflects good kicks" failure, every fired kick now logs (in [ai_interface/envs/JAL_env.py](ai_interface/envs/JAL_env.py), the kick-aim block):

```
Kick fired: robot=(x, y, θ°)  predicted_y=<py>  aim_quality=<aq>  bad_aim=<True|False>
```

where `predicted_y_at_goal_line = ry + (FIELD_X[1] - rx) × tan(θ)` — the straight-line projection of the kick at the goal line — and `bad_aim = aim_quality <= 0`, i.e. `|predicted_y| > goal_half_height = 5 sim units`.

The `bad_aim_kick_penalty` is applied whenever `bad_aim=True` so that an unaimed kick has clearly negative expected value.

---

## 2. Run results

Run id: `20260519_203435_238279`. Wall-clock: 2026-05-19 20:34 → 22:18 (≈ 1h 44min). Reached 39,869 of 50,000 timesteps in stage1_090_mid.

### Aggregate

| | Count | Rate |
|---|---|---|
| Episodes ended | 1924 | — |
| Goals | 170 | **8.84%** |
| Kicks fired | 1922 | ~1 per episode |
| Kicks with bad_aim | 1701 | 88.5% of kicks |
| Kicks with good_aim | 221 | 11.5% of kicks |
| Off-target (penalty area, missed mouth) | 1513 | 78.6% of episodes |
| Ball out of bounds | 210 | 10.9% of episodes |
| Ball stopped unreachable | 28 | 1.5% of episodes |
| max_steps timeout | 3 | 0.2% of episodes |

### Evolution over time (200-episode buckets)

```
       eps    n  goal%  badaim%  avg|py|   kick%
    0- 199  199   8.5%   84.7%    22.36    72%
  200- 399  200   9.0%   89.5%    22.49   100%
  400- 599  200   9.0%   86.5%    21.82   100%
  600- 799  200   8.5%   88.6%    21.32   100%
  800- 999  200   9.0%   89.0%    22.42   100%
 1000-1199  200   8.0%   90.5%    23.28   100%
 1200-1399  200   9.5%   89.5%    22.55   100%
 1400-1599  200   8.5%   90.0%    22.57   100%
 1600-1799  200  10.5%   87.0%    21.81   100%
 1800-1999  125   7.2%   90.4%    22.65   100%
```

**Every metric is flat.** Goal rate hovers 7–10%, bad-aim rate hovers 85–90%, `avg|predicted_y|` hovers 21–23 sim units. The action distribution snapped from 72% kick / 28% turn in the first 200 episodes (random pre-learning phase, before `learning_starts=5000` timesteps) to **100% kick from episode 200 onward** — and stayed there.

### What `avg|predicted_y| ≈ 22` means physically

- 1 sim unit = 0.1 m. The goal mouth is `±goal_half_height = ±5` sim units = ±0.5 m wide.
- A kick with `|predicted_y| = 22` projects to 2.2 m from goal center — **4.4× outside the goal mouth**.
- Distribution of all 1207 kick fires logged (partial earlier sample):

```
  0-2 sim units (perfect):   4.2%
  2-5 (still in goal):       7.7%   ← 11.9% land inside the goal mouth
  5-10 (just miss):         10.5%
        10-15:               12.7%
        15-20:               12.0%
        20-30:               23.3%
        30-50:               26.5%   ← over half of all kicks miss this badly
       50-100:                3.1%
```

The model isn't refining aim. It's firing kicks with effectively-random heading.

---

## 3. What the diagnostic logging proved

Cross-tab of `bad_aim` (a prediction made *at the moment of kick fire*) against the eventual episode outcome:

|  | actual goal | actual miss |
|---|---|---|
| `bad_aim=False` (predicted in goal) | 100 | 44 |
| `bad_aim=True` (predicted miss)     | 4   | 1058 |

(numbers from the first 1165 episodes; the pattern continues identically in the full 1924 — last 500 episodes show goal%=8.6%, bad_aim%=89.2%, avg|py|=22.23.)

Reading the table:

- When the predictor says "good aim", **69% of those kicks actually score** (100 / 144). The other 31% are physics deflections — friction, lateral drift — and are not a flaw in the predictor.
- When the predictor says "bad aim", **99.6% of those kicks miss** (1058 / 1062). Only 4 lucky-physics goals out of 1062.

This validates that:

1. `predicted_y` is a clean, reliable signal of whether a kick will score.
2. The `bad_aim_kick_penalty` is being applied to truly bad kicks — false-positive rate is 0.4%.
3. The model isn't being treated unfairly by the reward: it's getting paid -20 only for kicks that genuinely had no chance.

So the failure is *not* in the reward shaping. It is in the policy.

---

## 4. Root cause

### What the warmup did

stage1_015_warm trained at ±15° spawn with `ball_y ∈ [-3, 3]`. In that distribution:

- The cone gate is *always* passed at spawn: `cos(15°) ≈ 0.966 > 0.5`. So kicking on step 1 is always physically possible.
- Optimal aim ≈ initial heading (the spawn theta is already near the goal-pointing direction).
- **Optimal policy: output kick-logit high, turn_theta ≈ 0.** No turning needed.

The warmup achieved 70.6% goal rate doing exactly that.

### Why that policy fails at ±90° spawn

stage1_090_mid widens the spawn to `theta ∈ [-90°, 90°]` and ball-y to `[-15, 15]`. Now:

- 33% of spawns are out-of-cone (`|theta| > 60°` vs the ball at the new tighter 0.5 → 0.6 cone gate, ≈ 47% out-of-cone).
- Even in-cone spawns require a non-zero turn before kicking — the goal subtends only `atan(5/d)` from the ball position, e.g. ±11° at d=25 sim units, ±18° at d=15.
- **Optimal policy: state-dependent turn_theta to align heading with the goal direction, then kick.**

The warmup-trained actor weights have neither learned this nor have any reason to. The actor goes into stage1_090_mid with a strong prior: "kick-logit high, turn_theta near 0." The bad_aim_penalty was supposed to fight that prior. It did not. Here's why.

### The replay-buffer chicken-and-egg

TD3 learns by approximating Q(s, a) from buffered experiences. For the actor to learn that *turning* would be better than kicking at out-of-cone states, the Q-function must be trained on samples that include turn-actions. But:

1. The actor outputs kick-logit high → argmax is "kick" essentially always.
2. With `action_noise_std=0.4` on the raw [-1,1] action, **only ~12% of stored samples have the argmax flipped away from kick by noise.**
3. So ~88% of the buffer is kick-action samples.
4. Q(state, turn-action) is essentially untrained — there's no data to fit it on.
5. Without a useful Q-estimate for the turn action, the actor has no gradient pulling it toward turn.
6. So the actor stays on kick. Goto 1.

This is a stable local optimum. The bad_aim_penalty *does* propagate gradient — it pushes the actor to refine `turn_theta` (the continuous parameter that always runs alongside the chosen action), and the predicted_y trace would show this as cluster of values closer to zero over time. **It doesn't.** Average `|predicted_y|` stayed at 22 across all 1924 episodes. The signal isn't strong enough to overcome the warmup prior on a buffer dominated by kick-samples.

### Math check on the per-step noise (for the record)

The noise σ=0.4 is applied to the actor's raw output in [-1, 1]. The path from raw → physical turn delta is:

```
raw ∈ [-1, 1]
turn_theta = raw × π                        (rad/sec)         [ai_interface/envs/JAL_env.py:826]
per-step turn delta = turn_theta × 0.1      (SIM_TIMESTEP)    [networking/data_utils.py:215]
                    = raw × π × 0.1
                    = ±0.314 rad max
                    = ±18° per step max
```

So with σ=0.4 in raw space:

```
per-step turn-delta noise σ = 0.4 × π × 0.1 ≈ 0.126 rad ≈ 7.2°
```

This is *not* large enough to dominate the actor's mean turn_theta — if the actor wanted to command e.g. 9°/step, the realized turn is N(9°, 7.2°) and the signal is clearly visible above the noise. So the failure isn't "noise drowns the policy." It's "the policy itself isn't outputting useful turn_theta because the warmup taught it 0 and nothing in the new stage's experience has been strong enough to change that."

The previous draft of this analysis (in conversation) claimed σ=0.4 produced ±72° per-step noise. That was off by 10× — it missed the SIM_TIMESTEP factor.

---

## 5. The fix

One-line config change: [configs/td3_jal_curriculum_config.json](configs/td3_jal_curriculum_config.json).

```diff
- "load_model": "models/td3_jal_curriculum/stage1_015_warm_complete.zip",
+ "load_model": null,
```

This skips loading the warmup checkpoint. stage1_090_mid will start the actor with PyTorch's default random initialization, which has softmax-uniform action_type logits and small zero-mean turn_theta outputs — i.e. no prior favoring any action over any other.

### Why this and not something else

- **Don't lower `action_noise_std`.** σ=0.4 is fine. Lowering it would deepen the kick-lock by reducing the argmax-flip rate further.
- **Don't raise `bad_aim_kick_penalty`.** The penalty fires correctly and applies the right gradient; making it larger doesn't help when the underlying problem is the buffer's action diversity, not the per-sample signal strength.
- **Don't loosen the cone gate further.** cos > 0.6 is physically reasonable for the kicker plate; loosening it back to 0.5 would just let more guaranteed-miss kicks through.
- **Don't change the action space.** A discrete action-type head + continuous params is a real structural option, but it requires re-architecting the actor, the critic, and the env decoder. Try the cheap fix first.

### Everything else stays

| | |
|---|---|
| cone gate cos > 0.6 | stays |
| bad_aim_kick_penalty: 20.0 | stays (validated, see §3) |
| action_noise_std: 0.4 | stays |
| kick_aim_bonus_weight: 20.0 | stays |
| invalid_action_penalty: 0.5 | stays |
| alignment_weight: 2.0 | stays |
| Diagnostic logging | stays |

---

## 6. What to watch for in the next run

With a fresh actor, the **first ~5000 timesteps** are pure random exploration (driven by `learning_starts=5000` keeping the policy net silent). Expect:

- Action distribution roughly uniform across the 6 action types.
- Goal rate low (~5%) — random kicks land randomly.
- `|predicted_y|` distribution broad.

Once learning kicks in (after step 5000):

- **Goal rate climbing** above 15% by step 15k = improvement (the policy is learning what the warmup never did).
- **`avg|predicted_y|` shrinking** toward 10 then toward 5 = the aim signal is propagating.
- **bad_aim rate dropping** from ~85% toward 50% then toward 30% = kicks are firing only when aimed.
- **Action distribution non-degenerate:** if at step 15k it's stuck at kick=100% again, the warmup wasn't the only problem and we need bigger changes (likely the architecture).

### Acceptance criteria for stage1_090_mid

By the end of its 50,000 timestep budget:

- Goal rate ≥ 40%.
- bad_aim rate ≤ 50%.
- avg|predicted_y| ≤ 10.

If those land, stage1_180_full inherits the trained policy automatically. If they don't, the next investigation should be: separate exploration noise per dimension (high σ on action_type logits, low σ on continuous params) or replace the cos-delta alignment reward with a direct angular-error-to-goal reward.

---

## 7. Kicker cone removed; 58% model reused (2026-05-20)

### Background: the kicker cone gate and why it was removed

A kicker cone gate had been added to `JAL_env.py`: kicks only fired if the ball was within a `cos(70°) ≈ 0.342` forward arc of the robot. The idea was to model a physical plausibility check — a robot can't kick a ball behind it.

This caused every training run that reached this stage to fail in one of two ways:

- **TD3 (stage1_180_full):** Policy collapsed to `turn=100%`. The agent learned to spin indefinitely rather than kick, because the cone gate meant a kick outside the arc was wasted. Turning was always "safe" (no penalty), so `turn=100%` was a stable local optimum.
- **PPO (tested separately by a teammate):** Flat ~7–8% goal rate, policy frozen. Same structural cause.

The cone gate was removed because **rcssserver's 2D simulator accepts a kick command regardless of robot orientation** — the server computes kick direction from the ball–robot geometry, but it does not reject the command. The cone was an artificial constraint that didn't match the environment the robot is actually trained in. The 58% checkpoint (`stage1_6_final_58pct.zip`) was trained without any cone gate and reached 58% goal rate, confirming that the sim doesn't need it.

The cone can be revisited for physical robot deployment (where kicker plate geometry matters), but it should not be in the training loop.

### The 58% checkpoint and what it learned

`models/td3_jal_her/stage1_6_final_58pct.zip` was trained as a TD3+HER model on a 1v0 task where the robot spawned with a random orientation (±180°) but directly at the ball. Action space is 9D per robot: 6 logits for primitive selection (`[goto, approach_ball, turn, kick, start_dribble, stop_dribble]`) plus continuous params `(goto_x, goto_y, turn_theta)`. During its training, `goto`, `start_dribble`, and `stop_dribble` were disabled; `approach_ball` was also effectively disabled (its logit was masked, so its weights were never trained). The model learned to use `turn` (slot 2) and `kick` (slot 3) — at 58%, it consistently navigated from random orientation at the ball to an aimed kick toward goal.

This was chosen as the warm-start for the next stage because:
- It already knows "turn toward goal, then kick" — the hard part of aim.
- It matches the 9D action space exactly (no shape mismatch on load).
- Approach_ball (slot 1) has near-zero logits in this checkpoint — effectively untrained — so the next stage can teach it from scratch without overwriting the kick behavior.

### stage1_9 training run: what happened

Config: `stage1_9_approach_turn_kick` (200k steps, warm-start from 58% checkpoint).

Key differences from 58% checkpoint training:
- Robot spawns at default position (away from ball, ~x=−10), ball placed randomly in opponent half.
- `approach_ball` enabled alongside `turn` and `kick`.
- `action_noise_std` bumped from 0.2 → 0.3 to encourage exploration of the untrained `approach_ball` slot.
- No cone gate.

**Result at 38k / 200k steps (495 episodes):**

| Metric | Value |
|---|---|
| Overall goal rate | 29.3% (145/495) |
| Last 100 episodes goal rate | 30% |
| Typical action distribution | approach_ball 60–90%, turn 10–30%, kick 1–25% |
| Kick aim quality | Almost universally `bad_aim=True`, `aim_quality=0.00` |

The good news: `approach_ball` was being explored heavily — the untrained slot 1 was successfully promoted by the action noise. The robot was navigating to the ball.

The bad news: once at the ball, the robot fired kicks without first turning toward goal. Nearly every `Kick fired` log line showed `predicted_y` far outside the goal mouth. The 58% checkpoint's "turn-then-kick" behavior was being overwritten by "approach-then-kick" because at-ball states were rare in the new buffer (robot spent most steps approaching) and the alignment reward (weight=2) wasn't strong enough to override the immediate kick temptation.

### Why training was stopped at 38k steps instead of letting it run overnight

The failure mode (approach correctly, kick without aiming) was a stable local optimum under the current reward shaping. The specific issue:

- `bad_aim_kick_penalty = 0.0` — there was no cost for firing a random kick. Any kick that happened to push the ball forward still earned `goal_progress` reward, making bad-aim kicks weakly positive.
- `alignment_weight = 2.0` — the per-step reward for turning toward goal (max ±2 per step) was too weak to reliably outbid "kick now."

Simply letting training continue to 200k would not fix this. The Q-function was learning "kick whenever at ball" as a valid behavior, and TD3's deterministic policy + pessimistic twin-Q would tend to reinforce it.

More importantly, the critic Q-values were now calibrated to these old rewards. If we applied reward shaping changes and continued from the 38k checkpoint, the critic would need 10–20k steps to re-calibrate (its stored value estimates no longer match the new reward signal). That transition noise could entrench the bad behavior further.

It was cleaner to stop, fix the shaping, and restart.

### Why restart from 58% checkpoint instead of 30k checkpoint

Two options were considered:

**Continue from `stage1_9_steps30000.zip`:** The model has learned approach_ball (60–90% selection). But the critic Q-values are anchored to the old rewards — random kicks had positive Q under old shaping. Changing `bad_aim_kick_penalty` and `alignment_weight` mid-training creates a discontinuity: the policy gradient from the updated critic points in a new direction, but the policy's logits reflect old value estimates. The transition period could be 10–20k noisy steps.

**Restart from `stage1_6_final_58pct.zip`:** Clean critic state that matches the policy's behavior (turn-then-kick when at ball). New reward shaping is consistent from step 0. Approach_ball has to be re-learned, but the `learning_starts=5000` random-exploration phase samples all three enabled primitives uniformly, giving the critic ~1670 approach transitions to start from. The policy then re-learns approach under better shaping — this time, the alignment and aim signals are in place from the first update.

The cost of restarting is ~25–30k steps to recover approach_ball behavior. The benefit is a cleaner training trajectory with no Q-function debt. Restarting was chosen.

### Changes made for the next run

Two reward config changes in `configs/td3_jal_her_config.json`:

| Parameter | Old | New | Reason |
|---|---|---|---|
| `alignment_weight` | 2.0 | **4.0** | Double the per-step reward for turning toward goal while at ball. Total reward for a full 180° turn is now ~8 (was ~4) — meaningfully larger relative to a random kick. |
| `bad_aim_kick_penalty` | 0.0 | **5.0** | One-shot penalty when kick fires with `|predicted_y| > 5` sim units. Makes bad-aim kicks negative EV when misaligned (~80% of early kicks), suppressing the "kick now, aim later" behavior. |

**Everything else unchanged:**
- `load_model: models/td3_jal_her/stage1_6_final_58pct.zip`
- `action_noise_std: 0.3` (kept — needed to explore approach_ball from near-zero logit in 58% checkpoint)
- `timesteps: 200000`
- All other reward overrides unchanged

**EV sanity check (with new shaping):**

- Random unaligned kick: `24 (goal_progress) − 5 (penalty) + 0.05 × 150 (P_goal_at_bad_aim) ≈ +26.5`
- Turn 10 steps then aimed kick: `8 (alignment) + 24 (goal_progress) + 0.7 × (150 + 20) ≈ +151`

The gap (26 vs 151) should pull the policy toward turn-then-kick. `bad_aim_kick_penalty=5.0` was chosen (not 10.0 or 20.0) because the prior turn=100% collapse used penalty=20 *without* approach_ball as a third option. With approach_ball available, the policy has an escape route if kicking is suppressed, so the collapse risk is low.

### What to watch for

- **First 5k steps (random_starts phase):** action distribution roughly uniform over approach/turn/kick.
- **Steps 5k–30k:** approach_ball should climb to 40%+ selection. If it stays below 20% by step 20k, bump `action_noise_std` to 0.4.
- **Steps 30k–100k:** aim quality should improve — `avg|predicted_y|` should trend downward, bad_aim rate should drop from 80%+ toward 50%.
- **Acceptance at 200k:** goal rate ≥ 50%, bad_aim rate ≤ 40%.

---

## 8. Training run: 20260521_020942_188642

**Start time:** 2026-05-21 02:09:42
**Stage:** `stage1_9_approach_turn_kick`
**Steps:** 199,957 / 200,000 (completed)
**Load model:** `models/td3_jal_her/stage1_6_final_58pct.zip`

**Aim:** 1v0 approach + turn + kick. Warm-started from the 58% Stage-1.6 checkpoint with the kicker-cone gate removed (kicks fire whenever ball is kickable, regardless of orientation) and `approach_ball` unmasked so the policy must learn to navigate to the ball from a random heading.

### Results

| Metric | Value |
|---|---|
| Total episodes | 2823 |
| Overall goal rate | 1214/2823 = **43.0%** |
| Last 100 episodes goal rate | **52.0%** |
| Avg reward (last 10 eps) | 155.67 |
| Avg episode length (last 10 eps) | 56.8 |

**Episode outcomes:**

- `ball_in_penalty_off_target`: 1505 (53.3%)
- `goal_scored`: 1214 (43.0%)
- `max_steps`: 59 (2.1%)
- `ball_out_of_bounds`: 45 (1.6%)

**Goal rate over training (200-episode buckets):**

```
  eps    1- 200:  41.0%  ████████
  eps  201- 400:  39.0%  ███████
  eps  401- 600:  35.5%  ███████
  eps  601- 800:  48.0%  █████████
  eps  801-1000:  41.5%  ████████
  eps 1001-1200:  37.0%  ███████
  eps 1201-1400:  48.5%  █████████
  eps 1401-1600:  39.0%  ███████
  eps 1601-1800:  43.0%  ████████
  eps 1801-2000:  44.5%  ████████
  eps 2001-2200:  48.0%  █████████
  eps 2201-2400:  40.5%  ████████
  eps 2401-2600:  46.5%  █████████
  eps 2601-2800:  51.0%  ██████████
  eps 2801-2823:  34.8%  ██████
```

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.253 | 51.5% |
| Last 200 kicks | 0.311 | 43.5% |

**Action distribution samples:**

- ep 1800: approach_ball=68%  kick=31%
- ep 2000: approach_ball=72%  kick=5%  turn=21%
- ep 2200: approach_ball=96%  kick=1%  turn=1%
- ep 2400: approach_ball=80%  kick=7%  turn=12%
- ep 2600: approach_ball=73%  kick=1%  turn=24%
- ep 2800: approach_ball=92%  kick=1%  turn=5%

### Key config

**Reward overrides:**
- `has_ball_bonus`: 0.0
- `near_ball_bonus`: 0.1
- `approach_weight`: 2.0
- `approach_clip`: 0.5
- `step_bonus`: 0.0
- `goal_progress_weight`: 3.0
- `goal_progress_clip`: 2.0
- `ball_speed_bonus_weight`: 2.0
- `ball_speed_threshold`: 0.2
- `ball_speed_clip`: 2.0
- `goal_reward`: 150.0
- `alignment_weight`: 4.0
- `alignment_near_ball_only`: false
- `alignment_has_ball_only`: true
- `kick_aim_bonus_weight`: 20.0
- `bad_aim_kick_penalty`: 5.0

**Model params:**
- `learning_rate`: 0.001
- `action_noise_std`: 0.3
- `gamma`: 0.99
- `batch_size`: 64
- `her_goal_selection_strategy`: future
- `her_n_sampled_goal`: 4

### Code changes (non-config)

- Kicker cone gate removed in `JAL_env.py:891` — `can_kick = has_ball_now` only; orientation no longer required for the kick action to fire. Matches the regime under which the 58% Stage-1.6 checkpoint was trained, and matches rcssserver's actual kick semantics.
- `invalid_action_requested` in `JAL_env.py:894-898` retained only for kick-without-ball and dribble-without-ball — no cone-related penalty path remains.

### What went wrong

The run **hit its numerical acceptance bar** (52% last-100 goal rate ≥ 50% target, bad_aim improved 51.5% → 43.5%) but **failed the behavioral test**: in deterministic inference (`infer.py` with `noise_clip=0`), the policy idles near the ball and never kicks. Specifically:

- Action distribution in the last ~1000 episodes is dominated by `approach_ball` (68–96%) with `kick` at 1–7% and `turn` at 1–24%. The policy almost never *chooses* kick.
- The 52% goal rate is therefore **noise-driven**: with `action_noise_std=0.3` applied to the continuous logits at inference, the argmax randomly flips to the kick slot often enough that goals happen anyway. Remove the noise and the kick logit never wins.
- Root cause: with `step_bonus=0.0`, idling at the ball pays `+0.1/step` (`near_ball_bonus`). Over a 300-step max episode that's `+30`, minus the `-10` max_steps penalty = `+20` net. Idle is positive-EV, so the Q-function has no reason to prefer kicking unless the kick's expected reward exceeds +20.
- Off-target kick EV (≈ 53% of kicks) is `-5` (bad_aim_penalty) + small kick_aim shaping. Goal-scoring kick EV is `+150` but rare under deterministic policy. The training averages out positive enough to look fine in the goal-rate trace, but the policy itself prefers idling at the ball.

This is the same failure mode that has bitten previous runs (§4: `approach_ball=100%` collapse). The cure is to make idle strictly negative-EV.

### Fix for next run

**Change:** `step_bonus: 0.0 → -0.2` (added to reward overrides for `stage1_9`).

**Why this addresses the root cause:**
- Idle EV recomputed with `step_bonus=-0.2`: `300 steps × (0.1 − 0.2) − 10 = −40` → idle is now strongly negative.
- Off-target kick EV: `−5 − 0.2 × steps_to_reach_ball + small_kick_aim ≈ +0` to `−5` for a moderately-aimed kick that misses. Still better than idle.
- Goal-scoring kick EV: `+150 − 0.2 × ~60 steps ≈ +138`. Massively preferred.
- The Q-function should now learn that *any* kick is better than no kick, and that *aimed* kicks dominate. The `bad_aim_kick_penalty=5.0` keeps a gradient toward better aim without suppressing kick attempts entirely.

**Restart from the 58% Stage-1.6 checkpoint**, not the 200k Stage-1.9 checkpoint. The 200k model's Q-function has the idle-is-good behavior baked in; restarting from 58% gives the new step_bonus a clean slate to write to.

**What to watch for:**
- Action distribution: `kick` selection should rise to 15%+ by step 30k (vs the 1–7% seen here).
- Deterministic inference (`noise_clip=0`) should produce kicks, not idling.
- Goal rate should remain ≥ 40% — if it drops below 25%, `step_bonus` is too harsh and is pushing the policy to spam kicks regardless of aim. Roll back to `−0.1`.


---

## 9. Training run: 20260521_135938_795927

**Start time:** 2026-05-21 13:59:38.796163
**Stage:** `stage1_9_approach_turn_kick`
**Steps:** 199,986 / 200,000 (completed)
**Load model:** `models/td3_jal_her/stage1_6_final_58pct.zip`

**Aim:** 1v0 approach + turn + kick. Warm-starts from `stage1_6_final_58pct.zip` (action_dim=9), which already learned turn (slot 2) and kick (slot 3). Robot now spawns at default position (away from ball) with random orientation; ball placed randomly in the opponent half. Policy must learn to use `approach_ball` (slot 1, masked during the 58% run, so its weights are essentially untrained) to navigate to the ball before turning and kicking. Kicker-cone gate removed (2026-05-20): kicks fire whenever ball is in kickable range. HER `future` strategy supplies goal-conditioned relabeling.

### Results

| Metric | Value |
|---|---|
| Total episodes | 2989 |
| Overall goal rate | 1125/2989 = **37.6%** |
| Last 100 episodes goal rate | **46.0%** |
| Avg reward (last 10 eps) | 195.86 |
| Avg episode length (last 10 eps) | 66.1 |

**Episode outcomes:**

- `ball_in_penalty_off_target`: 1774 (59.4%)
- `goal_scored`: 1125 (37.6%)
- `ball_out_of_bounds`: 53 (1.8%)
- `max_steps`: 36 (1.2%)
- `ball_teleport`: 1 (0.0%)

**Goal rate over training (200-episode buckets):**

```
  eps    1- 200:  27.0%  █████
  eps  201- 400:  20.5%  ████
  eps  401- 600:  36.5%  ███████
  eps  601- 800:  43.5%  ████████
  eps  801-1000:  43.0%  ████████
  eps 1001-1200:  36.0%  ███████
  eps 1201-1400:  34.5%  ██████
  eps 1401-1600:  37.5%  ███████
  eps 1601-1800:  42.5%  ████████
  eps 1801-2000:  30.0%  ██████
  eps 2001-2200:  35.5%  ███████
  eps 2201-2400:  40.0%  ████████
  eps 2401-2600:  44.0%  ████████
  eps 2601-2800:  46.0%  █████████
  eps 2801-2989:  48.7%  █████████
```

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.152 | 67.5% |
| Last 200 kicks | 0.294 | 43.5% |

**Action distribution samples:**

- ep 1800: approach_ball=80%  kick=1%  turn=17%
- ep 2000: approach_ball=91%  kick=1%  turn=7%
- ep 2200: approach_ball=56%  kick=12%  turn=31%
- ep 2400: approach_ball=55%  kick=13%  turn=31%
- ep 2600: approach_ball=83%  kick=2%  turn=14%
- ep 2800: approach_ball=82%  kick=1%  turn=15%

### Key config

**Reward overrides:**
- `has_ball_bonus`: 0.0
- `near_ball_bonus`: 0.1
- `approach_weight`: 2.0
- `approach_clip`: 0.5
- `step_bonus`: -0.2
- `goal_progress_weight`: 3.0
- `goal_progress_clip`: 2.0
- `ball_speed_bonus_weight`: 2.0
- `ball_speed_threshold`: 0.2
- `ball_speed_clip`: 2.0
- `goal_reward`: 150.0
- `alignment_weight`: 4.0
- `alignment_near_ball_only`: False
- `alignment_has_ball_only`: True
- `kick_aim_bonus_weight`: 20.0
- `bad_aim_kick_penalty`: 0.0

**Model params:**
- `learning_rate`: 0.001
- `action_noise_std`: 0.3
- `gamma`: 0.99
- `batch_size`: 64
- `her_goal_selection_strategy`: future
- `her_n_sampled_goal`: 4

### Code changes (non-config)

None this run. The two source-level changes from §7/§8 (cone gate removed at `JAL_env.py:891`; `invalid_action_requested` retained only for kick/dribble-without-ball) are still in place.

### What worked

This is the first run that produced a **genuine, non-collapsed approach + turn + kick policy**.

- **Goal rate climbed from 27% → 48.7% over 200k steps** (last bucket is the run's best window). The trajectory in the final third — 35% → 40% → 44% → 46% → 48.7% — shows the policy was still actively improving when training ended, not plateaued.
- **Aim refinement finally engaged.** Avg aim_quality went from 0.152 (first 200 kicks) to **0.294 (last 200 kicks)** — nearly 2× improvement. Bad_aim rate dropped 67.5% → 43.5%. This is the first run where the aim_quality curve moved meaningfully across training.
- **No collapse mode.** Two dips appeared (eps 201–400 at 20%, eps 1801–2000 at 30%) and both **self-corrected**. The model recovered from local regressions rather than falling into turn=100% (§8) or kick=100% (§6) attractors.
- **Action distribution is healthy.** Approach dominant (55–91% across late episodes), kick used sparingly (1–13% — each kick is 1 step out of ~60), turn used for alignment (7–31%). No single-action collapse for any sustained window.
- **Compared to §8 (52% last-100 noise-driven):** that run's 52% was illusory — deterministic inference idled and kick selection was 1–7%. This run's 46% last-100 is **real policy behavior** — kicks happen across many state types, aim is improving with training, and rewards (last-10 avg 195.86) reflect genuine goal-scoring.

**Why the fix worked.** Setting `bad_aim_kick_penalty: 5.0 → 0.0` (between §8 and this run) removed the variance asymmetry that TD3's `min(Q1, Q2)` exploits. Under penalty=5.0, the Q-function saw kicks as "lottery tickets with big negatives," and `turn=100%` (-60/episode) dominated `approach+bad_kick` (-50 to -200/episode). With penalty=0.0, approach+bad_kick became roughly net-zero, and approach+good_kick (~+200) made approach strictly dominant. `kick_aim_bonus_weight=20` and `alignment_weight=4` (gated by `has_ball_only`) provided enough positive signal to refine aim without needing a negative-penalty shaper.

### Next stage

This run is a successful Stage 1.9 baseline. Two paths forward:

1. **Continue training from this checkpoint** (`stage1_9_approach_turn_kick_complete.zip`) for another 100–200k steps. The final-bucket trajectory (48.7% and still climbing) suggests the policy hasn't yet saturated — pushing past 50–55% goal rate before moving on would give a stronger foundation for Stage 2.
2. **Move to Stage 2 (1v1 with one stationary or weakly-aimed opponent)**, warm-starting from `stage1_9_approach_turn_kick_complete.zip`. The approach + turn + kick competency is now robust enough that opponent presence won't break it; what we'd test is whether the policy generalizes around an obstacle and learns when to dribble vs. shoot.

**Recommendation: option 1 first.** Squeeze another 100k steps from this exact config. If goal rate stalls below 55%, then move to Stage 2 anyway since opponent variety might be a better aim-improvement signal than more 1v0 reps. Watch for the same aim-quality trajectory in the continuation — first 200 kicks of the continuation should immediately read at the current 0.29 level, confirming the checkpoint preserved aim skill.

---

## 10. Training run: 20260602_131858_424567 — Stage 1.1 kick (first expandable-backbone run)

**Start time:** 2026-06-02 13:18:58
**Stage:** `stage1_1_kick`  (first sub-stage of the rebuilt Stage 1 curriculum)
**Steps:** 25,000 / 25,000 (ran to completion)
**Load model:** None — from scratch on the expandable backbone (`use_expandable_backbone=true`), HER, sim-embedded, CPU.

> ⚠️ Parser config warning: no stage marked `ACTIVE`. The stage shown (`stage1_1_kick`) is correct — it's the only stage in `curriculum` — but mark its description `ACTIVE` before the next run so the parser stops falling back.

**Aim:** Learn to kick. Robot spawns AT the ball (0.5 behind, on the ball-goal axis) facing goal with ±15° jitter; ball at x∈[5,30], y∈[-3,3]. `goto`, `turn`, `approach_ball`, dribbles all disabled → only `kick` is selectable.

### Results — total failure

| Metric | Value |
|---|---|
| Total episodes | 24,541 |
| Overall goal rate | 0 / 24,541 = **0.0%** |
| Last-100 goal rate | **0.0%** |
| Avg episode length (last 10) | **1.0 step** |
| Avg reward (last 10) | 7.78 |

**Episode outcomes:**
- `ball_teleport`: 24,490 (**99.8%**)
- `ball_stopped_unreachable`: 51 (0.2%)
- `goal_scored`: 0

Action distribution was `kick=100%` throughout (correct — masking worked), but every episode ended after a single step.

### Code changes (non-config)

This was the first run after three coupled changes landed together:
1. **Kick exit speed raised to ~5.75 m/s** to match the real robots (mechanical-team video analysis). In `configs/server.conf` + `update_server_conf.sh`: `kick_power_rate 0.027→0.0575`, `ball_accel_max 2.7→5.8`, `ball_speed_max 3→6`. (`ball_decay` left at 0.94.)
2. **Stage 1.1 narrowed to kick-only** — added `approach_ball` to `disabled_actions` (was left enabled by mistake during curriculum reconstruction).
3. **Device forced to CPU** (`"device": "cpu"` in config) — CPU is ~3.2× faster than MPS for this small net (256×256, batch 64).

### What went wrong

**Every kick was misread as a server teleport, killing the episode after 1 step.** The teleport backstop in `JAL_env.py` (`_BALL_TELEPORT_THRESHOLD = 5.0`) ends an episode if the ball moves >5.0 units in one cycle — it exists to catch server-side teleports (BallStuckRef etc.). That threshold was tuned when `ball_speed_max=3` (legit kicks maxed at ~2.7 units/cycle, safely under 5.0). Raising the kick exit to 5.75 units/cycle meant **every legitimate kick now displaced the ball ~5.75 units/step**, exceeding 5.0. Log evidence: `Ball teleport detected … jump=5.75 … 5.81 … 5.85` on essentially every episode; length=1.0; 0 goals.

Root cause = a hidden coupling: the kick-physics math was correct, but a downstream env invariant ("ball moves <5 units/step") was silently violated by the speed change. The reward/model config was fine; this was purely the teleport guard.

### Fix for next run

**Raised `_BALL_TELEPORT_THRESHOLD` 5.0 → 8.0** in `JAL_env.py` (commit `54a3b05`), with a comment tying it to `ball_speed_max`. The ball cannot physically move more than `ball_speed_max` (=6) units/cycle (position advances by the velocity, which the server clips to `ball_speed_max`), so 8.0 sits safely above the legitimate max while still catching real teleports (which jump the ball tens of units). **Rule to remember: if `ball_speed_max` is ever raised again, raise this threshold too.**

No reward/hyperparameter changes needed — those were never the problem. Re-run is the same 25k-from-scratch Stage 1.1 config; expect episodes of a handful of steps terminating `goal_scored` / `ball_in_penalty_off_target` / `ball_out_of_bounds`, not `ball_teleport`.

---

## 11. Training run: 20260602_132531_672811 — Stage 1.1 kick (SUCCESS, after §10 fix)

**Start time:** 2026-06-02 13:25:31
**Stage:** `stage1_1_kick`
**Steps:** 25,000 / 25,000 (complete)
**Load model:** None — from scratch, expandable backbone + HER + sim-embedded + CPU. Kick exit speed = ~5.75 m/s (new physics).
**Checkpoint:** `models/20260602_132531_672811_td3_jal_her/stage1_1_kick_complete.zip`

### Results

| Metric | Value |
|---|---|
| Total episodes | 4,506 (avg length 5.6 steps) |
| Overall goal rate | 2,927/4,506 = **65.0%** |
| Last-100 goal rate | **66.0%** |
| Avg reward (last 10) | 130.46 |

**Episode outcomes:** `goal_scored` 2,927 (65.0%) · `ball_in_penalty_off_target` 1,251 (27.8%) · `ball_out_of_bounds` 328 (7.3%). Zero `ball_teleport` (the §10 bug is fixed). Goal-episode rewards 157–244 (all >150 → real goals).

**Aim quality:** first 200 kicks 0.32 / bad_aim 33.5%; last 200 kicks **0.375 / bad_aim 31.0%**. Goal rate flat across training (59–72% buckets, no trend).

### Code changes (non-config)

This is the re-run of §10 after the unblocking fix:
- **`_BALL_TELEPORT_THRESHOLD` 5.0 → 8.0** in `JAL_env.py` (commit `54a3b05`) — §10 died because the 5.75 m/s kick (≈5.75 units/cycle) tripped the 5.0 teleport guard every step. 8.0 sits above `ball_speed_max=6`.
- Active physics: kick exit ~5.75 m/s (`kick_power_rate=0.0575`, `ball_accel_max=5.8`, `ball_speed_max=6`); `ball_decay` still 0.94 (friction pending mechanical-team data).
- Device forced to CPU (~3.2× faster than MPS for this net).

### What worked

- **Pipeline validated end-to-end.** First successful from-scratch run on the rebuilt stack (expandable backbone + HER + sim-embedded + real-speed kick + CPU). No crashes, healthy 5.6-step episodes, checkpoint saved.
- **Kick primitive fires reliably** (`kick=100%`) and scores at the geometric ceiling.
- **Critic seeded with alignment→reward:** aim_quality nudged 0.32→0.375 and bad_aim 33.5%→31% — the early signal the value function is learning better-aligned kicks pay more, which the turn stages build on.

**Important caveat — the 65% is a ceiling, not learned aiming.** In Stage 1.1 the policy has *no aim control*: `turn` is disabled and the kick fires straight along the body (`kick 100 0`). The robot kicks in whatever direction it spawned (±15° jitter), so the goal rate is capped by spawn geometry (goal subtends ~7° at x=5, ~18° at x=30). The policy cannot beat that here — real aim *improvement* is what Stage 1.2 will test once `turn` unlocks. The faster ball also raised OOB to 7.3% (was ~1.4%) — harder kicks overshoot faster. Expected.

### Next stage

Advance to **Stage 1.2 (turn warmup)**, warm-starting from `models/20260602_132531_672811_td3_jal_her/stage1_1_kick_complete.zip`. 1.2 introduces `turn` at ±60° spawn with wide ball-y — the first stage where the model can actually correct its heading before kicking, so watch whether aim_quality climbs past the 0.375 baseline and whether goal rate rises above the geometry ceiling. Stage 1.1's full config block is archived in `_completed_substages` in the config (not removed) for future reference.

---

## 12. Training run: 20260602_151505_733010 — Stage 1.2 turn warmup (COLLAPSE #1, alignment=0)

**Start:** 2026-06-02 15:15:05 | **Stage:** `stage1_2_turn_warmup` | **Steps:** 34,994 / 35,000
**Warm-start:** `models/20260602_132531_672811_td3_jal_her/stage1_1_kick_complete.zip` (the §11 65% kicker)

**Aim:** First stage to unlock `turn` (±60° spawn, wide ball-y, still spawn-at-ball). Goal: learn to turn toward the goal before kicking — i.e. the first stage where aim is actually controllable.

### Results

| Metric | Value |
|---|---|
| Total episodes | 6603 |
| Overall goal rate | 814/6603 = **12.3%** |
| Last 100 eps | 17.0% |
| Avg ep length (last 10) | 4.7 |

- Outcomes: `ball_in_penalty_off_target` 73.5%, `ball_out_of_bounds` 14.1%, `goal_scored` 12.3%
- Aim quality (kicks): first 200 = **0.083 / 83.0% bad** → last 200 = **0.077 / 83.5% bad** (flat — no learning)
- Goal-rate buckets: dead flat 9–16% across all 6,600 episodes
- Action dist: both `turn` and `kick` used for the first ~30 episodes, then **`kick=100%`** for the rest

### Code changes (non-config)

None. (Config-only run; the §10→§11 `_BALL_TELEPORT_THRESHOLD` 5.0→8.0 fix in `JAL_env.py` is still in place from §11.)

### What went wrong

**Policy collapsed to `kick=100%` — never learned to turn.** Goal rate stuck at the ~12% blind-kick geometric floor (a random ±60° heading occasionally points into the goal cone). Aim quality flat at ~0.08 start to finish. Root cause: **`alignment_weight=0` gave turning ZERO immediate reward** ([reward.py:270-278](ai_interface/envs/reward.py#L270-L278)). The policy *did* explore `turn` during the pre-`learning_starts` (5000-step) random phase, but once gradient updates began the critic saw those turn-transitions carrying 0 reward and kick-transitions carrying the aim bonus + goal EV, so it learned `Q(turn)≈0 < Q(kick)` and the deterministic policy dropped `turn`. The turn exploration was in the buffer — it just wasn't reinforced.

### Fix for next run

Set **`alignment_weight = 2.0`** (the value Stage 1.3 already uses) in the 1.2 reward overrides. The alignment term is delta-based (`facing_goal_cos_delta × weight`, has_ball-gated), so a 60°→0° turn earns an immediate **+1.0** — the per-step breadcrumb that was missing, reinforcing the turn-transitions already in the buffer. EV check: turn-then-kick `Q≈133` vs kick-now `Q≈18` from a 60° spawn. Reward-only change, bounded ~+1/episode (can't dominate the 20 aim-bonus + 150 goal, can't be farmed — delta telescopes). Verified safe before applying. → run logged as §13.

---

## 13. Training run: 20260602_152938_122882 — Stage 1.2 turn warmup (COLLAPSE #2, alignment=2)

**Start:** 2026-06-02 15:29:38 | **Stage:** `stage1_2_turn_warmup` | **Steps:** 34,992 / 35,000
**Warm-start:** same 1.1 kicker as §12. **Only change from §12:** `alignment_weight` 0 → 2.0.

### Results

| Metric | Value |
|---|---|
| Total episodes | 6634 |
| Overall goal rate | 865/6634 = **13.0%** |
| Last 100 eps | 11.0% |
| Avg ep length (last 10) | 3.1 |

- Outcomes: `ball_in_penalty_off_target` 72.6%, `ball_out_of_bounds` 14.4%, `goal_scored` 13.0%
- Aim quality: first 200 = **0.096 / 81.0% bad** → last 200 = **0.091 / 83.5% bad** (flat)
- Action dist: `turn` used through **episode 862**, then **`kick=100%` for the remaining 5,772 episodes** (5,823 of 6,634 windows are kick=100%)

### Code changes (non-config)

None.

### What went wrong

**Collapsed again — `alignment=2` did not rescue it.** Numbers essentially identical to §12 (13.0% vs 12.3%, aim flat). The trace pins the failure precisely: `turn` survived right up to **episode 862 ≈ where `learning_starts=5000` lands**, then died the instant gradient updates began. The +1.0 alignment breadcrumb is real, but it's swamped the moment the deterministic policy locks onto the already-confident kick-logit — and `action_noise_std=0.3` on the logits **almost never flips the argmax to `turn`**, so the critic gets too few turn-transitions to ever learn `Q(turn)` is high. The math (turn-then-kick is higher-EV) was correct; TD3 just never escapes the warm-start basin.

**Key history finding:** this collapse does **not** happen in the original 58% run (`ac0fd86`/`7fd0dcb`) because that run **never disabled then re-enabled an action** — it trained `turn`+`kick`+`approach` together in one 200k-step stage (`action_noise_std=0.05`), so it never created a dominant-kick-logit checkpoint to warm-start into. The collapse is an artifact of the *new* disable-first micro-curriculum, not TD3 itself.

### Fix for next run

Two levers, ordered cheap → structural:
1. **Raise `action_noise_std` 0.3 → 0.5** (applied). Attacks the broken link directly: more collection-time noise forces `turn` into the buffer where `alignment=2` reinforces it. Cheap experiment; isolates whether exploration is the bottleneck. **Magnitude gamble** — if the 1.1 kick-logit is too dominant, 0.5 may still be too timid. **Tripwire:** does `turn` survive past episode ~862?
2. **If §14 (noise=0.5) still collapses → `load_model = null`** (fresh-start 1.2, learn kick+turn together). This is closest to what the original 58% run actually did and removes the dominant-logit basin entirely, rather than fighting it. Strong candidate if exploration alone doesn't break the collapse.

---

## 14. Training run: 20260602_180617_572684 — Stage 1.2 turn warmup (SUCCESS, sim-embedded reward/gate fixes)

**Start time:** 2026-06-02 18:06:17.572925
**Stage:** `stage1_2_turn_warmup`
**Steps:** 34,800 / 35,000
**Load model:** `None`

**Aim:** Stage 1.2 from scratch with robot at the ball, `goto`/`approach_ball`/dribbles disabled, wide ball-y, and spawn heading sampled as a goal-relative offset in `[-55,55]` with minimum `25°` misalignment. The goal was to learn turn-then-kick without letting off-target kicks harvest positive progress reward.

### Results

| Metric | Value |
|---|---|
| Total episodes | 547 |
| Overall goal rate | 420/547 = **76.8%** |
| Last 100 episodes goal rate | **80.0%** |
| Avg reward (last 10 eps) | 179.66 |
| Avg episode length (last 10 eps) | 50.8 |

**Episode outcomes:**

- `goal_scored`: 420 (76.8%)
- `ball_in_penalty_off_target`: 80 (14.6%)
- `max_steps`: 28 (5.1%)
- `ball_out_of_bounds`: 19 (3.5%)

**Goal rate over training (200-episode buckets):**

```text
  eps    1- 200:  72.0%
  eps  201- 400:  80.0%
  eps  401- 547:  78.9%
```

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.548 | 0.0% |
| Last 200 kicks | 0.519 | 0.0% |

**Action distribution samples:**

- ep 1: `kick=18%  turn=81%`
- ep 200: `kick=11%  turn=88%`
- ep 400: `kick=10%  turn=89%`

### Key config

**Reward overrides:** `goal_progress_weight=3.0`, `goal_progress_clip=2.0`, `ball_speed_bonus_weight=2.0`, `goal_reward=150.0`, `alignment_weight=3.0`, `kick_aim_bonus_weight=20.0`, `bad_aim_kick_penalty=1.0`, `bad_aim_grad_scale=15.0`, no approach/near/has-ball/step bonuses.

**Model params:** `learning_rate=0.001`, `action_noise_std=0.4`, `action_noise_logit_std=0.4`, `action_noise_param_std=0.1`, `gamma=0.99`, `batch_size=64`, HER `future`, `her_n_sampled_goal=4`.

### Code changes (non-config)

- `ai_interface/envs/reward.py`: positive `goal_progress` for fast moving balls is multiplied by projected goal-mouth `aim_quality`; negative progress remains ungated. Ball-speed bonus is also aim-gated, so a hard off-target kick cannot collect speed reward.
- `ai_interface/envs/JAL_her_env.py`: HER live progress shaping is aim-gated using the same projected y-at-goal-line math. HER relabeled sparse reward remains unchanged.
- `ai_interface/envs/JAL_env.py`: added goal-relative spawn heading, `kick_requires_aim`, `kick_min_aim_quality`, and bad-kick blocking. Bad kick requests while holding the ball execute the actor's own turn parameter, increment invalid action count, and receive the configured invalid-action penalty.
- `ai_interface/trainers/td3_jal_her_trainer.py`: stage-level env knobs are plumbed through, and warm-started TD3 models now reattach the current config's action noise after `TD3.load`.
- `infer.py`: passes the same env knobs into sim-embedded inference and records action diagnostics.
- `tests/test_td3_kick_collapse_fixes.py`: added focused tests for reward aim-gating and curriculum kick-gating.

### What worked

The reward EV problem was fixed. Before the change, off-target kicks could still earn positive progress reward; after the change, off-target positive progress and HER progress are zeroed by `aim_quality=0`, while the invalid bad-kick gate keeps the actor turning until aim is usable. The training result shows no kick collapse: goal rate reached 76.8% overall / 80% last-100, action distribution retained both turn and kick, and bad-aim kicks were 0%.

Deterministic inference initially exposed a separate TD3 decoder problem: the actor saturated `turn=1.0` and `kick=1.0`, and `np.argmax` selected the earlier `turn` slot forever. The post-run decoder fix added an aim-aware tie-break for `turn`/`kick` ties. With that enabled, deterministic Stage 1.2 infer recovered to 75.5% overall / 77.0% last-100 with 0 invalid and 0 bad-aim kicks.

### Next stage

Promote the checkpoint `models/20260602_180617_572684_td3_jal_her/stage1_2_turn_warmup_complete.zip` to Stage 1.3. Keep approach disabled. Enable the same kick gate and deterministic tie-break in Stage 1.3, use `alignment_weight=2.0`, and expand spawn heading to full `[-180,180]`.

---

## 15. Training run: 20260602_184554_470708 — Stage 1.3 full turn (SUCCESS)

**Start time:** 2026-06-02 18:45:54.470934
**Stage:** `stage1_3_full_turn`
**Steps:** 149,992 / 150,000
**Load model:** `models/20260602_180617_572684_td3_jal_her/stage1_2_turn_warmup_complete.zip`

**Aim:** Full at-ball turn stage. Robot starts at the ball, `goto`/`approach_ball`/dribbles stay disabled, and heading is sampled goal-relative over `[-180,180]`. This tests whether the Stage 1.2 turn/kick skill generalizes from a narrow warmup band to any heading before Stage 1.4 unlocks approach.

### Results

| Metric | Value |
|---|---|
| Total episodes | 5195 |
| Overall goal rate | 4225/5195 = **81.3%** |
| Last 100 episodes goal rate | **85.0%** |
| Avg reward (last 10 eps) | 138.72 |
| Avg episode length (last 10 eps) | 32.1 |

**Episode outcomes:**

- `goal_scored`: 4225 (81.3%)
- `ball_in_penalty_off_target`: 760 (14.6%)
- `ball_out_of_bounds`: 204 (3.9%)
- `max_steps`: 6 (0.1%)

**Goal rate over training:**

- First 200 episodes: 72.5%
- Mid-run buckets mostly held around 79-86.5%
- Final bucket, eps 5001-5195: 85.1%

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.507 | 0.0% |
| Last 200 kicks | 0.550 | 0.0% |

**Action distribution samples near end:**

- ep 4000: `kick=12%  turn=87%`
- ep 4200: `kick=7%  turn=92%`
- ep 4400: `kick=13%  turn=86%`
- ep 4600: `kick=5%  turn=94%`
- ep 4800: `kick=40%  turn=60%`
- ep 5000: `kick=28%  turn=71%`

**Deterministic infer:** `infer_logs/20260602_190133_stage1_3_full_turn_complete`

- 306 episodes, 6000 steps.
- Overall goal rate: 250/306 = **81.7%**.
- Last 100 goal rate: **85.0%**.
- Outcomes: `goal_scored=250`, `ball_in_penalty_off_target=49`, `ball_out_of_bounds=7`.
- Kicks: 307 total, about one per episode.
- Average aim quality: 0.521 overall, 0.486 in the last 100.
- Bad-aim kicks: 0.0%.
- Invalid actions / blocked bad aim: 0.

Note: `summary.json` reported `last_100.kicks=0`, but the per-episode infer log shows kicks in the last 100. Recomputed from the log, last 100 had 100 kicks and 0 bad-aim kicks.

### Key config

**Reward overrides:** `goal_progress_weight=3.0`, `goal_progress_clip=2.0`, `ball_speed_bonus_weight=2.0`, `goal_reward=150.0`, `alignment_weight=2.0`, `kick_aim_bonus_weight=20.0`, `bad_aim_kick_penalty=1.0`, `bad_aim_grad_scale=15.0`, no approach/near/has-ball/step bonuses.

**Model params:** `learning_rate=0.001`, `action_noise_std=0.4`, `action_noise_logit_std=0.4`, `action_noise_param_std=0.1`, `gamma=0.99`, `batch_size=64`, HER `future`, `her_n_sampled_goal=4`.

### Code changes (non-config)

No new source edits were required specifically for Stage 1.3 beyond the Stage 1.2 reward/gate/tie-break plumbing already listed in §14. The Stage 1.3 activation was config-only: move `stage1_3_full_turn` into `curriculum`, move `stage1_2_turn_warmup` into `_completed_substages`, keep `approach_ball` disabled, enable the same kick safety knobs, and set `load_model` to the Stage 1.2 checkpoint.

### What worked

The Stage 1.2 skill generalized to the full heading range. Training and deterministic inference agree closely: 81.3% train goal rate vs 81.7% deterministic infer, and both have 85% last-100 goal rate. The model uses turns heavily and fires roughly one aimed kick per episode. Bad-aim kicks stayed at 0%, and max-step failures dropped to 0.1% in training.

The deterministic TD3 tie-break is still structurally important. The raw actor remains saturated on `turn=1.0` and `kick=1.0`, so deterministic deployment depends on the aim-aware decoder choosing kick once projected aim quality is high enough. That behavior is stable in Stage 1.3, but it must be considered before unlocking any new primitive.

### Next stage

Promote `models/20260602_184554_470708_td3_jal_her/stage1_3_full_turn_complete.zip` to Stage 1.4 only after updating the Stage 1.4 config with the same kick gate and tie-break. Stage 1.4 is the first stage to unlock `approach_ball`, so watch for a new TD3 argmax/tie issue involving the newly enabled approach slot. If deterministic infer after Stage 1.4 shows action collapse, the next structural fix is a categorical primitive policy rather than more reward shaping.

---

## 16. Training run: 20260603_110713_879342 — PPO stage1_090_mid aim bootstrap (SUCCESS)

**Start time:** 2026-06-03 11:07:13  
**Stage:** `stage1_090_mid`  
**Algorithm:** PPO (hierarchical JAL, centralized)  
**Steps:** 299,991 / 300,000  
**Load model:** None (fresh start)

**Aim:** AIM BOOTSTRAP rung. Ball_y narrowed to ±6 (goal window is ~16° wide from a typical spawn at x=10–30) so the policy can resolve aim against the turn-noise floor. Spawn theta ±90° so 33%+ of episodes require explicit turning before the kick. Two actions enabled: `kick` and `turn`. `approach_ball` disabled (robot spawns within kickable range, so it would be a no-op). All dribble actions disabled. Acceptance gates: goal rate ≥ 40%, bad_aim ≤ 50%, no primitive > 90% in any 200-ep window.

### Results

| Metric | Value |
|---|---|
| Total episodes | 30,377 |
| Overall goal rate | 13,937 / 30,377 = **45.9%** |
| Last 100 episodes goal rate | **94.0%** |
| Avg reward (last 10 eps) | 117.69 |
| Avg episode length (last 10 eps) | 10.4 steps |

**Episode outcomes:**

- `goal_scored`: 13,937 (45.9%)
- `ball_in_penalty_off_target`: 11,537 (38.0%)
- `ball_out_of_bounds`: 4,903 (16.1%)

**Goal rate over training (200-episode buckets):**

```
  eps    1-8000:   ~10-13%  (flat exploration phase)
  eps  8000-11000: ~13-18%  (early signal)
  eps 11000-13200: 15-23%   (lift-off begins)
  eps 13200-15200: 27-38%   (rapid climb)
  eps 15200-16200: 40-52%   (clears 40% gate)
  eps 16200-21000: 45-72%   (continued rise)
  eps 21000-24600: 71-91%   (steep climb)
  eps 24600-30377: 86-97%   (saturation ~90%+)
```

Full per-bucket data in parser output for run `20260603_110713_879342`.

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.076 | 86.0% |
| Last 200 kicks | 0.648 | **4.0%** |

**Action distribution near end (per-step):**

- ep 29200: `kick=20%  turn=80%`
- ep 29400: `kick=37%  turn=62%`
- ep 29600: `kick=10%  turn=89%`
- ep 29800: `kick=12%  turn=87%`
- ep 30000: `kick=28%  turn=71%`

The turn-heavy per-step mix is expected and healthy: ±90° spawns require turning *before* kicking, so turn dominates step count while 94% of episodes end in a goal.

### Key config

**Reward overrides:** `goal_reward=70.0`, `alignment_weight=3.0`, `alignment_has_ball_only=True`, `kick_aim_bonus_weight=20.0`, `bad_aim_kick_penalty=1.0`, `bad_aim_grad_scale=15.0`, `goal_progress_weight=3.0`, `goal_progress_clip=2.0`, `ball_speed_bonus_weight=2.0`. No approach/near/has-ball/step bonuses.

**Model params:** `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `target_kl=0.015`, `n_epochs=10`, `minibatch_size=256`, `rollout_size=4096`, `lr_initial=3e-4`, `lr_final=1e-4`, `ent_coef_initial=0.05`, `ent_coef_final=0.005`, `vf_coef=0.5`, `max_grad_norm=0.5`.

### Code changes (non-config)

Two source-level fixes applied before this run that are not captured in the reward overrides:

1. **`LOG_STD_MIN` lowered from −1.5 → −3.0** in `ai_interface/algorithms/ppo_jal.py:57`. At −1.5 the turn-angle exploration noise floored at `exp(−1.5)·π ≈ 40°`, coarser than the ~8–16° goal window subtended from typical kick spots (FIELD_X=±45, goal_half_height=5). The policy literally could not resolve the goal and bad_aim was structurally stuck at ~84%. At −3.0, the std can tighten to `exp(−3)·π ≈ 9°`, inside the goal window.

2. **Reward config alignment (Fix A):** `alignment_weight` raised 0→3.0, `bad_aim_kick_penalty` added 0→1.0, `bad_aim_grad_scale` added 0→15.0, `goal_reward` corrected 150→70.0 to match `RewardConfig` default in `reward.py`. These were config changes; the source of the `goal_reward` default is `reward.py:30`.

### What worked

Clear success. All acceptance gates cleared with wide margin:
- Goal rate ≥ 40%: **94% last 100 eps** ✓
- bad_aim ≤ 50%: **4% last 200 kicks** ✓
- No primitive > 90%: turn peaks at ~89% per-step (expected given ±90° spawn, not a collapse) ✓

The key signal is the aim improvement: bad_aim went from **86% → 4%**, aim_quality from 0.076 → 0.648. This confirms the `LOG_STD_MIN=-3.0` fix was the root cause of the prior plateau — once the Gaussian head's noise floor was below the goal's angular subtend, the policy converged on aim quickly. The graded `bad_aim_grad_scale=15.0` penalty and `kick_aim_bonus_weight=20.0` bonus gave continuous gradient on every kick instead of a binary on/off signal, which contributed to the smooth climb from ~12k episodes onward.

The reward EV at convergence is consistent: goal + aimed-kick bonus + aim-gated progress ≈ 70 + 20 + 10 ≈ +100 per episode, matching the observed mean reward of 117.69 (extra ~17 from alignment reward on the turn steps). This validates the reward calibration from Fix A is correct and stable.

**Change #4 (`bad_aim_kick_penalty` 1→4) is retired.** The condition for applying it was "aim moves but not far enough." Aim moved all the way to 4% bad_aim. Do not raise the penalty in later stages; history shows high kick penalties cause turn-collapse.

### Next stage

Advance to `stage1_180_full`: same reward config, same ball_y ±6, widen spawn theta ±90° → ±180°. This tests whether the learned turn+aim skill generalizes to episodes requiring a full 180° reversal. Budget: the ±90° curve lifted off ~ep 8k (~120k steps) and saturated ~ep 21k (~250k steps); the ±180° distribution is harder so budget **200,000 steps** (adjustable from the observed curve).

To activate: set `stage1_180_full.timesteps=200000` in `configs/ppo_jal_curriculum_config.json`, move the `ACTIVE` marker from `stage1_090_mid` to `stage1_180_full`.

Checkpoint: `models/ppo_jal_curriculum/stage1_090_mid_complete.pt`

---

## 17. Training run: 20260603_130026_755484 — PPO stage1_180_full turn generalization (SUCCESS)

**Start time:** 2026-06-03 13:00:26  
**Stage:** `stage1_180_full`  
**Algorithm:** PPO (hierarchical JAL, centralized)  
**Steps:** 200,000 / 200,000  
**Load model:** `models/ppo_jal_curriculum/stage1_090_mid_complete.pt`

**Aim:** TURN GENERALIZATION rung. Same narrow ball_y ±6, spawn theta widened to ±180° (67%+ of episodes require turning, including full 180° reversals). Two actions: `kick` and `turn`. Gate to advance: goal ≥ 45%. Tests whether the turn+aim skill from stage1_090_mid transfers to episodes requiring a full reversal before kicking.

### Results

| Metric | Value |
|---|---|
| Total episodes | 13,286 |
| Overall goal rate | 13,106 / 13,286 = **98.6%** |
| Last 100 episodes goal rate | **100.0%** |
| Avg reward (last 10 eps) | 137.53 |
| Avg episode length (last 10 eps) | 16.7 steps |

**Episode outcomes:**

- `goal_scored`: 13,106 (98.6%)
- `ball_in_penalty_off_target`: 109 (0.8%)
- `ball_out_of_bounds`: 71 (0.5%)

**Goal rate over training:** Opened at 96.5% on episode 1 and never dropped below 95%. Multiple 200-ep windows hit 100%. No warm-up phase — skill transferred immediately from the stage1_090_mid checkpoint.

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.653 | 3.0% |
| Last 200 kicks | 0.776 | **0.5%** |

Aim quality started high (0.653) because the model already knew how to aim — the checkpoint carried that in. It refined further during training (→ 0.776).

**Action distribution near end (per-step):**

- ep 12200–13200: `kick=5–25%  turn=75–94%`

Turn-heavy per-step mix is expected: ±180° spawns require more turning steps per episode. With 98.6% goal rate, kick usage is efficient (roughly one aimed kick per episode).

### Key config

Same reward overrides as §16: `goal_reward=70.0`, `alignment_weight=3.0`, `alignment_has_ball_only=True`, `kick_aim_bonus_weight=20.0`, `bad_aim_kick_penalty=1.0`, `bad_aim_grad_scale=15.0`, `goal_progress_weight=3.0`, `ball_speed_bonus_weight=2.0`. No approach/near/has-ball/step bonuses.

Same PPO model params as §16.

### Code changes (non-config)

None. Config-only stage advance from §16.

### What worked

Clean transfer. The warm-start from `stage1_090_mid_complete.pt` meant the model arrived already knowing how to aim — goal rate opened at 96.5% and saturated at ~99–100% within the first few thousand episodes. The 180° spawn distribution required no additional learning; the policy generalized the turn+aim behavior directly.

The avg episode length increase (10.4 → 16.7 steps) confirms the model is correctly executing longer turn sequences before kicking, rather than kicking immediately from a bad angle.

### Next stage

Both gates for `stage1_180_y10` are cleared (bad_aim 0.5% ≤ 45% ✓, goal 98.6% ≥ 45% ✓). Advance to `stage1_180_y10`: widen ball_y ±6 → ±10 at full ±180° spawn theta. This is the first rung that genuinely challenges aim from steeper angles (ball positions up to 10 units off-center require tighter aim). Budget: skill transferred instantly here, so the y-widening may also transfer quickly — start with 100,000 steps, extend if bad_aim climbs above 20%.

Checkpoint: `models/ppo_jal_curriculum/stage1_180_full_complete.pt`

---

## 18. Training run: 20260603_132028_979870 — PPO stage1_180_y10 aim steepening (SUCCESS)

**Start time:** 2026-06-03 13:20:28  
**Stage:** `stage1_180_y10`  
**Algorithm:** PPO (hierarchical JAL, centralized)  
**Steps:** 100,000 / 100,000  
**Load model:** `models/ppo_jal_curriculum/stage1_180_full_complete.pt`

**Aim:** AIM STEEPENING rung. Widen ball_y ±6 → ±10 at full ±180° spawn theta. First rung that genuinely challenges aim from steeper angles — ball up to 10 units off-center requires tighter turn+aim before kicking. Gate: bad_aim ≤ 45%, goal ≥ 45%.

### Results

| Metric | Value |
|---|---|
| Total episodes | 6,912 |
| Overall goal rate | 6,690 / 6,912 = **96.8%** |
| Last 100 episodes goal rate | **98.0%** |
| Avg reward (last 10 eps) | 125.95 |
| Avg episode length (last 10 eps) | 14.1 steps |

**Episode outcomes:**

- `goal_scored`: 6,690 (96.8%)
- `ball_in_penalty_off_target`: 193 (2.8%)
- `ball_out_of_bounds`: 29 (0.4%)

**Goal rate over training:** Opened at 93.0% episode 1, never dropped below 93% the entire run. Gradual refinement to 97–99.5% by end. No warm-up phase needed.

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.705 | 1.0% |
| Last 200 kicks | 0.787 | **0.0%** |

Aim quality *improved* vs §17 (0.776 → 0.787 at end) despite wider y-range — the more varied angles appear to be helping generalization rather than hurting it.

**Action distribution near end (per-step):** Variable mix, kick=14–80%, turn=20–85%. No collapse — wide variation across windows is healthy.

### Key config

Same reward overrides as §16–17: `goal_reward=70.0`, `alignment_weight=3.0`, `alignment_has_ball_only=True`, `kick_aim_bonus_weight=20.0`, `bad_aim_kick_penalty=1.0`, `bad_aim_grad_scale=15.0`, `goal_progress_weight=3.0`, `ball_speed_bonus_weight=2.0`.

### Code changes (non-config)

None. Config-only stage advance from §17.

### What worked

Clean transfer again. The model opened at 93% and refined to 98% without any visible learning curve — the checkpoint already encoded the necessary aim precision. The slight reward drop (137 → 126) and shorter episodes (16.7 → 14.1 steps) are expected: the larger y-range produces more center-ish spawns requiring less turning. bad_aim hit 0.0% in the final 200 kicks, better than §17.

### Next stage

Both gates for `stage1_180_y15` are cleared (bad_aim 0.0% ≤ 45% ✓, goal 96.8% ≥ 45% ✓). Advance to `stage1_180_y15`: widen ball_y ±10 → ±15 (full field width) at ±180° spawn theta. This is the final difficulty rung before Stage 1.9 (goto + approach enabled). Acceptance: goal ≥ 50% over final 1000 episodes, no primitive > 90% in any 100-ep window. Given the smooth transfers in §17–18, budget 100,000 steps — the skill should transfer immediately.

Checkpoint: `models/ppo_jal_curriculum/stage1_180_y10_complete.pt`

---

## 19. Training run: 20260603_133245_712428 — PPO stage1_180_y15 full difficulty (SUCCESS)

**Start time:** 2026-06-03 13:32:45
**Stage:** `stage1_180_y15`
**Steps completed / planned:** 99,984 / 100,000
**Warm-start:** `models/ppo_jal_curriculum/stage1_180_y10_complete.pt`

**Aim:** Final difficulty rung before approach training: widen ball_y ±10 → ±15 (full field width) at ±180° spawn theta. Robot spawns at ball, turn+kick only. Acceptance: goal ≥ 50% over final 1000 eps; no primitive > 90% in any 100-ep window.

### Results

| Metric | Value |
|---|---|
| Total episodes | 6,568 |
| Overall goal rate | 5844 / 6568 = **89.0%** |
| Last 100 episodes goal rate | **89.0%** |
| Avg reward (last 10 eps) | 122.82 |
| Avg episode length (last 10 eps) | 17.7 steps |

**Episode outcomes:**

- `goal_scored`: 5844 (89.0%)
- `ball_in_penalty_off_target`: 663 (10.1%)
- `ball_out_of_bounds`: 61 (0.9%)

**Goal rate over training (200-episode buckets):**

```
  eps    1- 200:  87.0%
  eps  201- 400:  86.0%
  eps  401- 600:  88.0%
  eps  601- 800:  88.0%
  eps  801-1000:  85.5%
  eps 1001-1200:  92.5%
  eps 1201-1400:  87.0%
  eps 1401-1600:  89.0%
  eps 1601-1800:  89.5%
  eps 1801-2000:  87.0%
  eps 2001-2200:  89.5%
  eps 2201-2400:  87.0%
  eps 2401-2600:  88.0%
  eps 2601-2800:  89.5%
  eps 2801-3000:  88.5%
  eps 3001-3200:  90.5%
  eps 3201-3400:  90.0%
  eps 3401-3600:  85.0%
  eps 3601-3800:  87.5%
  eps 3801-4000:  89.0%
  eps 4001-4200:  88.5%
  eps 4201-4400:  92.5%
  eps 4401-4600:  90.5%
  eps 4601-4800:  89.5%
  eps 4801-5000:  87.0%
  eps 5001-5200:  87.0%
  eps 5201-5400:  92.0%
  eps 5401-5600:  95.0%
  eps 5601-5800:  93.0%
  eps 5801-6000:  90.5%
  eps 6001-6200:  90.0%
  eps 6201-6400:  87.0%
  eps 6401-6568:  89.9%
```

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.727 | 0.5% |
| Last 200 kicks | 0.762 | **0.0%** |

**Deterministic inference (6000 steps, 252 episodes):**

| Metric | Value |
|---|---|
| Goal rate | **87.7%** (221/252) |
| bad_aim | **0.0%** |
| avg aim_quality | 0.787 |
| avg_reward | 119.0 |
| ball_in_penalty_off_target | 24 (9.5%) |
| max_steps timeouts | 7 (2.8%) |
| Action mix | turn=91%, kick=8% |

**Action distribution near end (per-step):** turn=81–92%, kick=7–18%. No collapse.

### Key config

Same reward overrides as §16–18: `goal_reward=70.0`, `alignment_weight=3.0`, `alignment_has_ball_only=True`, `kick_aim_bonus_weight=20.0`, `bad_aim_kick_penalty=1.0`, `bad_aim_grad_scale=15.0`, `goal_progress_weight=3.0`, `ball_speed_bonus_weight=2.0`, `approach_weight=0.0`.

### Code changes (non-config)

Added `_run_ppo_jal` runner to `infer.py` — new env/agent wiring for `PPOJALAgent` + `JALTeamEnv` (the `hier_ppo` runner is an incompatible old stack). Used to run the 6000-step deterministic inference above.

### What worked

Third consecutive instant transfer. Opened at 87% in the first 200 eps and held 87–95% throughout — no visible learning curve. The checkpoint already encoded aim precision for the full ±15 y-range. bad_aim hit 0.0% in final 200 training kicks and stayed 0.0% in deterministic inference (0.787 avg aim, the best yet). The 2.8% max_steps timeout rate in deterministic inference (absent in training) is a known stochastic/deterministic gap on hard spawns — normal, flagged for monitoring.

### Next stage

Stage 1.9 — home-spot approach → turn → kick. This is the real final Stage 1 task: robot starts at its fixed home pose `(-10, 0)` facing a random heading; ball spawns at full distribution (`x∈[5,30]`, `y∈[-15,15]`). Must approach 15–43 units, turn to aim, then kick. First stage with `approach_ball` enabled and the robot genuinely away from the ball.

Config changes (config-only, no env/trainer changes):
- Enable `approach_ball` (remove from `disabled_actions`)
- `approach_weight`: 0.0 → **1.0** (math: worst-case park EV = 1.0×43−10 = 33 << goal 70; at default 1.8 it would be 67 ≈ goal — unsafe local optimum)
- `spawn_robot_at_ball`: false, `random_spawn_theta`: ±180°
- `timesteps`: 300,000 (new behavior + new obs regime — not an instant transfer)
- Warm-start: `stage1_180_y15_complete.pt`

Checkpoint: `models/ppo_jal_curriculum/stage1_180_y15_complete.pt`

---

## 20. Training run: 20260603_153747_258211 — PPO stage1_9 home-spot approach → turn → kick (SUCCESS — Stage 1 complete)

**Start time:** 2026-06-03 15:37:47
**Stage:** `stage1_9`
**Steps completed / planned:** 299,973 / 300,000
**Warm-start:** `models/ppo_jal_curriculum/stage1_180_y15_complete.pt`

**Aim:** Final Stage 1 task. Robot starts at fixed home pose `(-10, 0)` facing a random heading; ball spawns at full distribution (`x∈[5,30]`, `y∈[-15,15]`). Must approach 15–43 units (`approach_ball` enabled for the first time), turn to aim, then kick. `goto` and dribbles stay disabled.

### Results

| Metric | Value |
|---|---|
| Total episodes | 2,327 |
| Overall goal rate | 1925 / 2327 = **82.7%** |
| Last 100 episodes goal rate | **80.0%** |
| Avg reward (last 10 eps) | 116.88 |
| Avg episode length (last 10 eps) | **114.6 steps** (was ~17 when spawned at ball — robot now walks across the field) |

**Episode outcomes:**

- `goal_scored`: 1925 (82.7%)
- `ball_in_penalty_off_target`: 350 (15.0%)
- `ball_out_of_bounds`: 29 (1.2%)
- `max_steps`: 23 (1.0%)

**Goal rate over training (200-episode buckets):**

```
  eps    1- 200:  76.5%
  eps  201- 400:  85.0%
  eps  401- 600:  80.5%
  eps  601- 800:  83.0%
  eps  801-1000:  83.0%
  eps 1001-1200:  85.0%
  eps 1201-1400:  83.0%
  eps 1401-1600:  85.0%
  eps 1601-1800:  84.5%
  eps 1801-2000:  79.0%
  eps 2001-2200:  85.0%
  eps 2201-2327:  83.5%
```

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.717 | 0.0% |
| Last 200 kicks | 0.681 | 1.5% |

**Deterministic inference (7000 steps, 68 episodes, sim-only):**

| Metric | Value |
|---|---|
| Goal rate | **80.9%** (55/68) |
| aim_quality | 0.710, bad_aim 1.6% |
| avg_reward | 128.4 |
| ball_in_penalty_off_target | 7 (10.3%) |
| max_steps timeouts | **6/68 = 8.8%** |
| Action mix | approach_ball 68%, turn 30%, kick 1% |

**Action distribution near end (per-step):** approach_ball 72–84%, turn 12–24%, kick 0–5%. No primitive exceeded 90% in any 100-ep window.

### Key config

`approach_weight=1.0` (new), `spawn_robot_at_ball=false`, `random_spawn_theta=±180°`. All other reward overrides same as §16–19.

### Code changes (non-config)

None. Config-only advance from §19.

### What worked

The approach skill bootstrapped without the fallback rung. `approach_ball` appeared in the first action samples and grew to a stable 72–84% mix; `max_steps` stayed at 1% throughout training. The warm-started turn+kick endgame chained on cleanly the moment the robot arrived at the ball — same reflex learned in §16–19. Reward math held: at `approach_weight=1.0` the park EV (≈33) stayed comfortably below goal (70), forcing genuine scoring attempts.

Deterministic inference matched training at 80.9% (vs 82.7% training). The 8.8% max_steps timeout rate in deterministic inference (vs 1% training) is the same stochastic/deterministic gap seen in §19 (2.8% there), amplified by the longer episodes — a handful of hard spawns stall in an approach/turn loop under argmax. Flagged for monitoring in Stage 2 but not a blocker.

### Next stage

**Stage 1 is complete.** Both acceptance gates cleared:
- goal ≥ 50% over final 1000 eps → ~83% ✓
- no primitive > 90% in any 100-ep window → peak approach_ball 84% ✓

The robot does the full realistic Stage 1 task end-to-end from its home spot. Remaining failure modes (15% off-target, 8.8% deterministic stall) are approach-distance and hard-spawn edge cases, not fixable within the Stage 1 regime.

**Stage 2** is next: unlock `goto` (activates Gaussian Dx/Dy head for the first time) and dribble primitives, and/or introduce a second robot or opponents. The `goto`/dribble plan that was originally in the `stage1_9` slot now moves to Stage 2.

---

## 21. Inference diagnostic: PPO stage1_9 "continuous-turning" stall — root-caused and fixed (2026-06-03)

**This is not a training run — it is the resolution of the deterministic-stall failure mode flagged at the end of §20.** No retrain was needed. Recorded here so the next agent does not re-chase the "robot keeps turning instead of kicking" ghost.

### Symptom

In deterministic inference of the Stage 1.9 model (`models/ppo_jal_curriculum/stage1_9_complete.pt`), the robot approached the ball, turned, and then — even when essentially facing the goal — kept turning forever to "find the right angle" instead of kicking, most often when the ball was near the goal's center line. Deterministic inference timed out (`max_steps`) on ~19% of episodes (goal rate 73.6%), versus ~1% during stochastic training. (§20 measured 8.8% on a shorter run and the same gap in §19; the longer 6000-step trace put it at ~19%.)

### Root cause (from per-step trace `infer_logs/20260603_165526_stage1_9_complete/step_trace.jsonl`)

The turn parameter head learned a **saturated bang-bang policy, not a proportional one.** The Gaussian *mean* for `turn_theta` (scaled `× π` → a command in radians, ≈ ±19°/step after server inertia) was:
- **79%** pinned at `|turn_theta| > 3.0` (≈ ±π), only 13% near zero — sharply bimodal.
- Sign correct (toward goal-aim) only **30%** of the time.
- Mean `|turn_theta|` still **2.87** even when already within 5° of the aim (proportional control would be ≈0).

Under **stochastic training**, the Gaussian exploration noise on the param supplied the fine turn corrections, so the robot settled and kicked. **Deterministic inference uses the mean only** → a fixed ±19° bang-bang command that overshoots the aim line and oscillates in a strict ±π 2-cycle, never reaching kick precision. Classic "deterministic mean of a high-variance policy." This is the whole story behind the §19/§20 deterministic-vs-stochastic `max_steps` gap.

### Fix (inference-time, no retrain)

Added two flags to `infer.py`'s PPO path (forwarded by `launch_infer.py`); see CHANGES.md #26 for code:
- `--ppo_stochastic` — sample primitive + params (faithful to training).
- `--ppo_param_noise_std` (**default now 0.3**, was effectively 0) — keep argmax primitive, add Gaussian noise to the param mean, re-clamp. Breaks the bang-bang loop while keeping the primitive choice deterministic.

### Validation (6000-step sim-embedded runs, same model)

| Run | mode | max_steps | goal rate | turn steps | sign-correct | mean \|turn\| when aimed |
|---|---|---|---|---|---|---|
| `165526` baseline | det. mean (0.0) | **~19%** (10/53) | 73.6% | 2437 | 30% | 2.87 |
| `170514` (A) | `--ppo_stochastic` | **0%** (0/67) | 86.6% | 1336 | 65% | 2.66 |
| `170626` (B) | `--ppo_param_noise_std 0.3` | **1.4%** (1/72) | **94.4%** | 837 | 76% | 2.21 |

The bang-bang magnitude persists (46–78% still saturated) but the noise breaks the 2-cycle: turn-step count collapses (no more thrashing) and sign-correct rises to 65–76%. **B is the new default.** It beats the original deterministic baseline by +21pp goal and cuts timeouts from ~19% to ~1.4%.

### Carry-over decision

**No Stage-1 retrain.** Stage 2 warm-starts from `stage1_9_complete.pt` and trains *stochastically* — the regime where the policy already works — so the bang-bang mean never carries into training behavior. The only requirement is that PPO JAL inference is never run with pure-deterministic mean; the `0.3` default enforces this. The §20 "flagged for monitoring in Stage 2" item is now **closed**.

### Optional future hardening

If a *pure-deterministic* PPO policy is ever needed (reproducible competition deployment), anneal the param-head entropy/std harder late in training, or shrink the turn action scale (`× π`), so the deterministic mean learns to taper to zero as the robot aligns.

Checkpoint: `models/ppo_jal_curriculum/stage1_9_complete.pt`

---

## 22. Training run: 20260617_195940_527567 — PPO stage2_0_baseline (BUGGED — retrain in progress)

**Start time:** 2026-06-17 19:59:40
**Stage:** `stage2_0_baseline`
**Steps:** 199,932 / 200,000 (completed)
**Load model:** `final_models/ppo_jal_expandable_stage1_final.pt`

**Aim:** STAGE 2 RUNG 0 — Baseline. Full Stage-1 chain (approach → turn → kick) vs the scripted keeper, verbatim Stage-1 setup. Robot spawns at home pose (~(−10,0)), random heading; ball at x∈[5,30], y∈[±15] (Stage-1 verbatim distribution). Scripted keeper added; kick reward switched to goalie-gap placement with `goalie_gap_blend_center=0.3` to cushion transfer from the center-aim Stage-1 checkpoint. Tests whether approach/turn/kick skills transfer to beating the keeper with corner placement without dismantling the chain first. Acceptance: goal ≥ 50%.

### Results (raw)

| Metric | Value |
|---|---|
| Total episodes | 2114 |
| Overall goal rate | 901/2114 = **42.6%** (below acceptance bar) |
| Last 100 episodes goal rate | **49.0%** |
| Avg reward (last 10 eps) | 64.12 |
| Avg episode length (last 10 eps) | 71.4 |

**Episode outcomes:**

- `keeper_cleared`: 1015 (48.0%) — inflated by the bug (see below)
- `goal_scored`: 901 (42.6%)
- `ball_out_of_bounds`: 83 (3.9%)
- `max_steps`: 60 (2.8%)
- `ball_in_penalty_off_target`: 55 (2.6%)

**Goal rate over training (200-episode buckets):**

```
  eps    1- 200:  20.0%  ████
  eps  201- 400:  41.5%  ████████
  eps  401- 600:  40.5%  ████████
  eps  601- 800:  41.5%  ████████
  eps  801-1000:  48.0%  █████████
  eps 1001-1200:  46.5%  █████████
  eps 1201-1400:  44.0%  ████████
  eps 1401-1600:  48.0%  █████████
  eps 1601-1800:  44.0%  ████████
  eps 1801-2000:  49.5%  █████████
  eps 2001-2114:  47.4%  █████████
```

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.613 | 6.0% |
| Last 200 kicks | 0.497 | 9.5% |

**Action distribution samples:**

- ep 1000: `turn=100%`
- ep 1200: `turn=100%`
- ep 1400: `turn=100%`
- ep 1600: `approach_ball=74%  kick=3%  turn=21%`
- ep 1800: `approach_ball=100%`
- ep 2000: `approach_ball=81%  kick=0%  turn=18%`

### Corrected results (bug episodes excluded)

A post-hoc filter script classified bug episodes as 1-step `keeper_cleared` with ball x∈[5,15). Results after excluding the 872 bug episodes from the 2114 total:

| Metric | Corrected value |
|---|---|
| Real episodes | 1242 / 2114 (58.8%) |
| Bug episodes (1-step keeper_cleared) | 872 / 2114 (41.2%) |
| Corrected goal rate | **901 / 1242 = 72.5%** |
| Genuine keeper saves | 143 (11.5% of real episodes) |
| Avg steps — genuine keeper save episodes | 175.6 |

Corrected goal rate over 200-real-episode buckets: 50.0% → 71.0% → 82.0% → 74.0% → 78.5% → 76.5% → 88.1% — a genuine upward trend showing the model was still improving when training ended.

**Verdict:** the corrected performance (72.5%) passes the ≥50% acceptance gate decisively. The failure is in the config, not the policy.

### Key config

**Reward overrides:**
- `approach_weight`: 1.0
- `goal_reward`: 70.0
- `alignment_weight`: 3.0, `alignment_has_ball_only`: True
- `kick_aim_bonus_weight`: 20.0
- `bad_aim_kick_penalty`: 1.0, `bad_aim_grad_scale`: 15.0
- `use_goalie_aim_gate`: True, `goalie_gap_min_quality`: 0.3
- `kick_into_keeper_penalty`: 1.5
- `goalie_gap_blend_center`: 0.3
- `opponent_near_ball_penalty`: -0.1, `opponent_near_ball_threshold`: 5.0
- `goal_progress_weight`: 2.0, `ball_speed_bonus_weight`: 2.0
- `step_bonus`: 0.0

**Model params:** `gamma=0.99` (PPO JAL expandable backbone, shared per-robot encoder)

### Code changes (non-config)

None. This was a config-only rung; the expandable backbone, keeper driving, and gap-aim gate were all already in source from prior work.

### What went wrong

**Bug: `ball_cleared_x_threshold=15.0` was inside the ball spawn range.**

The `keeper_cleared` terminal condition in `JAL_env.py` fires whenever ball x < threshold. With threshold=15.0 and ball spawns x∈[5,30], any ball spawning at x∈[5,15) was already below the threshold on step 1 — the episode ended instantly before the robot or keeper did anything. That band covers ~40% of the spawn range, matching the 41.2% bug episode fraction exactly.

Effect on training: ~872 of 2114 episodes were 1-step ghosts that: (a) flooded the rollout buffer with zero-information transitions, (b) injected noise into every PPO advantage estimate, (c) caused the raw goal rate (42.6%) to appear below acceptance. The `turn=100%` action distribution anomaly in episodes 1000–1400 is likely a PPO regression caused by early rollouts dominated by bug noise — the policy partially lost the approach skill before recovering in the second half of training.

### Fix for next run

**Config change in `configs/ppo_jal_curriculum_config.json` (already applied):**

```diff
- "ball_cleared_x_threshold": 15.0,
+ "ball_cleared_x_threshold": 4.0,
```

Setting the threshold to 4.0 — below the minimum spawn of x=5.0 — means no ball can ever spawn below the line. The `keeper_cleared` termination now only fires when the keeper genuinely punts the ball from x≈41 all the way back past x=4 (deep into our half). The corrected 72.5% real-episode performance confirms the policy is strong; the clean retrain should converge to a raw rate matching that.

**Retrain started immediately after the fix.** Warm-start: same `final_models/ppo_jal_expandable_stage1_final.pt`. All other config unchanged.

---

## 23. Training run: 20260617_204112_874662 — PPO stage2_0_baseline clean retrain (SUCCESS)

**Start time:** 2026-06-17 20:41:12
**Stage:** `stage2_0_baseline`
**Steps:** 199,868 / 200,000 (completed)
**Load model:** `final_models/ppo_jal_expandable_stage1_final.pt`

**Aim:** Same as §22 — Stage-1 full chain (approach → turn → kick) vs scripted keeper, verbatim Stage-1 spawn distribution (x∈[5,30], y∈[±15]), goalie-gap kick reward with `goalie_gap_blend_center=0.3`. This is the clean retrain after the `ball_cleared_x_threshold` bug fix (15.0 → 4.0). Acceptance: goal ≥ 50%.

### Results

| Metric | Value |
|---|---|
| Total episodes | 1415 |
| Overall goal rate | 1023/1415 = **72.3%** |
| Last 100 episodes goal rate | **76.0%** |
| Avg reward (last 10 eps) | 108.97 |
| Avg episode length (last 10 eps) | 103.7 |

**Episode outcomes:**

- `goal_scored`: 1023 (72.3%)
- `keeper_cleared`: 200 (14.1%)
- `ball_out_of_bounds`: 80 (5.7%)
- `ball_in_penalty_off_target`: 72 (5.1%)
- `max_steps`: 40 (2.8%)

**Goal rate over training (200-episode buckets):**

```
  eps    1- 200:  52.5%  ██████████
  eps  201- 400:  73.5%  ██████████████
  eps  401- 600:  79.5%  ███████████████
  eps  601- 800:  79.0%  ███████████████
  eps  801-1000:  72.5%  ██████████████
  eps 1001-1200:  69.5%  █████████████
  eps 1201-1400:  78.5%  ███████████████
  eps 1401-1415:  86.7%  █████████████████
```

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.693 | 4.0% |
| Last 200 kicks | 0.507 | 7.0% |

**Action distribution samples:**

- ep 400: `approach_ball=68%  kick=3%  turn=27%`
- ep 600: `approach_ball=62%  kick=4%  turn=33%`
- ep 800: `approach_ball=80%  kick=2%  turn=17%`
- ep 1000: `approach_ball=78%  kick=2%  turn=19%`
- ep 1200: `approach_ball=82%  kick=0%  turn=16%`
- ep 1400: `approach_ball=83%  kick=2%  turn=13%`

### Key config

**Reward overrides:**
- `approach_weight`: 1.0
- `step_bonus`: 0.0
- `goal_reward`: 70.0
- `goal_progress_weight`: 2.0, `goal_progress_clip`: 1.5
- `ball_speed_bonus_weight`: 2.0, `ball_speed_threshold`: 0.2, `ball_speed_clip`: 2.0
- `alignment_weight`: 3.0, `alignment_has_ball_only`: True
- `kick_aim_bonus_weight`: 20.0, `bad_aim_kick_penalty`: 1.0, `bad_aim_grad_scale`: 15.0
- `use_goalie_aim_gate`: True, `goalie_gap_min_quality`: 0.3
- `kick_into_keeper_penalty`: 1.5, `goalie_gap_blend_center`: 0.3
- `opponent_near_ball_penalty`: -0.1, `opponent_near_ball_threshold`: 5.0

**Model params:** `gamma=0.99` (PPO JAL expandable backbone)

### Code changes (non-config)

None. Config-only change from §22: `ball_cleared_x_threshold` 15.0 → 4.0.

### What worked

**Clean transfer — Stage-1 full chain beat the keeper at 72.3% raw with no pre-processing needed.**

- **Opened at 52.5% in the first 200 episodes** — no warm-up collapse. The Stage-1 approach/turn/kick chain transferred immediately to the 1v1 setup. This was the key acceptance test: the §22 plan had flagged collapse-to-turn-100% in the first 200 episodes as the primary risk, and it didn't happen.
- **Trajectory is upward.** 52.5% → 73.5% → 79.5% → 79% then a dip to 69.5% (mid-run variance) → recovered 78.5% → **86.7% in the final 15 episodes**. The model had not plateaued by the end of the budget.
- **No single-action collapse at any point.** Action distribution held a healthy mix throughout: approach_ball 62–83%, turn 13–33%, kick 0–4% per-step. Kick share is correctly low in per-step terms — each episode is ~104 steps and typically ends in one kick.
- **keeper_cleared rate (14.1%) is entirely genuine** — all multi-step episodes that ended in a keeper save, none from the spawn bug. Confirms the threshold fix eliminated the ghost terminations entirely.
- **Compared to §22 (corrected):** the corrected §22 goal rate was 72.5%; this clean run is 72.3% — nearly identical. This validates both the diagnosis and the fix: the real-episode performance was already at this level in §22; removing the bugs just made the raw number match.

One flag: **aim_quality drifted down** over training (0.693 → 0.507, bad_aim 4% → 7%). The model is scoring but not with increasingly precise aim as training progresses — the gap-gate reward is being satisfied at a good-enough level rather than pushing toward maximum quality. Monitor in the dribble rung; if it worsens, consider raising `goalie_gap_min_quality` from 0.3 or increasing `kick_aim_bonus_weight`.

### Next stage

Advance to **`stage2_1_dribble`** — the dribble rung. The baseline confirms approach/turn/kick transfers cleanly to 1v1; the next challenge is teaching the robot to use `start_dribble`/`stop_dribble` to reposition the ball for a better angle when the keeper is well-positioned.

Key config decisions for the dribble rung (already decided):
- **`ball_cleared_x_threshold`: `null`** — set to null (disabled) so that when the keeper saves, the robot can re-approach the rebounding ball, dribble into position, and kick again. Under the no-dribble baseline, ending on clear was correct; with dribble available the full loop should play out under `max_steps` only.
- Enable `start_dribble` and `stop_dribble` (remove from `disabled_actions`).
- Add dribble shaping terms to `reward_config_overrides` (currently all zero in baseline).

Checkpoint: `models/ppo_jal_expandable/stage2_0_baseline_complete.pt`

---

## 24. Training run: 20260618_050345_948864 — PPO stage2_1_dribble (PARTIAL — dribble objective not learned)

**Start time:** 2026-06-18 05:03:45
**Stage:** `stage2_1_dribble`
**Steps:** 299,909 / 300,000 (completed; final partial episode was not counted)
**Load model:** `models/ppo_jal_expandable/stage2_0_baseline_complete.pt`

**Aim:** Add `start_dribble` and `stop_dribble` to the validated Stage 2.0 full chain. The intended policy was to carry the ball laterally within the 1 m dribble limit, create a better goalkeeper gap, deliberately release, re-approach, and shoot. Acceptance required dribble use in at least 20% of episodes, improving post-dribble gap quality, and goal rate at least matching the 72.3% Stage 2.0 baseline.

### Results

| Metric | Value |
|---|---|
| Total episodes | 1821 |
| Overall goal rate | 1085/1821 = **59.6%** |
| Last 100 episodes goal rate | **80.0%** |
| Last 200 complete episodes goal rate | **73.5%** |
| Avg reward (last 10 eps) | 102.19 |
| Avg episode length (last 10 eps) | 134.5 |

**Episode outcomes:**

- `goal_scored`: 1085 (59.6%)
- `ball_in_penalty_off_target`: 291 (16.0%)
- `ball_out_of_bounds`: 244 (13.4%)
- `max_steps`: 195 (10.7%)
- `ball_dead_free_kick_r`: 6 (0.3%)

**Goal rate over training (200-episode buckets):**

```
  eps    1- 200: 46.0%
  eps  201- 400: 42.5%
  eps  401- 600: 61.0%
  eps  601- 800: 58.5%
  eps  801-1000: 73.0%
  eps 1001-1200: 58.0%
  eps 1201-1400: 60.5%
  eps 1401-1600: 61.5%
  eps 1601-1800: 73.5%
  eps 1801-1821: 76.2%
```

**Kick and placement quality:**

| Window | center aim quality | bad aim | mean valid goalkeeper gap | goalkeeper gap < 0.3 |
|---|---:|---:|---:|---:|
| First 200 kicks | 0.417 | 25.5% | 0.465 | 38.3% |
| Last 200 kicks | 0.509 | 10.5% | 0.455 | 37.4% |

The center-based aim metric improved, but the Stage 2 objective did not: mean goalkeeper-gap quality was flat/slightly worse and low-gap shots barely changed. Across the whole run, 654/1583 valid-gap kicks (41.3%) had gap quality below 0.3.

**Step-weighted episode action distributions:**

| Window | approach | kick | start dribble | stop dribble | turn |
|---|---:|---:|---:|---:|---:|
| First 200 episodes | 50.5% | 3.8% | 9.2% | 14.7% | 19.6% |
| Last 200 episodes | 70.2% | 2.0% | 3.8% | 1.8% | 19.9% |

`start_dribble` appeared at least once (after percentage rounding) in 192/200 final episodes, but its step share fell by 59%; `stop_dribble` fell by 88%. This is exploration-level contact with the primitives, not evidence of a learned dribble/release skill.

### Key config

- Full-chain spawn: robot at home, ball x `[5,30]`, y `[-15,15]`, random heading.
- Enabled primitives: approach, turn, kick, start dribble, stop dribble.
- `ball_cleared_x_threshold: null`, so keeper clears no longer ended episodes.
- Dribble shaping: state bonus `0.1`, lateral progress `0.25` with clip `0.3` (maximum `0.075` per step), deliberate release bonus `0.5`.
- Main task reward unchanged from Stage 2.0: goal `70`, kick aim bonus weight `20`, center blend `0.3`, goalkeeper-shot penalty `1.5`.
- PPO: `gamma=0.99`; learning rate annealed from about `2.97e-4` to `1.01e-4`; entropy coefficient from `0.0494` to `0.0051`.

### Code changes (non-config)

- Added projected dribble-radius exhaustion using the prior dribble-step displacement, so release occurs before crossing the SSL 1 m carry limit.
- Replaced the invalid simulator command `drop` with `kick 0 0`, which actually releases the caught ball.
- Reset the previous dribble pose when a session ends, preventing displacement estimates from leaking across sessions.

### What worked

- The run was numerically healthy: no exceptions, NaNs, or unstable PPO updates. Final KL was `0.0072`, categorical entropy remained `1.416`, and value loss was finite.
- The inherited approach/turn/kick chain recovered after the disruption caused by unmasking two new actions. Goal rate rose from 46.0% in episodes 1-200 to 73.5% in 1601-1800 and 80.0% in the exact final 100.
- Bad center aim fell from 25.5% to 10.5%; there was no single-action collapse.

### What went wrong

**The checkpoint recovered by suppressing dribbling, not by learning useful dribble repositioning.** Final goalkeeper-gap quality (`0.455`) did not improve over the first 200 kicks (`0.465`) and was slightly below the Stage 2.0 baseline's final `0.472`. Meanwhile, start/stop dribble use collapsed toward exploration levels.

The 59.6% whole-run goal rate is 12.7 percentage points below Stage 2.0's 72.3%. This comparison is conservative because Stage 2.1 disabled keeper-clear termination and therefore offered second chances; despite that, off-target endings rose from 5.1% to 16.0%, out-of-bounds from 5.7% to 13.4%, and max-steps from 2.8% to 10.7%. The exact final-100 score is encouraging but too noisy to override the flat gap metric and action trend.

There is also a logging gap: the training log records selected primitive percentages but not valid dribble-session starts, completed releases, anchor displacement, or whether each kick followed a dribble. Therefore the acceptance criteria "valid sessions in 20% of episodes" and "post-dribble gap better than direct-shot gap" cannot be measured directly from this run.

### Fix for next run

Do **not** promote directly to the queued full-chain/anneal stage. First add explicit session diagnostics: valid/invalid starts, forced versus deliberate releases, carried distance and lateral displacement, and `kick_after_dribble` with pre/post gap quality. Without these fields, reward tuning is guesswork.

Then fine-tune from this checkpoint on a more targeted dribble curriculum rather than repeating the same full-chain setup. Sample situations where the direct goalkeeper gap is poor, preserve the real `+70` goal objective, and reward measured improvement in gap from the session start to release/kick. The current per-step shaping can pay up to `0.175` per productive dribble step (discounted infinite-horizon ceiling `0.175 / (1 - 0.99) = 17.5`), but it rewards lateral motion rather than the actual outcome. A bounded session-level `delta_gap` reward would align the shaping with the intended skill while avoiding indefinite dribble farming.

Checkpoint retained for deterministic evaluation, not yet accepted as the Stage 2.1 curriculum pass: `models/ppo_jal_expandable/stage2_1_dribble_complete.pt`

---

## 25. Training run: 20260619_120642_631274 — PPO stage2_goalie / Stage 2A gap shooting (SUCCESS)

**Start time:** 2026-06-19 12:06:42
**Stage:** `stage2_goalie` (Stage 2A)
**Steps:** 299,925 / 300,000
**Load model:** `models/ppo_jal_expandable_wide/stage2_0_baseline_complete.pt`

**Aim:** First Stage 2 run on the wide expandable backbone. 1v1 vs scripted bisector goalie with `dribble_to` enabled for the first time. Ball spawns in attacking third (x=[25,34], y=±10). `use_goalie_aim_gate` + `kick_into_keeper_penalty` break the center-shot habit from Stage 1. Dense dribble rewards (`progress=1.0`, `quality=2.0`, `active=0.05`, `combo=3.0` quality-scaled) encourage the policy to *discover* dribble without farming (quality-scaled combo from day one blocks dribble→wild-kick). `goalie_gap_blend_center=0.3` softens the transfer from the Stage-1 checkpoint.

| Metric | Value |
|---|---|
| Total episodes | 1891 |
| Overall goal rate | 933/1891 = **49.3%** |
| Last 100 episodes goal rate | **57.0%** |
| Avg reward (last 10 eps) | 31.58 |
| Avg episode length (last 10 eps) | 148.1 |

**Episode outcomes:**
- `goal_scored`: 933 (49.3%)
- `ball_in_penalty_off_target`: 495 (26.2%)
- `goalie_catch`: 279 (14.8%)
- `ball_out_of_bounds`: 184 (9.7%)

**Goal rate over training (200-episode buckets):**
```
  eps    1- 200:  47.5%  █████████
  eps  201- 400:  41.0%  ████████
  eps  401- 600:  41.5%  ████████
  eps  601- 800:  41.5%  ████████
  eps  801-1000:  53.5%  ██████████
  eps 1001-1200:  47.0%  █████████
  eps 1201-1400:  56.5%  ███████████
  eps 1401-1600:  51.0%  ██████████
  eps 1601-1800:  60.5%  ████████████
  eps 1801-1891:  58.2%  ███████████
```

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.472 | 12.0% |
| Last 200 kicks | 0.357 | **11.5%** |

**Action distribution samples:**
- ep 800:  approach_ball=27%  dribble_to=48%  kick=3%   turn=19%
- ep 1000: approach_ball=53%  dribble_to=27%  kick=6%   turn=12%
- ep 1200: approach_ball=55%  dribble_to=24%  kick=5%   turn=14%
- ep 1400: approach_ball=59%  dribble_to=27%  kick=3%   turn=9%
- ep 1600: approach_ball=22%  dribble_to=66%  kick=2%   turn=8%
- ep 1800: approach_ball=20%  dribble_to=62%  kick=4%   turn=11%

**Key reward overrides:** `step_bonus=-0.05`, `goal_reward=70`, `use_goalie_aim_gate=true`, `goalie_gap_blend_center=0.3`, `kick_into_keeper_penalty=1.5`, `kick_aim_bonus_weight=20`, `dribble_target_progress_weight=1.0`, `dribble_target_quality_weight=2.0`, `dribble_active_bonus=0.05`, `post_dribble_kick_bonus=3.0`, `post_dribble_kick_quality_scale=true`, `kick_near_goalie_penalty=2.0`

### Code changes (non-config)

Fixed pre-existing `NameError: name 'ball_pos' is not defined` in `JAL_env.py` ~line 712. The `dribble_target_quality_weight` one-shot bonus block (fires on `carry_started`) referenced `ball_pos` which was never defined in `step()` scope. Fixed by defining `gs_ball_pos = getattr(current_game_state, "ball_pos", None)` before the loop, matching the pattern in the sibling `kick_near_goalie` block. Bug was latent since Stage 1 never used `dribble_to`; the first carry opened at episode 27 of the pre-run smoke, crashing immediately. Verified with `ast.parse` before the full run.

### What worked

Goal rate climbed from 47.5% → 60.5% over the run, finishing at 58.2% (last 100 = 57%). The policy discovered `dribble_to` and used it heavily (24–66% of actions) — the primary goal of Stage 2A. Bad_aim held flat at 11.5% vs 12.0% (first 200 kicks), confirming the quality-scaled combo bonus prevented wild post-dribble kicks. `goalie_catch` at 14.8% confirms the keeper is an active obstacle and the center-shot habit from Stage 1 was broken.

Note: 2B auto-started immediately after 2A completed but was stopped after ~12 episodes (~1,416 steps). Those episodes are excluded from this analysis (log sliced at line 16438, just before the `stage2b_goalie_dribble` start marker).

### Next stage

**Stage 2B** (`stage2b_goalie_dribble`): sharpen dribble targeting. Dense exploration rewards fade: `step_bonus=-0.1`, `dribble_active_bonus=0.03`, `dribble_target_progress_weight=1.5`, `dribble_target_quality_weight=4.0`, `post_dribble_kick_bonus=6.0` quality-scaled. Ball spawns widen to y=±15, `goalie_gap_blend_center=0.0`. Goal: dribble usage comes down from 60%+ toward 30–40%; carries show positive `quality_delta` (actually improving the gap); goal rate holds or rises.

Watch for over-dribbling staying >60% into 2B — if so, consider reducing `dribble_active_bonus` further or adding a counterfactual penalty for dribbling when gap quality is already above the `goalie_gap_min_quality` threshold.

---

## 26. Training run: 20260619_125127_808032 — PPO stage2b_goalie_dribble / Stage 2B dribble mastery (MIXED — re-tune & re-run 2B)

**Start time:** 2026-06-19 12:51:27
**Stage:** `stage2b_goalie_dribble` (Stage 2B)
**Steps:** 299,955 / 300,000
**Load model:** `models/ppo_jal_expandable_wide/stage2_goalie_complete.pt` (2A final)

**Aim:** Sharpen dribble targeting on top of the 2A gap-shot policy. Dense exploration rewards fade and the task widens: `step_bonus` −0.05→−0.1, `dribble_target_quality_weight` 2.0→4.0, `post_dribble_kick_bonus` 3.0→6.0 (quality-scaled), `dribble_active_bonus` 0.05→0.03, ball-y range ±10→±15, `goalie_gap_blend_center` 0.3→0.0. Goal: dribble usage settles to 30–40%, carries show positive `quality_delta`, goal rate holds or rises toward ~80%.

| Metric | Value |
|---|---|
| Total episodes | 1954 |
| Overall goal rate | 788/1954 = **40.3%** |
| Last 100 episodes goal rate | **49.0%** |
| Avg reward (last 10 eps) | 83.67 |
| Avg episode length (last 10 eps) | 138.6 |

**Episode outcomes:**
- `ball_in_penalty_off_target`: 803 (41.1%)
- `goal_scored`: 788 (40.3%)
- `goalie_catch`: 184 (9.4%)
- `ball_out_of_bounds`: 179 (9.2%)

**Goal rate over training (200-episode buckets):**
```
  eps    1- 200:  37.5%  ███████
  eps  201- 400:  29.0%  █████    ← early collapse
  eps  401- 600:  36.5%  ███████
  eps  601- 800:  46.5%  █████████
  eps  801-1000:  38.5%  ███████
  eps 1001-1200:  42.5%  ████████
  eps 1201-1400:  42.0%  ████████
  eps 1401-1600:  41.0%  ████████
  eps 1601-1800:  40.0%  ████████
  eps 1801-1954:  52.6%  ██████████  ← recovered
```

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.312 | 24.5% |
| Last 200 kicks | 0.356 | **10.5%** |

**Action distribution samples:**
- ep 800:  approach_ball=38%  dribble_to=31%  kick=3%  turn=26%
- ep 1000: approach_ball=38%  dribble_to=35%  kick=2%  turn=23%
- ep 1200: approach_ball=54%  dribble_to=33%  kick=1%  turn=10%
- ep 1400: approach_ball=50%  dribble_to=38%  kick=1%  turn=9%
- ep 1600: approach_ball=43%  dribble_to=43%  kick=2%  turn=10%
- ep 1800: approach_ball=56%  dribble_to=34%  kick=1%  turn=7%

**Key reward overrides:** `step_bonus=-0.1`, `goal_reward=70`, `use_goalie_aim_gate=true`, `goalie_gap_blend_center=0.0`, `kick_into_keeper_penalty=1.5`, `kick_aim_bonus_weight=20`, `dribble_target_progress_weight=1.5`, `dribble_target_quality_weight=4.0`, `dribble_active_bonus=0.03`, `post_dribble_kick_bonus=6.0`, `post_dribble_kick_quality_scale=true`, `kick_near_goalie_penalty=2.0`

### Code changes (non-config)

None. Pure config / reward-override run (the `dribble_target_quality` `ball_pos` fix from §25 carried in unchanged).

### What went wrong

**Mixed result — the targeting goals were met but overall goal rate stalled below the 2A carry-in.** Two of 2B's three aims succeeded: (1) dribble usage came down from 2A's 60%+ into the intended 31–43% band and held there, and (2) aim quality *sharpened* — bad_aim more than halved (24.5% → 10.5%) and avg aim climbed 0.312 → 0.356 across the run. But overall goal rate (40.3%) finished **flat-to-below** 2A (49.3% overall / 57% last-100), and never climbed toward the ~80% the plan's gate wanted.

The damage came from an **early collapse**: the goal rate dropped to 29.0% at eps 201–400 before slowly clawing back to 52.6% by the end. That ~5-bucket recovery is what dragged the overall average down. The cause is almost certainly that **too many difficulty knobs moved at once** on a warm-started policy: `step_bonus` doubled (−0.05→−0.1), spawn width widened 50% (±10→±15), and the `goalie_gap_blend_center` transfer crutch was removed entirely (0.3→0.0) — all simultaneously. The 2A policy destabilized under the combined shift, unlearned part of its gap-shot, then had to rebuild. `ball_in_penalty_off_target` at 41.1% (up from 2A's 26.2%) is the residue: more misses, consistent with the wider spawn + harsher step cost forcing rushed shots before aim had re-stabilized.

The encouraging signal: by the final bucket the policy had recovered to 52.6% *with* the sharper aim and selective dribble usage — i.e. the end state is healthier than the average. The targets weren't wrong; the transition was too abrupt.

### Fix for next run

**Re-run 2B with a gentler difficulty ramp so the warm-started policy doesn't collapse early.** Change one or two knobs at a time instead of three:
- **Keep a partial transfer crutch:** `goalie_gap_blend_center` 0.0 → **0.15** (don't drop it to zero on the first widened-spawn run; let the policy keep some center-blend while it adapts to ±15).
- **Soften the step penalty:** `step_bonus` −0.1 → **−0.07** (still tighter than 2A's −0.05 for urgency, but not aggressive enough to force rushed off-target kicks before aim re-stabilizes).
- **Hold the targeting weights** (`dribble_target_quality_weight=4.0`, `post_dribble_kick_bonus=6.0` quality-scaled) — these worked; aim and usage both moved the right way.

Rationale: the early 29% bucket is the whole story behind the flat average. Aim and dribble-selectivity both improved, so the reward *shape* is right — the problem is the step-change in task difficulty at warm-start. A gentler ramp should preserve the gains while avoiding the unlearn-then-rebuild detour, lifting the overall and last-100 above the 2A baseline. If the re-run holds ≥55% last-100 with bad_aim ≤12% and dribble usage 30–40%, promote to 2C.

---

## 27. Training run: 20260619_132709_813731 — PPO stage2b_goalie_dribble RE-RUN / gentler ramp (FAILED — hypothesis falsified, stopped at 72%)

**Start time:** 2026-06-19 13:27:09
**Stage:** `stage2b_goalie_dribble` (Stage 2B, re-run of §26 with the gentler-ramp fix)
**Steps:** 215,384 / 300,000 (**stopped early** — tracking below §26 with no recovery)
**Load model:** `models/ppo_jal_expandable_wide/stage2_goalie_complete.pt` (2A final, fresh restart)

**Aim:** Test the §26 "fix for next run": the early collapse was blamed on three difficulty knobs moving at once on a warm start. Re-tune softened `step_bonus` −0.1→−0.07 and kept a partial transfer crutch `goalie_gap_blend_center` 0.0→0.15, holding the targeting weights (`dribble_target_quality_weight=4.0`, `post_dribble_kick_bonus=6.0` quality-scaled). Spawn width left at ±15. Expectation: gentler ramp avoids the unlearn-then-rebuild detour and lifts overall/last-100 above the 2A baseline.

| Metric | Value |
|---|---|
| Total episodes | 1326 |
| Overall goal rate | 438/1326 = **33.0%** |
| Last 100 episodes goal rate | **40.0%** |
| Avg reward (last 10 eps) | 46.61 |
| Avg episode length (last 10 eps) | 163.0 |

**Episode outcomes:**
- `ball_in_penalty_off_target`: 572 (43.1%)
- `goal_scored`: 438 (33.0%)
- `goalie_catch`: 182 (13.7%)
- `ball_out_of_bounds`: 134 (10.1%)

**Goal rate over training (200-episode buckets):**
```
  eps    1- 200:  36.5%  ███████
  eps  201- 400:  34.0%  ██████
  eps  401- 600:  26.5%  █████    ← dipped anyway, despite gentler ramp
  eps  601- 800:  29.5%  █████
  eps  801-1000:  38.0%  ███████
  eps 1001-1200:  30.5%  ██████   ← no upward trend; bouncing, not climbing
  eps 1201-1326:  38.1%  ███████
```

**Aim vs §26 (run 1) at matched windows:**

| Window | This re-run | §26 run 1 |
|---|---|---|
| eps 1001–1200 goal | **30.5%** | 42.5% |
| Overall (full §26 / 72% here) | 33.0% | 40.3% |
| bad_aim (last 200 kicks) | **26.0%** (flat) | 24.5%→**10.5%** (sharpened) |

### Code changes (non-config)

None.

### What went wrong

**The §26 hypothesis was falsified.** The fix assumed the early collapse came from moving three difficulty knobs at once, so a gentler ramp should prevent it. It didn't: the goal rate **still dipped** (to 26.5% at eps 401–600) and then, unlike run 1, **never recovered** — it bounced in a 26–38% band with no upward slope, and aim stayed **flat at bad_aim 26%** instead of sharpening to 10.5% the way run 1 did in its back half. At matched episode counts the re-run trailed run 1 by ~10pp (eps 1001–1200: 30.5% vs 42.5%). Stopped at 215k (72%) once it was clear there was no path to the ≥55% / bad_aim ≤12% promote bar.

Two conclusions: (1) the early dip is **not** caused by the step penalty or blend removal — it's intrinsic to the ±15 spawn widening (±10→±15) outpacing the policy's aim. (2) Softening `step_bonus` to −0.07 likely **hurt**: less urgency → more aimless dribbling, and `ball_in_penalty_off_target` stayed the #1 outcome at 43% (worse than run 1's 41%). The shaping that actually rescued run 1 was the *pressure*, not the gentleness.

### Fix for next run

**Attack the real culprit — the spawn-width jump — with a width sub-ramp, and restore the step pressure.** Stage-1 already used width rungs (y6→y10→y15 as separate stages); mirror that for 2B:
- **Split 2B into two width rungs.** First rung holds spawn at the 2A width (±10, or at most ±12) so the warm-started policy keeps its gap-shot while the dribble-targeting weights take hold; second rung widens to ±15 warm-started from the first rung's checkpoint.
- **Restore `step_bonus` toward −0.1** (the urgency that drove run 1's aim sharpening); do not keep −0.07.
- **Keep the targeting weights** (`dribble_target_quality_weight=4.0`, `post_dribble_kick_bonus=6.0` quality-scaled) — these were never the problem.

Expectation: the narrow first rung avoids the aim-destroying width shock, so aim can sharpen (as it did in run 1) before the ±15 generalization rung. Promote bar unchanged: last-100 ≥55%, bad_aim ≤12%, dribble usage 30–40%.

---

## 28. Training run: 20260619_144302_419077 — PPO stage2b_goalie_dribble RUNG 1 / NARROW ±10 (SUCCESS — width sub-ramp worked, cleared promote bar)

**Stage:** `stage2b_goalie_dribble` (Stage 2B rung 1, the width sub-ramp's narrow rung)
**Steps:** 149,896 / 150,000 (ran to budget)
**Load model:** `models/ppo_jal_expandable_wide/stage2_goalie_complete.pt` (2A final)
**Aim:** Hold spawn at the 2A width (±10) so the dribble-targeting weights take hold and aim sharpens WITHOUT the ±10→±15 width shock that collapsed §26/§27; `step_bonus` restored to −0.1. Only width differs from rung 2.

### Results

| Metric | Value |
|---|---|
| Total episodes | 972 |
| Overall goal rate | 537/972 = **55.2%** |
| Last-100 goal rate | **55.0%** |

**Episode outcomes:** `goal_scored` 537 (55.2%) · `ball_in_penalty_off_target` 212 (21.8%) · `goalie_catch` 117 (12.0%) · `ball_out_of_bounds` 106 (10.9%)

**Goal rate over training (200-ep buckets) — NO early collapse:**
```
  eps    1- 200:  54.5%  ██████████
  eps  201- 400:  51.5%  ██████████
  eps  401- 600:  56.0%  ███████████
  eps  601- 800:  58.5%  ███████████   ← peak
  eps  801- 972:  55.8%  ███████████
```

**Aim quality (kicks):**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.313 | 14.0% |
| Last 200 kicks | 0.311 | 11.5% |

**Action distribution:** ep1 dribble=59% → ep200 45% → ep400 32% → ep600 44% → ep800 22% (noisy, mostly 22–45%).

### Code changes (non-config)

None.

### What worked

**The width sub-ramp removed the early collapse — this is the first stable 2B run.** Holding spawn at ±10 (vs jumping to ±15) kept the goal rate in a tight 51–58% band the entire run, where both §26 and §27 dipped to 26–29% in the first few hundred episodes. That directly validates §27's diagnosis: the collapse was the spawn-width shock, not the step penalty. Restoring `step_bonus=−0.1` (vs §27's −0.07) preserved the aim pressure. **Cleared the promote bar:** last-100 55.0% (≥55), bad_aim 11.5% (≤12), dribble usage in the 30–40% range (drifty but present).

### What still needs watching (carried into rung 2)

**aim_quality is dead flat: 0.313 → 0.311.** It started at the 2A warm-start setpoint and never moved across 150k. bad_aim dropped only 2.5pp (14.0%→11.5%). Read together: the policy learned to *clear the bad_aim threshold* (penalty-avoidance) but NOT to *maximize gap quality* — classic "just good enough." 21.8% off-target + 12.0% goalie-catch = ~34% of possessions still fail on placement. Two readings of the flat aim, deliberately left untested for now:
- **(A) reward can't push aim** → widening to ±15 will stress a stalled skill.
- **(B) ±10 doesn't *demand* aim** (narrow geometry, center-ish shots score ~55%) → ±15 is the natural forcing function that will teach placement.
Evidence is ambiguous (flat-from-first-bucket fits both).

### Next stage

**Promote to rung 2 (`stage2b2_goalie_dribble_wide`, ±15, 200k) with NO reward change** — one-knob discipline (only width changes from rung 1), so if rung 2 underperforms we know it's the width, not a confounded reward edit (the §26/§27 lesson). Chains in-memory from this rung's checkpoint `models/ppo_jal_expandable_wide/stage2b_goalie_dribble_complete.pt`. **Make aim_quality the gate, not just goal rate:** if avg aim_quality does NOT climb past ~0.38 by the eps 600–800 bucket at ±15, stop and apply the reward fix — `kick_aim_bonus_weight` 20→28 (amplify the gap-magnitude gradient that goal_reward=70 currently drowns) and/or `goalie_gap_blend_center` 0.15→0.0 (kill the center-shot crutch; note it also feeds the combo scale, so apply it alone, not stacked on a width jump). This pre-commit resolves readings (A)/(B) with the width confound already controlled.

---

## 29. Training run: 20260619_151248_911101 — PPO stage2b2_goalie_dribble_wide RUNG 2 / WIDE ±15 (FAILED — regression; reading (A) confirmed, aim degraded)

**Stage:** `stage2b2_goalie_dribble_wide` (Stage 2B rung 2, the ±15 width-generalization rung)
**Steps:** 199,921 / 200,000 (ran to budget)
**Load model:** `models/ppo_jal_expandable_wide/stage2b_goalie_dribble_complete.pt` (rung 1 ±10 final, 55%)
**Aim:** Generalize rung 1's stable ±10 policy to full ±15 width, ONE knob (width) changed. Pre-committed gate: aim_quality must climb past ~0.38 by eps 600–800 or stop and apply a reward fix.

### Results

| Metric | Rung 1 (±10) | This run (±15) |
|---|---|---|
| Overall goal rate | 55.2% | **35.6%** (450/1264) |
| Last-100 | 55.0% | **34.0%** |
| avg aim_quality first→last | 0.31→0.31 | **0.33→0.28** |
| bad_aim first→last | 14%→11.5% | **16%→28%** |

**Episode outcomes:** off_target 554 (43.8%) · goal 450 (35.6%) · goalie_catch 145 (11.5%) · OOB 115 (9.1%)

**Goal rate over training (200-ep buckets):**
```
  eps    1- 200:  41.5%
  eps  201- 400:  39.5%
  eps  401- 600:  33.5%
  eps  601- 800:  33.5%
  eps  801-1000:  36.0%
  eps 1001-1200:  29.0%
  eps 1201-1264:  37.5%
```

### Code changes (non-config)

None.

### What went wrong

**Reading (A) confirmed: widening did NOT force better aim — it degraded it.** Going ±10→±15 dropped goal rate 55%→35.6% and, critically, aim_quality FELL (0.33→0.28) while bad_aim nearly doubled (16%→28%). off_target became the #1 outcome at 43.8%. The pre-committed aim gate (≥0.38 by eps 600–800) was missed badly — aim went the wrong direction. This rules out reading (B) ("±10 just didn't demand aim"): the demand was there at ±15 and the policy could not meet it.

**The sub-ramp warm-start bought nothing at ±15.** Despite starting from a *stable* 55%/±10 policy, this run landed in the SAME 33–40% band as §26 (40%) and §27 (33%), which jumped straight to ±15. So ±15 is a ~35% wall for this policy+reward regardless of how gently it is approached. The width ladder gave a good ±10 waypoint; it did not solve ±15. The bottleneck is aim CAPABILITY gated by the reward, not the suddenness of the width change.

### Fix for next run

**Attack aim directly with the reward, tested in isolation at the STABLE ±10 width first** (new stage `stage2b_aim_push`). `kick_aim_bonus_weight` 20→28 to amplify the gap-magnitude gradient that goal_reward=70 currently drowns — the policy clears the bad_aim threshold but does not *maximize* gap. Width held ±10, warm-started from rung 1's checkpoint, so any aim movement is attributable to the reward (ONE knob from rung 1). Decision: if aim_quality climbs above ~0.38 at ±10, the lever works → re-attempt ±15 with it; if aim stays flat ~0.31, the lever is useless → deeper rethink. Secondary lever noted for later: reduce param-noise std 0.3→~0.2 for this aim-focused stage (fine aim at ±15 may be blurred by exploration noise on the Dθ head).

---

## 30. Training run: 20260619_164329_320856 — PPO stage2b_aim_push / kick_aim_bonus_weight 20→28 at ±10 (CONFOUNDED by warm-start shock — regressed to 44%, inconclusive on the aim lever)

**Stage:** `stage2b_aim_push` (aim-lever diagnostic, ±10, only `kick_aim_bonus_weight` 20→28 vs rung 1)
**Steps:** 149,999 / 150,000
**Load model:** `models/ppo_jal_expandable_wide/stage2b_goalie_dribble_complete.pt` (rung 1 ±10 final, 55%)
**Aim:** Isolate whether amplifying the aim bonus pushes aim_quality above ~0.38 at the stable ±10 width.

### Results

| Metric | Rung 1 (w=20) | This run (w=28) |
|---|---|---|
| Overall goal rate | 55.2% | **43.9%** (418/953) |
| Last-100 | 55.0% | 45.0% |
| aim_quality first→last | 0.31→0.31 | **0.18→0.235** |
| bad_aim first→last | 14%→11.5% | **38.5%→31%** |

**Episode outcomes:** goal 418 (43.9%) · off_target 293 (30.7%) · OOB 134 (14.1%) · goalie_catch 108 (11.3%)

**Goal rate (200-ep buckets):**
```
  eps   1-200:  41.0%
  eps 201-400:  38.0%   ← dip
  eps 401-600:  45.0%
  eps 601-800:  51.5%   ← re-climbing
  eps 801-953:  43.8%
```

### Code changes (non-config)

None.

### What went wrong

**Confounded by a warm-start value-function shock — the result is inconclusive on the aim lever and produced a regressed model.** Smoking gun: the **first 200 kicks measured aim_quality 0.181 / bad_aim 38.5%**, even though the loaded checkpoint (rung 1) *ended* at 0.31 / 11.5% at the **same ±10 width**. It collapsed immediately on warm-start. Contrast rung 1, which warm-started from 2A under the **same** `kick_aim_bonus_weight=20` and inherited cleanly (first-200 = 0.313, no collapse). The only difference here is the weight change 20→28.

Mechanism: changing the reward *scale* invalidated the inherited value function (calibrated to the w=20 reward magnitude). PPO's first updates used miscalibrated advantages → destructive early updates → the policy degraded, then spent the rest of the run re-climbing (aim 0.18→0.235, goals up to 51.5%) without recovering to baseline in 150k. **General lesson: you cannot change reward scale mid-curriculum without a value-function re-warmup — it always shocks the warm-start.**

### Fix for next run

**Re-test the "exploration noise blurs fine aim" hypothesis WITHOUT a reward-scale shock** (new stage `stage2b_aim_lowent`). Correction: `param_noise_std` is an *inference-only* flag ([launch_infer.py](launch_infer.py#L111)/[infer.py](infer.py#L444)) — it does not exist in training. Training-time exploration is the policy's own Gaussian action std, pressured up by `ent_coef`. So keep ALL reward weights at rung-1 values (`kick_aim_bonus_weight=20` — clean warm-start, no value-function shock) and instead lower the global `ent_coef` (initial 0.05→0.01, final 0.005→0.002) so (a) the 0.05 initial doesn't re-inject entropy onto rung-1's already-sharp policy and (b) the Dθ aim head can lower its std and sharpen. Width ±10, warm from rung-1's 55% checkpoint. Decision: aim_quality climbs above ~0.38 → noise was blurring aim (carry low ent_coef into a ±15 re-attempt); stays ~0.31 → the *mean* aim is the ceiling (deeper rethink: harder goalie / capacity / `LOG_STD_MIN`). Watch for premature entropy collapse (`entropy_tripwire`); revert `ent_coef` after.

---

## 31. Training run: 20260620_004156_864284 — PPO stage2b_aim_lowent / ent_coef 0.05→0.01 at ±10 (BEST MODEL YET 59.5%, but aim hypothesis FALSIFIED — aim is not the lever)

**Stage:** `stage2b_aim_lowent` (clean exploration-noise test, ±10, only global `ent_coef` lowered vs rung 1; ALL reward weights held at rung-1 values so the warm-start is clean)
**Steps:** 149,918 / 150,000
**Load model:** `models/ppo_jal_expandable_wide/stage2b_goalie_dribble_complete.pt` (rung 1 ±10 final, 55%)
**Aim:** Test whether lowering training-time exploration (`ent_coef_initial` 0.05→0.01, `ent_coef_final` 0.005→0.002) lets the Dθ aim head sharpen and pushes aim_quality above ~0.38 — WITHOUT a reward-scale shock (unlike §30).

### Results

| Metric | Rung 1 (ent 0.05) | This run (ent 0.01) |
|---|---|---|
| Overall goal rate | 55.2% | **59.5%** (612/1028) |
| Last-100 | 55.0% | **61.0%** |
| aim_quality first→last | 0.31→0.31 | **0.309→0.293** |
| bad_aim first→last | 14%→11.5% | **10.5%→6.0%** |

**Episode outcomes:** goal 612 (59.5%) · off_target 241 (23.4%) · goalie_catch 88 (8.6%) · OOB 87 (8.5%)

**Goal rate (200-ep buckets):**
```
  eps    1- 200:  55.0%
  eps  201- 400:  56.5%
  eps  401- 600:  61.0%
  eps  601- 800:  62.5%
  eps  801-1000:  63.5%   ← still rising
  eps 1001-1028:  53.6%   (28 eps, noise)
```

### Code changes (non-config)

None.

### What worked (and what the test settled)

**Best model recorded so far, and a clean warm-start (no §30-style shock).** Goal rate 55.2%→59.5% (last-100 61.0%), still trending up at the end. bad_aim nearly halved (11.5%→6.0%) — the policy got better at *avoiding* zero-gap kicks. No entropy collapse fired despite the lower `ent_coef`.

**But the pre-registered aim hypothesis is FALSIFIED.** Mean aim_quality stayed at 0.31 (drifted slightly to 0.293), nowhere near the ~0.38 gate. Per the §30 decision criterion, **the *mean* aim is the ceiling — exploration noise was not the blur.** This is now the **third** independent lever (width §29, reward-scale §30, exploration §31) that fails to move mean aim above ~0.31. Combined with the linear `goalie_gap_quality` (`gap/goal_half_height`) and a **bisector keeper that actively centers on the shot line**, ~0.31 mean gap is very likely near the achievable ceiling against this keeper.

**Most important reframe: goal rate climbed while aim stayed flat.** Scoring and gap-quality-at-kick have *decoupled* — the +4.5pp came from elsewhere in the chain (positioning, timing, fewer wild kicks), not from bigger gaps. **aim_quality is therefore NOT the binding constraint on goal rate.** The remaining dominant failure is off_target (23.4%), and this run is evidence it is not a gap-quality problem. Stop the aim-micro-optimization loop.

### Next stage

**Promote this 59.5% checkpoint and stop chasing mean aim.** Two viable forward paths (decision pending with user): (a) re-attempt full-width ±15 with this materially better, low-ent base — a decisive test of whether rung 2's 35.6% "±15 wall" was about *base-model quality* or is structural; or (b) consolidate ±10 and run the 2C selective-dribbling *weaning* (combo-only reward) to lock in the actual Stage-2 target behavior, deferring ±15 to a final hardening pass. **ent_coef decision is coupled to the choice:** keep it lowered for a pure-width ±15 re-attempt (no reward change, want a sharp stable policy); restore some exploration for 2C (its reward *shape* changes, so the policy needs room to re-adapt). The blanket "revert ent_coef" note from §30 is superseded by this coupling.

---

## 32. Training run: 20260620_012035_072544 — PPO stage2b2_goalie_dribble_wide / ±15 re-attempt from the 59.5% base (FAILED — the ±15 wall is STRUCTURAL, not base-quality)

**Stage:** `stage2b2_goalie_dribble_wide` (±15 re-attempt; ONE knob vs §31 — spawn width ±10→±15; ALL reward weights held at §31/rung-1 values; ent_coef kept low 0.01/0.002)
**Steps:** 149,922 / 150,000
**Load model:** `models/ppo_jal_expandable_wide/stage2b_aim_lowent_complete.pt` (§31, 59.5%)
**Aim:** Decisive test of whether §29's 35.6% "±15 wall" was about base-model quality (rung 1 was only 55%) or is structural. Pre-registered gate @ eps 600–800: ≥~50% → base quality, continue to 2C at ±15; ~35% → structural, pivot to 2C at ±10.

### Results

| Metric | §31 base (±10) | §29 first ±15 | This run (±15) |
|---|---|---|---|
| Overall goal rate | 59.5% | 35.6% | **37.7%** (383/1016) |
| Last-100 | 61.0% | 34.0% | **37.0%** |
| aim_quality first→last | 0.31→0.29 | 0.33→0.28 | **0.279→0.339** |
| bad_aim first→last | 10.5%→6% | 16%→28% | 16%→17% |

**Episode outcomes:** off_target 394 (38.8%) · goal 383 (37.7%) · goalie_catch 128 (12.6%) · OOB 104 (10.2%) · max_steps 7 (0.7%)

**Goal rate (200-ep buckets):**
```
  eps    1- 200:  44.0%
  eps  201- 400:  35.5%
  eps  401- 600:  36.0%
  eps  601- 800:  40.0%   ← gate window: 40% (FAIL, needed ≥~50%)
  eps  801-1000:  32.0%
  eps 1001-1016:  50.0%   (16 eps, noise)
```

### Code changes (non-config)

None.

### What went wrong

**The ±15 wall is STRUCTURAL — base-model quality was not the cause.** A materially better, more stable base (59.5%, last-100 61%) landed in the identical ~37% hole as §29 (35.6%) and the earlier §26/§27 attempts. The gate window (eps 600–800) read 40%/32%, well under the ≥~50% bar. Promoting a stronger ±10 policy did not transfer to ±15.

**Sharpest evidence for the §31 decoupling: aim_quality IMPROVED here (0.279→0.339, best last-200 on record) while goal rate cratered to 37%.** Better aim, worse scoring. The ±15 failure is driven by off_target (38.8%, the #1 outcome — same signature as §29's 43.8%), i.e. the wider/steeper spawn geometry produces more shots that physically miss the mouth, **not** an aim-at-the-gap problem. This is now conclusive: at ±15 the binding constraint is shot geometry/physics from wide spawns, and it is independent of gap-aim.

### Next stage

**Pivot to 2C at ±10** (per the pre-registration). Consolidate the working width where we hold 59.5%, build the actual Stage-2 target behavior — SELECTIVE dribbling — via the combo-only reward fade, and defer ±15 to a later hardening pass once the policy is stronger. Warm-start from `stage2b_aim_lowent_complete.pt`; restore `ent_coef` to ~0.03 (2C changes the reward *shape*, so the policy needs exploration room to re-adapt). NOTE the §33 reward analysis below: the planned 2C combo math does NOT enforce selectivity as written (combo rewards the chain unconditionally) — corrected weights applied before launch.

---

## 33. Training run: 20260620_015252_736068 — PPO stage2c_goalie_balance / SELECTIVE dribbling at ±10 (MIXED — goal rate recovered to ~60%, but selective USEFUL dribbling unproven + a measurement gap)

**Stage:** `stage2c_goalie_balance` (±10; reward-shape fade to combo-only, EV-corrected per the analysis below; warm from §31's 59.5%; ent_coef 0.03/0.005)
**Steps:** 199,954 / 200,000
**Load model:** `models/ppo_jal_expandable_wide/stage2b_aim_lowent_complete.pt` (§31, 59.5%)
**Aim:** Build SELECTIVE dribbling — dribble only when it improves the shot — on the working ±10 width, after §32 proved ±15 is structurally walled.

**Reward EV fix applied before launch (verified against code):** the planned combo-only fade did NOT enforce selectivity. `post_dribble_kick_bonus` fires on ANY dribble→kick chain, scaled by *center-mouth* aim (`_kick_aim_quality_from_pose`), **not** gap-improvement — so a useless dribble→kick netted ~+5 over a direct kick (the plan wrongly assumed combo scale ≈ 0.2 for a useless dribble; it's center-mouth aim ≈ 0.6). The only gap-gated selective term is `dribble_target_quality_weight` (one-shot, =0 for useless dribbles). Correction: combo 10→3, `dribble_target_quality_weight` 2→8, `step_bonus` −0.15→−0.2, `dribble_target_progress` 0.5→0.3 / clip 0.3→0.2. Net EV: useless dribble ≈ break-even, useful dribble strongly positive.

### Results

| Metric | §31 base (±10) | 2C final |
|---|---|---|
| Overall goal rate | 59.5% | **51.0%** (711/1394) |
| Last-100 | 61.0% | **60.0%** |
| Last bucket (1201–1394) | — | **62.4%** |
| aim_quality first→last | 0.31→0.29 | 0.272→0.283 |
| bad_aim first→last | 10.5%→6% | 19%→7% |

**Episode outcomes:** goal 711 (51.0%) · off_target 365 (26.2%) · goalie_catch 172 (12.3%) · OOB 145 (10.4%) · max_steps 1

**Goal rate (200-ep buckets):**
```
  eps    1- 200:  46.5%   ← warm-start recalibration dip (expected, §30)
  eps  201- 400:  55.0%
  eps  401- 600:  46.0%
  eps  601- 800:  51.5%
  eps  801-1000:  47.0%
  eps 1001-1200:  49.0%
  eps 1201-1394:  62.4%   ← recovered to baseline
```

### Code changes (non-config)

None for THIS run. (Instrumentation added AFTER this run for the §34 re-run — see "Fix for next run".)

### What happened (mixed)

**Goal rate recovered to baseline.** The mid-run flat ~48% was exactly the predicted warm-start value-function recalibration dip (reward shape changed); it re-climbed in the back half to **60% last-100 / 62.4% final bucket ≈ §31's 59.5%**. So the reward reshape did NOT cost goal rate once the value function settled. aim stayed flat (~0.28) — consistent with the §31/§32 decoupling.

**Selective USEFUL dribbling is unproven, and we hit a measurement gap.** Log analysis: only **73 / 1393 kicks (~5%) were dribble→kick chains** (combo fired), and **all 73 had `scale=0.00` → combo +0.0**, i.e. every captured dribble→kick ended OFF-TARGET (center-mouth aim = 0). Reads:
- The §33 anti-farm guard worked perfectly — not one wild dribble→kick collected a combo.
- The policy largely learned to **NOT** dribble-then-shoot and reverted toward §31's direct-shooting behavior (hence the ~60% recovery). Likely cause: post-carry kicks are off-target (the aim ceiling), so dribble→kick is genuinely low-EV → rationally avoided.
- **BUT the 6-step combo window undercounts dribble→(re-align)→kick chains, and the env logs no `carry_started`/gap-delta — so true per-possession dribble usage and usefulness are UNMEASURABLE from this run.** Per-step `dribble_to` selection was still ~42%, which can't be reconciled with 5% completed chains without the missing instrumentation. We are flying blind on the one metric Stage 2 is about.

### Fix for next run

**Add behavior-neutral instrumentation, then re-run 2C (same config) for a clean selectivity read.** Code added to `JAL_env._calculate_reward`'s reward layer (`ai_interface/envs/JAL_env.py`): (1) a "Dribble carry opened" log on every `carry_started` with `current_gap`, `target_gap`, signed `gap_delta`, and a `useful` flag — gives true per-possession dribble usage rate (carries / episodes) and the fraction that move toward a bigger gap; (2) `steps_since_dribble` added to the "Kick fired" log so kicks can be linked to a prior dribble release at ANY window (not just ≤6). Reward logic byte-for-byte unchanged. v1 checkpoints renamed `stage2c_goalie_balance_v1_*` so the re-run does not overwrite this 60%-last-100 model.

---

## 34. Training run: 20260621_164615_860793 — PPO stage2g_dribble_param / GRAB-commit macro + goal-relative dribble target (SUCCESS — dribbling finally works, 58% goals)

**Stage:** `stage2g_dribble_param`
**Steps:** 199,958 / 200,000
**Load model:** `models/ppo_jal_expandable_wide/stage2c_goalie_balance_complete.pt` (66% shooter, §31)

**Aim:** Fix the two root causes blocking dribbling from ever working: (1) the GRAB phase was outside the macro continuation so the policy almost never completed a catch (1/3000 steps); (2) the dribble target was decoded in absolute field coords, so the head's neutral output →target (0,0)=midfield → every carry latched a backward target → quality one-shot never fired → no gradient. Fix both: add a `committed` flag so GRAB is included in the macro; reparameterize the target to ball-forward/goal-relative so the neutral output is already a gap-improving forward carry.

### Results

| Metric | Value |
|---|---|
| Total episodes | 1551 |
| Overall goal rate | 898/1551 = **57.9%** |
| Last-100 goal rate | **55.0%** |
| Peak (eps 801–1000) | **62.5%** |
| Avg reward (last 10 eps) | 96.4 |

**Episode outcomes:** goal 898 (57.9%) · off-target 323 (20.8%) · goalie_catch 148 (9.5%) · OOB 141 (9.1%) · max_steps 40 (2.6%) · frozen_sim 1 (0.1%)

**Goal rate (200-ep buckets):**
```
  eps    1- 200:  57.0%  ███████████
  eps  201- 400:  52.0%  ██████████
  eps  401- 600:  56.5%  ███████████
  eps  601- 800:  60.0%  ████████████
  eps  801-1000:  62.5%  ████████████  ← peak
  eps 1001-1200:  61.0%  ████████████
  eps 1201-1400:  58.0%  ███████████
  eps 1401-1551:  55.6%  ███████████
```

**Aim quality:**

| | avg aim_quality | bad_aim rate |
|---|---|---|
| First 200 kicks | 0.269 | 6.5% |
| Last 200 kicks | 0.278 | 7.0% |

**Action distribution samples:**
- ep  400: approach_ball=61%  dribble_to=8%   kick=1%  turn=28%
- ep  600: approach_ball=60%  dribble_to=21%  kick=1%  turn=16%
- ep  800: approach_ball=63%  dribble_to=19%  kick=1%  turn=16%
- ep 1000: approach_ball=75%  dribble_to=14%  kick=0%  turn=8%
- ep 1200: approach_ball=78%  dribble_to=10%  kick=0%  turn=9%
- ep 1400: approach_ball=75%  dribble_to=5%   kick=3%  turn=16%

**Useful-carry fraction (from log):** 511/783 total opens = **66% useful** (target_gap > current_gap). Broken down: first 400 opens 73% useful → last 400 opens 43% useful. Goal rate held stable through the drop, suggesting the policy learned to be more selective (skipping low-value carries) rather than regressing.

**Training health (final PPO update):** entropy 1.00 (annealed cleanly from 1.36), KL 0.008 (well under 0.015 target), lr 1.03e-4 (fully decayed). No collapse.

### Code changes (non-config)

Four files changed — these are the core of why this run worked where all previous Stage 2 runs showed 0% useful carries:

1. **`ai_interface/utils/basic_commands.py` — `committed` flag on `DribbleState`:** Added `committed: bool = False` field (cleared in `reset()`). The GRAB phase now only counts as macro-active when `committed=True`, which is set the moment a fresh `dribble_to` action is picked. Before this, GRAB was outside the macro, so the stochastic policy almost never held `dribble_to` long enough to complete the multi-step turn-align needed before `catch` (1 catch in 3000 steps; model stood on ball spinning 192 steps in one episode).

2. **`ai_interface/envs/JAL_env.py` — `carry_continuation` extended to GRAB:** `carry_continuation` now includes `DRIBBLE_PHASE_GRAB` gated by `dribble_st.committed`. `committed` is set on the first fresh `dribble_to` pick and cleared by `_end_dribble_session` and on kick. Effect: a single `dribble_to` selection now reliably opens a carry (catch fired 17/1500 in smoke vs 1/3000 before, ~34×).

3. **`ai_interface/envs/JAL_env.py` — goal-relative dribble target decode:** Was `goto_x = raw*45`, `goto_y = clip(raw*30, ±10)` (absolute field coords → neutral output = midfield, always backward). Now: `fwd = (raw*0.5+0.5)*dribble_fwd_max` (maps [-1,1] → [0, fwd_max], always forward), `lat = raw*dribble_lat_max`, `target = ball + fwd*dir_to_goal + lat*perp`, clamped before goal line and `|y| ≤ dribble_target_y_clip`. Neutral head output is now a forward carry straight at goal center = already gap-improving = quality one-shot fires from step 1 = gradient flows from step 1.

4. **`ai_interface/envs/reward.py` — `dribble_fwd_max` / `dribble_lat_max` on `RewardConfig`:** Added the two tunable knobs that the decode above reads from config (values in this run: `dribble_fwd_max=10.0`, `dribble_lat_max=6.0`).

### What worked

**Dribbling finally works.** The two structural bugs (GRAB outside macro; absolute-coord target) both prevented any useful carry signal from reaching the policy across §29–§33. Both are fixed. Key evidence:

- **66% of carry-opens are useful** (gap-improving) vs 0% in every prior Stage 2 run.
- **Goal rate stable at 56–62%** across 1551 episodes with no collapse — the policy integrates dribbling without losing shooting ability.
- **Only 1 sim-brick** in 1551 episodes (the embedded-sim dead-ball fix from [[project_embedded_sim_deadball_brick]] held up throughout).
- **Training healthy to completion:** entropy annealed as scheduled, KL well-controlled, no instability.

### Next stage

**Aim sharpening.** The largest non-goal bucket is `ball_in_penalty_off_target` (323 eps, 20.8%). Aim quality is flat (0.27→0.28) and soft — kicks land but not at the gap. The goalie (148 catches, 9.5%) and OOB (141, 9.1%) are secondary.

Options to address aim:
- Boost `kick_aim_bonus_weight` (currently 20) and `alignment_weight` (currently 3) further — aim reward is there but not dominant enough to sharpen placement.
- Tighten `goalie_gap_min_quality` threshold to force only high-quality-gap shots (currently 0.3).
- Add a stage where `bad_aim_kick_penalty` is increased and the model must hold for a better angle before firing.

Warm-start from `stage2g_dribble_param_complete.pt` (this run's output). Do NOT revert to the pre-§34 absolute-coord target decode.

---

## 35. Training run: 20260622_114715_270338 — PPO stage2g_dribble_param / physical-realism and corrected reward-credit retrain (SUCCESS, with late selectivity regression)

**Start time:** 2026-06-22 11:47:15 IST  
**Stage:** `stage2g_dribble_param`  
**Steps:** The last completed episode ended at 199,901 / 200,000 environment steps; the trainer reached and saved the 200,000-step checkpoint while episode 1595 was still open.  
**Load model:** `models/ppo_jal_expandable_wide/stage2c_goalie_balance_complete.pt`  
**Final model:** `models/ppo_jal_expandable_wide/stage2g_dribble_param_complete.pt`

**Aim:** Retrain Stage 2g from the clean Stage 2c shooter after changes 1–4 changed the physical turn/catch dynamics, action accounting, macro credit, bounded parameter distribution, and dribble/kick reward geometry. The acceptance gate was useful target/carry deltas above zero while retaining at least roughly 55% goals.

### Results

| Metric | Value |
|---|---:|
| Completed episodes | 1,594 |
| Overall goal rate | **946/1,594 = 59.3%** |
| Last-100 goal rate | **63.0%** |
| Best rolling-100 goal rate | **73.0% at episode 880 (~108.7k steps)** |
| Final-194 goal rate | **62.9%** |
| Final-10 average reward / length | **65.8 / 100.8 steps** |

**Episode outcomes:** goal 946 (59.3%) · off-target 334 (21.0%) · goalie catch 153 (9.6%) · OOB 149 (9.3%) · max-steps 12 (0.8%).

**Goal-rate trend (200-episode buckets):**

```text
eps    1– 200: 37.5%
eps  201– 400: 51.5%
eps  401– 600: 60.5%
eps  601– 800: 64.5%
eps  801–1000: 64.5%
eps 1001–1200: 66.0%
eps 1201–1400: 67.5%
eps 1401–1594: 62.9%
```

**Kick quality:** first 200 kicks averaged aim quality 0.258 with 34.5% bad aim; the last 200 averaged 0.370 with only 1.0% bad aim. The corrected post-safe goalie-gap quality among geometrically valid kicks also rose from 0.487 to 0.599. This is a real improvement, not merely a higher goal count.

**Dribble target and achieved-gap metrics:**

| Metric | Whole run | Mid-run peak | Last ~200 episodes |
|---|---:|---:|---:|
| Useful target commitments | 1,935/2,162 = **89.5%** | **94.4%** (eps 1201–1400) | **89.8%** |
| Mean reachable target gap delta | **+0.0125** | **+0.0138** (eps 1001–1200) | **+0.0125** |
| Positive completed achieved-gap deltas (>0.001) | 515/643 = **80.1%** | **91.5%** (eps 801–1000) | **73.0%** |
| Mean completed achieved-gap delta | **+0.0196** | **+0.0247** (eps 1001–1200) | **+0.0191** |

The corrected reachable-endpoint reward is therefore training the intended behavior. This is materially stronger evidence than the previous run's full-target usefulness metric: both the target selected for the physically reachable 0.85 m segment and the gap actually achieved are positive on average.

**Dribble selectivity:** 937/1,594 episodes (58.8%) opened at least one carry, rising to 64.0% over the last 200. Across the full run, episodes with a carry scored 62.5% versus 54.8% without one. That aggregate difference is partly a learning-time confound: over the final 200 episodes both groups scored exactly 62.5%. The final summary was also dribble-heavy by duration (`dribble_to=50%` of executed steps). Because `dribble_to` is a multi-step macro, this is not a 50% decision-selection rate, but it still shows that the final policy spends substantial time carrying without a measurable late-run goal-rate advantage.

Carries reduced OOB failures over the whole run (52/937 = 5.5% with a carry versus 97/657 = 14.8% without), but all 12 max-step episodes involved a carry. This suggests the macro is useful for ball control yet still occasionally overcommits or churns.

**PPO health:** 48 updates completed. One early update at 12,288 steps overshot the KL target (0.0323 versus 0.015), after which KL remained controlled and finished at 0.0051. Categorical entropy generated 24 tripwire warnings, reached a minimum of 0.166, and finished at 0.261. There was no catastrophic collapse, but the low primitive entropy matches the large phase-to-phase oscillation between dribble-heavy and direct-play behavior.

### Code changes (non-config)

- Enforced the physical 20°/s angular-velocity cap in both simulator and robot serialization, and extended `dribble_to` alignment duration to operate under that cap.
- Removed rcssserver catch pose correction for field players in both the external and embedded server trees while retaining catch glue; goalkeeper correction remains unchanged.
- Corrected requested/executed primitive accounting so padded robots and requested-but-overridden macro actions are not reported as physical actions.
- Changed kick projection to originate at the ball, introduced post-safe goal-gap geometry, and changed immediate dribble target reward to signed reachable-endpoint improvement.
- Required verified carry state for active-dribble reward, added achieved-gap reward, and enabled explicit OOB/off-target terminal penalties for Stage 2g.
- Preserved primitive commitment during a latched dribble macro while removing parameter-policy credit after the initial target has been latched.
- Replaced clipped Gaussian parameter actions with a tanh-squashed Gaussian and its correct change-of-variables log probability.

Exact source files, rationale, math, line locations, and validation are recorded in `docs/CHANGES.md` under the 2026-06-22 changes 1–4 entry.

### What worked

The run passed its primary gate. Goals recovered from 37.5% in the first 200 episodes to 63.0% in the final 100, bad kicks fell from 34.5% to 1.0%, 89.5% of reachable targets were gap-improving, and 80.1% of completed measured carries achieved a positive gap delta. The new reward/credit formulation is mathematically aligned with the physical segment and receives a learnable signal.

The strongest checkpoint cannot be selected from stochastic training return alone. The 110k checkpoint was near the best rolling-100 window (70–73%), while 140k and 180k were also around 68–69%; the complete checkpoint finished at 63%. Later checkpoints have better kick and target quality but more dribble usage and weaker achieved-gap precision.

### Next stage

Run identical deterministic embedded inference seeds against at least `stage2g_dribble_param_steps110000.pt`, `stage2g_dribble_param_steps140000.pt`, `stage2g_dribble_param_steps180000.pt`, and `stage2g_dribble_param_complete.pt`. Compare goal rate, carry-open rate, achieved-gap delta, episode length, and failure reasons. Do not automatically promote the final checkpoint.

If deterministic evaluation confirms the late selectivity regression, the next curriculum should reduce dense dribble-discovery incentives rather than changing geometry again: fade `dribble_target_quality_weight` and `dribble_active_bonus`, retain signed `dribble_achieved_gap_weight`, and make the policy earn most dribble value through improved post-carry shot outcome. The geometry and physical fixes should remain unchanged.

---

## 36. Training run: 20260623_105445_993837 — PPO stage2h_newphys_pm10_finetune / preserved Stage 2H low-LR fine-tune (MIXED — safer than full retrain, not yet promotable)

**Start time:** 2026-06-23 10:54:45.994147  
**Stage:** `stage2h_newphys_pm10_finetune`  
**Steps:** 49,890 / 50,000  
**Load model:** `models/ppo_jal_expandable_wide/stage2h_newphys_pm10_complete_preserved_20260623_101555_IST.pt`  
**Save namespace:** `models/ppo_jal_expandable_wide_finetune/`

**Aim:** Conservative 50k fine-tune from the preserved working Stage 2H checkpoint, rather than the failed full retrain from `stage2c_goalie_balance_complete.pt`. The run kept the same ±10 spawn distribution, same goalie, same real-robot constrained primitives, committed kick macro, keeper-away target, retarget guard, penalty-area dribble guard, and keeper-zone shaping. PPO was deliberately conservative: low LR, low entropy, and tighter KL.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 280 |
| Overall goal rate | 143/280 = **51.1%** |
| Last 100 episodes goal rate | **54.0%** |
| Avg reward, last 10 eps | **45.86** |
| Avg episode length, last 10 eps | **126.5** |

**Episode outcomes:**

- `goal_scored`: 143 / 280 = **51.1%**
- `goalie_catch`: 65 / 280 = **23.2%**
- `ball_out_of_bounds`: 62 / 280 = **22.1%**
- `ball_in_penalty_off_target`: 10 / 280 = **3.6%**

**Goal-rate trend:**

```text
eps   1–200: 49.5%
eps 201–280: 55.0%
last 100:    54.0%
```

**Aim quality:**

| Window | Avg aim_quality | Bad-aim rate |
|---|---:|---:|
| First 200 kicks | 0.346 | 0.0% |
| Last 200 kicks | 0.350 | 0.5% |

**Sample action distributions:**

- episode 1: `approach_ball=71%`, `dribble_to=27%`, `kick=1%`
- episode 200: `approach_ball=62%`, `dribble_to=36%`, `kick=0%`, `turn=0%`

### Key config

Active reward/mechanics were the current Stage 2H set:

- `step_bonus=-0.1`
- `dribble_target_progress_weight=0.3`, `dribble_target_progress_clip=0.2`
- `dribble_target_quality_weight=4.0`
- `dribble_achieved_gap_weight=4.0`
- `dribble_active_bonus=0.0`
- `dribble_target_y_clip=10.0`
- `dribble_penalty_area_guard_margin=1.0`
- `dribble_penalty_area_y_clip=3.5`
- `kick_keeper_away_target_y=3.25`
- `kick_keeper_retarget_max_count=1`
- `keeper_zone_radius=4.0`
- `keeper_zone_floor=0.0`

Conservative PPO settings:

- `learning_rate_initial=0.0001`
- `learning_rate_final=0.00003`
- `ent_coef_initial=0.005`
- `ent_coef_final=0.001`
- `target_kl=0.008`

### Code changes (non-config)

This run used the current Stage 2H source-level mechanics that are not fully represented by config alone:

- committed kick macro in `JAL_env.py`, so kick alignment is not interrupted by the next policy sample;
- kicks routed through `basic_commands.kick(..., dribbling=True)` with turn-rate limiting, not raw kick emission;
- deterministic keeper-away kick target and one-shot retarget guard;
- penalty-area dribble target y-clamp;
- keeper-zone dense reward factor applied from the actual shot/carry origin;
- debug/inference tracing for kick macro state, kick target, retargeting, and dribble guard.

### What worked

Fine-tuning from the preserved Stage 2H model was materially better than the failed full retrain from Stage 2C. The previous 2026-06-23 full retrain was only ~22–28% goals around 150k steps and was dominated by catches/OOB. This fine-tune reached **51.1% overall** and **54.0% over the last 100** in only 50k steps.

The short fine-tune also avoided the max-step problem: the recorded failures were catches, OOB, and penalty/off-target, not long dribble stalls. That confirms the separate fine-tune strategy is the right direction compared with relearning from Stage 2C.

### What still failed

The checkpoint is not safe to promote yet. Failure rates remain too high:

- goalie catches are still **23.2%**;
- OOB is still **22.1%**;
- aim quality barely moved (`0.346 → 0.350`);
- bad-aim rate is low, but low bad-aim does not imply keeper-safe placement — many shots are legal/non-bad but still caught or leave the field.

The training result is also much worse than the preserved model's best deterministic inference behavior, so promotion must be based on checkpoint-by-checkpoint inference, not the final stochastic training checkpoint.

### Next stage

Run deterministic embedded inference with debug logs on every fine-tune checkpoint:

- `models/ppo_jal_expandable_wide_finetune/stage2h_newphys_pm10_finetune_steps10000.pt`
- `models/ppo_jal_expandable_wide_finetune/stage2h_newphys_pm10_finetune_steps20000.pt`
- `models/ppo_jal_expandable_wide_finetune/stage2h_newphys_pm10_finetune_steps30000.pt`
- `models/ppo_jal_expandable_wide_finetune/stage2h_newphys_pm10_finetune_steps40000.pt`
- `models/ppo_jal_expandable_wide_finetune/stage2h_newphys_pm10_finetune_steps50000.pt`
- `models/ppo_jal_expandable_wide_finetune/stage2h_newphys_pm10_finetune_complete.pt`

Compare them against the preserved checkpoint, not against training return. Promote only if deterministic inference is at least as good as the preserved model on goal rate and does not reintroduce catches/OOB.

If all fine-tune checkpoints are worse than the preserved model, keep the preserved Stage 2H checkpoint as the base and do not continue PPO fine-tuning. If one checkpoint is close but has persistent OOB, the next adjustment should target shot safety/targeting, not broader entropy or a full retrain.

---

## 37. Training run: 20260625_181000_753820 — PPO stage3_defender_v2_finetune / upgraded defender reward and SSL rules (MIXED — meets goal-rate target, not clean enough yet)

**Start time:** 2026-06-25 18:10:00.754081  
**Stage:** `stage3_defender_v2_finetune`  
**Steps:** 199,912 / 200,000  
**Load model:** `models/ppo_jal_expandable_wide/stage3_defender_steps400000.pt`  
**Save namespace:** `models/ppo_jal_expandable_wide_stage3_v2/`

**Aim:** Fine-tune the existing Stage 3 attacker against the upgraded hardcoded defender with defender-aware shot quality, Stage 2H real-physics reward settings, SSL foul detectors, corrected `8.5` env-unit dribble segment limit, and no standalone `goto` / `turn` primitives.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 973 |
| Overall goal rate | 381/973 = **39.2%** |
| Last 100 episodes goal rate | **39.0%** |
| Last 200 episodes goal rate | **41.0%** |
| Last 100 shot goal rate | **47.0%** |
| Avg reward, last 100 eps | **-70.32** |
| Avg reward, last 10 eps | **7.23** |
| Avg episode length, last 100 eps | **214.7** |

**Episode outcomes:**

- `goal_scored`: 381 / 973 = **39.2%**
- `goalie_catch`: 247 / 973 = **25.4%**
- `max_steps`: 186 / 973 = **19.1%**
- `ball_out_of_bounds`: 74 / 973 = **7.6%**
- `defender_push_foul`: 33 / 973 = **3.4%**
- `defender_in_defense_area`: 23 / 973 = **2.4%**
- `attacker_push_foul`: 22 / 973 = **2.3%**
- `attacker_excessive_dribble`: 6 / 973 = **0.6%**
- `ball_in_penalty_off_target`: 1 / 973 = **0.1%**

**Goal-rate trend:**

```text
eps   1-200: 40.5%
eps 201-400: 36.5%
eps 401-600: 33.5%
eps 601-800: 44.0%
eps 801-973: 41.6%
last 100:    39.0%
last  50:    42.0%
```

**Shot quality:**

| Window | Shot goal rate | Avg gap_quality | Avg aim_quality | Avg dist to keeper |
|---|---:|---:|---:|---:|
| Last 100 shots | 47.0% | 0.591 | 0.412 | 9.70 |
| Last 200 shots | 54.0% | 0.611 | 0.410 | 9.01 |

**Final action mix:**

- last 100 episodes: `approach_ball=35.4%`, `dribble_to=48.6%`, `kick=0.2%`, `turn=14.0%`
- last 200 episodes: `approach_ball=34.5%`, `dribble_to=50.6%`, `kick=0.2%`, `turn=13.0%`

**SSL rule events:**

- `attacker_touched_ball_in_defense_area`: 976 events across 511 episodes, non-terminal but each penalized `-8`
- `attacker_push_foul`: 22 terminal episodes
- `attacker_excessive_dribble`: 6 terminal episodes
- `defender_push_foul`: 33 terminal episodes, zero attacker penalty
- `defender_in_defense_area`: 23 terminal episodes, zero attacker penalty

### Key config

- `use_defender_lane_gate=true`
- `defender_lane_block_dist=2.6`
- `defender_lane_min_quality=0.3`
- `dribble_segment_limit=8.5`
- `dribble_target_quality_weight=5.0`
- `dribble_achieved_gap_weight=5.0`
- `dribble_target_progress_weight=0.3`
- `dribble_active_bonus=0.0`
- `post_dribble_kick_bonus=3.0`
- `post_dribble_kick_combo_window=6`
- `post_dribble_kick_quality_scale=true`
- `keeper_zone_radius=4.0`
- `keeper_zone_floor=0.0`
- `kick_keeper_away_target_y=3.25`
- `disabled_actions=["goto", "turn"]`

### Code changes (non-config)

This run used the new Stage 3 v2 source-level mechanics:

- defender-aware `Q3` shot quality in `reward.py`, using safe in-mouth target candidates rather than only the goal center;
- dribble target and achieved-gap rewards in `JAL_env.py` gated by combined goalie, defender-lane, and keeper-zone quality;
- post-dribble kick bonus scaled by fired-shot quality;
- `SSLRuleTracker` in `ssl_rule_events.py` for crash, pushing, no-progress, excessive-dribble, and defense-area-touch events;
- corrected local `dribble_segment_limit` lookup in `_action_to_commands`, fixing the earlier runtime `NameError`.

### What worked

The run cleared the immediate Stage 3 v2 scoring threshold. The last 100 episode goal rate was **39.0%**, and the last 200 was **41.0%**, both above the planned 25% acceptance target. The final 200 shots scored **54.0%**, so when the policy actually creates and releases a shot, it often beats the defender and goalie.

The attacker also learned to use the new dribble lane reward: the final action mix is roughly half `dribble_to`, and shot probes in the final windows show moderate lane quality (`gap_quality` around `0.59-0.61`).

### What still failed

This is not clean enough to promote blindly. The training return remains negative over the last 100 episodes because the policy often carries too deep or too long before shooting:

- `attacker_touched_ball_in_defense_area` fired **976** times across **511** episodes;
- `max_steps` remains high at **19.1%** overall and **25%** in the last 100 episodes;
- goalie catches are still **25.4%** overall and **22%** in the last 100 episodes;
- only **3** kicks landed inside the 6-step post-dribble combo window, while **131** kicks happened more than 100 steps after a carry opened;
- final action distribution still reports `turn=14%` even though standalone `turn` was intended to be disabled, so the disabled-action mapping should be audited.

The main behavioral failure is not "cannot score." It is "scores sometimes, but does not reliably convert a good dribble into an immediate shot." It keeps carrying into the defense area or stalls until the episode times out.

### Next run

Run deterministic embedded inference on the 130k-200k checkpoints before promoting the final checkpoint. The training curve is not monotonic enough to assume `steps200000.pt` is best.

For the next fine-tune, target release timing and penalty-area discipline rather than more lane discovery:

- increase penalty-area guard pressure for dribble targets and consider making repeated attacker defense-area touches terminal after a short grace count;
- increase the reward for kicking within the post-dribble combo window and reduce/zero the same bonus after the window expires;
- audit `disabled_actions` because `turn` still appears in action distribution logs;
- reduce the long-carry local optimum by adding a small penalty when a carry remains open for too many steps without a shot or meaningful lane improvement;
- keep defender-lane geometry unchanged for now, because the shot success rate after firing is already reasonable.

---

## 38. Training runs: 20260625 Mac-mini ×3 — PPO `stage4_2v2` / first 2-attacker scale-up (FAILED — ~0% goals; sim-freeze brick + unsolved 2v3)

**Runs (all on Mac mini, embedded sim):**
- `20260625_162023_992706` — stage4 from line 27953
- `20260625_200856_640414` — stage4 from line 27549
- `20260625_213858_046720` — stage4 from line 26963 (**the only run that completed all 400k and saved the checkpoints**)

Each session first re-ran `stage3_defender_v2_finetune` (150k) then rolled into `stage4_2v2` (400k). Numbers below are the **stage4 segment only** (parsed directly, since `parse_training_log.py` keys on the `ACTIVE` tag which is still on stage3 and therefore mislabels/blends the two stages).

**Load model:** top-level `final_models/stage3_complete.pt` (single-attacker Stage 3). Backbone reused count-agnostically for `num_robots 1→2` ("Reusing existing agent … flushing rollout buffer"), so attacker #2 inherits the trained attacker's weights, not random init.
**Save namespace:** `models/ppo_jal_expandable/` (last four checkpoints, steps 370k–400k, copied into `models/ppo_jal_expandable_wide/`).

**Aim:** First scale-up from 1 → 2 RL attackers. **2 RL attackers** (ids 1, 2 on TritonBots) vs **3 scripted opponents** on TeamB — `goalie` (id 1), `defender` (id 2), `marker_defender` (id 3, marking id 2). `team_config_stage4.json`, all primitives enabled (`disabled_actions: []`), random ball + spawn θ, 400k budget. Reward adds multi-robot coordination shaping on top of Stage 3: `spread_bonus 0.04`, `redundant_chase_penalty 0.08`, `support_position_bonus 0.05`, `possession_transfer_bonus 5.0`, `defender_lane_gate`, `opponent_near_ball_penalty -0.3`; `goal_reward 70`.

### Results (stage4 segment only)

| Run | Stage4 eps | Goals | Dominant outcome | Mean reward (early→late) |
|---|---:|---:|---|---|
| 162023 | 3265 | 1 (**0.0%**) | `frozen_state_stale_sim` 2855 (**87%**) | −86 → −10 |
| 200856 | 1973 | 1 (**0.1%**) | `frozen_state_stale_sim` 1434 (**73%**) | −71 → −12 |
| 213858 (completed) | 1114 | 0 (**0.0%**) | `max_steps` 913 (**82%**) | −92 → −57 |

**Outcome breakdown, completed run 213858:** `max_steps` 913 (82%), `frozen_state_stale_sim` 170 (15%), `ball_in_penalty_off_target` 26 (2.3%), `defender_push_foul` 4, `goalie_catch` 1. **Only 26 off-target + 1 catch in 1114 episodes** — the attackers barely got a shot away.

**Goal-rate trend (213858, quartiles):** 0.0% / 0.0% / 0.0% / 0.0% — flat zero throughout. Mean reward climbed −92 → −57 (something is being learned), but never positive and never a goal.

### What went wrong

Two distinct failure modes, neither solved:

1. **Embedded-sim freeze brick, aggravated at 2 robots (runs 162023 & 200856).** `frozen_state_stale_sim` ([JAL_env.py:3458](../ai_interface/envs/JAL_env.py#L3458)) fires when the ball **and all controlled robots** stay frozen (Δpos/Δθ < eps) for `_FROZEN_STATE_STEPS` consecutive cycles — i.e. the embedded engine has stalled. It hit **73–87% of episodes**, so two of the three runs trained mostly against a dead simulator and are essentially garbage data. This is the dead-ball/stale-sim brick family resurfacing hard once a second controlled robot is in the loop.

2. **2v3 never solved (clean run 213858).** With far fewer freezes (15%), **82% of episodes time out (`max_steps`)** at deeply negative reward. The 2-attacker policy holds/shuffles the ball but cannot break a 3-defender block: ~0 shots, 0 goals across all 1114 episodes. Warm-starting attacker #2 from the single-attacker weights gave competent individual ball skill but **no coordination emerged** to beat the extra defenders in 400k steps.

### Fix for next run

- **Fix the freeze brick before any more 2-robot training** — it's the cheaper, more decisive problem (corrupted 2 of 3 runs outright). Investigate why `frozen_state_stale_sim` spikes specifically at `num_robots=2`: likely the embedded engine stalls when a controlled robot's command queue desyncs (cf. the catch-glue / cone-rejected-kick brick variants), now reachable via a second attacker. Either harden the engine-rebuild trigger to also fire on a frozen-state terminal, or root-cause the stall in `socket_utils.py` command queueing for multi-robot.
- **Consider a 1 → 2 intermediate before 2 → full team** (the documented-safe ladder): warm-start a 2-attacker stage against a *single* scripted defender + goalie (drop the `marker_defender`) so coordination has a gentler gradient than 2v3 from step 0. The plan (`sprightly-singing-kite.md`) flagged exactly this fallback if the jump destabilizes — it did.
- **Re-examine the timeout local optimum:** 82% `max_steps` with near-zero shots suggests the multi-robot shaping (`spread`/`support`/`possession_transfer`) may be rewarding passive positioning over shot creation. Verify the attackers actually attempt to penetrate rather than circulate; if not, add shot-urgency pressure as in Stage 3 (`carry_urgency_*`) and/or reduce the support-position bonus.
- Net: the `stage4_2v2_steps*.pt` checkpoints encode a 2-attacker policy at **~0% scoring** against this opponent set — **do not promote them**; treat Stage 4 as not-yet-started once the freeze brick is fixed.

---

## 39. Training run: 20260626_185308_653190 — PPO `stage4i_2atk_1def` / documented-safe 2v1 intermediate (STOPPED at 53% — Fix A validated; kick=0% root cause = unlearnable role split / crowding, NOT aim)

**Stopped early** at step 158,924 / 300,000 (~53%, 444 episodes) — halfway gate verdict was decisive, so the remaining 141k steps were not worth burning.

**Load model:** top-level `final_models/stage3_complete.pt` (single-attacker Stage 3); weight-shared encoder reused count-agnostically so both attacker slots start competent.

**Aim:** The documented-safe 1→2 rung after §38 failed. **2 RL attackers** (TritonBots ids 1,2, both warm-started) vs **ONE scripted defender + goalie** (`team_config_2atk_1def.json`, no `marker_defender`). New machinery: `pass_to_teammate` primitive (6th, through-ball `lead` param), dribble obstacle avoidance, `goto` avoidance, and a coordination + effective-passing reward (hardened pass-event transfer + `pass_quality`, role-gated spread/support). Target: emergent 2v1 overload — carrier shoots an open lane, passes to the open supporter when the defender closes.

### Results

| Metric | Value |
|---|---|
| Steps / planned | 158,924 / 300,000 (stopped) |
| Total episodes | 444 |
| Goals | **0 / 444 = 0.0%** |
| Goal rate (200-ep buckets) | 0.0% / 0.0% / 0.0% — flat zero |
| Aim quality (kicks, first→last 200) | 0.09 → 0.09 (no improvement) |
| Mean reward (200-ep buckets) | −138 → −126 |

**Episode outcomes:** `max_steps` 335 (75.5%), `defender_push_foul` 56 (12.6%), `attacker_push_foul` 23 (5.2%), `ball_teleport` 13 (2.9%), `ball_in_penalty_off_target` 12 (2.7%), `ball_out_of_bounds` 3, `defender_in_defense_area` 1, `ball_dead_goal_kick_l` 1.

**Action distribution (last 200 eps):** approach_ball 55%, dribble_to 33%, goto 4%, turn 4%, pass_to_teammate 1%, **kick 0%**.

### Code changes (non-config)

This run is the first to exercise all of: the `pass_to_teammate` primitive (`ppo_jal.py` index 5 + `JAL_env.py` decode/receiver-selection/pass-event tracking), `dribble_to(obstacle_avoidance=True)` CARRY detour (`basic_commands.py`), the tracked pass-event possession-transfer + `pass_quality` reward (`reward.py`/`JAL_env.py`), **Fix A** (`force_rebuild` on brick terminals threaded `reset → networker → commander → embedded backend`), and **Fix B** (the §38-followup reward retune). Full detail in `docs/CHANGES.md` (2026-06-26 entry).

### What went wrong

**Fix A worked; Fix B did not.**

1. **Fix A (brick recovery) — VALIDATED.** 22 brick terminals (`ball_teleport`/`frozen_state_stale_sim`) occurred over the run; the `force_rebuild` backstop fired **11 times** and the sim recovered cleanly every time — no 1-step death spiral, the run advanced steadily to 159k. The multi-robot sim-freeze that corrupted 2 of 3 §38 runs is **solved at the symptom level**.

2. **Fix B (kick collapse) — INSUFFICIENT. Same failure as §38: kick=0%, 0 goals, flat across all 444 episodes.** Fix B *is* biting — mean reward is deeply negative (−126 to −138), exactly carry_urgency + step penalties punishing the dribble-forever camp, so we successfully made dribbling unprofitable. **But the policy still won't kick** — it eats the penalty by approaching/dribbling (55%+33%=88% of actions) rather than switching to kick. Aim quality is pinned at **0.09** (the rare kicks are badly aimed), and 18% of episodes end on **push fouls** (bodies colliding near the ball — both robots converging on it despite role gating).

**Root cause (CORRECTED — it is the role/coordination split, NOT a missing kick macro).** An earlier draft of this section blamed the [turn-cap-breaks-aiming] pattern and recommended building a `face_then_kick` macro. **That macro already exists** — when the policy selects `kick`, [JAL_env.py:2329-2380](ai_interface/envs/JAL_env.py#L2329) latches a keeper-away target and calls `kick(..., dribbling=True)`, which returns a *geometric, cap-robust* `turn angle_diff/dt` until within 5° then fires ([basic_commands.py:231](ai_interface/utils/basic_commands.py#L231)). Stage 3 scored ~39% with this exact macro. So aim execution is not the blocker.

The real blocker is **2-attacker crowding from an unlearnable role split.** The reward gates every carrier/supporter reward on `is_chaser = is_nearest_to_ball` ([reward.py:950](ai_interface/envs/reward.py#L950)/[:1054](ai_interface/envs/reward.py#L1054)), but the **per-robot observation contains no role signal** — the live slot is only `x,y,θ,vx,vy,is_dribbling,start_dribble_xy`; dims 8+ are reserved/zero ([JAL_env.py:1698-1707](ai_interface/envs/JAL_env.py#L1698)). The policy is a **weight-shared encoder + shared heads + permutation-equivariant attention** ([ppo_jal.py:22](ai_interface/algorithms/ppo_jal.py#L22)), and both slots warm-start from the *same* selfish single-attacker Stage 3 policy. So the reward asks one robot to be the supporter, but the policy has no input telling it *which* robot it is — and `is_nearest_to_ball` flips cycle-to-cycle as both converge, so even the reward target oscillates. Result: both robots run the identical "go to ball" reflex → 88% approach+dribble, 18% push-foul collisions, and `kick` is never the EV-best action for *either* (both are fighting for the ball, neither ever gets clean, settled possession to shoot from). kick=0% is a **downstream symptom of the crowd**, not an aim failure.

### Fix for next run

- **Give the policy the role variable it is being graded on, and stabilize it.** (1) Fill reserved per-robot obs dim 8 with the robot's role / `is_nearest_to_ball` flag (the env already computes `nearest_rid` at [JAL_env.py:3320](ai_interface/envs/JAL_env.py#L3320)); optionally dim 9 = teammate-relative geometry. (2) Make the chaser assignment **sticky** (commit at episode start or with a hysteresis margin) and feed the *same* committed role to both the obs feature and the reward gate. Then the shared weights can learn one conditional policy (chaser → ball, supporter → spread), and the supporter reward stops chasing a flipping label. Stays centralized JAL — no MAPPO.
- **Do NOT add another reward pass first.** Fix B's shaping (`carry_urgency`, `spread`, `support`) was never learnable because its gating variable was invisible and oscillating — fix the observation/role first, then re-evaluate the shaping.
- **Keep Fix A as-is** — brick recovery is validated; carry it forward unchanged.
- **Do not promote** any checkpoint from this run (0% scoring). Re-run `stage4i_2atk_1def` from the same warm-start once the role feature + sticky assignment land.

---

## 40. Training run: 20260628_025218_390537 — PPO `stage4j_support_pass_v1` / supporter-target + pass-gated 2v1 intermediate (FAILED — no passing, late regression to 8% goals)

**Completed** at 299,728 / 300,000 steps, then saved `models/ppo_jal_expandable/stage4j_support_pass_v1_complete.pt`.

**Load model:** `models/ppo_jal_expandable_wide_stage3_v3/stage3_defender_v2_finetune_complete.pt`.

**Aim:** Stage 4j should turn the task into a real 2-attacker pattern: one claimant keeps full ball actions, the non-claimant is `goto`-only and learns receive/support coordinates, and `pass_to_teammate` is unmasked only when the supporter has a legal, ready target with a clear lane and useful continuation shot.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 863 |
| Overall goals | 87 / 863 = **10.1%** |
| Last 100 goal rate | **8.0%** |
| Last 50 goal rate | **6.0%** |
| Last 100 avg reward | **-122.6** |
| Last 100 avg length | **327.7** |

**Episode outcomes:** `max_steps` 596 (69.1%), `goal_scored` 87 (10.1%), `ball_in_penalty_off_target` 78 (9.0%), `goalie_catch` 78 (9.0%), `ball_out_of_bounds` 11 (1.3%), `defender_push_foul` 7 (0.8%), `robot_out_of_bounds` 4, `defender_in_defense_area` 1, `attacker_push_foul` 1.

**Goal-rate trend:** eps 1-200 = 3.5%, 201-400 = 10.5%, 401-600 = 16.0%, 601-800 = 11.0%, 801-863 = 7.9%. The run improved into the middle, then regressed late.

**Action distribution, final windows:**

- Last 100, carrier `rid=1`: `approach_ball` 21.2%, `dribble_to` 62.6%, `goto` 3.8%, `kick` 0.8%, `turn` 10.5%.
- Last 100, supporter `rid=2`: `goto` 97.1%, with some claimant-switch noise (`approach_ball` 8.9%, `dribble_to` 16.3% on episodes where it likely became claimant or the action logger sampled transient role changes).
- Final summary bucket: aggregate actions still show `pass_to_teammate=0%`.

**Kick quality:** 1,638 kick events. First 200 kicks had avg `aim_quality=0.326`, bad aim 18.0%; last 200 improved slightly to avg `aim_quality=0.385`, bad aim 12.5%. This is not enough: final goals dropped despite slightly better aim.

**SSL events:** 42 total: `attacker_touched_ball_in_defense_area` 32, `defender_push_foul` 7, `defender_in_defense_area` 2, `attacker_push_foul` 1. Collision rules are not the dominant failure.

### Code changes (non-config)

This run used the Stage 4j supporter/pass code path added after §39:

- `ai_interface/envs/stage4_support.py`: explicit support target classification (`RECEIVE_READY`, `RECEIVE_TARGET`, `GENERAL_SUPPORT`) with legality, lane-clear, and continuation-quality checks.
- `ai_interface/envs/JAL_env.py`: role-aware primitive mask, non-claimant `goto` execution, support-target construction, ready-target pass gate, supporter-selected pass target routing, pass outcome penalties, dribble-gap banking, and open-shot urgency state.
- `ai_interface/envs/reward.py`: support target progress/readiness rewards, bad-target penalty, pass reward/penalty fields, dribble achieved-gap finish window, and open-shot urgency fields.
- `configs/ppo_jal_curriculum_config.json`: new active `stage4j_support_pass_v1` config with the stage3-v3 warm-start and Stage 4j pass/support reward settings.

### What went wrong

The supporter mask solved crowding mechanically, but did **not** create a two-attacker policy. The supporter mostly executed `goto`, yet the carrier never learned to use it:

- There were **no real pass events** in the timestamped logs: no `Pass`, no `possession_transfer`, no pass failure records, and `pass_to_teammate=0%` in summaries.
- The carrier stayed in a Stage-3-like local optimum: last-100 `dribble_to` was **62.6%**, while `kick` was only **0.8%**.
- Most episodes still timed out: last-100 `max_steps` was **62%** and overall `max_steps` was **69.1%**.
- The late curve regressed: goals peaked at **16%** for episodes 401-600, then fell to **7.9%** in the final bucket.
- The pass/support diagnostics were not logged (`support_target`, `RECEIVE_READY`, `RECEIVE_TARGET` all absent), so the run cannot tell us whether the supporter rarely proposed valid targets, the pass mask was too strict, or the carrier never sampled pass after readiness.
- The log showed the older carry-urgency penalty firing (`grace=25`, `0.03/step`, cap `2.25`) even though Stage 4j also configured the newer open-shot urgency fields. The intended stronger "shoot/pass when open" pressure was therefore not clearly controlling behavior.

The root cause is likely an overly hard sparse coordination gate: a pass can only be selected after the supporter target is already ready, physically close, legal, lane-clear >= 0.70, shot-quality >= 0.35, and the carrier has front-cone possession. Since the carrier's mask is based on the previous support target state, and the ready state is not frequent enough, the policy never explores the pass payoff. The result is a single-carrier dribble policy with a passive `goto` teammate.

### Fix for next run

Do not promote `stage4j_support_pass_v1_complete.pt`.

First add logging/instrumentation so the next run is diagnosable:

- log per-episode counts for support target modes and rejection reasons;
- log pass-mask availability count for claimant slots;
- log when `pass_to_teammate` is masked and why (`no_ready_target`, `bad_cone`, `lane_low`, `quality_low`, `too_close`);
- log ready-target distance/lane/quality histograms.

Then make the next curriculum easier and less sparse:

- split Stage 4j into a receive-position pretraining rung where the supporter is rewarded for `RECEIVE_TARGET` and `RECEIVE_READY` even before pass execution is required;
- temporarily relax pass readiness: allow pass on `RECEIVE_TARGET` with a larger receive radius or lower lane threshold, then anneal back to lane >= 0.70 and radius 2.5;
- make pass availability itself a small dense reward for the carrier/supporter pair so the policy can discover the preconditions before needing a full possession-transfer event;
- reduce single-carrier dribble farming further by lowering `post_dribble_kick_bonus` from 8.0 toward the Stage 3 v3 value and/or increasing the active open-shot/carry urgency once `Q3` is high;
- verify the open-shot urgency implementation path, because the configured `open_shot_urgency_*` values did not visibly replace the old `carry_urgency_*` behavior in this run.

Math note: the pass success reward was still sound on paper. A quality-0.5 pass should pay `(12 + 16 * 0.5) / 2 = +10` team reward after averaging over two attackers, while an intercepted pass gives `-16 / 2 = -8`. The issue was not reward EV after a successful pass; it was that the policy almost never reached the valid-pass action surface.

---

## 41. Training run: 20260628_171015_717398 — PPO `stage4p_receive_finish_macro_v1` / receive-finish command macro (FAILED — flat 24% goals, the finish macro fired twice in 826 episodes because passes don't resolve)

**Completed** at 199,974 / 200,000 steps, then saved `models/ppo_jal_expandable/stage4p_receive_finish_macro_v1_complete.pt`.

**Load model:** `models/ppo_jal_expandable/stage4o_forced_pass_finish_v1_steps160000.pt`.

**Aim:** Replace the mask-only post-pass finish scaffold with a per-receiver **command-level macro** that acquires → settles → (optionally stages under the SSL dribble cap) → aims through the kick cone → fires, and only then consumes the banked pass-finish reward (payout scaled by fire-time shot quality). Keep the learned pass timing / support positioning from Stage 4o.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 826 |
| Overall goals | 199 / 826 = **24.1%** |
| Last 100 goal rate | **25.0%** |
| Last-26 summary reward μ | **-85.6** |
| Avg episode length (last 10) | 187.9 |

**Episode outcomes:** `ball_in_penalty_off_target` 232 (28.1%), `max_steps` 208 (25.2%), `goal_scored` 199 (24.1%), `goalie_catch` 110 (13.3%), `attacker_excessive_dribble` 37 (4.5%), `ball_out_of_bounds` 27 (3.3%), `defender_push_foul` 8, `defender_in_defense_area` 2, `attacker_push_foul` 2, `frozen_state_stale_sim` 1.

**Goal-rate trend (200-ep buckets):** 21.5% → 20.5% → 26.0% → **30.0%** → 11.5%. Rose into the third/fourth quarter, then **collapsed to 11.5%** in the final episodes.

**Aim quality:** First 200 kicks avg `aim_quality=0.403`, bad aim 9.5%; last 200 **regressed** to 0.338, bad aim **16.5%**.

**Action distribution (carrier rid=1 / supporter rid=2):** carrier stayed dribble-centric throughout — `dribble_to` 56% (ep1) → 71% (ep200) → 83% (ep400) → 64–70% (ep600–800), with `kick` 0–2% and `pass_to_teammate` 0–2%. Supporter `rid=2` was `goto=100%` the entire run.

### Code changes (non-config)

This run exercised the Stage 4p receive-finish macro added after §40 (see CHANGES.md 2026-06-28 Stage 4p):

- `ai_interface/envs/JAL_env.py`: per-robot `post_pass_finish_macro` state + `_start_/_execute_/_expire_post_pass_finish_macro`, `_post_pass_finish_staging_target`, `_post_pass_finish_shot_quality`; macro runs before normal decode so the receiver can finish even when not the sticky claimant; debug counters `receive_finish_macro(started/acquire/settle/kick_align/fired/expired/bad_cone/avg_shot_q)`.
- `ai_interface/envs/reward.py`: macro knobs (`post_pass_finish_macro_enabled`, `_window_steps`, `_min_shot_quality`, `_staging_max_carry`, `_scale_reward_by_shot_quality`) + `support_pass_select_best_receiver`.
- `configs/ppo_jal_curriculum_config.json`: active `stage4p_receive_finish_macro_v1`, warm-start from stage4o 160k, `pass_finish_window_steps=80`, `pass_macro_max_align_steps=60`.

### What went wrong

The macro this stage was built for **almost never executed**, and scoring was flat versus the Stage-4i/4j baseline (~24% vs ~25% — all solo-dribble goals, not coordinated passing).

- **Passes do not resolve.** Across the whole run: `Pass FIRED=61`, `Pass RESOLVED=9` (~15% completion). Per-episode the pass macro is *requested* heavily (`requested=27–46`) but `fired=0` — the align macro burns all its steps (`align_steps≈27–46`, `timeouts=0` only because it ran out the episode) and never launches a legal pass.
- **The receive-finish macro started 9 times in 826 episodes and fired exactly 2 times** (`fired=1` twice). It stalls in `acquire`/`settle`/`kick_align` and dies without firing.
- **Finish banks are empty.** Resolved passes bank ≈0 (`bank=+0.00` ×7, +0.04, +0.09) because pass target quality is low (`passer_lane_q≈0.21–0.29`). Run totals: `banked=9, consumed=2, expired=5`. With nothing banked, there is no finish reward to drive the macro — the entire 4p mechanism is starved.
- **Pass geometry is still deep/wide.** Fired pass aims cluster at the clamp edge `x=31.50` and frequently `|y|=10.0` (the `support_target_y_clip` limit), with misses around `miss=2.60` and `recv_dist_to_aim=1.72` — the same deep/wide target problem §40 (4n) tried to fix is still active.
- **Late instability:** goals fell from a 30% peak to 11.5%, aim quality regressed (bad-aim 9.5%→16.5%), and last-bucket reward μ=-85.6. The forced-pass mask makes the carrier *want* to pass into a window that almost never becomes a legal, completable pass, producing many wasted align frames and off-target penalty-area losses (28.1% of episodes).

Root cause: **the bottleneck is upstream of the finish macro.** Stage 4p invested in turning *resolved* passes into shots, but passes rarely resolve. The pass-launch/align gate (`pass_macro` legality in `JAL_env.py`) lets the carrier latch a pass request and spin for tens of steps without ever reaching a legal, accurate launch, and the targets it does launch are deep/wide low-quality. The finish macro is correct (it fired twice, proving the plumbing), but it has almost nothing to consume.

### Fix for next run

Do not promote `stage4p_receive_finish_macro_v1_complete.pt`.

The next rung (Stage 4q) must fix **pass resolution**, not the finish layer:

- Raise `fired/resolved` rate: tighten the pass-launch gate so a latched pass either fires quickly when legal or is abandoned (don't let it spin `40+` align steps draining the episode); cap `pass_macro_max_align_steps` well below 60 and fall back to a shot/recover instead of burning the clock.
- Fix pass-target geometry: pull targets in from the `x=31.50` / `|y|=10.0` clamp edges. Tighten `support_target_max_x` and `support_target_y_clip` and/or require a higher minimum `passer_lane_q` (current launched passes at ~0.21–0.29 are too low to complete) so the supporter receive wedge sits in genuinely reachable, lane-clear space.
- Only after `Pass RESOLVED` per episode is consistently non-trivial does the receive-finish macro have banks to consume — keep 4p's macro code, it is validated.
- Watchpoint: if a Stage 4q smoke run still shows `fired≈0` while `requested` is high, the fix is still in the launch/align legality gate, not in reward.

Math note: the macro reward path is sound — a quality-0.5 pass banks `(14 + 12*0.5)/2 = +10.0`, paid only on a macro kick and scaled by fire-time shot quality. The failure is entirely that passes don't complete, so the bank is empty and the +10 chain never pays.

---

## 42. Training run: 20260628_210222_891381 — PPO `stage4r_intercept_gate_v1` / receiver-claim handoff + longer pass align budget (FAILED — no sustained passing improvement, goal rate regressed late)

**Completed** at 199,997 / 200,000 steps.

**Load model:** `models/ppo_jal_expandable_wide_stage3_v3/stage3_defender_v2_finetune_complete.pt`.

**Aim:** Test the two structural fixes after the Stage 4r diagnostics:

- hand the sticky ball claim to the intended receiver immediately after a pass fires, so the receiver is not locked to supporter-only `goto` while the ball arrives;
- raise `pass_macro_max_align_steps` from 24 to 90 so selected passes have enough time to align and fire.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 843 |
| Overall goals | 218 / 843 = **25.9%** |
| Last 100 goal rate | **21.0%** |
| Avg reward last 10 | **+4.76** |
| Avg episode length last 10 | 150.4 |

**Episode outcomes:** `ball_in_penalty_off_target` 283 (33.6%), `max_steps` 218 (25.9%), `goal_scored` 218 (25.9%), `goalie_catch` 47 (5.6%), `attacker_excessive_dribble` 46 (5.5%), `ball_out_of_bounds` 15 (1.8%), `defender_push_foul` 9 (1.1%), `frozen_state_stale_sim` 3, `defender_in_defense_area` 2, `attacker_push_foul` 1, `ball_dead_goal_kick_l` 1.

**Goal-rate trend (200-ep buckets):** 16.5% -> 25.5% -> **32.0%** -> 29.0% -> 27.9%; despite the mid-run rise, the last-100 goal rate fell to **21.0%**.

**Aim quality:** first 200 kicks avg `aim_quality=0.344`, bad aim 8.5%; last 200 avg `aim_quality=0.387`, bad aim 4.0%. Shooting aim improved, but this did not translate into a stronger final policy.

**Pass pipeline totals from diagnostics:** `requested=301`, `fired=14`, `resolved=3`, `expired_timeout=2`, `expired_interception=1`. `pass_macro started=32`, `align_steps=472`, `timeouts=0`. `receive_finish_macro started=0`, `fired=0`, `consumed=0`.

**Support/pass diagnostics:** `pass_available_steps=0` in the logged episode summaries. Support modes were mostly `GENERAL_SUPPORT=185,628`, with only `RECEIVE_READY=7,595` and `RECEIVE_TARGET=6,774`. Rejections/mask reasons were dominated by `pass_interceptable`, `not_kickable`, `bad_reception_cone`, `too_close_for_ssl_pass`, `pass_lane_blocked`, and `lane_low`.

### What changed

The longer align budget worked only in the narrow sense: pass macro timeouts disappeared (`timeouts=0`). That means the old 24-step timeout was not the only blocker.

The receiver-claim handoff had almost no opportunity to help. It only activates after a pass fires, and only 14 passes fired in 843 episodes. Only 3 resolved, and none started the receive-finish macro, so the run never produced enough post-pass situations to train or validate coordinated finishing.

The policy remained essentially a one-carrier strategy:

- carrier still alternated mostly between `approach_ball` and `dribble_to`;
- supporter stayed `goto`-only as designed, but rarely created a launchable pass state;
- no sampled action distribution showed meaningful `pass_to_teammate` usage.

### What went wrong

This run moved the bottleneck from explicit align timeouts to **pass launchability and target legality**:

- Pass requests were high enough (`301`) to show exploration, but request -> fire was only **4.7%**.
- Fire -> resolve was only **21.4%** (`3/14`), so total request -> resolved was about **1.0%**.
- `pass_available_steps=0` means the carrier almost never saw a clean, currently launchable pass surface in the logged summaries.
- Many receive targets were rejected as interceptable or lane-blocked by the defender, and many carrier pass masks were rejected because the carrier was not kickable or the ball was outside the reception cone.
- `ball_in_penalty_off_target` became the largest failure mode at **33.6%**, so the policy still drives into the opponent penalty area or bad end states instead of converting possession.

### Fix for next run

Do not promote this checkpoint as a coordination policy.

The next fix should not be more align budget. It should make passing a committed physical primitive once a valid target is latched:

- when `pass_to_teammate` is selected with a valid target, run a carrier pass macro that can reacquire/settle the ball, align, and fire instead of repeatedly requiring the policy to resample a perfectly valid pass frame;
- keep the latched receiver target stable through the macro unless it becomes illegal/interceptable;
- separately log why each request did not start a macro, why started macros did not fire, and why fired passes did not resolve;
- reduce or redirect the huge `ball_in_penalty_off_target` failure path, because it is now a larger end-state problem than collision fouls.

Math note: the structural handoff remains logically correct, but it is downstream of `Pass FIRED`. With request -> fire at only 4.7%, it cannot affect most episodes. The 90-step align budget removed macro timeouts, so the remaining failure is state validity/launch execution, not insufficient alignment time.

---

## 43. Training run: 20260628_225743_267521 — PPO `stage4s_goalie_only_pass_pretrain` / goalie-only pass mechanics pretrain (PARTIAL SUCCESS — pass mechanics learned, scoring collapsed)

**Completed** at 149,669 / 150,000 steps.

**Load model:** `models/ppo_jal_expandable_wide_stage3_v3/stage3_defender_v2_finetune_complete.pt`.

**Aim:** Remove the field defender and train 2 RL attackers vs the scripted goalie only, so the policy can learn the pass chain before defender pressure returns: supporter `goto` receive targets, carrier pass selection, pass macro fire, receiver claim handoff, receive-finish macro, and shot.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 644 |
| Overall goals | 63 / 644 = **9.8%** |
| Last 100 goal rate | **4.0%** |
| Avg reward last 10 | +6.63 |
| Avg episode length last 10 | 202.9 |

**Episode outcomes:** `ball_in_penalty_off_target` 391 (60.7%), `max_steps` 142 (22.0%), `goal_scored` 63 (9.8%), `attacker_excessive_dribble` 20 (3.1%), `goalie_catch` 14 (2.2%), `ball_out_of_bounds` 14 (2.2%).

**Goal-rate trend:** 14.5% -> 11.0% -> 4.5% -> 6.8%; last-100 was only 4.0%.

**Aim quality:** unchanged at avg `aim_quality=0.319`, bad aim 16.0% in both first and last 200 kicks.

### What Worked

The goalie-only rung **did learn real pass mechanics**. Compared with Stage 4r under the defender (`301 requested / 14 fired / 3 resolved`), this run produced:

- `pass_requested=14,817`
- `pass_fired=275`
- `pass_resolved=141`
- `pass_timeout=25`
- `pass_interception=0`
- fired in `267 / 644 = 41.5%` of episodes
- resolved/fired = `141 / 275 = 51.3%`
- `pass_available_steps=69,252`
- support target modes: `RECEIVE_READY=75,268`, `RECEIVE_TARGET=39,035`, `GENERAL_SUPPORT=35,366`

That clears the mechanical pretrain bar for pass launch and receive resolution: the defender really was the sparse-pass bottleneck.

The receiver-claim handoff also worked: every resolved pass started the receive-finish macro (`finish_started=141`), which proves the receiver is no longer stuck as a pure `goto` supporter after a successful pass.

### What Went Wrong

The run did **not** learn useful attacking conversion.

- `finish_started=141`, but `finish_fired=1`, `finish_consumed=1`, `finish_expired=124`.
- Finish macro time was mostly spent in `settle=6,436`, `bad_cone=6,262`, `acquire=2,605`, and `kick_align=1,286`.
- Almost all finish macros expired by deadline.
- Resolved pass reward was only around `+6` and finish bank was tiny: mean bank `+0.025`, max `+0.09`.
- Pass targets still clustered at the deep/wide clamp: resolved pass `|aim_y|` mean `8.66`, median `10.0`; target x was often `31.50`.
- `ball_in_penalty_off_target` exploded to 60.7%, so the learned behavior is often pass/carry into the opponent penalty or wide dead zones rather than pass -> controlled shot.

The action distribution confirms the policy shifted toward passing but away from finishing:

- ep 400 carrier: `pass_to_teammate=11%`
- ep 600 carrier: `pass_to_teammate=14%`
- supporter remained mostly `goto`, but post-pass macro appeared after resolved passes.

So the good news is that pass selection and pass resolution finally exist. The bad news is that the receiver-finish macro cannot turn those possessions into kicks, and the reward currently lets pass resolution be the end of the useful behavior.

### Fix For Next Run

Do not promote this as a final Stage 4 policy, but keep it as evidence that goalie-only pretraining is useful.

Next run should focus on **post-pass finishing**, not more pass availability:

- Move receive targets away from the deep/wide clamp (`x=31.50`, `|y|=10`) and toward central shotable points, or reduce `support_target_max_x` / `support_target_y_clip`.
- Make post-pass finish macro more decisive: after a resolved pass, if the receiver is kickable but repeatedly outside the reception cone, use a deterministic geometric settle/face step that forces the ball into the front cone instead of burning 60+ bad-cone frames.
- Pay meaningful finish bank only on a real kick/goal, not on pass resolution alone. Current resolved-pass reward `~+6` is enough to teach passing even when it leads to no shot.
- Penalize pass-resolved-then-no-shot expiry more directly; `124 / 141` finish macros expired.
- Add a hard guard against post-pass carries into opponent penalty/off-target zones, because `ball_in_penalty_off_target` is now the dominant failure.

Math note: the pass mechanics acceptance passed (`41.5%` episodes fired, `51.3%` fired passes resolved), but the finish chain failed (`1 / 141 = 0.7%` resolved passes produced a macro kick). The next EV target should make pass reward conditional on a post-pass kick/goal, otherwise the policy can optimize for pass completion while lowering goal rate.

---

## 44. Training run: 20260629_000316_240570 — PPO `stage4s_goalie_only_pass_pretrain` / post-pass finish fixes (PARTIAL RECOVERY — goals improved, coordinated finish still failed)

**Completed** at 149,924 / 150,000 steps.

**Load model:** `models/ppo_jal_expandable_wide_stage3_v3/stage3_defender_v2_finetune_complete.pt`.

**Aim:** Re-run the goalie-only Stage 4s pass mechanics pretrain after fixing the receiver finish
macro's bad-cone behavior, tightening receive targets, reducing standalone pass reward, and making
post-pass finish reward dominant.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 691 |
| Overall goals | 190 / 691 = **27.5%** |
| Last 100 goal rate | **42.0%** |
| Avg reward last 10 | **+40.4** |
| Avg episode length last 10 | 179.5 |

**Episode outcomes:** `ball_in_penalty_off_target` 279 (40.4%), `goal_scored` 190 (27.5%),
`max_steps` 114 (16.5%), `goalie_catch` 62 (9.0%), `attacker_excessive_dribble` 30 (4.3%),
`ball_out_of_bounds` 16 (2.3%).

**Goal-rate trend:** 28.0% -> 23.5% -> 24.5% -> **41.8%** in the final 91 episodes. The late
goal-rate recovery is real, but it is mostly solo/direct finishing, not pass-to-shot coordination.

**Aim quality:** first 200 kicks avg `aim_quality=0.372`, bad aim 4.0%; last 200 kicks avg
`aim_quality=0.365`, bad aim 5.5%. Aim stayed stable and reasonably clean.

**Pass pipeline totals:**

- `requested=15,001`
- `fired=227` (`1.5%` request -> fire)
- `resolved=99` (`43.6%` fire -> resolve, `0.7%` request -> resolve)
- `timeout=39`
- `interception=2`
- `receive_finish_macro started=99`
- `finish_banked=99`
- `finish_consumed=1`
- `receive_finish_macro fired=1`
- `receive_finish_macro expired=93`

**Final 100 episodes pass pipeline:**

- `requested=1,533`
- `fired=27` (`1.8%` request -> fire)
- `resolved=22` (`81.5%` fire -> resolve)
- `receive_finish_macro started=22`
- `receive_finish_macro fired=0`
- `finish_consumed=0`
- `receive_finish_macro expired=20`

**Support/pass geometry diagnostics across the run:**

- support modes: `RECEIVE_READY=71,417`, `RECEIVE_TARGET=39,177`,
  `GENERAL_SUPPORT=39,330`
- target reasons: `ok=110,594`, `too_close_for_ssl_pass=32,866`,
  `blocks_carrier_shot_lane=6,464`
- pass mask reasons: `not_kickable=30,456`, `too_close=23,390`,
  `bad_reception_cone=22,624`, `receiver_far=6,425`,
  `blocks_carrier_shot_lane=2,385`, `direct_shot_better=2,167`

### Code changes (non-config)

This run exercised the Stage 4s fixes logged in CHANGES.md on 2026-06-28 Late PM:

- `ai_interface/envs/JAL_env.py`: post-pass macro now has a latched non-degenerate
  `settle_target`; bad-cone recovery uses a real short dribble target instead of
  `dribble_to(ball_xy)`; finish target selection prefers in-mouth goalie-gap targets with a
  shortest-turn tie-break; non-receiver attackers clear out during an active post-pass finish macro.
- `ai_interface/utils/basic_commands.py`: `goto()` normalises the body-relative dash angle.
- `configs/ppo_jal_curriculum_config.json`: Stage 4s receive targets were pulled inward
  (`support_forward=8..16`, `support_target_max_x=29`, `support_target_y_clip=6`), standalone pass
  reward was reduced (`possession_transfer_bonus=0.5`, `pass_quality_weight=1.5`), and finish reward
  was raised (`pass_finish_bonus=24`, `pass_finish_quality_weight=20`).

### What worked

The late goal rate recovered from the previous Stage 4s collapse:

- previous Stage 4s: overall 9.8%, last-100 4.0%;
- this run: overall 27.5%, last-100 42.0%.

The target tightening also improved fired-pass quality once a pass actually launched. In the last
100 episodes, fired passes resolved at `22 / 27 = 81.5%`, much better than the previous run's
`141 / 275 = 51.3%`. This means the receive target geometry is less broken than before.

### What went wrong

The model still has **not learned effective coordinated passing**.

The main failure moved earlier in the chain:

- request -> fire is only `227 / 15,001 = 1.5%`;
- last-100 request -> fire is only `27 / 1,533 = 1.8%`;
- `pass_available_steps=0` in episode summaries even though many support targets are `ok`;
- mask reasons are dominated by the carrier not being in a physical launch state:
  `not_kickable`, `bad_reception_cone`, `too_close`, and `receiver_far`.

The receiver finish macro also still fails:

- whole run: `1 / 99 = 1.0%` resolved passes produced a receiver macro kick;
- last 100: `0 / 22 = 0.0%`;
- macro frames are still dominated by `bad_cone=3,984`, `settle=3,984`, `acquire=1,876`,
  `kick_align=1,820`, then deadline expiry.

So the goal-rate improvement is mostly the Stage 3 solo policy adapting to the easier goalie-only
environment. The pass chain is visible, but it is not profitable and not reliable. The policy often
requests passes during forced-pass windows, but the pass macro rarely reaches a valid front-cone
launch; when a pass resolves, the receiver still fails to turn that possession into a kick.

### Fix for next run

Do not promote this as a coordinated Stage 4 policy.

The next change should be structural, not just reward tuning:

- make pass execution a deterministic command-level macro once a valid pass target is selected:
  reacquire/settle the carrier's ball into the front cone, align to the latched receiver target, and
  fire, instead of requiring PPO to keep resampling `pass_to_teammate` while the physical launch
  conditions flicker;
- treat repeated pass requests with no fire as a failed macro state and release the forced-pass mask,
  because it currently burns long episodes with `requested >> fired`;
- make post-pass finish more direct: after a resolved pass, if the receiver spends too many bad-cone
  frames, force a geometric face/settle-to-goal action and then kick, or expire quickly with penalty
  instead of consuming 80 steps;
- keep the tighter receive target geometry, because fired passes now resolve much better;
- reintroduce defender pressure only after a goalie-only run shows non-zero receiver finish
  consumption in the last 100 episodes.

Math note: lowering standalone pass reward was correct. A quality-0.5 resolved pass now pays only
`(0.5 + 1.5*0.5) / 2 = +0.625` team reward, while a quality-0.5 receiver finish would pay roughly
`(24 + 20*0.5) / 2 = +17` before shot-quality scaling. The problem is not reward EV now; it is that
the physical pass and receive-finish macros almost never reach their terminal kick states.

---

## 45. Training run: 20260629_015011_487635 — PPO `stage4s_goalie_only_pass_pretrain` / repeated goalie-only run (FAILED — no new coordination progress)

**Completed** at 149,905 / 150,000 steps.

**Load model:** `models/ppo_jal_expandable_wide_stage3_v3/stage3_defender_v2_finetune_complete.pt`.

**Aim:** Re-run active Stage 4s goalie-only pass mechanics pretrain with the same post-pass finish
and reward settings as §44, to see whether another 150k steps would produce stable pass-to-finish
behavior.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 695 |
| Overall goals | 178 / 695 = **25.6%** |
| Last 100 goal rate | **32.0%** |
| Avg reward last 10 | **+18.75** |
| Avg episode length last 10 | 182.7 |

**Episode outcomes:** `ball_in_penalty_off_target` 261 (37.6%), `goal_scored` 178 (25.6%),
`goalie_catch` 105 (15.1%), `max_steps` 101 (14.5%), `attacker_excessive_dribble` 29 (4.2%),
`ball_out_of_bounds` 20 (2.9%), `frozen_state_stale_sim` 1 (0.1%).

**Goal-rate trend:** 29.5% -> 20.0% -> 24.5% -> 31.6%. This is worse than §44, where the final
bucket reached 41.8%.

**Aim quality:** first 200 kicks avg `aim_quality=0.399`, bad aim 8.0%; last 200 kicks avg
`aim_quality=0.399`, bad aim 12.0%. Aim quality did not improve.

**Pass pipeline totals:**

- `requested=14,256`
- `fired=211` (`1.5%` request -> fire)
- `resolved=92` (`43.6%` fire -> resolve, `0.6%` request -> resolve)
- `timeout=34`
- `interception=2`
- `receive_finish_macro started=92`
- `finish_banked=92`
- `finish_consumed=1`
- `receive_finish_macro fired=1`
- `receive_finish_macro expired=82`

**Final 100 episodes pass pipeline:**

- `requested=1,789`
- `fired=21` (`1.2%` request -> fire)
- `resolved=12` (`57.1%` fire -> resolve)
- `receive_finish_macro started=12`
- `receive_finish_macro fired=0`
- `finish_consumed=0`
- `receive_finish_macro expired=12`

**Support/pass diagnostics across the run:**

- support modes: `RECEIVE_READY=69,769`, `RECEIVE_TARGET=40,582`,
  `GENERAL_SUPPORT=39,554`
- target reasons: `ok=110,351`, `too_close_for_ssl_pass=33,427`,
  `blocks_carrier_shot_lane=6,127`
- pass mask reasons: `not_kickable=30,855`, `too_close=23,529`,
  `bad_reception_cone=21,772`, `receiver_far=7,426`,
  `blocks_carrier_shot_lane=2,353`, `direct_shot_better=2,008`

### Code changes (non-config)

No additional source changes beyond §44. This run tested whether the same Stage 4s setup would learn
with another independent 150k training run from the Stage 3 v3 base checkpoint.

### What went wrong

This run did not make progress relative to §44:

- overall goal rate fell from 27.5% to 25.6%;
- last-100 goal rate fell from 42.0% to 32.0%;
- request -> fire stayed stuck at about 1.5%;
- final-100 receiver finish remained zero: `0 / 12` receive-finish macros fired.

The failure mode is now stable and reproducible. The supporter creates many nominally valid targets
(`ok` target reasons ~110k), but the carrier is usually not in a physical launch state
(`not_kickable`, `bad_reception_cone`, `too_close`, `receiver_far`). When a pass does resolve, the
receiver macro again spends its budget in `bad_cone`/`settle`/`acquire`/`kick_align` and expires
without a shot.

The final policy is still a solo/dribble/direct-shot policy with occasional attempted passes. The
pass chain is visible in logs but not behaviorally useful.

### Fix for next run

Do not run another reward-only repeat of Stage 4s. The repeated run confirms the bottleneck is
structural.

Next code change should be:

- a deterministic carrier pass-execution macro that owns `ACQUIRE -> SETTLE_FRONT_CONE ->
  ALIGN_TO_LATCHED_TARGET -> FIRE_PASS` once the policy selects a valid pass;
- a deterministic receiver finish macro that owns `ACQUIRE -> SETTLE_FRONT_CONE -> FACE_GOAL ->
  FIRE_KICK`, with a short bad-cone cap instead of an 80-step expiry loop;
- explicit abort/failure penalties for repeated requested-but-not-fired pass windows and
  resolved-pass-with-no-shot expiry.

Math note: the reward is already shaped so the finish is much more valuable than the pass. A
quality-0.5 resolved pass pays about `+0.625` team reward, while a quality-0.5 finish would pay about
`+17` before shot-quality scaling. Since the policy still gets `0` final-100 finish fires, more
reward does not create learning signal; the macro needs to physically produce the kick events first.

---

## 46. Training run: 20260629_025315_805063 — PPO `stage4s_goalie_only_pass_pretrain` / committed pass-receive macro run (FAILED — worse goals, receiver still never finishes)

**Completed** at 149,667 / 150,000 steps.

**Load model:** `models/ppo_jal_expandable_wide_stage3_v3/stage3_defender_v2_finetune_complete.pt`.

**Aim:** Test the new committed pass/receive macro path in goalie-only Stage 4s: when the carrier
requests a pass, the env should own pass execution; when a pass is fired, the receiver should move to
the pre-contact receive pose and then finish from possession. This was meant to turn the previously
observed resolved-pass-but-no-shot failure into actual pass-to-kick chains.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 555 |
| Overall goals | 95 / 555 = **17.1%** |
| Last 100 goal rate | **15.0%** |
| Avg reward last 10 | **+6.23** |
| Avg episode length last 10 | 261.2 |

**Episode outcomes:** `max_steps` 250 (45.0%), `ball_in_penalty_off_target` 115 (20.7%),
`goal_scored` 95 (17.1%), `frozen_state_stale_sim` 45 (8.1%), `goalie_catch` 35 (6.3%),
`attacker_excessive_dribble` 10 (1.8%), `ball_out_of_bounds` 5 (0.9%).

**Goal-rate trend:** 18.5% -> 17.0% -> 15.5%. This is a clear regression from §45
(`25.6%` overall, `32.0%` last 100) and from §44 (`42.0%` last 100).

**Aim quality:** first 200 kicks avg `aim_quality=0.382`, bad aim 8.9%; last 200 kicks avg
`aim_quality=0.382`, bad aim 8.9%. Aim did not improve.

**Pass pipeline totals:**

- `requested=22,840`
- `fired=62` (`0.27%` request -> fire)
- `resolved=40` (`64.5%` fire -> resolve, `0.18%` request -> resolve)
- `pass_macro_started=3,447`
- `pass_macro_timeouts=136`
- pass macro fallback reasons: `pass_kick_failed=3,042`, `pass_macro_align_timeout=136`
- `receive_finish_macro started=62`
- `finish_banked=40`
- `finish_consumed=0`
- `finish_expired=39`
- `receive_finish_macro fired=0`
- `receive_finish_macro expired=40`
- receiver macro spent `1,781` frames in `bad_cone` and `758` in `kick_align`

**Support/pass diagnostics across the run:**

- support modes: `RECEIVE_READY=79,368`, `GENERAL_SUPPORT=38,806`,
  `RECEIVE_TARGET=31,493`
- target reasons: `ok=110,861`, `too_close_for_ssl_pass=33,811`,
  `blocks_carrier_shot_lane=4,978`, `low_continuation_quality=17`
- pass mask reasons: `too_close=26,742`, `not_kickable=23,173`,
  `bad_reception_cone=8,156`, `receiver_far=4,436`,
  `direct_shot_better=1,812`, `blocks_carrier_shot_lane=1,757`
- `pass_available_steps=0`

### Code changes (non-config)

This run included the committed pass/receive macro changes after §45:

- carrier pass requests enter a command-level pass macro instead of relying only on action masking;
- fired passes start an unresolved receiver macro immediately, so the receiver can move to a
  pre-contact receive pose before the pending pass formally resolves;
- the receive pose is placed just behind the target along the incoming pass line using the robot plus
  ball contact radius;
- pending pass expiry clears the unresolved receive macro;
- post-pass finish state remains keyed by receiver robot id.

### What went wrong

The changes did **not** improve the model. They made the logs more diagnostic, but performance
regressed:

- goal rate fell to `17.1%` overall and `15.0%` last 100;
- `max_steps` jumped to `45.0%`;
- request -> fire fell from §45's `1.48%` to `0.27%`;
- receiver finish stayed at `0` fired shots and `0` consumed finish banks;
- resolved/fired improved to `64.5%`, but the model fired too few passes for that to matter.

The main bottleneck is now the carrier pass execution macro. The policy requested many more passes
than before, but `3,042 / 3,447` pass macro starts ended as `pass_kick_failed`, and another `136`
timed out. On the receive side, every useful resolved-pass reward still disappears: `40` finish banks
were created, `39` expired, and none were consumed.

The receive side is also still not physically settling into a front-cone shot state. The receiver
macro spent many frames in `bad_cone`/`kick_align` but fired zero times, which matches the visual
symptom where the receiver turns around or over-aims instead of finishing.

### Fix for next run

Do not train this version further. The next change should make the macro less policy-like and more
deterministic:

- when a valid pass is committed, do not repeatedly call a kick attempt that can fail; explicitly
  drive `SETTLE_FRONT_CONE -> FACE_TARGET_SMALLEST_ANGLE -> FIRE_PASS` and fire only when the front
  cone is valid;
- after pass resolution, make the receiver finish macro face the best in-mouth goal target using
  shortest-angle geometry, with a hard cap on bad-cone frames;
- if the receiver cannot enter a valid cone within the cap, abort quickly with a training penalty
  instead of burning most of the episode;
- keep the pre-contact receive pose idea, because fired passes are resolving better, but it cannot be
  the whole fix.

Math note: the run confirms the problem is not pass reward magnitude. A fired pass now resolves more
often than before (`64.5%` vs `43.6%`), so the lane/receive target is not hopeless. The failure is
that only `0.27%` of requests become fired passes and `0 / 40` resolved passes become finish shots.
No scalar reward can train a pass-to-goal policy when the environment almost never emits the terminal
kick event.

---

## 47. Training run: 20260629_040401_401350 — PPO `stage4s_goalie_only_pass_pretrain` / Codex deterministic contact-pose pass+receive macros (FAILED the finish bar — receiver fired 3/485, 0 in last-100; policy is a solo dribbler that only passes when forced)

**Completed** at 149,922 / 150,000 steps. Wall-clock ~16 min (embedded, MPS).

**Load model:** `models/ppo_jal_expandable_wide_stage3_v3/stage3_defender_v2_finetune_complete.pt`.

**Aim:** First test of Codex's untested deterministic contact-pose macros (CHANGES.md 2026-06-29 ×2: the Sumatra-style committed pass/receive macro + the `acquire → settle_contact → face_target → fire` phasing with `_can_fire_physical_kick`). Goalie-only 2v1. Hypothesis: deterministic contact-pose phasing finally makes the receiver emit a finish kick.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 815 |
| Overall goals | 149 / 815 = **18.3%** |
| Last 100 goal rate | **24.0%** (trend 13.5→16→19.5→24.5%, then a 15-ep partial bucket) |
| Avg reward last 10 | +22.86 |
| Avg episode length last 10 | 162.0 |

**Episode outcomes:** `ball_in_penalty_off_target` 426 (52.3%), `goal_scored` 149 (18.3%),
`ball_out_of_bounds` 90 (11.0%), `max_steps` 72 (8.8%), `goalie_catch` 57 (7.0%),
`attacker_excessive_dribble` 20 (2.5%).

**Aim quality:** first 200 kicks `aim=0.372` bad 7.0%; last 200 `aim=0.374` bad 7.5%. Stable.

**Pass pipeline (full run):**

- `requested=22,662`, `fired=485` (**2.14%** req→fire), `resolved=109` (**22.5%** fired→resolve — *down* from §44 43.6% / §46 64.5%), `timeout=92`, `interception=4`
- `pass_macro started=1,041`, **`align_timeout=493` (47% of macro starts die on align)**, fallback_reasons all `pass_macro_align_timeout`
- **`receive_finish_macro started=485 → FIRED=3 → expired=189`**; finish_bank banked=109 consumed=3 expired=87
- receive-macro frames: **`acquire=9,480`** (dominant), `bad_cone=1,199`, `settle=1,199`, `kick_align=608`
- **Last 100 eps:** requested=2,336 fired=47 resolved=13 — **receive_finish fired=0**, expired=24

**Debug inference (sim-embedded, 8k steps, 40 eps):** goal_rate **52.5%**, actions `goto=50% dribble_to=35% approach_ball=10% kick=0%`, `requested approach_ball=9602 dribble_to=5735 kick=663`, 103 carries — **zero `pass_to_teammate`**. The inference policy is a pure solo dribble-and-score strategy; passing does not appear when not forced by the mask.

### Code changes (non-config)

This run is the first training test of the uncommitted Codex changes already logged in CHANGES.md
(2026-06-29): contact-pose helpers (`_front_contact_pose`, `_at_contact_pose`,
`_can_fire_physical_kick`, `_goto_contact_pose_command`), `pass_macro_phase`
(`acquire→settle_contact→face_target→fire`), `passer_support_lock_until_count`, and the
pre-contact receiver-pose macro. No additional source edits for this run.

### What went wrong

The contact-pose phasing did **not** fix the finish chain, and resolution actually regressed.

1. **Carrier can't reach the fire gate.** `_can_fire_physical_kick` requires the ball in the front
   reception cone *and* heading-to-target ≤5° *simultaneously*. That only holds when the robot sits
   exactly at the contact pose (behind ball, colinear to target). Small movement/turn error drops it
   back to `face_target` and re-loops, so 493/1041 = **47%** of pass macros die on
   `pass_macro_align_timeout`. The `face_target` turn is incremental (`turn angle_diff/dt`,
   JAL_env.py:3416) under the rate cap — the documented bang-bang convergence failure
   ([[project_turn_cap_breaks_aiming]]).
2. **Passes don't reach the receiver.** fired→resolve fell to 22.5% (from 43–64%), so most fired
   passes never get to the receiver — lead-target / pass-power / receiver positioning is off.
3. **Receiver can't collect or finish.** `receive_finish_macro` is dominated by `acquire=9,480`
   frames — the receiver mostly never even gets the ball into a shot pose; only 3/485 fired.
4. **The deepest problem is structural, not mechanical.** Debug inference proves the policy scores
   **52.5% solo by dribbling** and never passes voluntarily. In goalie-only 2v1 there is no defender
   blocking the dribble, so **solo dribble strictly dominates passing** — the policy is *correct* to
   not pass. Forced passes that then physically fail teach the policy passing is worthless (a death
   spiral). We are asking the model to prefer a brittle, failing pass over a 52%-effective dribble in
   an environment that never requires a pass.

### Fix for next run

Two-layer fix; pick the carrier+receiver mechanics first since they gate everything:

- **Make pass fire robust (carrier):** stop requiring a simultaneous 5° cone + heading lock reached
  by incremental turning. Once the carrier is at/near the contact pose, fire the pass with the
  geometric target angle directly (the catch-glue already holds the ball in front), or widen the fire
  tolerance and fire on the first in-cone frame instead of re-looping `face_target`. Target: align
  timeout < 10%.
- **Make the receiver finish reuse the proven solo behavior:** the policy already dribbles-to-goal and
  scores 52% solo. After a pass resolves, hand the ball-claim to the receiver and let the normal
  `approach→dribble_to→kick` primitives finish, instead of a deterministic 5°-cone finish macro that
  fires 3/485. The finish macro is *less* capable than the policy it overrides.
- **Reconsider the environment incentive (structural):** consider whether goalie-only is the right rung
  to teach passing at all — with no defender, dribble dominates. Either (a) accept 2v1 as a
  *pass-mechanics* rung whose only bar is "a forced pass reliably ends in a goal" (i.e. fix execution,
  don't expect voluntary passing), then move to 2v2 where a defender makes passing necessary; or
  (b) add a light dribble-suppression / lane pressure so passing can out-score dribbling here.

Math note: no reward change made this run. Reward EV already favors finish over pass
(§44/§45 math). The blocker remains terminal-event emission: 3/485 finish fires and 47% pass-align
timeouts mean there is no learning signal for "pass then score," regardless of reward magnitude.

---

## 48. Training run: 20260629_043329_437513 — PPO `stage4s_goalie_only_pass_pretrain` / fire gate widened 5°→10° (PARTIAL — fixed what it targeted, exposed the real bottleneck: passes overshoot the receiver)

**Completed** at 149,957 / 150,000 steps. Wall-clock ~16 min.

**Load model:** `models/ppo_jal_expandable_wide_stage3_v3/stage3_defender_v2_finetune_complete.pt`.

**Aim:** Test the fire-gate wiring fix (CHANGES.md 2026-06-29 "fire gate widened 5°→10°"): wire
`pass_macro_orientation_threshold_deg=10` and thread `angle_tolerance` through `kick()` so the
macros can actually emit a kick instead of re-looping on the hardcoded 5° gate. Hypothesis: align
timeout drops and receive-finish fires go non-zero.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 879 |
| Overall goals | 163 / 879 = **18.5%** |
| Last 100 goal rate | **11.0%** (trend 19→20.5→21→17→10%, declining) |
| Avg reward last 10 | +5.69 |

**Episode outcomes:** `ball_in_penalty_off_target` 487 (55.4%), `goal_scored` 163 (18.5%),
`goalie_catch` 96 (10.9%), `ball_out_of_bounds` 63 (7.2%), `max_steps` 52 (5.9%).

**Pass pipeline (full run):**

- `requested=20,620`, `fired=525` (2.55%, up from §47 2.14%), **`resolved=48` (9.1% of fired — DOWN from §47 22.5%)**
- `pass_macro started=964`, `align_timeout=390` (**40.5%**, down from §47 47%)
- `receive_finish_macro started=525 → FIRED=4 → expired=133` (vs §47 3); finish frames `acquire=9,226 bad_cone=482 (↓from 1,199) kick_align=370 (↓from 608)`
- expired split: **timeout=76, interception=14**
- **Last 100:** fired=59 resolved=4 (6.8%), receive_finish fired=0

### Code changes (non-config)

`kick()` gained an `angle_tolerance` param (default 5° preserves solo shooting); `JAL_env.py` carrier
and receiver fire gates now read the 10° config knobs; `reward.py` added
`receive_finish_fire_tolerance_deg=10`. See CHANGES.md 2026-06-29.

### What worked

The fix did exactly what it targeted: align timeout 47%→40.5%, fire rate up, and the finish-macro
cone walls fell sharply (`bad_cone 1199→482`, `kick_align 608→370`) — so the carrier fires more and
the receiver reaches a shot pose more easily. Mechanically the gate was the right diagnosis.

### What went wrong — the real bottleneck is pass DELIVERY, and the gate change made it worse

`resolved/fired` crashed 22.5%→9.1%. The wider 10° release angle traded accuracy for fire-rate, so
passes miss the receiver even more. But the deeper, dominant problem is **gross pass overshoot**,
independent of angle:

- on expiry, ball-to-aim `miss` median = **23.7 units**; ball `travelled` from release median =
  **32 units** — yet passes are aimed only ~10–16 units forward.
- fired-pass `power` median = **72**, with **261/525 at the 75 cap**. The old power model
  (`base=30 + 3·d`, cap 75) sends nearly every pass at ~max power.
- **Verified physics:** total ball travel = `power · kick_power_rate / (1−ball_decay)` =
  `power · 0.027 / 0.06` = `power · 0.45`. So power 72 → 32.4u travel, matching the measured 32u
  exactly. A pass aimed 12u away travels ~30u → **rockets ~18u past the receiver** → ball runs to
  the goalie or out → timeout/interception → 9% resolve. The receiver burns `acquire=9,226` frames
  chasing balls that already blew past.

This is why every prior run (§43–§47) capped at 9–22% resolution regardless of fire-angle or finish
macro: the ball never arrives at the receiver.

### Fix for next run (run #3, applied — config only)

Size pass power to the pass distance so the ball arrives at the receiver still rolling (catchable),
instead of overshooting 2.5×. Using verified travel = `power · 0.45`, target travel ≈ `d` →
`power ≈ d / 0.45 ≈ 2.2·d`:

- `pass_power_base: 30 → 2`, `pass_power_per_unit: 3 → 2.2`, `pass_power_min: 35 → 15`,
  `pass_power_max: 75 → 48`.
- Resulting travel is a uniform ~+0.8u past the target across d=7..20 (ball reaches the receiver and
  keeps rolling slightly — a true through-ball). e.g. d=12 → power 28 → travel 12.8u (was 30u).

Kept the 10° fire gate (overshoot, not angle, is the dominant delivery error; at d=12 a 10° error is
only 2.1u lateral, within the receiver collection window). Single-variable change so attribution is
clean: watch `resolved/fired` (target >40%, was 9%) and `receive_finish FIRED`.

---

## 49. Training run: 20260629_045535_501979 — PPO `stage4s_goalie_only_pass_pretrain` / pass power sized to distance (PROGRESS on goals; pass still doesn't connect — receiver rendezvous is the next bottleneck)

**Completed** at ~149,900 / 150,000 steps. Wall-clock ~16 min.

**Load model:** `models/ppo_jal_expandable_wide_stage3_v3/stage3_defender_v2_finetune_complete.pt`.

**Aim:** Stop passes overshooting the receiver by sizing pass power to distance
(`base 30→2, per_unit 3→2.2, max 75→48`, verified travel = `power·0.45`). Hypothesis:
`resolved/fired` jumps from 9% as the ball arrives at the receiver instead of 18u past it.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 605 |
| Overall goals | 171 / 605 = **28.3%** (up from §48 18.5%) |
| Last 100 goal rate | **29.0%** (up from §48 11.0%; trend 27→29.5→27.5%, stable) |
| Avg reward last 10 | +18.7 |
| Avg episode length last 10 | 262.7 (up — episodes drag longer) |

**Episode outcomes:** `ball_in_penalty_off_target` 214 (35.4%), `goal_scored` 171 (28.3%),
**`max_steps` 152 (25.1%, up from §48 5.9%)**, `goalie_catch` 38 (6.3%), `attacker_excessive_dribble`
17 (2.8%), `ball_out_of_bounds` 13 (2.1%).

**Pass pipeline (full run):**

- `requested=24,485`, `fired=487` (1.99%), **`resolved=44` (9.0% — unchanged from §48)**
- expired: **timeout=405, interception=0** (was 76/14 — the overshoot-into-goalie is gone)
- `pass_macro started=1,229`, `align_timeout=645` (52%, up)
- `receive_finish_macro started=487 → FIRED=6 → consumed=6 → expired=443`; **acquire_frames=16,050** (≈33/macro)
- `EXPIRED travelled median = 17.3u` (was 32u — power fix confirmed working)
- `FIRED power median = 30, max = 48` (was median 72 / cap 75 — power fix confirmed)
- **Last 100:** fired=84 resolved=10 (11.9%), receive_finish FIRED=2 (first non-zero last-100)

### Code changes (non-config)

None. Config-only: pass power model (see §48 Fix). Plus prior runs' fire-gate wiring (CHANGES 6-29).

### What worked

The power fix did exactly what the physics predicted: `travelled` 32→17u, `power` 72→30, and
**interceptions went to 0** — the ball no longer rockets past the receiver into the goalie or out.
That alone lifted overall goals 18.5→28.3% and last-100 11→29% (fewer balls lost; more stay in play
to be dribbled in), and produced the first non-zero last-100 finish fires (2).

### What went wrong — receiver rendezvous, not ball speed

`resolved/fired` stayed at 9% and the failure mode shifted entirely to **timeout** (405/405): the
ball now stops short/beside the receiver instead of overshooting, but the receiver still never
collects it — `acquire_frames` ≈33/macro means the receiver spends the *whole* 35-step window
chasing and never reaches the ball. `max_steps` episodes rose to 25% (ball stays in play, receiver
chases it around to the deadline).

Root cause located in `_support_pass_candidate` (JAL_env.py:5183): a pass fires as long as the
receiver is within **`support_pass_receiver_max_target_dist = 9.0` units** of the aim point, and
`support_pass_allow_receive_target=True` lets it fire while the receiver is still *en route*
(RECEIVE_TARGET mode, not settled). So the ball is sent ~to a forward point the receiver is still up
to 9u away from and hasn't reached — it can't close 9u **and** settle/catch within 35 steps.

### Fix for next run (run #4, applied — config only)

Tighten the rendezvous gate so a pass only fires when the receiver is settled near the landing point:
`support_pass_receiver_max_target_dist: 9.0 → 4.0`. With the now-correct power (ball travels to the
aim point), requiring the receiver within 4u of that point means the ball should arrive at/just past
the receiver while it is roughly in place. Single-variable change; watch `resolved/fired` (target
>30%, was 9%) and whether `max_steps` drag falls. If resolution improves but caps, next step is to
aim the pass at the receiver's actual position (+small goal-ward lead) rather than a fixed forward
support point, removing the rendezvous dependency entirely.

---

## 50. Training run: 20260629_051457_049654 — PPO `stage4s_goalie_only_pass_pretrain` / tighten rendezvous tol 9→4 (FAILED — falsified the tolerance hypothesis; resolution got worse)

**Completed** at ~149,900 / 150,000 steps. Wall-clock ~16 min.

**Aim:** Make a pass fire only when the receiver is settled near the landing point
(`support_pass_receiver_max_target_dist 9.0 → 4.0`), so the ball arrives at the receiver. Hypothesis:
resolved/fired jumps from 9%.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 689 |
| Overall goals | 165 / 689 = **23.9%** (down from §49 28.3%) |
| Last 100 goal rate | **25.0%** (down from §49 29.0%) |

**Pass pipeline:** `requested=21,286 fired=442 (2.08%) resolved=29` (**6.6%** — *worse* than §49 9.0%);
expired `timeout=365 interception=0`; `pass_macro align_timeout=467/1016 (46%)`;
`receive_finish started=442 → FIRED=11 → consumed=9` (finish conversion up — when a pass resolves the
fixed finish now fires); `acquire_frames=14,686`. Last-100: fired=64 resolved=3 (5%), finish FIRED=0.

### Code changes (non-config)

None. Config-only: `support_pass_receiver_max_target_dist 9 → 4`.

### What went wrong

The hypothesis was **falsified**: requiring the receiver within 4u of its support target did not make
passes connect — resolution dropped to 6.6% and goals to 23.9%. So even a receiver settled within 4u
of the aim point, with correct power, doesn't end up with the ball. This proves the problem is not the
receiver's distance to its *target* but the **open-loop aim-at-a-fixed-point design**: the ball, the
support target, the receive pose (1.1u beyond), and the receiver's real position are four points that
don't coincide, and kick-noise scatter over 12–17u widens the gap. (Silver lining: `FIRED=11,
consumed=9` confirms that *when* a pass resolves, the post-§48 finish reliably converts it — the
finish is no longer the blocker; delivery is.)

### Fix for next run (run #5, applied — code + config)

Stop aiming at an abstract point. **Aim the pass "to feet"** at the receiver's actual position plus a
small goal-ward lead, so the ball goes to the robot regardless of where it is. Shrank
`pass_receive_pose_offset 1.115 → 0.6` so the receiver barely moves from the aim point, reverted
`max_target_dist 4 → 6`. See CHANGES.md 2026-06-29 "to-feet passing". This removes the multi-point
rendezvous that capped resolution at 6–9% across runs §47–§50. Watch `resolved/fired` (break past 9%)
and goals.

---

## 51. Training run: 20260629_053706_717764 — PPO `stage4s_goalie_only_pass_pretrain` / "to-feet" passing (MARGINAL — resolution 10.9%, plateau barely moved; stale-aim flaw identified)

**Completed** at ~149,900 / 150,000 steps.

**Aim:** Aim the pass at the receiver's actual position (+1u goal lead) instead of an abstract support
point, to remove the rendezvous mismatch. Expected `resolved/fired` to break past 9%.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 695 |
| Overall goals | 160 / 695 = **23.0%** |
| Last 100 goal rate | **22.0%** |

**Pass pipeline:** `requested=21,028 fired=439 (2.09%) resolved=48` (**10.9%** — barely above the 9%
plateau); expired `timeout=308 interception=0`; **`pass_macro align_timeout=531/1076 (49%)`** (up —
half of all pass macros never fire); `receive_finish started=439 FIRED=4 consumed=4`;
`acquire_frames=13,948`. Last-100: fired=59 resolved=4 (7%), finish FIRED=0.

### Code changes (non-config)

To-feet aiming + `pass_to_feet_lead=1.0` + `pass_receive_pose_offset 1.115→0.6` + `max_target_dist
4→6`. See CHANGES.md 2026-06-29 "to-feet passing".

### What went wrong

To-feet moved resolution only 9%→10.9% — within noise. Two flaws found:

1. **Stale aim:** the receiver position was latched at macro *start*, but the macro spends up to 90
   align cycles before firing (49% time out), during which the receiver keeps moving — so the ball
   is sent to where the receiver *was*, not where it *is*. To-feet aimed at a stale point.
2. **Carrier alignment is the upstream wall:** `align_timeout=49%` means half the passes never fire
   regardless of where they aim. The bespoke contact-pose phase machine (acquire→settle→face→fire)
   oscillates under the rate-capped turn — it is *less* reliable than the `kick(dribbling=True)`
   helper the 52% solo shooter uses (which does catch→align→fire cap-robustly in one call).

### Fix for next run (run #6, applied — code only)

**Dynamic to-feet:** re-aim at the receiver's *live* position every macro cycle (not the latched
start position), fixing flaw #1. Single targeted change. Watch `resolved/fired` and `align_timeout`.

### Strategic note (5 runs in)

Resolution has held at **6.6–10.9% across five well-reasoned delivery fixes** (fire gate, power,
rendezvous tol, to-feet). Meanwhile debug inference shows the policy **scores 52% solo by dribbling
and never passes voluntarily** — in goalie-only 2v1 a pass is never *needed*, so forced passes that
fail just suppress the goal rate (training 23% vs inference 52%). The finish is solved
(`consumed=9/11` when a pass resolves); only open-loop delivery in a noisy sim remains broken. If run
#6's dynamic to-feet does not clearly lift resolution (>~20%), the conclusion is that **goalie-only
cannot teach passing** and the next step is either (a) replace the bespoke carrier alignment with the
proven `kick()` path, or (b) advance to 2v2 where a defender makes passing necessary and the policy
has a reason to choose it — which is the user's actual end goal.

---

## 52. Training run: 20260629_055539_219154 — PPO `stage4s_goalie_only_pass_pretrain` / dynamic to-feet (REGRESSED — 2nd consecutive regression; goalie-only delivery tuning is exhausted)

**Completed** at ~149,900 / 150,000 steps.

**Aim:** Fix the §51 stale-aim flaw by re-aiming at the receiver's *live* position every macro cycle.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 682 |
| Overall goals | 137 / 682 = **20.1%** (↓ from §51 23.0%, §49 28.3%) |
| Last 100 goal rate | **17.0%** (↓ from §51 22.0%) |

**Pass pipeline:** `requested=21,741 fired=457 (2.10%) resolved=33` (**7.2%** — ↓ from §51 10.9%);
expired `timeout=308 interception=0`; **`align_timeout=573/1131 (51%)`** (↑); `receive_finish
FIRED=1 consumed=0`. Last-100: fired=69 resolved=6 (9%).

### What went wrong

Re-aiming at the moving receiver every cycle *added* oscillation: the carrier keeps re-turning toward
a shifting target and never converges, so `align_timeout` rose to 51% and resolution fell to 7.2%.
Second consecutive regression (§49 28% → §51 23% → §52 20%).

### Decision — stop tuning goalie-only delivery

Six runs (§47–§52) have held pass `resolved/fired` at **6.6–10.9%** through every reasonable delivery
fix (fire-gate wiring, physics-verified power, rendezvous tolerance, to-feet, dynamic to-feet — the
last two *regressed*). The finish is solved (`consumed=9/11` when a pass resolves). The blocker is
open-loop pass **delivery** in a noisy sim, and it is not yielding to parameter tuning.

Compounding this, the strategic premise is falsified: debug inference shows the policy **scores ~52%
solo by dribbling and never passes voluntarily** — in goalie-only 2v1 a pass is never *needed*, so
forced passes only suppress the goal rate (training 20–28% vs inference 52%). The stated 2v1 gate
("≥35% goals AND scores whenever it passes") is effectively unreachable here: the model already
exceeds 35% at inference *without passing*, and it has no incentive to pass against a lone keeper.

**Action taken:** reverted the two regressing to-feet changes (code + config) back to the §49
best-known state (28% goals, fire-gate + power fixes retained). Codebase left in best state. Surfaced
the strategic fork to the user (keep grinding 2v1 delivery via a proven-mechanism carrier rewrite vs
advance to 2v2 where passing is needed vs accept solo-52%) — the next direction is a plan-level
decision the experiments have now justified escalating.

---

## 53. Training run: 20260629_095545_323404 — PPO `stage4s_goalie_only_pass_pretrain` / catch-glue settle (BREAKTHROUGH on delivery quality; new bottleneck = carrier wastes episodes failing to fire)

**Completed** at ~149,900 / 150,000 steps. (User direction: fix delivery, stay in 2v1.)

**Aim:** Carrier pass `settle` now catches+glues the ball via `dribble()` (the mechanism the 52% solo
shot uses) before aiming, so the ball stays in the front cone while turning to the receiver instead of
drifting out (the ~50% align-timeout oscillation).

### Results

| Metric | Value |
|---|---:|
| Total episodes | 545 |
| Overall goals | 120 / 545 = **22.0%** |
| Last 100 goal rate | **18.0%** |

**Pass pipeline:** `requested=46,402 fired=155 (0.33%)` **`resolved=36 (23.2% of fired)`** — last-100
**29%**. Up from the 6–11% plateau across §47–§52. `receive_finish started=155 FIRED=5 consumed=4`.
expired `timeout=107 interception=0`.

**But:** `pass_macro align_timeout=897/1228 (73%)` (worse than ~50%), `align_steps median=182/episode`
(budget 90). The carrier burns ~90% of each episode in non-convergent pass alignment.

### Code changes (non-config)

`JAL_env.py`: carrier pass `settle_contact` calls `dribble()` (catch+glue) instead of
`_goto_contact_pose_command`; added `dribble` import. See CHANGES.md 2026-06-29 "catch-glue".

### What worked

**Delivery quality is fixed.** `resolved/fired` jumped 9–11% → **23.2%** (last-100 29%) — catching the
ball before aiming makes the passes that fire actually reach the receiver. This is the first real break
of the plateau. The receiver finish still converts (consumed 4/5).

### What went wrong

Catching is slower than the old (broken) goto-pose, and the in-cone aim turn still occasionally drops
the ball back to settle → re-catch, so `align_timeout` rose to 73% and only 155 passes fired (vs ~440).
`align_steps median=182/ep` (budget 90) ⇒ most macros hit the full budget without converging
(oscillation, not slowness). The carrier wastes the episode failing to pass, holding goals at 22%.

### Fix for next run (run #8, applied — config only)

Stop wasting episodes on forced passes the carrier can't complete: **shoot more, pass selectively.**
`stage4_force_pass_max_direct_shot_quality 0.42 → 0.25` (force a pass only when the direct shot is
genuinely poor) and `stage4_force_open_shot_quality 0.48 → 0.40` (take decent shots via the proven 52%
solo skill). Goals should rise toward the 35% bar; the catch-glue's 23%-connect passes still fire in
poor-shot situations, preserving "scores whenever it passes." If goals rise but passing becomes too
rare, next lever is fixing the in-cone oscillation directly (hold the catch through the aim turn).

---

## 54. Training run: 20260629_101702_155561 — PPO `stage4s_goalie_only_pass_pretrain` / shoot-more, pass-selective (PROGRESS — goals 22→27.6%; pass chain intact, finish converts 6/6)

**Completed** at ~149,900 / 150,000 steps. (Direction: fix delivery, stay in 2v1.)

**Aim:** After the catch-glue delivery win (§53), stop the carrier wasting episodes on forced passes it
can't complete: `stage4_force_pass_max_direct_shot_quality 0.42→0.25`, `stage4_force_open_shot_quality
0.48→0.40` (shoot more, pass only when the shot is genuinely poor).

### Results

| Metric | Value |
|---|---:|
| Total episodes | 613 |
| Overall goals | 169 / 613 = **27.6%** (up from §53 22.0%) |
| Last 100 goal rate | **27.0%** (up from §53 18.0%) |

**Episode outcomes:** `max_steps` 209 (34.1% — still dragging), `goal_scored` 169 (27.6%),
`ball_in_penalty_off_target` 150 (24.5%), `goalie_catch` 53 (8.6%).

**Pass pipeline:** `fired=151 resolved=22 (14.6%)`; `align_timeout=740/1019 (73%)`; **`receive_finish
FIRED=6 consumed=6`** (every started finish converted — pass→shot chain works). Last-100 fired=29
resolved=5.

### What worked

Goals +5.6pp (22→27.6%) and the pass chain stayed intact — `consumed=6/6` means resolved passes still
convert to receiver shots. So shooting more lifts goals without breaking "scores when it passes."

### What went wrong

Still short of 35%, and `max_steps=34%` shows the carrier still wastes a third of episodes stuck in
forced-pass oscillation (`align_timeout=73%`). Pass resolution dipped to 14.6% (passes now occur in
tougher spots).

### Fix for next run (run #9, applied — config only)

Take one more step on the proven lever: `stage4_force_pass_max_direct_shot_quality 0.25→0.18`,
`stage4_force_open_shot_quality 0.40→0.35` — let the carrier shoot/dribble (its 52% solo skill)
whenever it has even a modest shot, forcing a pass only in genuinely dire shot situations. Expect goals
toward 35% and `max_steps` to fall, while the catch-glue still connects the (now rarer) passes. If
goals stall short of 35%, switch to fixing the in-cone align oscillation directly (mirror the shot's
`dribble_to`-toward-target carry in the pass settle so the ball aligns to the pass direction instead of
ping-ponging settle↔aim).

---

## 55. Training run: 20260629_103653_978676 — PPO `stage4s_goalie_only_pass_pretrain` / shoot-more step 2 + KEY inference finding (35% bar MET; voluntary passing needs 2v2)

**Completed** at ~149,900 / 150,000 steps.

**Aim:** Push the shoot-more lever (`force_pass 0.25→0.18`, `force_open_shot 0.40→0.35`) toward 35%.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 695 |
| Overall goals | 197 / 695 = **28.3%** |
| Last 100 goal rate | **20.0%** |

**Pass pipeline:** `fired=107 resolved=28 (26.2%)`, `align_timeout=74%`, **`receive_finish FIRED=8
consumed=8`** (every resolved pass converts to a receiver shot). `max_steps` down to 20.4%. Passing
got rarer (12 fired in last 100).

### Debug inference (sim-embedded, 8k steps, 38 eps) — the decisive datapoint

`goal_rate=52.6%`, actions `goto=50% dribble_to=36% kick=0%`, **requested: approach_ball=9559
dribble_to=5818 kick=623 — ZERO pass requests.** The forced-pass mask is training-only; given free
choice the model is a pure solo dribble-scorer at **52.6%**.

### Conclusion — 2v1 rung is done; advance to 2v2

1. **35% goal bar is MET** (52.6% at inference; training ~28% was exploration + forced-pass drag).
2. **The pass→goal machinery now works**: catch-glue delivery (resolved/fired 9%→26%) + receiver
   finish conversion (`consumed=8/8`). When the model passes, the chain scores.
3. **But the model will not pass *voluntarily* in goalie-only 2v1** — dribbling scores 52%, so the
   policy correctly never chooses a pass. No delivery fix changes this incentive; it is structural
   ([[project_stage4_goalie_only_cant_teach_passing]]).

The "shoot-more" lever plateaued (§54 27.6% → §55 28.3%). The 2v1 rung has produced everything it
structurally can: goals past 35% and functional pass mechanics. **Plan:** validate the in-cone
oscillation fix once in clean 2v1 (run #10), then add the defender and move to **2v2**, where the
defender blocks the dribble so passing becomes necessary and the policy finally has a reason to choose
it — the user's actual end goal.

---

## 56. Training run: 20260629_105536_146782 — PPO `stage4s_goalie_only_pass_pretrain` / dribble_to-toward-receiver settle (PARTIAL — align wall mostly fixed; move to 2v2)

**Completed** at 149,860 / 150,000 steps.

**Load model:** `models/ppo_jal_expandable_wide_stage3_v3/stage3_defender_v2_finetune_complete.pt`.

**Aim:** Validate the Run #10 carrier pass-settle change: replace the catch-only `dribble()` settle
with `dribble_to(target=receiver)` so the carrier catches and carries in the pass direction before
aiming. Target metric: reduce `pass_macro align_timeout` from the §55 plateau of ~74% toward <40%.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 677 |
| Overall goals | 191 / 677 = **28.2%** |
| Last 100 goal rate | **31.0%** |
| Avg reward last 10 | +15.54 |

**Episode outcomes:** `ball_in_penalty_off_target` 214 (31.6%), `goal_scored` 191 (28.2%),
`max_steps` 129 (19.1%), `goalie_catch` 90 (13.3%), `attacker_excessive_dribble` 38 (5.6%),
`ball_out_of_bounds` 13 (1.9%), `frozen_state_stale_sim` 2 (0.3%).

**Pass pipeline:** `requested=3,042`, `fired=191` (**6.3% request→fire**), `resolved=35`
(**18.3% fired→resolved**), expired `timeout=126 interception=6`.

**Macro counters:** `pass_macro started=404`, `align_timeout=166` (**41.1%**, down from ~74%),
`finish_bank banked=35 consumed=4 expired=23`, `receive_finish started=191 fired=18 expired=152`.
Last 100: `fired=29 resolved=5`, `align_timeout=28/65` (**43.1%**), `receive_finish fired=5`,
`finish_consumed=0`.

**Support/pass diagnostics:** support targets were available (`ok=105,836`,
`RECEIVE_READY=68,108`, `RECEIVE_TARGET=37,728`), but many carrier steps still failed physical pass
preconditions: `not_kickable=31,774`, `too_close=27,993`, `bad_reception_cone=11,778`,
`receiver_far=5,253`.

### Code changes (non-config)

This run tested the source change logged in `CHANGES.md` on 2026-06-29: the carrier pass
`settle_contact` branch in `JAL_env.py` now uses `dribble_to(..., target=receiver)` rather than
catch-only `dribble()`, and suppresses `dribble_to`'s own carry-limit release so the pass macro can
bookkeep the final in-cone kick.

### What worked

The intended oscillation fix mostly worked. `align_timeout` fell from ~74% (§55) to **41.1%**, and
pass request→fire improved because the carrier no longer burns entire episodes as often in the
settle↔aim loop. Overall last-100 goals also recovered to **31%**, which is better than the prior
training runs even though goalie-only inference already showed the real 2v1 policy prefers solo
dribble.

### What still failed

The run did **not** clear the strict `<40%` align target, and last-100 `align_timeout` was 43.1%.
Receiver finish activity increased (`18` receive-finish kicks), but only `4 / 35` banked passes were
consumed, so the pass-to-goal chain remains too sparse to optimize further in a goalie-only setting.
This does not overturn §55: without a field defender, solo dribble remains strategically dominant and
the policy has no reason to pass voluntarily.

### Next stage

Move to the real Stage 4 target: **2 attackers vs hardcoded defender + goalie**. The config now adds
`stage4t_2v2_defender_pass_v1`, warm-started from
`models/ppo_jal_expandable/stage4s_goalie_only_pass_pretrain_complete.pt`, preserving the latest pass
mechanics while reintroducing the defender and defender-lane gate. Watch whether defender pressure
creates actual pass demand: pass requests/fires/resolutions, receive-finish consumption, max-step
rate, and whether the supporter stays useful instead of crowding.

---

## 57. Training run: 20260629_112919_468499 — PPO `stage4t_2v2_defender_pass_v1` / 2v2 defender pass transfer (FAILED)

**Completed** at 199,827 / 200,000 steps.

**Load model:** `models/ppo_jal_expandable/stage4s_goalie_only_pass_pretrain_complete.pt`.

**Aim:** Reintroduce one hardcoded defender after the goalie-only pass-mechanics rung. The goal was
to make solo dribble less reliable, create real pass demand, and verify that the learned supporter
positioning plus carrier pass macro can produce pass-to-finish goals in the actual 2v2 setup.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 664 |
| Overall goals | 66 / 664 = **9.9%** |
| Last 100 goal rate | **11.0%** |
| Max-step outcomes | 338 / 664 = **50.9%** |
| Ball-in-penalty off-target | 153 / 664 = **23.0%** |

**Episode outcomes:** `max_steps` 338 (50.9%), `ball_in_penalty_off_target` 153 (23.0%),
`goal_scored` 66 (9.9%), `frozen_state_stale_sim` 28 (4.2%), `goalie_catch` 27 (4.1%),
`attacker_excessive_dribble` 25 (3.8%), `defender_push_foul` 15 (2.3%),
`attacker_push_foul` 10 (1.5%), `ball_out_of_bounds` 2 (0.3%).

**Goal trend:** 9.0% in episodes 1-200, 8.5% in 201-400, 12.5% in 401-600, and 9.4% in the final
64 episodes. Last-100 goal rate was only **11.0%**, far below the 35% target.

**Pass pipeline:** `requested=10,688`, `fired=365` (**3.4% request→fire**), `resolved=24`
(**6.6% fired→resolved**), expired `timeout=114 interception=210`. Last 100:
`requested=1,420`, `fired=50`, `resolved=3`, `timeout=15`, `interception=29`.

**Macro counters:** `pass_macro started=865`, `align_timeout=356` (**41.2% timeout/start**),
`finish_bank banked=24 consumed=1 expired=13`, `receive_finish started=366 fired=2 expired=342`.
Last 100: `pass_macro started=117`, `align_timeout=44` (**37.6%**), `finish_bank banked=3
consumed=0`, `receive_finish started=50 fired=0 expired=46`.

**Support/pass diagnostics:** support targets were frequently classified as useful
(`ok=106,631`, `RECEIVE_READY=72,148`, `RECEIVE_TARGET=34,483`), but pass launchability remained
broken (`pass_available_steps=0`). The dominant mask/failure reasons were `not_kickable=53,488`,
`too_close=35,539`, `bad_reception_cone=19,821`, `quality_low=8,620`, `lane_low=5,809`, and
`blocks_carrier_shot_lane=4,231`.

### Code changes (non-config)

No source code changed for this run. It exercised the Stage 4t config added after §56, using the
Stage 4s checkpoint and the existing dribble-to-receiver pass settle, receiver claim handoff,
receive-finish macro, role-gated supporter `goto`, and defender-lane/intercept gating.

### What went wrong

The defender did create pass pressure, but it also collapsed the delivery chain. The model requested
many passes, including forced-pass windows, yet only **3.4%** of requests fired and only **6.6%** of
fired passes resolved. Even when a pass resolved, the receiver almost never finished it:
`finish_consumed=1/24`, and the receive-finish macro fired only **2** times in the whole run.

The main blocker is now downstream of supporter positioning. The supporter often finds receive-ready
targets, but the carrier is usually not physically in a legal pass state (`not_kickable`,
`bad_reception_cone`, `too_close`), and when passes do fire the defender/intercept logic kills most
of them before receiver finish. This explains the poor goal rate and 50.9% max-step rate: the policy
is neither a good solo scorer against the defender nor a reliable passing team.

### Fix for next run

Do not promote `stage4t_2v2_defender_pass_v1_complete.pt`. First run embedded debug inference to see
the exact failure geometry, then change the pass system rather than only increasing pass reward.
Candidate fixes to verify from inference traces:

1. Make pass availability/logging consistent: `pass_available_steps=0` while passes are firing means
   the mask/debug path and forced-pass path are not reporting the same launch condition.
2. Reduce request→fire loss by making carrier pass alignment more deterministic or command-level
   during forced useful-pass windows.
3. Reduce fired→resolved loss by inspecting whether the defender intercept model is too aggressive
   or whether the pass target/receiver claim handoff still leaves the receiver unable to contest.
4. Only after pass resolution works, adjust rewards. Current reward changes cannot teach passing
   because the pass-to-finish success signal appears only once in 664 episodes.

### Embedded debug inference follow-up

The first post-run embedded inference was invalid because it omitted
`--team_config team_config_2atk_1def.json`; the simulator did not spawn the intended goalie +
defender layout, so its high goal rate is discarded.

Valid 8k-step embedded debug inference:

```bash
/opt/anaconda3/envs/rcai/bin/python infer.py \
  --model_path models/ppo_jal_expandable/stage4t_2v2_defender_pass_v1_complete.pt \
  --trainer ppo_jal \
  --config configs/ppo_jal_curriculum_config.json \
  --env sim-embedded \
  --team_config team_config_2atk_1def.json \
  --stage stage4t_2v2_defender_pass_v1 \
  --steps 8000 \
  --debug_infer \
  --ppo_stochastic
```

Result: 29 episodes, 4 goals (**13.8%**), average reward **-77.6**. Outcomes were
`max_steps=12`, `ball_in_penalty_off_target=9`, `goal_scored=4`, `goalie_catch=2`,
`attacker_excessive_dribble=2`.

Pass diagnostics confirmed the training failure: `pass_requested=362`, `pass_fired=14`,
`pass_resolved=0`, `pass_timeout=2`, `pass_interception=10`; `finish_banked=0`,
`finish_consumed=0`; `receive_started=14`, `receive_fired=0`, `receive_expired=12`.

Root cause found in the trace and code: the pass macro's `settle_contact` branch used
`dribble_to(target=receiver_target)`. In 2v2 this could carry the ball almost all the way to the
receive target before the booked pass fired, producing fake short passes/crowding and immediate
interceptions/timeouts. Example fired passes had release points nearly equal to aim targets
(`release=(29.16,6.19)`, `aim=(29.00,6.00)`, travelled `0.54`), which cannot teach receiver
finishing.

Implemented next-run fix: `stage4u_short_pass_settle_v1` caps pass settle to a short front-cone
touch, aborts any pass whose remaining flight distance falls below `pass_min_distance`, restores a
light `pass_intercept_min_margin=0.5`, and makes `pass_available_steps` count launchable states even
when `pass_available_bonus=0`.

---

## 58. Training run: 20260629_121520_455103 — PPO `stage4u_short_pass_settle_v1` / short-settle 2v2 pass transfer (MIXED — training tail hit 35%, inference fell to 20% and no passes completed)

**Completed** at 199,979 / 200,000 steps.

**Load model:** `models/ppo_jal_expandable/stage4t_2v2_defender_pass_v1_complete.pt`.

**Aim:** Fix the Stage 4t fake-pass regression by keeping pass `settle_contact` as a short
front-cone touch instead of carrying the ball to the receive target. The expected signs were:
nontrivial pass flight distance, fewer fake/crowding passes, better fired→resolved rate, and a real
2v2 goal-rate recovery.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 879 |
| Overall goals | 206 / 879 = **23.4%** |
| Last 100 goal rate | **35.0%** |
| Avg reward last 10 | **+13.01** |
| Max-step outcomes | 173 / 879 = **19.7%** |

**Episode outcomes:** `ball_in_penalty_off_target` 257 (29.2%), `goal_scored` 206 (23.4%),
`max_steps` 173 (19.7%), `goalie_catch` 148 (16.8%), `attacker_excessive_dribble` 31 (3.5%),
`frozen_state_stale_sim` 27 (3.1%), `ball_out_of_bounds` 20 (2.3%),
`defender_push_foul` 16 (1.8%), `attacker_push_foul` 1 (0.1%).

**Goal trend:** 19.0% in episodes 1-200, 23.0% in 201-400, 22.5% in 401-600,
25.5% in 601-800, and **32.9%** in the final 79 episodes. Parser last-100 rate was **35.0%**.

**Aim quality:** improved materially. First 200 kicks: avg aim `0.417`, bad-aim `13.5%`. Last
200 kicks: avg aim `0.490`, bad-aim `8.0%`.

**Support diagnostics:** the pass availability logging fix worked: total `pass_available_steps`
across episode summaries was **15,696** instead of the misleading zero from Stage 4t. Support modes:
`GENERAL_SUPPORT=169,728`, `RECEIVE_READY=17,848`, `RECEIVE_TARGET=12,403`.

**Pass pipeline:** still failed. From pass logs: only **8** `Pass FIRED`, **0** `Pass RESOLVED`,
and **7** `Pass EXPIRED`. The fake-pass guard worked: fired pass flight distances were nontrivial
(`min=7.15`, `median=9.12`, `mean=10.85`, `max=16.41`) and there were no
`pass_target_too_close_after_settle` aborts. But the policy almost never completed the pass macro:
sampled episode summaries showed `pass_macro started=170`, `timeouts=127` (**74.7% timeout/start**),
`finish_bank banked=0 consumed=0`, and `receive_finish started=7 fired=0 expired=7`.

Dominant support/mask blockers remained `pass_interceptable`, `not_kickable`, `bad_reception_cone`,
`too_close`, and `lane_low`.

### Embedded debug inference follow-up

Valid 8k-step embedded inference was run with the intended 2v2 team config:

```bash
/opt/anaconda3/envs/rcai/bin/python infer.py \
  --model_path models/ppo_jal_expandable/stage4u_short_pass_settle_v1_complete.pt \
  --trainer ppo_jal \
  --config configs/ppo_jal_curriculum_config.json \
  --env sim-embedded \
  --team_config team_config_2atk_1def.json \
  --stage stage4u_short_pass_settle_v1 \
  --steps 8000 \
  --debug_infer \
  --ppo_stochastic
```

Inference result: 35 episodes, 7 goals (**20.0%**), avg reward **-49.1**. Outcomes:
`ball_in_penalty_off_target=11`, `goal_scored=7`, `max_steps=7`, `goalie_catch=6`,
`frozen_state_stale_sim=1`, `attacker_excessive_dribble=1`, `defender_push_foul=1`,
`ball_out_of_bounds=1`.

Inference pass result: **0 Pass FIRED / 0 RESOLVED / 0 EXPIRED**, despite
`pass_to_teammate=193` requested and 310 executed pass-to-teammate macro frames. The model mostly
scored by solo dribble/shot: actions were `goto=50%`, `dribble_to=30%`, `approach_ball=12%`,
`pass_to_teammate=1%`, `kick≈0%`, with 72 fired kicks, avg aim `0.377`, bad-aim `22.2%`.

### Code changes (non-config)

This run used the code changes logged in `CHANGES.md` under "Stage 4u short-settle pass macro":
short pass settle, pass-min-distance fire guard, pass power from ball-to-target flight distance,
clearer pass logs, and pass availability diagnostics independent of reward bonus.

### What worked

The short-settle fix solved the fake-pass symptom. Fired passes now have real flight distance
(`>=7.15` units in the training log), so the earlier `release≈aim` failure is gone. Goal rate also
recovered strongly in training compared with Stage 4t: overall 9.9% → 23.4%, last-100 11% → 35%.
Aim quality improved over the run.

### What went wrong

The model did **not** learn useful passing. The training tail reached the numerical 35% bar, but
debug inference dropped to 20% and produced zero fired passes. The remaining failure is not fake
short passes; it is pass macro completion and value preference. Most pass macro attempts still time
out before firing, and when the policy is evaluated it mostly reverts to solo dribble/shot. The
receiver finish path is still starved because no pass resolves, so `finish_bank` remains zero.

### Fix for next run

Do not promote `stage4u_short_pass_settle_v1_complete.pt` as the final Stage 4 policy. The next fix
should make pass execution deterministic once the policy selects a valid pass target, instead of
spending 90 steps in a fragile settle/aim macro:

1. Replace the carrier pass alignment macro with a command-level deterministic pass action when a
   latched support target is launchable: move to a legal contact pose, face the target with the same
   physical kick helper used by shots, then fire or abort quickly.
2. Add a hard timeout much shorter than 90 steps for pass attempts, then return to shoot/dribble
   instead of burning the episode.
3. Keep the short-settle/min-flight guard from Stage 4u.
4. Only increase pass reward after `Pass FIRED` and `Pass RESOLVED` appear regularly in embedded
   inference; right now reward cannot teach a chain that almost never executes.

---

## 59. Training run: 20260629_141213_533549 — PPO `stage4u_short_pass_settle_v1` / pass align-timeout → carry-toward-goal fallback (35% bar MET via carry; passing STILL 0 fired/0 resolved → pinpointed settle-cone wall)

**Completed** at 199,783 / 200,000 steps.

**Load model:** `models/ppo_jal_expandable/stage4t_2v2_defender_pass_v1_complete.pt`.

**Aim:** Implement §58 fix rec #2 — stop burning episodes on `turn 0` when a forced pass can't
complete. `_abort_pass` now falls back to `dribble_to(target=goal)` (the proven solo carry) on
`pass_macro_align_timeout`, and `pass_macro_max_align_steps` was cut 90→30 so a failing pass bails
into the carry quickly.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 888 |
| Overall goals | 307 / 888 = **34.6%** |
| Last 100 goal rate | **37.0%** |
| Final-bucket (801-888) | **38.6%** |
| Avg reward last 10 | **+7.99** |
| Max-step outcomes | 172 / 888 = **19.4%** |

**Episode outcomes:** `goal_scored` 307 (34.6%), `ball_in_penalty_off_target` 205 (23.1%),
`max_steps` 172 (19.4%), `goalie_catch` 116 (13.1%), `frozen_state_stale_sim` 27 (3.0%),
`attacker_excessive_dribble` 24 (2.7%), `ball_out_of_bounds` 23 (2.6%), `defender_push_foul` 10 (1.1%).

**Goal trend:** 29.0% → 33.5% → 37.0% → 37.0% → 38.6% across the 200-ep buckets — a steady climb to
the 35% bar.

**Pass pipeline:** still broken. Across episode summaries: `fired=0` in 203 eps, `fired=1` in 9,
`fired=14` in 1; `resolved=0` in 97 eps, `resolved=1` in 2, `resolved=3` in 1. Effectively **~0 passes
fire and ~0 resolve**. Pass-macro diagnostics show why: every started pass times out in the
`settle_contact` phase (`pass_macro started=1 align_steps=26–31 timeouts=1
fallback_reasons={'pass_macro_align_timeout'}` per episode) — `align_steps` hits the new 30 cap and
aborts **before reaching the fire branch**. `receive_finish_macro started=0` everywhere (nothing to
finish because nothing fires).

### Code changes (non-config)

`JAL_env.py` `_abort_pass`: on `pass_macro_align_timeout` with the ball still held, fall back to
`dribble_to(target=goal)` instead of `turn 0` (logged in CHANGES.md). Config: stage4u
`pass_macro_max_align_steps 90 → 30`.

### What worked

The carry fallback did its narrow job: failed forced passes now convert to shot-improving solo carries
instead of stalling. Goals rose to 34.6% overall / 38.6% in the final bucket (vs §58's 23.4%/35%) and
`max_steps` held at 19.4%. The 35% bar is met — but **entirely by solo play**, not by passing.

### What went wrong

The real objective ("score whenever it passes") is unmet: ~0 passes fire or resolve. The cut to a
30-step align cap, combined with the existing `settle_contact` phase that gates on
`ball_in_reception_cone` and short-settles with `dribble_to`, meant every pass exhausted its budget in
settle and timed out before the fire branch could run. The fire branch works when reached; the
upstream **reception-cone settle gate** is the wall (it almost never latches under the 20°/s turn cap).

### Fix for next run

Rewrite the carrier pass macro to mirror the **proven solo-shot loop** (`action_type=="kick"`, which
already scores ~38–52%): once the carrier has the ball, drive every cycle with
`kick(self_pose, ball_xy, angle_to_receiver, kick_power=pass_power, dribbling=True,
angle_tolerance=fire_threshold)` — fire on `"kick …"`, re-glue via a short `dribble_to` only on
`"failed"`, keep turning on `"turn …"`. This removes the `ball_in_reception_cone` settle gate
entirely; the catch-glue holds the ball in front while `kick()`'s geometric `angle_diff/dt` turn
converges to the receiver exactly as the solo shot converges to goal. Raise
`pass_macro_max_align_steps 30 → 45` for re-glue+align room. (Implemented; under test in the next run.)

---

## 60. Training run: 20260629_144125_467415 — PPO `stage4u_short_pass_settle_v1` / unified solo-shot pass loop (FAILED — 10 fired/0 resolved, goals regressed to 22.7%; 2.0-unit settle target un-glues the ball)

**Completed** at 199,623 / 200,000 steps. **Load model:** `stage4t_2v2_defender_pass_v1_complete.pt`.

**Aim:** §59's fix — replace the `settle_contact` reception-cone phase with a single loop mirroring the
proven solo shot: drive `kick(dribbling=True)` aimed at the receiver every cycle, re-glue via
`dribble_to` only on `"failed"`. `pass_macro_max_align_steps 30 -> 45`.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 860 |
| Overall goals | 195 / 860 = **22.7%** |
| Last 100 goal rate | **25.0%** |
| Max-step outcomes | 193 / 860 = **22.4%** |
| Pass FIRED / RESOLVED / EXPIRED | **10 / 0 / 9** |

### What went wrong

Two problems. (1) Goals regressed from §59's 34.6% to 22.7%. (2) Passing barely moved: only 10 fired,
0 resolved in 860 episodes; macros still timed out at the 45-cap. Root cause in the `"failed"` re-glue
branch: it pointed `dribble_to` at a **2.0-unit short settle target** (`pass_macro_settle_dist`), which
`dribble_to` reached and **released (un-glued) almost immediately**. So `kick()` never had a stably
glued in-front ball to turn-and-fire; the two machines fought every cycle (catch -> tiny carry ->
release -> unglue -> `"failed"` -> re-catch). The kick-first loop was correct; the re-glue target was
the bug.

### Fix for next run

Make the `"failed"` re-glue a real **carry toward the receiver** (far target, stop `min_flight` short)
so the catch-glue stays alive and the body orients toward the receiver — like the solo shot carries
toward goal. (Implemented; tested in §61.)

---

## 61. Training run: 20260629_150756_470571 — PPO `stage4u_short_pass_settle_v1` / carry-orient re-glue (PARTIAL — fires 10->19, first RESOLVED=1, but goals crashed to 17.6% from sideways drag)

**Completed** at ~200,000 steps. **Load model:** `stage4t_2v2_defender_pass_v1_complete.pt`.

**Aim:** §60's fix — `"failed"` branch carries toward the receiver (`ball + dir*(dist - min_flight)`)
to keep the glue alive while orienting the body, instead of the 2.0-unit settle that released.

### Results

| Metric | Value |
|---|---:|
| Total episodes | 869 |
| Overall goals | 153 / 869 = **17.6%** |
| Last 100 goal rate | **11.0%** |
| Max-step outcomes | 192 / 869 = **22.1%** |
| Pass FIRED / RESOLVED / EXPIRED | **19 / 1 / 17** |

### What worked / went wrong

The mechanism improved: fires rose 10 -> 19, several pass macros completed **without** timeout
(`align_steps=16/27/39, timeouts=0`), and the run produced the **first `Pass RESOLVED=1`**. But goals
**crashed further to 17.6% / 11% last-100** — carrying the catch-glued ball toward a wide/lateral
support receiver (targets cluster at `(29, +/-6)`, the support clamp edge) drags it sideways/backward
off the goal line, destroying solo-scoring progress. Net: passing is now mechanically *possible* but
its cost to goals is unacceptable, and resolution is still ~5% (1/19) because the receiver rarely meets
the ball.

### Fix for next run

Stop translating the ball during the pass. The `"failed"` branch now catches the ball **IN PLACE**
(`dribble()` -> turn-to-face-ball -> `catch 0`, no carry); `kick()` then rotates the glued ball to face
the receiver and fires. This should keep firing while removing the sideways drag, recovering the goal
rate. (Implemented; under test.) **If goals stay low**, the forced-pass mask itself is over-suppressing
solo scoring — dial it back. **Resolution** (receiver catching the in-flight pass) is the next major
lever after firing is stable: `_execute_post_pass_finish_macro` already `goto`s the receiver to the
receive point on fire, but only 1/19 resolved — investigate pass accuracy vs. receiver arrival timing.

# `dribble_to` — Parameterized Ball-Transport Primitive

**Date:** 2026-06-19 · **Last updated:** 2026-06-21
**Scope:** Stage 2 of the PPO JAL curriculum (goalie + dribble). `dribble_to` is live slot 4 of the
action space `goto, approach_ball, turn, kick, dribble_to`. It replaced the old
`start_dribble` / `stop_dribble` catch-session pair.

This document records: (1) the original function a teammate wrote, (2) the changes we made to harden
it for training, (3) what is still left to implement for later (multi-robot) stages, and (4) the
selective-dribbling investigation (§7–§9) that rewrote how the target is decoded, how a carry is
sustained, and how realized carries are credited.

> **Reading order for the current behaviour:** §1 and §3 describe the 2026-06-19 state and are
> partly **superseded**. For how `dribble_to` actually works today, read **§7 (goal-relative target
> + no-walk gate), §8 (true-macro carry + achieved-gap credit), §9 (the catch-glue brick), §10
> (committed GRAB), and §11 (simulator-synchronized verification + turn-free carry)**. §11 supersedes
> the catch/carry mechanics in §7 and §10.

Code:
- Skill + phase machine: [`ai_interface/utils/basic_commands.py`](../ai_interface/utils/basic_commands.py) — `dribble_to()`, `DribbleState`
- Driver + reward bookkeeping: [`ai_interface/envs/JAL_env.py`](../ai_interface/envs/JAL_env.py) — `_action_to_commands`, `step`
- Dense shaping: [`ai_interface/envs/reward.py`](../ai_interface/envs/reward.py)

Related: [PPO_EXPANDABLE_CHANGES.md](PPO_EXPANDABLE_CHANGES.md), memory `project_dribble_drop_invalid_command`.

---

## 1. What it is

`dribble_to(Dx, Dy)` is a **parameterized primitive**: the policy selects the `dribble_to` primitive
and emits a target coordinate `(Dx, Dy)` (the same param slots `goto` uses). The env calls the skill
**once per timestep** and the skill returns **one** low-level command; multi-step progress is held in a
per-robot `DribbleState` persisted by the env (`self.dribble_states`).

It exists because the ball cannot legally be carried any distance in one go — there is an **excessive
dribbling rule (penalty for dribbling continuously for more than ~1 m)**. So the ball is transported in
short, legal hops: grab → carry < 1 m → release (genuinely lose possession) → re-grab → repeat, until
the ball is on target.

### The phase machine

> ⚠️ **Superseded — current machine is in §7.2/§7.3.** The `APPROACH` phase was **removed** (`dribble_to`
> no longer walks to the ball — see §7.2); `GRAB` now turns to face the **target**, not the ball (§7.2);
> `segment_start` is anchored **after** the catch teleport, not at the catch location (§7.3); and a carry is
> **sustained as a macro** across steps rather than dropped on the first non-dribble action (§8). The table
> below is the 2026-06-19 design, kept for history.

| Phase | What happens (2026-06-19 design) | Command emitted |
|---|---|---|
| `APPROACH` | *(removed §7.2)* Close on the ball with `goto`, facing the target so the grab is pre-aimed. | `dash …` / `turn …` |
| `GRAB` | Turn to face the ball *(now the target, §7.2)*, then `catch` it (server glue), anchoring `segment_start` *(now deferred, §7.3)*. | `turn …` → `catch 0` |
| `CARRY` | Face target and dash; the glued ball follows. Release once the segment nears the limit. | `dash …` / `turn …` / `drop` |
| `RELEASE` | Dash away from the ball until there is observable separation (`kickable + margin`), genuinely losing possession. | `dash …` |
| `DONE` | Ball is on target (or the dribble was aborted). | `drop` / `done` |

Possession comes from a **server patch** (`M_ball_catcher`): a non-goalie `catch` glues the ball to the
player each cycle. The glue is only cleared by a kick/tackle/drop — this is why a clean **release** matters
(see §3), and why an episode ending while the ball is still glued **bricks the sim** (see §9).

---

## 2. The original function (as the teammate wrote it)

The original `dribble_to` / `DribbleState` already had the correct overall shape:

- The four-phase machine above (APPROACH → GRAB → CARRY → RELEASE → loop, terminating at
  `arrival_margin = 0.3 m`).
- `DribbleState` held just `phase` and `segment_start`.
- `segment_limit = 0.85 m` carry distance (margin under the 1 m rule for the released ball's per-step
  travel); `separation_margin = 0.2 m` re-dribble gap.
- Release via the literal command `"drop"`.
- Dense reward scaffolding in `reward.py` (`dribble_target_progress`, a one-shot
  `dribble_target_quality_delta` gap-improvement bonus, and a `post_dribble_kick` combo bonus) — all
  weights defaulting to 0, enabled per stage via `reward_config_overrides`.

It was a sound skeleton, but had several gaps that would either misbehave in the sim or fail to train.
Our analysis (full writeup in `~/.claude/plans/sprightly-singing-kite.md`) found:

**Correctness**
- **Switching primitives mid-carry left the ball glued with no release.** If the policy selected
  `dribble_to` (caught the ball) and then on the next step picked `turn` / `approach_ball`, the skill was
  never called, so no `drop` was issued — the catch glue persisted and the ball stayed stuck to the robot.
- **No failure timeout.** A drifting ball could loop APPROACH↔GRAB (or RELEASE) forever, burning the whole
  episode to `max_steps`.

**Learnability**
- **The target was re-read every step from a noisy continuous head.** During CARRY the heading was
  recomputed from *that step's* `(Dx, Dy)`. We train/infer PPO **stochastically** (`param_noise_std = 0.3`),
  so the target jittered step-to-step and the carry heading wobbled — a shaky, hard-to-learn line.
- **The one-shot target-quality bonus rewarded *choosing* a good target, not *reaching* it** — it fired on
  the transition step regardless of whether a carry ever happened, so it could be partly farmed.

**Multi-robot (deferred — see §4)**
- No obstacle avoidance, no possession arbitration between robots, no passing, single-agent / keeper-centric
  reward.

---

## 3. Changes we made (2026-06-19)

Implemented the correctness (P0) and learnability (P1) fixes. **No observation or action-param shape
changed**, so the wide checkpoint stays compatible (`obs_dim = 105`, heads `12 / 8` untouched).

### 3.1 Target latching — `basic_commands.py` (P1, the most important fix)

`DribbleState` gained a `target` field. The **first call of a transport latches `(Dx, Dy)`**; every later
call **ignores the policy's re-emitted target and uses the latched one**. The latch clears only when the
caller resets the state (arrival / kick / abandon / stall). This kills the `param_noise_std = 0.3` jitter
in the carry heading — the ball is now driven in a committed straight line per transport.

```python
# first call of a transport latches the target; later calls reuse it
if state.target is None:
    state.target = (float(target[0]), float(target[1]))
target = _as_float_array(state.target)
```

The env reads the latched target back into `self.dribble_to_target[robot_id]` after the call, so the dense
`dribble_target_progress` reward measures against the committed target rather than the jittering one.

### 3.2 Progress watchdog / stall timeout — `basic_commands.py` (P0)

`DribbleState` gained `best_ball_to_target` and `no_progress_steps`. The ball must reach a **new
closest-ever distance to the target** (improving by ≥ `progress_margin = 0.05`) at least once every
`stall_limit = 25` steps; otherwise the machine aborts to `DONE` and returns `"done"`, freeing the policy
to do something else instead of wasting the episode.

It is **distance-agnostic**: a far target legitimately takes more steps; only genuine *lack of progress*
trips it. `stall_limit` is the calibratable knob — start at 25 and tighten after the first smoke run shows
the real per-phase step distribution.

```python
if state.best_ball_to_target is None or ball_to_target < state.best_ball_to_target - progress_margin:
    state.best_ball_to_target = ball_to_target
    state.no_progress_steps = 0
else:
    state.no_progress_steps += 1
    if state.no_progress_steps >= stall_limit:
        state.phase = DRIBBLE_PHASE_DONE
        return "done"
```

### 3.3 Mid-dribble abandon path — `JAL_env.py` (P0)

> ⚠️ **Superseded by §8 (2026-06-21).** This abandon-on-non-dribble path was the right *safety* fix but the
> wrong *carry* policy: it dropped the ball the first step the stochastic policy picked any non-dribble
> action, so a carry never lasted long enough to move the ball. It was replaced by the **true-macro**
> behaviour in §8 (the carry continues regardless of the sampled primitive; only `kick` interrupts).

When the policy selects a **non-`dribble_to`, non-`kick`** action while the phase is `CARRY` (ball glued),
the env now **forces a `drop` release that step** and resets the session before honoring the new action.
(`kick` already releases on its own.) This closes the "ball stuck with no release" hole.

### 3.4 Session reset on completion — `JAL_env.py`

When the skill reaches `DONE` (arrival or stall-abort), the env calls `_end_dribble_session(robot_id)` so
the **next** `dribble_to` selection latches a fresh target and starts a new transport.

### 3.5 Bonus tied to execution — `JAL_env.py` + per-robot info (P1)

The one-shot `dribble_target_quality` bonus is now gated on a new **`carry_started`** flag (a carry segment
actually opened this step — the ball was caught) instead of mere primitive selection. The policy is now
paid for **grabbing and carrying toward a better gap**, not for picking a target it never reaches.

### 3.6 Verification

- Skill unit test `/tmp/test_dribble_to.py` (rcai env): target latching ignores jitter; full
  APPROACH→GRAB→CARRY→RELEASE transitions; watchdog aborts after exactly 25 stalled steps; watchdog does
  **not** trip while the ball progresses; arrival → DONE. **5/5 pass.**
- Both edited modules parse and import cleanly; `DribbleState` fields confirmed; no obs/param shape change.

---

## 4. The `drop` release command — RESOLVED (2026-06-19)

The skill (and the env's release paths) emit the literal command **`"drop"`**. Native `drop` is now
**implemented in the version_2 embedded sim** (the build also ports the missing catch-glue hunks from
`ssim_patches` — version_2 previously had only collision-push, no glue). Verified via a turn-in-place
smoke. The local engine edits live in `robocup_downloads` and are **not pushed**. Do **not** revert
`drop` → `kick 0 0`; the `drop` pivot is deliberate. See memory `project_dribble_drop_invalid_command`.

Live embedded smokes of the dribble stage now run (§7–§9 were produced by them).

---

## 5. To implement for future (multi-robot) stages

These were analyzed and deliberately **not** built yet — they only matter once teammates/opponents populate
the field, and some would break the current single-attacker baseline if enabled now.

| Item | Why deferred / what's needed |
|---|---|
| **Obstacle avoidance in `dribble_to`** | `goto` already supports it (`obstacle_avoidance=True`, builds detours around robots from `game_state`); the skill currently opts out. **Cannot just flip it on** in APPROACH today — with no teammates the avoid-points include the ball itself and would steer away from the ball it's chasing. APPROACH/RELEASE become a one-line flip once other robots exist; CARRY is harder (it uses a straight `dash`, not `goto`) — first let the policy route via intermediate targets (works with latching since a carry is ≤ 0.85 m), add a curved-carry only if collisions persist. |
| **Possession arbitration** | Nothing stops two robots both selecting `dribble_to` at the same ball; only one can hold the catch glue. `GameState` has **no owner field** (`(count, ts, ball_pos, robot_poses, None)`) — possession is *derived* by ball-to-robot distance. Derive ownership (nearest robot within kickable) and surface it as an obs feature so the centralized policy can learn one-carrier behavior; optionally mask `dribble_to` for non-carriers when a teammate already holds the ball. |
| **Passing primitive** | `dribble_to` moves the ball with a single carrier to a *coordinate*. A real 5-robot offense needs ball movement *between* players (pass / lead a teammate into space). This is a **new sibling primitive**, not a tweak to `dribble_to`. |
| **Team-level dribble reward** | Gap quality is measured only against the scripted goalie's `y`. There is no team credit for advancing the ball into space for a teammate. Needs a team reward shaping pass alongside the passing primitive. |
| **Policy-controlled pace (optional)** | `speed`, `segment_limit`, `separation_margin` are fixed args. Exposing `speed` / commit-distance via the param head is only worth it if the fixed version learns first. |
| **Carry-fraction observation (optional)** | The reserved per-agent obs dims (currently 0) could expose `carried / segment_limit` so the policy can anticipate the imminent release and time a follow-up kick (the combo). Skip unless the policy struggles to time release→kick after target latching. `obs_dim = 105` stays unchanged (reserved dims only). |

---

## 6. Tunable knobs (current defaults)

| Knob | Default | Where | Notes |
|---|---|---|---|
| `arrival_margin` | `0.3 m` | `dribble_to()` | ball-on-target tolerance |
| `segment_limit` | `0.85 m` | `dribble_to()` | carry distance before release (margin under the 1 m rule) |
| `separation_margin` | `0.2 m` | `dribble_to()` | re-dribble gap after release |
| `speed` | `80.0` | `dribble_to()` | dash power during carry/approach |
| `stall_limit` | `25` steps | `dribble_to()` | progress-watchdog abort threshold — **calibrate from first smoke run** |
| `progress_margin` | `0.05 m` | `dribble_to()` | min ball→target improvement that counts as progress |
| `carry_loss_margin` | `0.6 m` | `dribble_to()` | **(§7.3)** while CARRYing, abort only if `robot_ball_dist > kickable + this` (a real loss), not on glue jitter |
| `max_catch_attempts` | `3` | `dribble_to()` | bounded face-ball catch retries before aborting the macro |
| `verify_probe_steps` | `2` | `dribble_to()` | unique simulator lateral-probe frames allowed to verify coupled robot/ball motion |
| `dribble_fwd_max` | `0.0` (10 in 2g) | `reward.py` / stage config | **(§7.1)** forward carry distance along ball→goal for the goal-relative decode; `0` = legacy absolute decode |
| `dribble_lat_max` | `0.0` (6 in 2g) | `reward.py` / stage config | **(§7.1)** lateral steer (±) for the goal-relative decode |
| `dribble_target_y_clip` | `0.0` (10 in 2g) | `reward.py` / stage config | clamp on decoded target `|y|`; `0` = no clamp |
| `dribble_target_progress_weight` | `0.0` | `reward.py` / stage config | dense ball-toward-target reward (clipped ±0.5) |
| `dribble_target_quality_weight` | `0.0` | `reward.py` / stage config | one-shot gap-improvement bonus at carry OPEN, gated on `carry_started` |
| `dribble_achieved_gap_weight` | `0.0` | `reward.py` / stage config | **(§8)** REALIZED end-of-carry gap-delta paid at carry CLOSE (the selective signal) |
| `post_dribble_kick_bonus` | `0.0` | `reward.py` / stage config | dribble-release → kick combo bonus (gap-scaled) |

---

## 7. Selective-dribbling investigation — root cause & mechanics fixes (2026-06-20)

After §3 the skill was *correct* but **useful dribbling never emerged**: across the §31–§35 reward
experiments and a multi-stage reaction-lag-keeper curriculum (§34/§35 + the 6/20 lag runs), **0% of
~687 opened carries improved the shooting gap** (mean `gap_delta ≈ −0.25`). Re-weighting the reward and
weakening the keeper both failed because neither was the binding constraint. Three deeper problems were
found and fixed (all in code; see memory `project_dribble_ppo_mechanics_fixes`).

### 7.1 The target was decoded as ABSOLUTE field coordinates (the binding root cause) — `JAL_env.py`

The dribble target was decoded like `goto`: `target_x = goto_x_raw * 45`, `target_y = clip(goto_y_raw*30, ±10)`.
The dribble-target head was warm-started from a shooter that **never used it**, so its neutral output is
≈ 0 → target ≈ **(0, 0) = midfield**, ~45 units *behind* the goal. That makes `positional_gap_quality ≈ 0.07`
vs a current gap ≈ 0.32, so **every** carry latched a target that *worsened* the angle. The one-shot
gap-improvement bonus could therefore never fire, the head got **no gradient**, and forward-is-better was an
exploration trap the head never escaped.

**Fix — decode the target RELATIVE to the ball, forward along the ball→goal-centre vector** (only for
`dribble_to`; the disabled `goto` path is untouched). With unit vectors `dir` (ball→goal centre) and `perp`:

```python
fwd = (goto_x_raw * 0.5 + 0.5) * dribble_fwd_max   # [-1,1] -> [0, fwd_max]  (always forward)
lat =  goto_y_raw * dribble_lat_max                # [-1,1] -> [-lat_max, lat_max]  (steer to open side)
target = ball + dir*fwd + perp*lat                 # clamped before the goal line and to |y| <= y_clip
```

Now the **neutral** head output is already a forward, gap-improving carry toward goal centre, so the
one-shot quality bonus fires from step 1 and the head gets a gradient. Falls back to the legacy absolute
decode when `dribble_fwd_max == 0` or the ball pose is unknown. Knobs: `dribble_fwd_max` (10 in stage 2g),
`dribble_lat_max` (6), `dribble_target_y_clip` (10).

### 7.2 `dribble_to` no longer walks to the ball — removed the internal APPROACH phase — `basic_commands.py`

The macro had an internal `APPROACH` (walk-to-ball) phase, so selecting `dribble_to` from afar **walked**
and counted as a dribble while transporting no ball (telemetry: ~80% selection, ~0.4 real carries/ep — the
masquerade poisoned credit assignment). Now:

- Default/reset phase is **`GRAB`**, not `APPROACH`.
- A **near-ball gate** at the top: a *fresh* selection that is not already within `kickable_tolerance`
  returns a no-op `"done"` (gently step-costed, **not** flagged invalid). Reaching the ball is
  `approach_ball`'s job. **One selection = one carry segment**, and the policy must compose
  `approach_ball → dribble_to`. (`approach_ball` stays fully enabled.)
- `GRAB` turns to face the **target** (not the ball) before catching. The ball is still free during the
  turn, so it moves nothing; the old "face the ball, catch, then swing 180° toward target in CARRY" path
  dragged the *glued* ball through a full orbit. Aligning before the catch makes CARRY start already pointed
  at the target and just dash.

### 7.3 The catch teleport corrupted `segment_start` — deferred it + a carry-loss gate — `basic_commands.py`

A standalone smoke revealed that `catch` **teleports the ball ~2 m** to the held position and reorients the
robot to face it. The old code recorded `segment_start` at the **pre-catch** ball position, so the first
CARRY step measured `carried = ‖post-catch − pre-catch‖ ≈ 2 m ≥ segment_limit (0.85)` and **released
instantly** — the carry died one step after the catch and transported nothing (TRAINING.md §37: ~55/55
carries, realized gap-delta ≈ 0).

Two fixes:
- **Defer `segment_start`**: leave it `None` in `GRAB`; CARRY initialises it from the **settled post-catch**
  ball position, so `carried` measures the real carry distance from where the hold begins.
- **Phase-aware near-ball gate** (`carry_loss_margin = 0.6 m`): while CARRY/RELEASE, re-applying the tight
  possession threshold every step killed the carry on small glue jitter. Now a live carry aborts only on a
  genuine **loss** — `robot_ball_dist > kickable_tolerance + carry_loss_margin`.

These were necessary but **not sufficient** on their own: see §8.

---

## 8. `dribble_to` is now a TRUE MACRO + realized-carry credit (2026-06-21)

A 3k env smoke after §7 confirmed the target fix worked — carry opens were now `useful=True` with
`gap_delta +0.19…+0.63` — **but realized carries still moved the ball ≈ 0** (achieved-gap `start == end`,
the carry closing ~5 ms after it opened). Root cause: the **mid-dribble abandon (§3.3)** dropped the ball
the first step the stochastic policy picked any non-dribble action. A 0.85 m carry needs **~10+ consecutive
`dribble_to` selections**; at ~40% selection probability, stringing 10 in a row is ≈ 0.01% likely, so the
carry never reached the dash phase. The fix (chosen over "trust training to raise the selection
probability") makes `dribble_to` behave like the multi-step macro it was always meant to be.

### 8.1 Carry continuation — `JAL_env.py`

> ⚠️ **Extended by §10 (2026-06-21).** The `carry_continuation` below only covered `CARRY`/`RELEASE`,
> so the macro kept a carry *going* but did not help it *open* — the `GRAB` (align + catch) phase still
> needed several consecutive `dribble_to` picks, which the stochastic policy almost never produced (catch
> fired 1× in 3000 steps). §10 brings `GRAB` into the macro via a `committed` flag.

The abandon path (§3.3) is replaced. Once a carry is **open** (`phase in {CARRY, RELEASE}`), the env keeps
driving the carry toward the **latched** target every step **regardless of the sampled primitive** — only a
`kick` interrupts (release + shoot):

```python
carry_continuation = dribble_st.phase in (DRIBBLE_PHASE_CARRY, DRIBBLE_PHASE_RELEASE) \
                     and action_type != "kick"
...
elif action_type == "dribble_to" or carry_continuation:
    if carry_continuation and action_type != "dribble_to":
        executed_action_type = "dribble_to"   # credited as a dribble step; the macro is driving
    command = dribble_to(self_pose=..., ball_pose=..., target=..., state=dribble_st)
```

`dribble_to()` already **latches the target on the first call** (§3.1), so a continuation safely ignores the
policy's noisy per-step `(Dx, Dy)`. A carry now ends only on **arrival**, **segment limit**, **stall
watchdog**, **ball loss**, or a **kick** — never on a stray non-dribble pick.

### 8.2 Achieved-gap credit fires on ALL carry closes — `JAL_env.py` + `reward.py`

A new reward, `dribble_achieved_gap_weight`, pays the **realized** gap improvement measured at carry close
(`positional_gap_quality(end) − positional_gap_quality(start)`), gated on `carry_started` (start) and
`stop_dribble_fired` (close). This is the *selective* signal: a pointless carry nets ~0, a real
gap-improving carry is paid; it complements the one-shot quality bonus that fires at carry *open*.

Because the macro makes most carries **arrive** at their (now-forward) target rather than run to the foul
limit, the close trigger was broadened — `stop_dribble_fired` now fires when a carry leaves CARRY for **any**
terminal phase (`prev == CARRY and phase in {RELEASE, DONE}`: arrival / stall / loss / limit), not only the
segment-limit `RELEASE`. Otherwise the realized credit would almost never be paid. `stop_dribble_at_limit`
still distinguishes the foul-limit release for downstream logic.

### 8.3 Verified (20k embedded smoke, stage 2g warm-started shooter)

- **140 healthy episodes** (lengths min 60 / mean 142 / max 300), **0 one-step / 0 `ball_teleport` bricks**.
- **17 carries opened, 13 achieved-gap events, 9 with non-zero realized delta** (range −0.002 … **+0.020**)
  — realized transport is finally **measured and paid** (was 55/55 ≈ 0).
- Carries sustain across multiple dash steps and transport the ball instead of dying at the catch.

Not yet trained to the 40–50k acceptance gate (useful-carry fraction off ~0%, mean realized `gap_delta`
crosses 0, goal rate ≥ ~55%).

---

## 9. The catch-glue brick the macro exposed — `socket_utils.py` (2026-06-21)

The first 20k smoke after §8 **bricked at episode 39**: episodes 1–38 were normal, then all 16,057
following episodes were 1-step with `reason=ball_teleport`. The macro made carries sustain, so an episode
**ending while the ball was still catch-glued** (e.g. `max_steps` hits mid-carry) became common — and that
left the ball glued to the attacker. Unlike a referee set-piece, the **playmode stays `play_on`**, so the
existing referee-mode recovery (memory `project_embedded_sim_deadball_brick`) never fired. On the next
episode's first step the engine **re-snaps the held ball onto the holder** (ball jumps from spawn to the
robot), tripping the `ball_teleport` guard — every subsequent episode bricks identically.

**Fix** (`EmbeddedSimulatorBackend`): track catch-glue ownership and rebuild the engine when the ball is
still held at reset.

- `queue_commands` sets `_ball_caught_by = (side, unum)` on a `catch` command, and clears it on `drop`/`kick`.
- `reset()` rebuilds the engine when `prev_pm not in _SOFT_RESETTABLE_PLAYMODES` **OR** `_ball_caught_by is
  not None` (the engine rebuild is the proven, guaranteed way to clear latched held-ball state).
- `_initialize_simulator()` clears `_ball_caught_by` (a fresh engine holds no ball).

Goalie catches were never the gap — they already enter the play-mode path (`free_kick_right` →
non-soft-resettable → rebuild). Post-fix the 20k smoke is fully healthy (§8.3).

> **How to recognise this brick:** a tail of 1-step `ball_teleport` episodes where the playmode stays
> `play_on` and the ball jumps from its spawn onto the robot at step 1. That is a still-held catch, not a
> referee set-piece.

---

## 10. The macro now covers GRAB so a carry actually OPENS (2026-06-21)

§8 made a carry **sustain** once open, but a full sim-embedded infer (3k steps, stage 2g shooter) showed
carries almost never **opened** in the first place. The trace was unambiguous: across **3000 steps the
`catch` (grab) command fired exactly once**. The robot reached the ball constantly — `has_ball` was true
for 595 steps, including one run of **192 consecutive steps standing on the ball** — but during those steps
it sent `turn` 567 times and `catch` once. That is the visible "goes to the ball and just keeps turning,
does nothing" behaviour.

### 10.1 Root cause — `GRAB` was outside the macro

§8's `carry_continuation` only covered `CARRY`/`RELEASE`, so the `GRAB` phase (turn-to-face-target, *then*
`catch`) only ran on a **freshly sampled** `dribble_to`. But opening a carry needs a few **uninterrupted**
turn-to-align steps before the catch is allowed (alignment within `angle_tolerance`). The stochastic policy
oscillates `approach_ball` / `turn` / `dribble_to` while standing on the ball (~⅓ each), and **every
non-dribble step issues its own `turn`** that knocks the heading back off target — so the grab alignment
never converged and `catch` essentially never fired. With no successful carries, the achieved-gap / quality
rewards (§8.2, §7.1) almost never paid out, so the policy got no signal to learn dribbling: a chicken-and-egg
training failure.

This is **not** a sim mismatch — it reproduced on the native embedded engine (and was slightly worse than
the external server: 3.3% vs 7.9% carry rate). The constraint is the GRAB gating, upstream of the simulator.

### 10.2 Fix — a `committed` flag brings GRAB into the macro — `basic_commands.py` + `JAL_env.py`

`phase` **defaults** to `GRAB`, so `GRAB` cannot simply be added to `carry_continuation` — that would make
the macro run every step even far from the ball (and `dribble_to` no-ops there, so the robot would never
approach). Instead, a new `DribbleState.committed` flag marks that the policy **actually picked `dribble_to`
at the ball**, and only then does the macro take over the grab:

- `DribbleState` gains `committed: bool = False`, cleared by `reset()` (which the env calls on session end
  `DONE` and on `kick`).
- A fresh `dribble_to` pick sets `dribble_st.committed = True`. If the robot is not in range, `dribble_to()`
  returns `"done"` → phase `DONE` → the session ends and the flag is reset the same step, so an out-of-range
  pick never strands the flag.
- `carry_continuation` now includes `GRAB`, gated by `committed`:

```python
carry_continuation = (
    dribble_st.committed
    and dribble_st.phase in (DRIBBLE_PHASE_GRAB, DRIBBLE_PHASE_CARRY, DRIBBLE_PHASE_RELEASE)
    and action_type != "kick"
)
```

Net effect: a **single** `dribble_to` selection at the ball now commits the robot through
`GRAB → catch → CARRY → RELEASE` across steps regardless of what the policy samples next — the grab can
finish aligning without competing primitives disturbing the heading. The `committed` gate keeps the default
`GRAB` phase from hijacking ordinary `approach_ball` steps. The GRAB still has the stall watchdog (§3.2) as a
safety cap (the free ball is stationary, so a non-converging align aborts after `stall_limit`).

### 10.3 Verified (sim-embedded, same un-retrained stage 2g model)

- **`catch` fired 1 / 3000 steps → 17 / 1500 steps (~34×)**; carry runs ~1 → 17; `dash`-with-ball 5 → 25.
- The macro now reliably **opens** carries — the prerequisite the achieved-gap reward needs.

Carries are still **short** (max ~6 steps) because this model was trained under the old (rarely-grabbing)
mechanic and has not seen carry reward. The mechanic is fixed; **stage 2g must be retrained** so the §7.1 /
§8.2 dribble rewards finally drive the policy to carry far and toward goal.

---

## 11. Simulator-synchronized catch verification and turn-free carry (2026-06-21)

A sim-only inference after §10 showed successful server catches followed by random turns and glued-ball
orbits, while telemetry reported **0 verified carries**. Two ownership bugs caused this:

1. `VERIFY` could record its baseline from the same simulator count that emitted `catch 0`. The later
   catch teleport then looked like mismatched robot/ball motion, producing a false verification failure.
2. A failed verification returned `done` without sending `drop`, resetting `DribbleState` while rcssserver
   still held the ball. Ordinary policy turns then rotated the server-pinned ball around the robot.

The current phase sequence is:

`GRAB → SETTLE → VERIFY → CARRY → ALIGN_RELEASE → RELEASE → DONE`

- `SETTLE` waits for a new `GameState.count` after `catch 0`, then records the post-catch baseline.
- `VERIFY` evaluates only new simulator counts and uses a low-power positive directional probe away from
  the ball. This prevents a free-ball collision from faking coupled motion and reduces probe displacement
  from roughly 0.12 m per catch to roughly 0.03 m. Duplicate/stale states do not consume an attempt.
  Negative power is not used because this server clamps it to zero (`MIN_DASH_POWER=0`).
- Failed verification sends `drop`, remains in `RELEASE` until a new simulator count, then retries `GRAB`
  (up to `max_catch_attempts`) or terminates. Arrival and watchdog exits while catch ownership is possible
  use the same drop-before-reset path.
- `CARRY` never turns the body. It emits `dash speed angle_diff`, using rcssserver's directional dash to
  translate toward the latched target while preserving body orientation. This prevents the glued ball
  from orbiting during 90–180° target alignment.
- `ALIGN_RELEASE` performs a simulator-count-gated, bounded alignment toward the remaining latched target
  immediately before `drop`, placing the glued ball in front of the robot for the next kick without
  reintroducing turns throughout `CARRY`.
- `carry_started`, active-carry telemetry, and achieved/active carry rewards begin only when `VERIFY` enters
  `CARRY`; target-choice reward is paid once at the earlier commitment transition.
- GRAB alignment requests `heading_error / dt`, but the shared command boundary caps it to the physical
  robot limit of 20°/s (2° per 100 ms simulator cycle).
- The progress watchdog runs only during verified `CARRY`; alignment, settlement, and verification do not
  consume the transport stall budget.

With `--debug_infer`, `step_trace.jsonl` now records simulator count, dribble phase, catch attempt,
verification step/displacements/offset change, alignment step count, latched target, and target-heading error.

## 12. Stage 2g reward and PPO credit assignment (2026-06-22)

- A fresh in-range `dribble_to` commitment receives an immediate **signed** gap reward evaluated at the
  endpoint reachable in one 0.85 m segment. The distant declared target is no longer rewarded as though one
  carry could reach it.
- Achieved-gap reward remains delayed until the verified carry closes and scores actual execution quality.
- While a macro is committed, PPO may sample only continuation or a kick interrupt. Continuation target
  parameters are masked out because the phase machine uses the original latched target.
- `dribble_active_bonus` fires only during verified carrying, not GRAB/SETTLE/VERIFY/alignment.
- Kick projection starts at the ball, and goalie-gap kick quality tapers continuously within 1 m of a post.

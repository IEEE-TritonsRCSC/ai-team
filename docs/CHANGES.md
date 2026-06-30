# RoboCup AI Training — Codebase Changes Log

All changes made to fix training issues, improve the environment, and implement curriculum learning for the TD3+HER trainer.

---

## 2026-06-30 — Defender clears the ball instead of dropping it to the attacker

Watching inference, the defender sometimes "handed" the ball to the attacker right after catching it.
Root cause in `ai_interface/defender.py` `_clear()`: the defender catch-glues the ball, then rotates
to aim the clear, but body rotation is capped at ~2°/cycle (MAXMOMENT). With `CLEAR_HOLD_CYCLE_LIMIT =
8` (and the orbiting glued ball quickly tripping `CLEAR_CARRY_LIMIT = 0.75`), it could not swing toward
the upfield clear target in time, so it fell through to **`drop`** — which releases the glued ball *at
the defender's own feet*, exactly where the pressing attacker is. Free possession to the attacker.

**Fix:**

- Replaced the `drop` fallback with a forward clear kick (`kick 70/85 0`). A drop is never the right
  release under pressure; a forward kick boots the ball away from our goal (the geometric turn has
  already swung the body roughly toward the upfield clear target by then). Kept units-safe — only
  `kick <power> 0` (direction 0), since the command pipeline does no rad/deg conversion and the rest of
  the codebase only ever kicks straight ahead.
- `CLEAR_HOLD_CYCLE_LIMIT` `8 → 24` and `CLEAR_CARRY_LIMIT` `0.75 → 1.4` so the geometric turn has time
  to point upfield before the forced release (cleaner clears; the glued ball orbiting the body no
  longer trips the carry limit after a few degrees of rotation).

**Validation:** `45 passed` (supporter + marker-defender, incl. `test_clear_when_ball_at_feet`).
6000-step `sim-embedded` smoke (24 eps): goal_rate **29.2%**, `hc_clearout_move=10%` (defender now
clears away), no `drop`. The attacker still scores via its own play rather than off defender gifts.

---

## 2026-06-30 — Supporter active during attack interlude + snappier opponent intercept

Two play-quality fixes from watching inference:

**1. The second attacker (supporter) sat still until the first attacker got the ball.** Root cause was
*not* `goto` — it was the claimant-follow interlude. While the gate is in `hardcoded_attack` mode
(robot 1 soloing toward the ball, PPO not in control), two things idled robot 2: (a) the env's
interlude loop drives only the carrier and defaulted every other pool robot to `turn 0`
([JAL_env.py](ai_interface/envs/JAL_env.py) ~L2810), and (b) `HardcodedSupporterCommandProvider`
suspends itself for `mode in {hardcoded_pass, hardcoded_attack}`. So robot 2 only began moving once
PPO regained control (`solo_finish`) — i.e. after robot 1 had the ball. Trace: first `hc_support_move`
at env_step ~54.

- Fix: the env now owns a `HardcodedSupporter` (`self.hardcoded_supporter`) and, during a
  `hardcoded_attack` interlude, drives the **non-carrier** pool robot with it instead of `turn 0`
  (records the `hc_support_move` event too). The aux provider stays suspended during the interlude,
  so there is no double-driving — the env supporter drives robot 2 during the attack interlude, the
  aux provider drives it during PPO control. Same `HardcodedSupporter` logic in both, and this path
  is shared by training and inference (`JALTeamEnv` is reused), so both get the fix.
  `hardcoded_pass` is unchanged (it already drives the receiver as robot 2 — suspension still required
  there to avoid a conflict).
- Verified: 1200-step `sim-embedded` smoke now shows the supporter emitting `hc_support_move` from
  **env_step 1** in every episode (was 54).

**2. The defender felt slow to intercept.** The crash/push speed-cap (now enforced on the defender's
full-speed INTERCEPT, which previously bypassed it) started decelerating from `5·PLAYER_SIZE` (4.5
units) out, which throttled the press too early.

- Fix: `PROXIMITY_SLOW_DIST` `5.0 → 3.8·PLAYER_SIZE`. The defender (and anything closing on an
  opponent) now holds full speed until ~3.4 units and only decelerates over the final ~1.6 units. The
  ramp + `0.10` floor still keep the contact closing speed low, so the anti-bang behavior holds.
- Verified: opponent-ahead speed profile is now `100` until 3.42u, then `74 (3u) → 43 (2.5u) → 10
  (contact)`.

**Validation:** `45 passed` (supporter + marker-defender). `py_compile` clean on `JAL_env.py` and
`basic_commands.py`. Both smoke runs clean, no exceptions.

---

## 2026-06-30 — Proximity cap correction: teammate vs opponent, no freeze, no supporter pin

The previous "hard crash/push stop" change (floor `0.0`, full stop at contact, applied equally to
teammates over a wide 4.5-unit / 150° cone) over-corrected and produced two regressions visible in
inference:

- **Robots froze.** A `0.0` floor means a dash heading at a robot inside the cone is zeroed at
  contact, so two robots whose targets pass through each other deadlock to a permanent stop.
- **The second attacker (supporter) waited for the first.** With the wide-cone full stop applied to
  *teammates*, the main attacker sitting a few units ahead pinned the supporter in place; it only
  started moving once the main attacker advanced toward the ball and cleared its forward cone. Trace
  evidence: a supporter episode opened `turn0 turn0 dash20 dash2 dash2 dash3 dash5 …` (crawling on
  the cap) instead of running its support route.

**Fix (`ai_interface/utils/basic_commands.py`, `_robot_proximity_speed_factor`):** the cap now
distinguishes the mover's own team from opponents (found by matching `self_pose` in
`game_state.robot_poses`):

- **Opponents — strict** (the SSL crash/push rule): `PROXIMITY_SLOW_DIST = 5·PLAYER_SIZE`, floor
  `PROXIMITY_MIN_SPEED_FACTOR = 0.10`. Ramps 95→~9 as it closes; the defender's full-speed intercept
  still cannot bang in.
- **Teammates — gentle**: `PROXIMITY_TEAMMATE_SLOW_DIST = 2.6·PLAYER_SIZE` (only bites in genuine
  close quarters), floor `PROXIMITY_TEAMMATE_MIN_FACTOR = 0.45`. A supporter eases past the attacker
  instead of being throttled/pinned, but still won't barge it.
- **Floors are non-zero on both paths**, so two robots can always creep and never deadlock to a
  freeze.

**Validation:** `45 passed` (supporter + marker-defender suites). Primitive check: teammate ahead →
`95` until 2.34u then 42.75 at contact (never pinned); opponent ahead → `95 → 9.5` ramp; contact
still creeps `>0` (no freeze). 1200-step `sim-embedded` smoke: the supporter now dashes at 95 from
step 1 of every episode (was crawling), run clean, no exceptions.

---

## 2026-06-30 — Hard crash/push stop and straight-in loose-ball approach

Sim-only inspection showed two movement problems the earlier proximity cap did not cover:

- The defender and the carrier still banged into each other — the defender when chasing/intercepting
  a ball the attacker held, and the attacker when both went for a loose ball. The proximity speed cap
  added earlier was gated behind `obstacle_avoidance`, but the defender's **full-speed INTERCEPT**
  path (`defender.py` `_move_to(..., full_speed=True)`) calls `goto(obstacle_avoidance=False)`, so the
  cap was bypassed entirely and the robot charged in at speed 100. The cap's 0.18 floor also let
  robots keep *creeping* into contact (a Pushing-rule violation) instead of stopping.
- The attacker visibly **vibrated back and forth** just before reaching a contested loose ball. Cause
  was the `goto` detour: it recomputes a left/right waypoint around the blocking defender every cycle,
  and near the ball the two sides are an almost-exact tie, so floating-point noise flipped the chosen
  side frame to frame.

**Fixes (`ai_interface/utils/basic_commands.py`):**

- Crash/push cap is now enforced on **every** player-aware dash, independent of `obstacle_avoidance`,
  via a new `goto(crash_speed_cap=True)` parameter (gated on `include_player_obstacles`). So even the
  defender's full-speed intercept now decelerates as it closes on a robot. SSL Crashing (8.4.2) and
  Pushing (8.4.1) are thus hard code constraints on all pursuit, not just detour-planned moves.
- `PROXIMITY_MIN_SPEED_FACTOR` 0.18 → **0.0**: the dash ramps to a full stop exactly at contact
  (`2·PLAYER_SIZE`), so robots no longer crawl into / push through each other. Turning and kicking are
  unaffected, so a robot at the contact standoff can still contest the ball (~kickable) without barging.
- New `DETOUR_SKIP_DIST = 3·PLAYER_SIZE`: on the **final approach** (target nearer than this) `goto`
  skips detour planning and goes straight in, letting the crash cap decelerate it instead of
  side-stepping. This removes the close-range side-flip vibration when reacquiring a loose ball.
- `_select_detour` now takes the robot `heading` and biases a near-tie toward the side the robot is
  already turned toward, so medium-range detours don't flip either.

**Keeper exemption (`ai_interface/goalie.py`):** `_goto` passes `crash_speed_cap=False` — the keeper
must dive freely across a crowded mouth and is never slowed by an attacker in front of it. (The
defender's intercept deliberately keeps the default `True` so it is capped.)

**Validation:**

```bash
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/utils/basic_commands.py ai_interface/goalie.py ai_interface/defender.py
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/test_hardcoded_supporter.py tests/test_marker_defender.py
```

Result: `45 passed`. Direct primitive check confirmed: full-speed dash toward a robot ramps
`100 → 81 (4u) → 7 (2u) → 0 (contact)`, the keeper stays at 100 with `crash_speed_cap=False`, and the
detour side holds steady (no flip) as the robot creeps forward. An 800-step `sim-embedded` smoke ran
clean (3 eps, attacker carries 12× and scores, no exceptions).

---

## 2026-06-30 — Loose-ball recovery gate, active defender clear, and attacker-side spawns

Follow-up sim-only inspection showed three related problems:

- When the ball was beside the attacker or had just been lost, the hybrid gate could still hand
  control to the Stage-3 PPO finisher even though the state was not Stage-3-ready. That made the
  robot vibrate or wait instead of immediately reacquiring the ball.
- The scripted defender entered `CLEAR` only when the ball was already kickable. If the ball was
  beside/behind the defender, `kick()` returned `failed` and `_clear()` collapsed to `turn 0`, so
  the defender waited for the attacker rather than collecting and clearing the loose ball.
- The active Stage 4 hybrid config spawned the ball in the attacking half while leaving the attacker
  at its default pose. This often produced starts where the ball was closer to the defender than to
  the attacking robot, which is not the Stage-3-like handoff condition we want.

**Fixes:**

- `ai_interface/envs/JAL_env.py`
  - Changed claimant-follow mapping so any non-Stage-3-ready carrier state uses `hardcoded_attack`,
    not `solo_finish`.
  - Specifically tags loose-ball cases with `reason="loose_ball_recover"` and
    `loose_ball_recover=true`, keeping PPO out until possession/reception-cone conditions are
    valid for the Stage-3 finisher.

- `ai_interface/defender.py`
  - Added `LOOSE_CLEAR_RADIUS`, so the defender actively handles nearby loose balls before they are
    perfectly kickable.
  - `_clear()` now moves to a legal contact point if the ball is just outside kick range.
  - If the ball is kickable but outside the front reception cone, the defender turns/catches first
    instead of returning `turn 0`.

- `configs/ppo_jal_curriculum_config.json`
  - Updated active `stage4_hardcoded_support_2v2` spawn settings:
    - `spawn_robot_at_ball=true`;
    - `spawn_offset_behind_ball=1.25`;
    - `spawn_theta_relative_to_goal=true`;
    - `random_spawn_theta_range_deg=[-45, 45]`.
  - This keeps the initial ball close to the attacking side and gives the hybrid controller a
    possession-like start instead of a defender-side scramble.

- `ai_interface/envs/JAL_env.py`
  - When claimant-follow mode is active and `spawn_robot_at_ball=true`, reset now also places the
    second attacker near the ball with a lateral support offset instead of leaving it at the distant
    default pose.

- `tests/test_hardcoded_supporter.py`
  - Updated the claimant-follow loose-ball test to require `hardcoded_attack` recovery.
  - Added defender regressions for side-ball collection and near-loose-ball clear movement.

**Validation:**

```bash
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/defender.py ai_interface/envs/JAL_env.py tests/test_hardcoded_supporter.py
/opt/anaconda3/envs/rcai/bin/python -m json.tool configs/ppo_jal_curriculum_config.json
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/test_hardcoded_supporter.py
```

Result: `tests/test_hardcoded_supporter.py` passed (`28 passed`).

---

## 2026-06-30 — Hardcoded attack catch-glue shot macro and stricter blocker veto

Follow-up sim-only inference (`infer_logs/20260630_202532_env0_stage4_hardcoded_support_2v2_complete`)
showed the previous fix was not sufficient. The run completed only 6 episodes, so it is not a stable
performance estimate, but it reproduced the two visible issues:

- Completed outcomes: `1/6 goals = 16.7%`, with `2` goalie catches, `2` off-target penalty-area
  endings, and `1` max-step.
- The stuck/slow-looking close-ball behavior was not `approach_ball`; episode 1 switched into
  `hc_attack_shot_align` for 22 steps before firing. That means the hardcoded attack bridge was
  rotating toward a shot target before explicitly catch-gluing the ball.
- The rebound/off-target issue was also in the hardcoded attack bridge: every terminal tracked shot
  was `hc_attack_shot_fired`.

**Fixes:**

- `ai_interface/hardcoded_attack.py`
  - Added per-robot `caught_for_kick` state, mirroring the already-working hardcoded pass macro.
  - `_kick_or_settle()` now does:
    1. face/catch the ball with `dribble()` until it emits `catch 0`;
    2. once caught, rotate the glued ball+robot assembly toward the target;
    3. fire only when the glued ball is still in the reception cone and heading error is within `5°`.
  - This prevents the robot from spinning toward the goal while the ball is merely touching the mouth.
  - Tightened direct-shot lane constraints:
    - `shoot_min_lane_clear: 0.70 -> 0.95`;
    - added wider `shot_lane_block_dist=4.5` env units for hardcoded fired shots.
  - Added `advance_min_lane_clear=0.85`; blocked long advance lanes now stage with a controlled
    dribble move instead of kicking into the defender.
  - Hardcoded attack events now include `lane_clear`, `nearest_blocker`, and `blocked_target` where
    relevant.

- `ai_interface/envs/JAL_env.py` and `infer.py`
  - Propagate hardcoded lane diagnostics into `step_trace.jsonl`:
    - `hardcoded_lane_clear`;
    - `hardcoded_nearest_blocker`;
    - `hardcoded_blocked_target`;
    - existing PPO kick traces also now include `kick_defender_lane_clear` and
      `kick_blocked_defender_lane`.

- `tests/test_hardcoded_supporter.py`
  - Added regressions that hardcoded attack catches before direct-shot alignment.
  - Added regression that a blocked advance lane stages instead of firing a kick.

**Validation:**

```bash
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/hardcoded_attack.py ai_interface/envs/JAL_env.py infer.py tests/test_hardcoded_supporter.py
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/test_hardcoded_supporter.py
```

Result: `tests/test_hardcoded_supporter.py` passed (`26 passed`).

---

## 2026-06-30 — Ball-facing approach and defender-lane shot veto

Latest sim-only debug inference exposed two concrete failure modes:

- In early episodes, the attacker reached the ball but stayed on `turn 0` even though the ball was
  outside the front reception cone. Trace rows showed `approach_ball` at about `1.65` env units from
  the ball with a ball-heading error around `116°`, so the robot was close enough for `goto()` to
  report done but was not physically facing the ball to grab it.
- The attacker repeatedly fired hardcoded shots through a defender lane. The 50k inference before
  that was not performing well: `14/97 goals = 14.4%`, with `33` max-step endings, `27` goalie
  catches, and `16` off-target penalty-area endings. The new hardcoded terminal-shot counters showed
  many `hc_attack_shot_fired` events ending as goalie catches, off-targets, or max-step stalls after
  defender/traffic rebounds.

**Fixes:**

- `ai_interface/utils/basic_commands.py`
  - `approach_ball()` now defaults its final heading target to the live ball bearing.
  - When the robot is already inside the approach margin, `goto()` therefore turns toward the ball
    instead of returning `done` and being converted to `turn 0`.

- `ai_interface/hardcoded_attack.py`
  - Added `shoot_min_lane_clear=0.70`.
  - Direct hardcoded attack shots now require both normal shot quality and a clear non-goalie
    defender lane before emitting `hc_attack_shot_*`.

- `ai_interface/hardcoded_supporter.py`
  - Added the same `shoot_min_lane_clear=0.70` guard for supporter finish shots.
  - If the lane is defender-covered, the supporter chooses return-pass/staging behavior instead of
    shooting into the blocker.

- `ai_interface/envs/JAL_env.py`
  - Added `_kick_defender_lane_clear()` for PPO/environment-owned kicks.
  - PPO kick macros now log `kick_defender_lane_clear` and `kick_blocked_defender_lane`.
  - A kick whose target lane is below `0.70` is vetoed before firing. When possible, the env falls
    back into a controlled `dribble_to` reposition step instead of kicking the ball into the defender.

- `tests/test_hardcoded_supporter.py`
  - Added regressions for close-ball approach turning, hardcoded direct-shot lane rejection, and
    PPO defender-lane detection.

**Validation:**

```bash
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/utils/basic_commands.py ai_interface/hardcoded_attack.py ai_interface/hardcoded_supporter.py ai_interface/envs/JAL_env.py tests/test_hardcoded_supporter.py
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/test_hardcoded_supporter.py
```

Result: `tests/test_hardcoded_supporter.py` passed (`24 passed`).

---

## 2026-06-30 — Terminal outcome tracking for hardcoded shots

The hardcoded bridge could emit many shot labels such as `hc_attack_shot_fired`, but the previous
diagnostics did not attribute the final episode result to those shots. This made conversion opaque:
we could see that hardcoded shots happened, but not whether they ended as goals, off-target shots,
goalie catches, or max-step stalls.

**Fixes:**

- `ai_interface/envs/JAL_env.py`
  - Added hardcoded terminal-shot classification for `hc_attack_shot_fired` and `hc_shot_fired`.
  - `record_external_action()` now stamps external events with `episode_num`, `env_step`, and
    best-effort `sim_count`.
  - At episode end, the env logs `Hardcoded shot terminal tracking` with:
    - total hardcoded fired shots in the episode;
    - counts by `label -> terminal outcome`;
    - the last hardcoded fired shot before termination, including target, quality, age, and outcome.
  - Terminal info now exposes `info["hardcoded_shot_terminal"]` for downstream inference/training
    diagnostics.

- `infer.py`
  - Aggregates `info["hardcoded_shot_terminal"]` across the run.
  - `summary.json` now includes:
    - `hardcoded_shot_terminal.all_fired_events`, e.g. `hc_attack_shot_fired->goal_scored`;
    - `hardcoded_shot_terminal.last_fired_per_episode`, which avoids over-counting episodes with
      multiple hardcoded shots and gives the most relevant conversion view.
  - Final inference logs print `[ALL] hardcoded shot terminal outcomes` when any hardcoded shot fires.

---

## 2026-06-30 — Hardcoded 2v2 pass/attack gate tightened after 12% inference regression

Latest 50k-step embedded inference for `stage4_hardcoded_support_2v2_complete` regressed to
**11/92 goals = 12.0%**. The hardcoded bridge was active, but it was being used from bad field
positions: fired hardcoded pass targets had mean `x=-4.72` and hardcoded attack staging targets had
mean `x=-6.98`, so the bridge was trying to build final scoring actions from our own half. PPO kicks
also fell (`85 → 55`) and bad aim rose (`29.4% → 38.2%`), confirming that the hardcoded layer was
stealing too many possessions before the Stage-3 finisher had a clean scoring state.

**Fixes:**

- `ai_interface/hybrid_pass.py`
  - Added an explicit attacking receive envelope for final scoring passes:
    `x >= 25`, `x <= 34`, `|y| <= 10` for the left-side attack.
  - `_clamp_receive_target()` now clamps to that attacking band instead of only applying field and
    max-x bounds. This prevents final pass targets in negative x / own-half positions.
  - Added `attacking_receive_target()` and `is_attacking_receive_target()` so the gate and pass
    lifecycle share the exact same target geometry.
  - `HardcodedPassCoordinator.start()` now prefers the gate-selected `pass_target` when present,
    keeping carrier aim and receiver movement consistent through the pass macro.

- `ai_interface/envs/JAL_env.py`
  - The claimant-follow gate now scores the teammate at the **concrete attacking pass target**, not
    at the teammate's raw current pose.
  - Pass candidates are rejected when the receiver is more than `10` env units from the attacking
    target, so a teammate sitting deep in our half cannot create a fake final pass by being clamped
    to `x=25`.
  - Added a short hardcoded-attack lock (`45` steps, replan allowed after `8`) to prevent
    frame-by-frame flicker while a physical macro is grabbing/aligning/carrying. The lock still
    permits a deliberate switch to hardcoded pass after the protected window if the teammate target
    is clearly better.
  - Reset/active-pass paths clear the attack lock so stale ownership cannot leak across episodes or
    pass lifecycles.

- `ai_interface/hardcoded_attack.py`
  - Added a near-Stage-3 bridge band: short staging carries are now reserved for possessions already
    near the attacking envelope (`x≈17..36`, `|y|<=16`).
  - Deep or wide possessions use `hc_attack_advance_*`: a direct controlled advance kick toward the
    Stage-3 envelope instead of repeated short dribble segments from negative x.
  - Direct hardcoded shots require a minimum attacking x (`x>=22`) so the bridge stops taking
    speculative long shots from poor field positions.

**Validation:**

```bash
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/test_hardcoded_supporter.py
```

Result: focused hardcoded-supporter suite passed (`21 passed`). New tests cover attacking receive
target clamps, rejection of deep teammates as final-pass targets, and deep hardcoded attack using
advance instead of staging.

---

## 2026-06-30 — Post-receive finish lock + safer hardcoded pass gate

Latest embedded inference after the catch-glue pass fix proved the pass mechanics were alive but the
team behavior regressed: `hc_pass_fired=149`, `hardcoded_pass_received=86`, but goals fell to **8.0%**
and kicks fell to **74** versus the passing-disabled fine-tuned baseline's **15.2%** and **273** kicks.
The completed passes were not becoming finish attempts. Episodes with multiple received passes often
timed out, while goal episodes were mostly not pass-assisted. The failure was control logic, not more
reward shaping: after a pass receive, the gate could immediately re-evaluate geometry and enter
another hardcoded pass instead of letting the receiver use the Stage-3 solo finisher.

**Fixes:**

- Added a real post-receive finish lock in `ai_interface/envs/JAL_env.py`.
  - On `hardcoded_pass_received`, PPO is mapped to the receiver and `ball_claimant_id` is pinned to
    that receiver.
  - The lock suppresses hardcoded pass entry and keeps the receiver as the active PPO robot until it
    fires a kick, loses the ball clearly to the teammate, expires, or the episode resets.
  - The old `_claimant_follow_pass_cooldown_until` only blocked `pass_mode`; it did **not** force
    receiver control. The new lock fixes the pass-ping-pong path directly.
  - Gate diagnostics now surface `reason=post_receive_finish_lock`, remaining lock steps, previous
    passer id, and receiver-ball distance.

- Tightened the active 2v2 pass gate in `configs/ppo_jal_curriculum_config.json`.
  - `claimant_follow_solo_lane_blocked_max`: `0.55 → 0.35` so passing happens only when the solo
    lane is genuinely blocked.
  - `claimant_follow_pass_lane_min`: `0.75 → 0.85` so the hardcoded pass lane must be cleaner.
  - `claimant_follow_teammate_margin`: `0.35 → 0.45` so the receiver must be meaningfully better
    than the carrier.
  - `claimant_follow_post_receive_cooldown_steps`: `30 → 120`; this now also controls the finish
    lock window, giving the receiver up to 12 seconds of PPO finish/recovery time.

- Narrowed hardcoded receive geometry in `ai_interface/hardcoded_supporter.py` and
  `ai_interface/hybrid_pass.py`.
  - `SupporterConfig.max_receive_y_abs`: `22 → 14` to stop receive/pass targets near the touchline.
  - Support candidate lateral offsets now use `±10/±6/±3` instead of `±12/±8/±4`.
  - Fallback support lateral offset narrowed from `8` to `6`.
  - `HardcodedPassCoordinator` now clamps both the receiver's initial pass target and defender-offset
    lead target through the same receive band; lead passes can no longer re-expand to `y≈20-24`.

**Why this is mathematically consistent with the inference:**

- Latest pass target analysis: 149 fired passes had mean `abs(y)=11.49`, with **49** targets at
  `abs(y)>18` and **47** at `abs(y)>20`. Since the goal mouth is centered and only about `±5` env
  units high, these wide receives produce poor continuation angles and many
  `ball_in_penalty_off_target` endings. Capping receive `|y|` at 14 keeps the supporter wide enough
  to avoid the defender but not so wide that finish geometry is destroyed.
- A 120-step finish window is 12 seconds (`0.1s/step`). That is deliberately larger than one align
  cycle under the 2°/step turn cap: a worst-case 180° finish turn needs about 90 steps before the kick
  fires, so the old 30-step cooldown could expire before the receiver had any realistic shot chance.

**Validation:**

```bash
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/envs/JAL_env.py ai_interface/hybrid_pass.py ai_interface/hardcoded_supporter.py tests/test_hardcoded_supporter.py
/opt/anaconda3/envs/rcai/bin/python -m json.tool configs/ppo_jal_curriculum_config.json
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/test_hardcoded_supporter.py
```

Result: focused hardcoded-supporter suite passed (`16 passed`). `graphify update .` was run after the
code changes.

---

## 2026-06-30 — Collision constraints + catch-glue pass align + lead pass (code, with passing re-enabled)

Re-enabled the hardcoded pass overlay for the 2v2 stage (`claimant_follow_solo_lane_blocked_max:
0.0 → 0.55`, the env default) and made three code changes so passing actually works under SSL rules.

**1. SSL collision constraints (code, not reward).** SSL rules 8.4.2 (Crashing: closing speed on
the line between two robots > 1.5 m/s = foul) and 8.4.1 (Pushing) are enforced at the
movement-primitive layer rather than penalized in reward:
- `basic_commands.goto` gained a **robot-proximity speed cap** (`_robot_proximity_speed_factor`):
  when a dash heads *toward* another robot within `PROXIMITY_SLOW_DIST` (5·PLAYER_SIZE), speed ramps
  down to a floor (0.18×) at contact. Applies to opponents (the SSL rule) **and** teammates (a
  supporter must not barge the attacker — not an SSL foul but it wrecks our own play). Gated on
  `obstacle_avoidance and include_player_obstacles`.
- The RL attacker's `approach_ball` at `JAL_env.py` now passes `obstacle_avoidance=True` — it was
  charging straight through the defender to reach the ball (avoidance defaulted off).

**2. Catch-glue pass align (root-cause fix for hc_pass_fired≈0%).** The carrier's pass-align
previously issued a bare `turn`, which spun the body off the **un-glued** ball; it fell out of the
mouth, the kick gate failed, and passes almost never fired (smoke: 2328 requested / **0 fired**).
`_carrier_pass_command` now mirrors the dribble GRAB: `dribble()` to face+`catch 0` (glue), a sticky
`HybridPassState.carrier_caught` flag, a geometric full-error `turn` to aim while the glued ball
revolves in the mouth, then a real `kick` on alignment. Smoke: **0 → ~30 fires per 6k steps.** A
re-grab spin bug (a premature "glue slipped" reset re-entered GRAB on the glued, orbiting ball and
spun forever) was removed — `carrier_caught` is sticky until the recover branch (ball gone) resets it.

**3. Lead pass around a blocked defender.** `_lead_target_around_defender` (called once in `start()`):
if an opponent sits on the ball→receiver lane (between them, within `2·PLAYER_SIZE+0.6`), the pass
target is shifted perpendicular away from the blocker by the clearance needed (bounded 2–5 units),
and the receiver follows the shifted target via `_receive_pose`. Lane clear → target unchanged.
Limitation: computed once at pass start (not re-evaluated as the defender moves) to keep the
carrier's aim target stable.

All three smoke-tested in `sim-embedded`; runs clean, attacker still carries the ball, passes fire,
goals still score. Real eval is the user's full inference run.

---

## 2026-06-30 — Stage renamed to 2v2 + passing-disabled solo-finisher fine-tune (SUCCESS)

Renamed the hybrid stage `stage4_hardcoded_support_2v3` → **`stage4_hardcoded_support_2v2`** in
`configs/ppo_jal_curriculum_config.json` (the live matchup is 2v2: 1 RL attacker + 1 hardcoded
supporter vs goalie + 1 defender). Set top-level `load_model` →
`final_models/stage3_complete.pt` and `save_path` → `models/ppo_jal_hybrid_support_2v2`.

With the hardcoded pass system disabled (`claimant_follow_solo_lane_blocked_max: 0.0`), the frozen
Stage-3 solo finisher dribbled until timeout and barely shot (11.0% goals, **37 kicks**, 48%
`max_steps`, aim 0.18). Fine-tuning that checkpoint for 200k steps in the 2v2 env (reward overrides
unchanged) fixed it: **15.2% goals (12/79), 273 kicks (7.4×), aim 0.33, bad-aim 46%→18%, timeouts
48%→35%, reward −201→−135.** Training goal rate climbed 4.5%→7.0%→14.0%. No code changes — config
only. See `docs/TRAINING.md` §67. The fine-tune scores by firing more well-aimed 10–15m shots, not
by working the ball closer.

---

## 2026-06-30 — Hybrid pass gate tuning: align-timeout fix + selective pass entry

### Problem

Inference analysis of the `stage4_hardcoded_support_2v3_complete` checkpoint revealed two issues in
the hardcoded pass coordinator (`ai_interface/hybrid_pass.py`):

1. **~50% align_timeout**: `max_align_steps=48` was too small. With the server body-rotation cap of
   2°/cycle (MAXMOMENT=±2, 20°/s @ 0.1 s/step), worst-case 180° correction needs
   `ceil((180−5)/2) = 88` cycles. The old value only covered ~96°, causing ~half of all pass
   attempts to time out before the carrier could face the pass lane.

2. **Pass monopolising possession**: The permissive gate (`solo_lane_blocked_max=0.55`,
   `pass_lane_min=0.65`, `teammate_margin=0.15`) triggered too eagerly. The hardcoded pass lifecycle
   consumed ~45% of sim steps (carrier aligning + receiver holding), starving the PPO solo-finisher
   which scores ~50% in its native stage. With passes firing every 4-5 steps on average, the solo
   finisher got ~3 kick opportunities per 6,000 steps instead of the ~33 it should get.

### Fix

**[ai_interface/hybrid_pass.py](../ai_interface/hybrid_pass.py)**:
- `max_align_steps: 48 → 110` — covers full 180° worst-case body rotation (88 cycles) plus margin
  for ball drift and minor re-approach. Verified mechanically: align_timeout fell from 33 → 5
  across 68 pass attempts.

**[ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py)**:
- Added `claimant_follow_post_receive_cooldown_steps` constructor param (default 30).
- After a `received` terminal, suppress re-pass for 30 sim steps so the new carrier (just received
  the ball) gets at least one PPO solo-finish attempt before the gate can trigger again.
- `_claimant_follow_pass_cooldown_until` tracked in state; reset to -1 on episode reset.
- Added `hybrid_pass_max_align_steps` constructor param, forwarded to `HybridPassConfig` at init.

**[ai_interface/trainers/ppo_jal_curriculum_trainer.py](../ai_interface/trainers/ppo_jal_curriculum_trainer.py)**:
- Plumbed `claimant_follow_post_receive_cooldown_steps` and `hybrid_pass_max_align_steps` from
  stage config via `_stage_or_top`.

**[infer.py](../infer.py)**:
- Same two params read from `stage_config` and forwarded to `JALTeamEnv`.

**[configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json)**
(`stage4_hardcoded_support_2v3`):
- `claimant_follow_solo_lane_blocked_max: 0.55 → 0.40` — carrier must be genuinely blocked before
  passing is considered.
- `claimant_follow_pass_lane_min: 0.65 → 0.75` — pass lane must be clearly safe.
- `claimant_follow_teammate_margin: 0.15 → 0.35` — teammate must be substantially better placed
  than the carrier before the gate engages.
- `claimant_follow_post_receive_cooldown_steps: 30` (new).
- `hybrid_pass_max_align_steps: 110` (new).

### Effect

Probe comparison (stage3 solo-finisher checkpoint, `stage4_hardcoded_support_2v3`):

| metric (per 1k steps)       | permissive gate | selective gate |
|-----------------------------|-----------------|----------------|
| PPO solo kicks              | 0.5             | 2.75 (+5.5×)   |
| total shots (kick+hc_shot)  | 1.2             | 3.0 (+2.5×)    |
| hc passes fired             | 4.8             | 4.25 (−12%)    |
| avg_reward                  | −123.9          | −101.2         |
| pass align_timeout share    | ~50%            | ~16%           |

Passing now only fires when the carrier is genuinely blocked and a teammate is substantially better
placed. The PPO solo-finisher gets the ball and shoots 2.5–5× more often.

### Validation

```bash
/opt/anaconda3/envs/rcai/bin/python -c "import json; json.load(open('configs/ppo_jal_curriculum_config.json')); print('JSON valid')"
# JSON valid
```

---

## 2026-06-29 — Robot-agnostic solo-finish gate for 2v1 hybrid PPO

### Problem

The 2v1 hardcoded-support hybrid was still too brittle because the PPO scorer was pinned to
TritonBots robot 1. When robot 2 received or won the ball, it could not become the learned
Stage-3-style finisher. The opposite failure also existed: if the carrier's goal lane was blocked,
the PPO policy still tried to solo-dribble instead of handing that possession to the hardcoded
passing logic.

### Fix

Added a robot-agnostic claimant-follow mode in
[ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py):
- `claimant_follow_robot_ids` defines the physical attacker pool, currently `[1, 2]`.
- The single PPO action slot is remapped each step to the robot that should be the solo finisher.
- A geometric gate chooses the mode:
  - `solo_finish`: PPO controls the current carrier when its lane is not blocked, or no useful pass
    exists.
  - `hardcoded_pass`: if the carrier's goal lane is blocked, the teammate has a better continuation
    lane, and the pass lane is open, PPO is held out and the hardcoded controller drives the carrier
    to pass.
- Per-robot dribble/kick/reward state is now allocated over the full follow pool so robot 1/2 swaps
  do not lose macro state or hit missing dictionary keys.
- `env.step()` now reports `active_robot_id`, `ppo_control_active`, and `claimant_follow` gate
  details for debug inference and training logs.

Updated the hardcoded supporter plumbing in
[ai_interface/trainers/policy_control.py](../ai_interface/trainers/policy_control.py),
[ai_interface/trainers/ppo_jal_curriculum_trainer.py](../ai_interface/trainers/ppo_jal_curriculum_trainer.py),
and [infer.py](../infer.py):
- `HardcodedSupporterCommandProvider` can now control the non-active robot from a pool instead of a
  fixed robot id.
- Trainer and inference loops call `set_active_robot_id()` before sending aux commands, so the
  hardcoded side always owns the robot PPO is not currently driving.
- The curriculum trainer stores PPO transitions only when `ppo_control_active=true`; hardcoded-pass
  interlude rewards are accumulated into the next PPO-controlled transition, so PPO is trained only
  on solo-finish decisions.

Updated [configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):
- enabled `claimant_follow_robot_ids: [1, 2]` on `stage4_hardcoded_support_2v3`;
- added gate thresholds for blocked solo lane, open pass lane, teammate improvement margin, and
  minimum pass distance;
- set the hardcoded supporter `robot_pool_ids` to `[1, 2]`;
- changed `load_model` to the latest local hybrid checkpoint
  `models/ppo_jal_hybrid_support_2v1/stage4_hardcoded_support_2v3_complete.pt`.

Added gate tests in [tests/test_hardcoded_supporter.py](../tests/test_hardcoded_supporter.py):
- clear carrier lane keeps PPO on the carrier;
- blocked carrier lane with an open teammate switches to hardcoded-pass mode and maps PPO to the
  receiver as a non-trainable interlude.
- regression coverage for the startup crash where hardcoded-pass mode zeroed all live primitives
  for inactive PPO rows, causing `primitive_valid_mask disabled every primitive for a slot`.
  Hardcoded-pass mode now leaves the runtime primitive mask sampleable because the sampled action is
  ignored and not stored during the interlude.

---

## 2026-06-29 — Hybrid supporter inference triage (theta units bug + frozen-scorer OOD)

Diagnosed the two failed hybrid inference runs (`stage4_hardcoded_support_2v3`,
logs `20260629_195910` embedded and `20260629_200442` sim-only). Three reported symptoms:
supporter goes off the field, no passing, and the solo PPO attacker performs far worse than
in pure-AI play. Root causes, with ground truth from the `.rcg` game log + obs dumps:

### 1. Supporter goes off the field — FIXED (theta degrees-as-radians)

`GameState.robot_poses` stores headings in **degrees** (`sim_get_robot_poses`); every env
consumer converts via `np.deg2rad`. The new `HardcodedSupporter` read `pose[2]` **raw** and
treated it as radians. A spawn heading of e.g. 121.85° was fed into `goto` as 121.85 *radians*,
so `goto`'s dash direction `atan2(dy,dx) - theta` became a ~120-radian value — after the sim
wraps it modulo 2π the robot dashes in a garbage direction and drifts to the boundary. The
`.rcg` confirmed the supporter (`l 2`) roaming x∈[-48,48], y∈[-33,33] and pinning in the
bottom-left corner regardless of where the ball was.

Fix: convert degrees→radians once at the boundary in `_extract_pose` and `_team_poses`
([ai_interface/hardcoded_supporter.py](../ai_interface/hardcoded_supporter.py)). Verified in
isolation: from the same spawn the dash direction now resolves to ~45° toward the forward
receive target instead of a 55-radian value.

### 2. Solo attacker much worse — ROOT-CAUSED (frozen Stage 3 model is out-of-distribution)

The PPO scorer is the **frozen** `stage3_defender_v2_finetune` checkpoint, which scores 54.5%
in its native stage (embedded, stochastic). Dropped into the hybrid it produces **0 kicks /
0 goals** and dribble-locks (`dribble_to` ~80%, has-ball ~72%, ball never advancing past
x≈26). Findings:
- It is **not** an inference-flag issue: `--ppo_stochastic` does not restore kicking here
  (it does on native Stage 3).
- The hybrid stage sets **`num_opponents=3`** (goalie + defender + marker_defender). The obs
  fills `num_opponents` opponent context slots, and `num_opponents` is derived from the
  opponent team size (`infer.py:764`), not the stage field. The frozen policy trained with
  context slot 3 **always zero**; the marker populates it → out-of-distribution → it stops
  kicking. Obs dump: native Stage 3 `ctx_mask=[0,1,1,0,…]`, hybrid `ctx_mask=[0,1,1,1,…]`.
- Reducing the opponent team to 2 (goalie + defender) makes the reset obs **structurally
  identical** to Stage 3, but the frozen model **still** does not kick — the second same-team
  body (the supporter) perturbs the in-episode dynamics (ball contention/lanes/defender
  tracking) that a single-agent policy never trained against.

Conclusion: a frozen single-agent policy cannot be dropped into the hybrid and finish
reliably. The `stage4_hardcoded_support_2v3` stage must be **fine-tuned** (warm-started from
Stage 3) so the attacker adapts to playing alongside the supporter. "No passing" (symptom 3)
follows from the scorer never kicking and the supporter never legally receiving.

---

## 2026-06-29 IST — Hardcoded Stage 4 hybrid supporter fallback

### Problem

The learned Stage 4 passing stack remained unreliable under the deadline: the RL policy could score
solo with the Stage 3 behavior, but repeated Stage 4 attempts failed to learn a dependable
supporter/pass/receive/finish chain. We need a Wednesday-safe fallback that keeps the working Stage 3
scorer and removes the second attacker's low-level coordination burden from PPO.

### Fix

Added [ai_interface/hardcoded_supporter.py](../ai_interface/hardcoded_supporter.py):
- `HardcodedSupporter` controls one same-team supporting attacker.
- Off ball, it searches a compact Sumatra-inspired receive wedge ahead/lateral to the carrier,
  scores candidates by carrier-to-support lane clearance, continuation shot quality, opponent
  separation, and travel cost, then latches the chosen point for stability.
- If the ball is already moving toward the supporter, it moves to an earliest reachable intercept
  point instead of waiting passively at the support spot.
- If the supporter gets the ball, it immediately shoots through the best legal in-mouth target, or
  returns a clear pass to the learned attacker if the shot is poor.
- Geometry helpers (`lane_clear_quality`, `legal_support_point`, `segment_distance`) are standalone
  and covered by tests.

Added `HardcodedSupporterCommandProvider` in
[ai_interface/trainers/policy_control.py](../ai_interface/trainers/policy_control.py):
- pads commands to the supporter's simulator unum, so a Stage 3 PPO model can keep controlling
  TritonBots robot 1 while the hardcoded supporter controls robot 2 on the same team;
- supports `controller_type: "hardcoded_supporter"` through the generic aux policy factory.

Wired `controller_type: "hardcoded_supporter"` into:
- [ai_interface/trainers/ppo_jal_curriculum_trainer.py](../ai_interface/trainers/ppo_jal_curriculum_trainer.py)
  for training/eval loops;
- [infer.py](../infer.py) for direct inference and `launch_infer.py` subprocesses.

Added a preserved config rung in
[configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):
- `stage4_hardcoded_support_2v3`;
- one PPO-controlled attacker (`robot_ids: [1]`) plus same-team hardcoded supporter (`TritonBots`
  robot 2);
- TeamB keeps the existing scripted goalie + ball defender + marker defender.

Added [team_config_hardcoded_support_2v3.json](../team_config_hardcoded_support_2v3.json) for
`launch_infer.py`:
- starts 2 TritonBots players so robot 1 can run the Stage 3 PPO attacker while robot 2 is driven by
  the hardcoded supporter aux policy;
- starts 3 TeamB players with TeamB robot 1 as goalie, matching the scripted goalie + two-defender
  opponent setup used by `stage4_hardcoded_support_2v3`.

Added [tests/test_hardcoded_supporter.py](../tests/test_hardcoded_supporter.py) for lane scoring,
legal support-point bounds, forward separated receive selection, and same-team command padding.

---

## 2026-06-29 IST — Pass macro rewritten to mirror the proven solo-shot loop (kill the reception-cone settle wall)

### Problem

The first abort-fallback run (§59 stage4u, run 20260629_141213) hit the 35% goal bar (34.6% overall,
38.6% last bucket — the carry fallback worked) but passing was still **0 fired / 0 resolved**. The
pass-macro diagnostics localized the failure precisely: every started pass timed out in the
`settle_contact` phase (`align_steps` 26–31, hitting the 30 cap; `fallback_reasons={'pass_macro_align_timeout'}`),
**never reaching the fire branch**. The `settle_contact` phase gated on `ball_in_reception_cone` and
short-settled with `dribble_to`; under the 20°/s turn cap that separate cone almost never latched, so
the macro burned its whole alignment budget there. The fire branch itself works when reached — the
upstream settle gate was the wall.

### Fix

[ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py) pass action branch: **replaced the
`settle_contact`/`face_target`/`fire` phase machine with a single unified loop that mirrors the proven
solo-shot path** (`action_type=="kick"`, which scores ~38–52%). Once the carrier has the ball, every
cycle now calls `kick(self_pose, ball_xy, angle_to_receiver, kick_power=pass_power, dribbling=True,
angle_tolerance=fire_threshold)`:
- returns `"kick …"` → register the pending pass + fire (unless the receiver drifted inside
  `pass_min_distance`, which bails to the carry fallback);
- returns `"failed"` (ball not glued in front) → re-acquire with a short, min-flight-preserving
  `dribble_to` settle touch (the only place `dribble_to` is still used);
- returns `"turn …"` → still aligning; the geometric `angle_diff/dt` turn converges despite the cap.

This removes the `ball_in_reception_cone` settle gate entirely. The catch-glue keeps the ball in front
while `kick()`'s geometric turn aligns the heading toward the receiver — the exact convergence the solo
shot already achieves. (Highest-EV untried fix from `project_stage4_goalie_only_cant_teach_passing`.)

[configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json) stage4u:
`pass_macro_max_align_steps 30 → 45` to give the converging loop room for a re-glue grab plus the
geometric alignment (the solo `kick_macro` cap is 120; 45 is generous for a pass while still bailing a
genuinely-stuck attempt into the carry fallback quickly).

**Follow-up (same day, after the first unified-loop run 20260629_144125):** the unified loop still
fired only 10 passes / 860 eps and goals regressed to 22.7%. Root cause in the `"failed"` re-glue
branch: it pointed `dribble_to` at a **2.0-unit short settle target**, which `dribble_to` reached and
**released (un-glued) almost immediately** — so `kick()` never had a stably-glued in-front ball to
turn-and-fire; the two machines fought (catch → tiny carry → release → unglue → "failed" → re-catch).
Fix: the `"failed"` branch now drives a `dribble_to` **carry in the receiver's direction** (a far
target = `ball + dir·(dist − min_flight)`), exactly like the solo shot carries toward goal — keeping
the catch-glue alive and rotating the body to face the receiver, stopping `min_flight` short so the
carry can never collapse the pass below the legal minimum. `kick()` (first each cycle) fires the moment
the heading aligns, before the carry reaches the stop point.

**Follow-up #2 (same day, after carry-orient run 20260629_150756):** carry-orient raised fires (10→19)
and got the first `Pass RESOLVED=1`, but goals **crashed further to 17.6%** — carrying the ball toward
a wide/lateral receiver drags it sideways/backward off the goal line. Fix: the `"failed"` re-glue
branch now catches the ball **IN PLACE** with `dribble()` (turn-to-face-ball → `catch 0`), no
translation. `kick()` then rotates the glued ball to face the receiver and fires, so the pass no longer
sacrifices forward progress toward goal. If the ball drifts out of kickable range, re-approach with
`approach_ball`. (Goal trend across these runs: 34.6% settle-cone wall → 22.7% un-glue fight → 17.6%
sideways drag → in-place catch under test.)

---

## 2026-06-29 IST — Stage 4 pass align-timeout falls back to carry-toward-goal (don't burn the episode in 2v2)

### Problem

In the 2v2 run (§58 stage4u), pass-macro completion stayed broken: `align_timeout=74.7%`, 8 fired /
0 resolved. Under defender pressure the carrier cannot finish the settle/aim, and the old abort just
emitted `turn 0` and re-entered the forced-pass macro next step — burning whole episodes
(`max_steps≈20%`) without scoring or passing. (Implements the parallel worker's §58 "fix for next
run" rec #2: short timeout + fall back to shoot/dribble.)

### Fix

[ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py) `_abort_pass`: on
`pass_macro_align_timeout` (and when the carrier still has the ball), the abort now falls back to
`dribble_to(target=goal)` — the proven solo-carry skill — instead of `turn 0`. This advances the ball
toward goal so a failed forced pass turns into shot-improving progress (and the `dribble` carry can
continue via the existing carry-continuation path), instead of stalling. Essential dribble bookkeeping
(`dribble_session_active`, anchor, target) is set; any dribble_to `failed`/`done` safely reverts to
`turn 0`.

[configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json) stage4u:
`pass_macro_max_align_steps 90 → 30` so a failing pass bails into the productive carry quickly
(successful 2v1 alignments completed in a ~17-step median, so 30 keeps real passes while cutting the
wasteful 90-step stalls).

### Validation

```text
py_compile: passed
tests/test_stage4_support_pass.py + tests/test_ball_action_recovery.py: 56 direct tests passed
config json: valid
```

Watch in 2v2: `max_steps` drag should fall and goals rise (failed passes now score via carry); passes
that DO align (<30 steps) still fire. Reward changes deferred until passes fire+resolve regularly (per
§58 rec #4).

---

## 2026-06-29 IST — Stage 4u short-settle pass macro (fix fake short passes and pass diagnostics)

### Problem

Stage 4t completed at only **9.9%** training goals and a valid 8k-step embedded debug inference
reached only **13.8%** goals. The model requested many passes, but the pass chain still failed:
`requested=362`, `fired=14`, `resolved=0` in inference.

The root cause was in the carrier pass macro. During `settle_contact`, the macro used
`dribble_to(target=receiver_target)`. That helped the earlier goalie-only alignment wall, but in
2v2 it could carry the ball almost all the way to the receiver target before the pass fired. Those
were not useful passes; they became short crowding dribbles or immediate defender interceptions
(`release` nearly equal to `aim`, e.g. `release=(29.16,6.19)`, `aim=(29.00,6.00)`).

One diagnostic was also misleading: `pass_available_steps` was only incremented when
`pass_available_bonus > 0`, so stages with the bonus disabled logged zero launchable-pass frames
even if the mask/path made passes available.

### Fix

[ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py):

- changed pass `settle_contact` from full `dribble_to(target=receiver_target)` to a **short
  settle touch** in the pass direction;
- capped that settle touch by the remaining legal pass flight distance, preserving
  `pass_min_distance`;
- added a fire-time guard: if the remaining ball-to-target distance is below `pass_min_distance`,
  abort with `pass_target_too_close_after_settle` instead of logging a fake pass;
- compute pass power from the true remaining ball-to-target flight distance, not robot-to-target
  distance;
- log fired passes with `flight`, `passer_shot_q`, `target_q`, and `lane_q`;
- keep the old `passer_lane_q` pending-pass key for reward compatibility, but also write
  `passer_shot_q` so future diagnostics do not confuse shot quality with pass-lane quality;
- increment `pass_available_steps` for actual `support_pass_launchable` frames independently of
  whether `pass_available_bonus` is enabled.

[configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):

- preserved failed `stage4t_2v2_defender_pass_v1` with `timesteps=0`;
- added active `stage4u_short_pass_settle_v1`, warm-starting from
  `models/ppo_jal_expandable/stage4t_2v2_defender_pass_v1_complete.pt`;
- kept `pass_macro_settle_dist=2.0`;
- restored a light defender-intercept margin with `pass_intercept_min_margin=0.5`, so forced passes
  are still possible but no longer fire through obviously reachable defender lanes;
- set the top-level `load_model` to the Stage 4t complete checkpoint and made Stage 4u the only
  nonzero-timestep curriculum rung.

### Validation

```bash
/opt/anaconda3/envs/rcai/bin/python -m json.tool configs/ppo_jal_curriculum_config.json
```

Result: JSON parses successfully. The next run should use
`stage4u_short_pass_settle_v1` and watch the new pass logs for nontrivial `flight` distance,
lower `pass_target_too_close_after_settle`, higher `fired→resolved`, and nonzero
`pass_available_steps` when launchable states exist.

---

## 2026-06-29 IST — Stage 4t 2v2 defender pass transfer config

### Problem

Run §56 showed the `dribble_to(target=receiver)` pass-settle fix did what it was meant to do
mechanically: `pass_macro align_timeout` dropped from ~74% to 41%. But goalie-only 2v1 still cannot
teach voluntary passing because solo dribble remains the best strategy when no field defender blocks
the lane.

### Fix

Added an active `stage4t_2v2_defender_pass_v1` curriculum rung in
[configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):

- warm-starts from `models/ppo_jal_expandable/stage4s_goalie_only_pass_pretrain_complete.pt`;
- runs `2` RL attackers vs TeamB scripted goalie + one hardcoded defender using
  `team_config_2atk_1def.json`;
- preserves the latest Stage 4s pass machinery: physical pass power (`base=2`, `per_unit=2.2`,
  `max=48`), `dribble_to` carrier settle, support target rewards, forced open-shot finish,
  receiver finish macro, and role-gated supporter `goto`;
- enables defender-aware reward gates: `use_defender_lane_gate=true` and
  `defender_lane_penalty=0.5`;
- keeps the defender intercept gate permissive enough for exploration
  (`pass_intercept_defender_speed=0.9`, `pass_intercept_min_margin=0.0`) instead of restoring the
  old Stage 4r over-strict valid-pass surface.

`stage4s_goalie_only_pass_pretrain` is preserved with `timesteps=0`; `stage4t_2v2_defender_pass_v1`
is the only active Stage 4 rung.

### Validation

```bash
/opt/anaconda3/envs/rcai/bin/python -m json.tool configs/ppo_jal_curriculum_config.json
```

Result: config JSON parses successfully and reports one active Stage 4 rung,
`stage4t_2v2_defender_pass_v1`.

---

## 2026-06-29 IST — Stage 4 carrier pass settle: dribble_to-toward-receiver (kill the settle↔aim oscillation)

Refines the catch-glue fix below. Run §53/§55 showed catch-glue fixed delivery quality
(resolved/fired 9%→23–26%) but `align_timeout` rose to ~73%: `dribble()` faces the *ball*, which
fought the subsequent turn-to-*target*, ping-ponging settle↔aim. Now the pass `settle_contact` calls
`dribble_to(target=receiver)` — the exact carry the 52% solo shot uses — which catches AND carries the
ball in the pass direction, so it enters the front cone already facing the receiver and the in-cone
`kick()` fires immediately. dribble_to's own carry-limit release is suppressed (`kick`→`turn 0`) so the
bookkept in-cone pass-fire path registers the pass. [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py).
Validation: py_compile passed; 56 stage4 + ball-action tests passed. Watch `align_timeout` (target
<40%, was 73%).

---

## 2026-06-29 IST — Stage 4 carrier pass: catch-glue the ball before aiming (mirror the 52% shot)

### Problem

Across runs §47–§52, pass `resolved/fired` stayed at 6–11% and `pass_macro align_timeout` sat at
~50% — half of all pass macros never fired. Root cause: the carrier pass `settle_contact` phase used
`_goto_contact_pose_command` (drive to a pose behind the ball) which **never issues a `catch`**, so
the ball was never glued to the dribbler. When the macro then turned to aim at the receiver, the
unglued ball drifted out of the front reception cone → back to settle → **oscillation** that burned
the whole align budget. The proven solo shot (`action_type=="kick"`, ~52% goals) does not have this
problem because its `dribble_to` **catches** the ball first; once glued, bare turns from `kick()`
keep it in front while aligning.

The "to-feet" aiming experiments (previous CHANGES entry) were **reverted** — they regressed
(`resolved` 9%→7%, goals 28%→20%) because re-aiming at the moving receiver added more oscillation on
top of the unglued-ball problem. The real bug was the missing catch, not the aim point.

### Fix

[ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py): the carrier pass `settle_contact`
branch now calls `dribble(self_pose, ball_xy)` (turns to face the ball, then `catch 0` to glue it to
the dribbler) instead of `_goto_contact_pose_command`. Falls back to `approach_ball` only if the ball
is out of catch range. Once the ball is caught and in the front cone, the existing in-cone
`kick(dribbling=True, angle_tolerance=10°)` align+fire path runs exactly like the 52% shot — bare
turns keep the glued ball in front, so it reaches the fire gate instead of oscillating. Added
`dribble` to the basic_commands import.

### Validation

```text
py_compile: passed
tests/test_stage4_support_pass.py + tests/test_ball_action_recovery.py: 56 direct tests passed
```

Watch in training: `pass_macro align_timeout` should fall well below 50% and `resolved/fired` should
break past the 6–11% plateau (the receiver finish already converts resolved passes at consumed≈9/11).

---

## 2026-06-29 IST — Stage 4 "to-feet" passing (aim at the receiver, not an abstract support point) — REVERTED

> Reverted same day: to-feet aiming regressed (resolved 9%→7%, goals 28%→20%) because the real bug
> was the missing catch (see entry above), not the aim point. Kept for the record.

### Problem

After the pass-power fix (run §49) stopped passes overshooting (`travelled` 32→17u,
interceptions→0, goals 18→28%), `resolved/fired` stayed stuck at ~9% across four runs (§47–§50).
Diagnosis: every pass was aimed at an abstract forward **support target**, while the receiver was
driven there separately and then pushed to a receive pose ~1.1u *beyond* it. The ball, the support
target, the receive pose, and the receiver's actual position were four different points that never
reliably coincided — and kick-noise scatter over 12–17u widened the gap. Tightening the rendezvous
tolerance (§50, `max_target_dist 9→4`) made it *worse* (6.6%), confirming the open-loop
aim-at-a-point design was the problem, not the tolerance.

### Fix

Aim the pass **to the receiver's actual position** (with a small goal-ward lead) so the ball goes to
the robot, not a point the robot has to rendezvous with.

- [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py): when the pass macro registers, the
  aim target is overridden to `receiver_position + pass_to_feet_lead · unit(receiver→goal)` (using the
  receiver's live pose from `pose_by_robot_id`). The carrier then aligns to and fires at the receiver.
- [ai_interface/envs/reward.py](../ai_interface/envs/reward.py): added `pass_to_feet_lead = 1.0`.
- [configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json) (stage4s):
  `pass_to_feet_lead = 1.0`; shrank `pass_receive_pose_offset 1.115 → 0.6` so the receiver barely
  moves from where the ball is sent; reverted `support_pass_receiver_max_target_dist 4 → 6` (to-feet
  now handles delivery, so the gate only needs to confirm the receiver is in a sensible spot).

### Geometry

Receiver at R, aim = R + 1.0·(toward goal). With the §49 power model the ball travels ~aim_dist+0.8u,
so it stops ~1.8u past R along the pass line. The receive pose is aim + 0.6u (= ~1.6u past R), facing
the incoming ball, so the ball arrives ~0.2u in front of the receiver's mouth — inside the kickable
reception cone — and the receiver only steps ~1.6u forward onto it instead of chasing an abstract
point up to 9u away.

### Validation

```text
py_compile: passed (JAL_env.py, reward.py)
config json: valid
tests/test_stage4_support_pass.py + tests/test_ball_action_recovery.py: 56 direct tests passed
  (updated test_pass_fire_starts_inflight_receive_finish_macro to assert to-feet aiming:
   receive_target ≈ receiver position + ~1u goal-ward lead, not the old (32,6) support point)
```

---

## 2026-06-29 IST — Stage 4 pass/finish fire gate widened from 5° to 10° (wiring fix)

### Problem

Training run §47 (first test of the deterministic contact-pose macros) still failed the finish bar:
`pass_macro started=1,041` with **`align_timeout=493` (47%)**, and the receiver finish macro fired only
**3 / 485** (0 in the last-100 episodes). Debug inference showed the policy scores **52.5% solo by
dribbling** and never passes voluntarily — so the only deliverable left is making a *forced* pass
reliably end in a goal, which requires the macros to actually emit terminal kick events.

Root cause was a tight, mis-wired fire gate:

- The pass/receive macros decide "aligned enough to fire" but then call the shared `kick()` primitive,
  which **hardcoded a 5° heading tolerance** ([ai_interface/utils/basic_commands.py](../ai_interface/utils/basic_commands.py) `kick`).
  So even when `_can_fire_physical_kick` (carrier) passed at a wider tolerance, `kick()` re-gated at 5°
  and returned a `turn` → `pass_kick_failed` / `kick_align_timeout` → macro times out.
- Codex's `pass_macro_orientation_threshold_deg = 10.0` knob existed in `reward.py` but was **never
  wired into JAL_env** — every fire gate hardcoded `math.radians(5.0)` (JAL_env.py:3413, 3449, 4494).
- 5° is far too tight for the rate-capped, glue-drift align dynamics; the contact-pose oscillates in
  and out of the gate and the macro burns its whole budget.

### Fix

- [ai_interface/utils/basic_commands.py](../ai_interface/utils/basic_commands.py): `kick()` gained an
  `angle_tolerance` parameter (default `radians(5)` — the solo-shoot path that already scores 52% is
  **unchanged**). Only the pass/receive macros pass a wider value.
- [ai_interface/envs/reward.py](../ai_interface/envs/reward.py): added
  `receive_finish_fire_tolerance_deg = 10.0` (carrier reuses the existing
  `pass_macro_orientation_threshold_deg = 10.0`).
- [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py):
  - carrier fire gate (`fire_threshold`) now reads `pass_macro_orientation_threshold_deg`;
  - the carrier `kick(...)` call passes `angle_tolerance=fire_threshold` (the actual wiring bug);
  - the receiver finish `kick(...)` call passes `angle_tolerance=receive_finish_fire_tolerance_deg`.
- [configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json): stage4s sets both
  knobs to `10.0` explicitly (so the log-training parser captures them).

### Shot geometry check

A 10° heading error at a 12-env-unit finish gives a lateral miss of `12·tan(10°) ≈ 2.1` units, well
inside the `5.0`-unit goal half-height — so a wider finish still hits the goal mouth. Pass release at
10° is comfortably inside the receiver collection window. No reward values changed (this is a fire-gate
wiring fix, not a reward change).

### Validation

```text
py_compile: passed (JAL_env.py, reward.py, basic_commands.py)
smoke: kick() returns "turn" at 8° with default 5° gate, "kick" at 8° with 10° gate, "turn" at 12°/10° gate
tests/test_stage4_support_pass.py + tests/test_ball_action_recovery.py: 56 direct tests passed
  (updated near-aligned test to 14° to keep guarding the "turn, not pass_kick_failed" path;
   added test_pass_macro_fires_within_widened_orientation_gate for the 7° fire case)
config json: valid
```

---

## 2026-06-29 IST — Stage 4 deterministic pass/finish contact-pose macros

### Problem

The latest Stage 4s goalie-only training run showed that pass exploration was no longer the main
blocker, but physical execution still failed:

- `22,840` pass requests produced only `62` fired passes (`0.27%` request -> fire);
- `3,042 / 3,447` pass macro starts ended as `pass_kick_failed`;
- `40` resolved passes produced `0` consumed finish banks and `0` receiver finish shots;
- the receiver macro spent many frames in `bad_cone` and `kick_align`, then expired;
- the passer could still reclaim/crowd the ball after a pass fired.

The code root cause was that the pass macro considered the carrier "aligned" at a configurable
`10 deg` threshold, then called the normal physical kick helper. That helper only emits a real
`kick` inside the front reception cone and within `5 deg`; otherwise it returns a turn/catch/failed
path. A pass in the `5-10 deg` band therefore reset as `pass_kick_failed`, destroying the committed
pass attempt.

### Fix

Updated [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py):

- Added shared contact-pose geometry helpers:
  - `_front_contact_pose(ball_xy, target_xy)`;
  - `_at_contact_pose(...)`;
  - `_target_heading_error(...)`;
  - `_can_fire_physical_kick(...)`;
  - `_goto_contact_pose_command(...)`.
- Extended pass macro state with `pass_macro_phase` and `pass_macro_phase_steps`.
- Reworked carrier pass execution into deterministic phases:
  `acquire -> settle_contact -> face_target -> fire`.
- The pass macro now fires only after the same physical checks as `kick()` pass:
  ball in front reception cone and heading error within `5 deg`.
- A one-frame failed fire check no longer resets the whole pass macro. The macro keeps aligning or
  times out through the explicit phase caps.
- Added short phase caps:
  `pass_macro_bad_cone_max_steps`, `pass_macro_face_max_steps`, and existing
  `pass_macro_max_align_steps`.
- Added `passer_support_lock_until_count`: after a pass fires, the passer is temporarily excluded
  from ball-claim selection so it cannot crowd the receiver.
- Reworked post-pass bad-cone recovery:
  - receiver now moves to the shot contact pose behind the ball instead of repeatedly running a
    dribble settle loop;
  - bad-cone and kick-align phases have short caps via
    `post_pass_finish_macro_bad_cone_max_frames` and
    `post_pass_finish_macro_kick_align_max_steps`.

Updated [ai_interface/envs/reward.py](../ai_interface/envs/reward.py):

- Added config knobs:
  `pass_macro_contact_pose_radius`, `pass_macro_bad_cone_max_steps`,
  `pass_macro_face_max_steps`, `pass_support_lock_steps`,
  `post_pass_finish_macro_bad_cone_max_frames`, and
  `post_pass_finish_macro_kick_align_max_steps`.

Updated [configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):

- Active `stage4s_goalie_only_pass_pretrain` now explicitly uses:
  - `pass_macro_bad_cone_max_steps: 12`;
  - `pass_macro_face_max_steps: 20`;
  - `pass_macro_contact_pose_radius: 0.45`;
  - `pass_support_lock_steps: 20`;
  - `post_pass_finish_macro_bad_cone_max_frames: 12`;
  - `post_pass_finish_macro_kick_align_max_steps: 20`.

Updated [tests/test_stage4_support_pass.py](../tests/test_stage4_support_pass.py):

- Added regression coverage that the contact pose puts the ball in the physical front cone.
- Added regression coverage that a near-aligned pass turns/continues instead of resetting as
  `pass_kick_failed`.
- Added regression coverage for short receive bad-cone expiry.
- Added regression coverage for the passer support lock handing ball claim to the teammate.

### Validation

```bash
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/envs/JAL_env.py ai_interface/envs/reward.py tests/test_stage4_support_pass.py
/opt/anaconda3/envs/rcai/bin/python - <<'PY'
import tests.test_stage4_support_pass as t
for name in [n for n in dir(t) if n.startswith("test_")]:
    getattr(t, name)()
print("stage4 support tests passed")
PY
/opt/anaconda3/envs/rcai/bin/python -m json.tool configs/ppo_jal_curriculum_config.json
```

Result:

```text
py_compile: passed
stage4 support tests: 36 direct test functions passed
config json: valid
pytest not run: rcai env has no pytest module
```

---

## 2026-06-29 IST — Stage 4 Sumatra-inspired committed pass/receive macro

### Problem

The Stage 4 passing pipeline had moved past the first blocker (`pass_to_teammate` could be sampled),
but the behavior was still physically brittle:

- pass requests often failed to become fired passes because the carrier had to already be in the
  front reception cone before the pass macro could even run;
- after a pass fired, the receiver only entered the post-pass finish macro once the pass had already
  resolved, which is too late for an incoming ball;
- the receiver behaved like a passive supporter/collector instead of preparing a receive pose before
  the ball arrived;
- the passer could continue normal support behavior around the same ball after release, recreating
  crowding near the receiver;
- the failure mode matched what Sumatra avoids: treating a pass as one carrier action rather than a
  committed two-robot play.

### Sumatra reference

Reviewed local TIGERs Mannheim/Sumatra code under `/private/tmp/sumatra` and adapted the executable
parts that fit this Python env:

- `StandardPassActionMove`: a pass only becomes a true release after ball contact plus small
  orientation error.
- `PassReceiverRole`: the receiver drives to a pre-contact pose behind the pass target and faces the
  incoming source.
- `ReceiveState`: receiving is an active interception skill, not passive `goto` waiting.
- `PassCreator`/`PassGenerator`: pass validity accounts for receiver reachability and preparation,
  not just a clear static line.

### Fix

Updated [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py):

- Relaxed the primitive mask so `pass_to_teammate` is not disabled solely because the carrier has the
  ball just outside the physical front cone. It is still masked when there is no valid receiver target
  or the carrier is out of possession range.
- Reworked the carrier pass execution into a committed physical macro:
  `acquire -> settle front cone -> align -> fire`.
- Added a settle phase that uses the dribble/catch macro to pull the ball into the front reception
  cone before pass release instead of falling back to `turn 0`.
- Changed the pass orientation gate to a Sumatra-style threshold (`10 deg` by default) before firing.
- When a pass fires, the env now immediately starts an unresolved receiver macro using the pass
  `release_pos` and `receive_target`.
- Added `_pass_receive_pose(...)`, which computes the receiver pre-contact pose:
  `receive_pose = receive_target + unit(release->target) * (PLAYER_SIZE + BALL_SIZE)`.
  With current constants this offset is `0.9 + 0.215 = 1.115` env units.
- The unresolved receiver macro now drives to that pre-contact pose, faces the pass source, waits
  there while the ball is still far, then actively approaches/collects once the ball reaches the
  receive window.
- When the pending pass resolves, the same receiver macro is marked `resolved=True` and continues into
  the existing settle/finish logic instead of being replaced with a geometry-less late macro.
- If a pending pass expires or is intercepted, the unresolved receiver macro is cancelled so the
  receiver does not chase a dead pass.

Updated [ai_interface/envs/reward.py](../ai_interface/envs/reward.py):

- Added configurable knobs:
  `pass_macro_orientation_threshold_deg`, `pass_macro_settle_dist`,
  `pass_receive_pose_offset`, `pass_receive_pose_radius`,
  `pass_receive_ball_obstacle_dist`, and `pass_receive_hold_ball_dist`.

Updated [tests/test_stage4_support_pass.py](../tests/test_stage4_support_pass.py):

- Added regression coverage for fired pass -> unresolved receiver macro creation.
- Added regression coverage for the in-flight receiver macro moving to the pre-contact receive pose.

### Validation

```bash
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/envs/JAL_env.py ai_interface/envs/reward.py tests/test_stage4_support_pass.py
/opt/anaconda3/envs/rcai/bin/python - <<'PY'
# direct smoke: aligned pass fires and starts unresolved receiver macro
PY
~/.local/bin/graphify update .
```

Result:

```text
py_compile: passed
direct smoke: pass fired, receiver macro started with resolved=False and receive_pose=(32.997..., 6.498..., -2.677...)
graphify update: passed
pytest not run: rcai env has no pytest module; base conda pytest lacks gymnasium
```

---

## 2026-06-28 (Late PM) — Stage 4s pass-finish behavior fixes

### Problem

The Stage 4s goalie-only debug inference exposed a different failure from the earlier pass-mask
starvation:

- the policy could request passes, but the receiver almost never converted them into shots;
- post-pass finish macros were starting and then expiring: `macro_started=17`, `macro_fired=0`,
  `macro_expired=16` in the latest 15k embedded debug run;
- macro diagnostics were dominated by `bad_cone`/`settle` frames, meaning the receiver had the ball
  near it but not inside the physical front reception cone;
- the previous bad-cone branch tried to `dribble_to(target=ball_xy)`, which is degenerate because the
  target is already at the ball, so the dribble macro can settle/complete without moving the ball into
  a shootable lane;
- support targets were still too deep/wide for reliable immediate finishing, contributing to
  `ball_in_penalty_off_target` and receivers turning away from the goal;
- standalone pass completion reward was large enough that PPO could learn "complete a pass" without
  learning "complete a pass, receive, and shoot";
- after a pass resolved, the old passer decoded a normal support wedge around the same ball point and
  could drive back toward the receiver, recreating attacker crowding;
- `goto()` could emit an unnormalised body-relative dash angle, which made some movement look like it
  was taking the long/reflex rotation path.

### Fix

Updated [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py):

- Added a latched `settle_target` to the post-pass finish macro state.
- Added `_post_pass_finish_settle_target(...)`, which chooses a short, legal, non-degenerate carry
  target instead of targeting the current ball position during bad-cone recovery.
- The settle target samples small forward/centerward candidates, filters field bounds and opponent
  defense-area/off-target positions, and scores candidates by current shot quality.
- The bad-cone macro branch now reuses the latched settle target and calls `dribble_to()` toward that
  point. If no legal settle target exists, it turns toward the ball using the shortest normalised
  angle instead of falling into a no-op dribble.
- Added `_post_pass_finish_kick_target_y(...)`, which chooses among center and keeper-away in-mouth
  targets using goalie-gap quality, then breaks ties by the smallest heading change from the receiver's
  current orientation. This prevents the receiver from aiming past the goal when an equivalent shorter
  finish exists.
- Added `_post_pass_clearout_target(...)` and wired it into `_decode_support_goto_target(...)`.
  During an active post-pass finish macro, non-receivers clear away from the receiver/ball lane instead
  of decoding a normal receive wedge around the same ball point.

Updated [ai_interface/utils/basic_commands.py](../ai_interface/utils/basic_commands.py):

- Normalised the body-relative dash angle in `goto(...)`, so path-following uses the shortest angular
  direction consistently.

Updated active Stage 4s settings in
[configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):

- tightened receive coordinates to more finishable lanes:
  `support_forward_min: 8.0`, `support_forward_max: 16.0`, `support_target_max_x: 29.0`,
  `support_target_y_clip: 6.0`, `support_target_y_clip_x_min: 26.0`;
- reduced dense support shaping so it does not dominate the actual pass-finish objective:
  `support_receive_target_bonus: 0.04`, `support_receive_ready_bonus: 0.08`;
- made the receiver need to be closer to its selected target before a pass launch is considered valid:
  `support_pass_receiver_max_target_dist: 9.0`;
- reduced standalone resolved-pass reward:
  `possession_transfer_bonus: 0.5`, `pass_quality_weight: 1.5`;
- disabled pass-available waiting reward: `pass_available_bonus: 0.0`;
- made the finish the main source of pass-chain value:
  `pass_finish_bonus: 24.0`, `pass_finish_quality_weight: 20.0`;
- lowered the post-pass macro shot-quality threshold to `0.20` so it shoots acceptable open finishes
  instead of over-staging until the deadline.

Reward math after this change:

- A resolved pass with quality `q=0.5` now pays only `0.5 + 1.5*0.5 = 1.25` total before team
  averaging, so it is not enough to train pass-only wandering.
- A receiver finish macro kick with quality `q=0.5` banks `24 + 20*0.5 = 34` total before team
  averaging and shot-quality scaling. With two attackers, that is `17` team reward before scaling, so
  the profitable behavior is pass→receive→shoot.
- The settle carry samples at most `6` env units, still below the `8.5` env-unit safety cap and the
  SSL `1m = 10` env-unit dribble limit.

### Validation

```text
/opt/anaconda3/envs/rcai/bin/python -m py_compile \
  ai_interface/envs/JAL_env.py \
  ai_interface/utils/basic_commands.py \
  ai_interface/envs/reward.py

/opt/anaconda3/envs/rcai/bin/python -m json.tool \
  configs/ppo_jal_curriculum_config.json
```

`pytest` is not installed in the `rcai` environment, so the closest Stage 4/pass and ball-action
regression tests were invoked directly:

```text
/opt/anaconda3/envs/rcai/bin/python - <<'PY'
import importlib
mods = ['tests.test_stage4_support_pass', 'tests.test_ball_action_recovery']
count = 0
for mod_name in mods:
    mod = importlib.import_module(mod_name)
    for name in sorted(dir(mod)):
        if name.startswith('test_') and callable(getattr(mod, name)):
            count += 1
            getattr(mod, name)()
print(f'{count} direct tests passed')
PY
```

Result:

```text
49 direct tests passed
```

---

## 2026-06-28 (PM) — Stage 4s goalie-only pass pretrain

### Problem

The latest Stage 4r defender run proved the receiver-claim handoff and longer pass-align budget were
not enough. The run finished at 25.9% overall goals and 21% last-100, with only
`301 requested -> 14 fired -> 3 resolved` passes. The key diagnostic was that the defender made the
valid-pass surface too sparse: `pass_available_steps=0` in logged summaries, with target/mask
rejections dominated by `pass_interceptable`, `not_kickable`, `bad_reception_cone`,
`too_close_for_ssl_pass`, and `pass_lane_blocked`.

That means Stage 4 was still asking PPO to learn supporter positioning, pass timing, pass execution,
receiver collection, and defender-aware lane creation all at once. The model never got enough
successful pass completions for the downstream receiver-finish macro to train.

### Fix

Added a new active curriculum rung in
[configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):

- `stage4s_goalie_only_pass_pretrain`
- `timesteps: 150000`
- `team_config: team_config_2atk_goalie.json`
- `num_opponents: 1`
- only one aux opponent controller: TeamB scripted goalie `robot_id=1`
- previous defender stage `stage4r_intercept_gate_v1` is preserved with `timesteps: 0`

The new rung keeps the Stage 4 coordination machinery active:

- `ball_action_recovery: true`
- sticky claimant split for attacker/supporter roles
- supporter remains `goto`-driven through the dynamic mask
- role-relative support targets
- pass macro with `pass_macro_max_align_steps: 90`
- pending-pass receiver claim handoff
- post-pass receive-finish macro

But it removes field-defender pressure:

- `use_defender_lane_gate: false`
- `defender_lane_penalty: 0.0`
- TeamB has only the goalie, so there is no hardcoded defender shadowing the pass lane
- pass intercept gating is relaxed for this mechanics rung with `pass_intercept_defender_speed: 0.0`
  and `pass_intercept_min_margin: 0.0`

Reward/config adjustments for the pretrain:

- support target shaping is denser: `support_receive_target_bonus: 0.08`,
  `support_receive_ready_bonus: 0.14`, `support_bad_target_penalty: 0.04`
- pass distance floor is relaxed but still meaningful: `support_min_pass_distance: 7.0`,
  `pass_min_distance: 7.0` (`0.7m` with the 10 env-units/m conversion)
- `support_receive_ready_radius: 6.0` and
  `support_pass_receiver_max_target_dist: 12.0` make the receive state easier to discover
- `pass_available_bonus: 0.02` for at most 6 steps teaches the precondition without making waiting
  profitable
- open-shot urgency remains active so the carrier still shoots when the direct shot is clearly better

This stage is not meant to be the final Stage 4 policy. It is a pass-mechanics pretrain: train
without the defender until pass fire/resolve/receive-finish is real, then reintroduce defender
pressure in a later rung.

### Validation

```text
/opt/anaconda3/envs/rcai/bin/python -m json.tool configs/ppo_jal_curriculum_config.json

/opt/anaconda3/envs/rcai/bin/python - <<'PY'
import json
from ai_interface.envs.reward import RewardConfig
cfg = json.load(open('configs/ppo_jal_curriculum_config.json'))
active = [(k, v.get('timesteps')) for k, v in cfg['curriculum'].items() if int(v.get('timesteps', 0)) > 0]
print(active)
RewardConfig(**cfg['curriculum']['stage4s_goalie_only_pass_pretrain']['reward_config_overrides'])
PY
```

Result:

```text
active [('stage4s_goalie_only_pass_pretrain', 150000)]
stage4s RewardConfig OK
```

---

## 2026-06-28 (PM) — Stage 4r pass receiver claimant handoff + pass-align budget

### Problem

Stage 4r fixed the pass-mask starvation bug: pass requests rose sharply, proving the dynamic
intercept gate no longer blocks exploration. The next failure moved downstream:

- pass requests were now visible, but request→fire conversion stayed very low;
- fired passes rarely resolved because the intended receiver remained the non-claimant supporter;
- the supporter mask kept the receiver `goto`-only, so it could stand near the receive target but
  could not `approach_ball` to collect a pass or recover a loose one;
- pass alignment still timed out under the physical 20 deg/s turn cap.

The important math for the second issue: at `0.1s/step` and `20 deg/s`, a turn command changes robot
heading by about `2 deg/step`. The previous active Stage 4r `pass_macro_max_align_steps=24` could
cover only about `48 deg`, but Stage 4 spawns use random headings over `[-180, 180]`, so many valid
pass attempts could need much more alignment time.

### Fix

Updated [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py):

- `_ball_claimant(...)` now treats an active pending pass as a temporary ownership transfer.
- While `_pending_passes` contains an in-window pass, the intended receiver becomes the claimant.
- That unmasks the receiver's ball-collection path (`approach_ball` when far; kick/dribble paths once
  close) instead of forcing it back to supporter `goto`.
- Active `post_pass_finish_macro` owners also become claimant, keeping observation/reward/mask role
  bits consistent while the receive-finish macro is running.
- Expired pending passes are ignored by the claimant handoff, so stale pass events do not trap the
  receiver as claimant forever.

Updated [configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):

- Active `stage4r_intercept_gate_v1` now uses `pass_macro_max_align_steps: 90`.
- Preserved historical Stage 4 rungs were left at their previous values.

### Validation

```text
conda run -n rcai python -m py_compile \
  ai_interface/envs/JAL_env.py ai_interface/envs/reward.py tests/test_stage4_support_pass.py

manual tests.test_stage4_support_pass runner: ran 30 tests, all passed
conda run -n rcai python tests/test_ball_action_recovery.py: 19 passed
conda run -n rcai python -m json.tool configs/ppo_jal_curriculum_config.json: OK
```

`tests/test_stage3_reward_rules.py` was not runnable in the current `rcai` environment because
`pytest` is not installed there.

---

## 2026-06-28 (PM) — Stage 4r sim-only inference team config

Added [team_config_2atk_goalie.json](../team_config_2atk_goalie.json) for evaluating the
`stage4r_intercept_gate_v1` checkpoint under `--env sim-only` with the **2 attackers vs.
goalie-only** matchup (`TritonBots` 2/0, `TeamB` 1/1 — one opponent, flagged as keeper, no field
defender). The existing root configs didn't cover this case: `team_config_2atk_1def.json` adds a
field defender (`TeamB` 2/1) and `team_config_stage2/3.json` only spawn a single attacker. Use it
with:

```bash
/opt/anaconda3/envs/rcai/bin/python launch_infer.py \
  models/ppo_jal_expandable/stage4r_intercept_gate_v1_complete.pt \
  --trainer ppo_jal --config configs/ppo_jal_curriculum_config.json \
  --stage stage4r_intercept_gate_v1 \
  --team_config team_config_2atk_goalie.json \
  --env sim-only --steps 3000
```

> Note: `--env sim-only` is the **external** rcssserver, which lacks the embedded catch-glue dribble
> patches — dribble carries won't reproduce the way they do under `sim-embedded`. This config is for
> watching pass/shoot behavior and the keeper matchup, not dribble evaluation.

---

## 2026-06-28 (PM) — Stage 4r v2: the intercept gate was over-rejecting and mask-starving pass exploration

### Problem (why "the model still won't explore `pass_to_teammate`")

The first Stage 4r gate (below) fixed the *concept* but was **mis-calibrated and over-rejected**, which
is strictly worse than the static gate it replaced: it made a legal pass target essentially never
exist, so the action mask zeroed `pass_to_teammate` ~99% of the time and the policy could not even
*sample* a pass — there is nothing to explore and no gradient toward passing.

Evidence (live, run `20260628_183635_702568`, confirmed `stage4r_intercept_gate_v1`):

- Over **485 episodes**: passes `requested=7`, `fired=1`. The policy almost never selects pass.
- Aggregated `pass_mask_reasons` (why `mask[slot,5]=0` for the carrier): `not_kickable` 35,938,
  `too_close` 32,973, **`pass_interceptable` 22,338**, `bad_reception_cone` 19,893, `lane_low` 6,522.
- Aggregated supporter `target_reasons` (classification of the receive target): of 118k,
  **`ok` = 1.1%**; `too_close_for_ssl_pass` 46.5%, **`pass_interceptable` 41.2%**, `pass_lane_blocked` 10.4%.

A grid sweep over receive targets reproduced it exactly: with the defender **central** (its normal
defending position, directly between ball and goal) the intercept gate rejected **100%** of forward
passes (`legal-ok = 0` at every short-pass defender pose); only a defender already drifted ≥6u
off-centre left any passable target.

Two distinct bugs in `pass_intercept_margin` (mine, from the first 4r):

1. **Constant ball speed.** I modeled the pass at a fixed `pass_speed`. A real kick decays
   (`BALL_DECAY=0.94`): the ball covers 12u in ~11.7 steps, not the ~8.6 a constant model assumes,
   so late-flight time was under-counted.
2. **No directional filter (the big one).** `t_def` let *any* defender run to the nearest lane point.
   A defender **behind the ball** (trailing the play) or **behind the receiver** (it cannot reach a
   ball already past it) still produced a negative margin and vetoed the pass. Concretely, a defender
   at `(30,0)` *behind* a receiver at `(27,0)` returned margin −6.24 → "interceptable", which is
   physically nonsense. Because some real defender is almost always *somewhere*, this rejected nearly
   everything.

These compounded with `min_pass_distance = 12` (46.5% `too_close`), leaving the ~1.1% feasible window.

### Fix (code + config recalibration)

**Code** — [ai_interface/envs/stage4_support.py](../ai_interface/envs/stage4_support.py)
`pass_intercept_margin(...)`:

- **Decaying-speed ball model** via new helper `_ball_travel_time(distance, pass_speed, decay)`:
  treats `pass_speed` as launch speed `v0`, inverts the geometric-series travel
  `s(k)=v0·(1−decay^k)/(1−decay)` to get the true step count to each lane point (`decay≈1`
  degenerates to the old `distance/pass_speed`). Honest, slower late-flight `t_ball`.
- **Directional eligibility filter**: a blocker only counts at a lane point if its projection onto the
  pass direction lies between the ball and the receiver (±`catch_radius`). Defenders behind the ball
  or beyond the receiver no longer veto. Returns `+inf` when no blocker is ever eligible.

**Config** — `stage4r_intercept_gate_v1` recalibrated (a `min_intercept_margin` sweep showed a sharp
knee: `0.0` = gate effectively off, `~0.5–1.0` = robust feasible region, higher = diminishing):

- `pass_intercept_min_margin` 3.0 → **1.0**
- `support_min_pass_distance` & `pass_min_distance` 12 → **8**
- `support_forward_min` 12 → **14**, `support_forward_max` 18 → **20** (receiver clears the distance floor)

**Result** (grid sweep, config-wired through the real `env._classify_support_target`): legal-ok pass
fraction **1.1% → ~9%** overall, rising to **18–50 targets** per open-lane defender pose, while a
defender dead-on a short forward lane is still (correctly) rejected, and behind-ball / behind-receiver
defenders now correctly read as `ok`. Tests: `tests/test_stage4_support_pass.py` +2 (directional
filter, decaying speed); 28/28 pass, 62/62 across the stage3/4 + accounting + recovery suites.

> Watch on the next run: passes **REQUESTED** per episode must climb off ~0 first (exploration
> unblocked) — *then* RESOLVED rises and interception share of non-resolved passes falls. If REQUESTED
> stays ~0, the block is no longer the gate. If passing volume looks too loose, raise `min_margin`
> toward 1.5–2.0 (tune this knob, **not** the gate geometry).

---

## 2026-06-28 — Stage 4r Dynamic Intercept-Time Pass Gate (the real fix)

### Problem (true root cause of the whole stage4j–4q passing saga)

A dozen Stage 4 rungs (4i→4q) never produced coordinated passing — every run collapsed back to
solo dribbling (`pass_to_teammate` 0–2%), and reward/mask tuning never broke it. Diagnosis across the
runs showed **interceptions dominate every non-resolved pass** (stage4o: `104` interceptions vs `55`
resolved; 4m: `82` vs `50`). The carrier's value function correctly learned that passing loses and
kept reverting to dribbling.

The root cause is **not** reward shaping and **not** a geometric impossibility (an earlier claim that
fired passes had "lane quality 0.16–0.38" was wrong — that metric, `positional_gap_quality`, is the
carrier's open-goal SHOT angle, which is *correctly* low when the carrier passes because it can't
shoot; it says nothing about interception).

The actual defect is in the **pass-legality gate**. `classify_support_target(...)` scored the pass
lane with `lane_clear_quality(...)`, a **static, instantaneous** metric: the defender's *current*
perpendicular distance to the ball→target line, divided by `block_dist`. It is blind to ball-flight
time. A pass takes ~1–2 s to travel, during which the defender runs *onto* the lane and touches the
ball (interception is adjudicated at resolution as "opponent touches ball", a true dynamic-pursuit
event in [JAL_env.py](../ai_interface/envs/JAL_env.py) `_resolve` path). So the gate green-lit passes
that were physically doomed. No config knob fixes a time-blind metric — which is exactly why every
rung failed, and why Stage 4q (raising the *static* threshold `0.35 → 0.45`) made it worse: it just
suppressed passing (`12` fired vs 4p's `61`, macro fired `0×`) without changing intercept physics.

### Fix (code, not config)

Added a **dynamic interception-time gate** and a new active stage `stage4r_intercept_gate_v1`.

Added [ai_interface/envs/stage4_support.py](../ai_interface/envs/stage4_support.py) `pass_intercept_margin(...)`:

- Models the pass as the ball traveling start→target in a straight line at constant `pass_speed`
  (env units/step). For sampled points `P` along the lane, ball arrival is
  `t_ball = dist(start,P)/pass_speed`; a defender `D` can cut the lane near `P` at
  `t_def = max(0, dist(D,P) - catch_radius)/defender_speed`.
- Returns `min over P,blockers of (t_def - t_ball)`: **positive** means every defender reaches every
  lane point only *after* the ball passes (safe); **negative** means a defender can sit on the lane
  first (interceptable). `+inf` when no blockers / disabled.
- `classify_support_target(...)` now rejects targets as `"pass_interceptable"` when
  `margin < pass_intercept_min_margin` (skipped when the margin knob is `0`, preserving old behavior).

Wired through [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py) `_classify_support_target(...)`
and added four `RewardConfig` knobs in [ai_interface/envs/reward.py](../ai_interface/envs/reward.py):
`pass_intercept_ball_speed`, `pass_intercept_defender_speed`, `pass_intercept_catch_radius`,
`pass_intercept_min_margin` (all default `0` → gate disabled / behavior unchanged).

Stage 4r config:

- Reverts Stage 4q's counterproductive geometry/threshold changes back to the proven 4p values
  (support wedge `forward 12–18`, `lateral 12`, `y_clip 10`; static `support_pass_lane_min_quality`
  back to `0.35` as a coarse pre-filter; `pass_macro_max_align_steps 24`).
- Enables the new gate: `pass_intercept_ball_speed=1.4`, `pass_intercept_defender_speed=0.9`,
  `pass_intercept_catch_radius=0.9`, `pass_intercept_min_margin=3.0` (derived from
  `KICK_POWER_RATE=0.027`, `BALL_DECAY=0.94`, `PLAYER_DECAY=0.4`; all tunable as config).
- Warm-starts from the Stage 4o 160k checkpoint, not the failed 4p/4q policies.
- Marks `stage4q_pass_resolve_v1` failed/withdrawn (`timesteps: 0`).

### Validation

```bash
conda run -n rcai python -m py_compile \
  ai_interface/envs/stage4_support.py ai_interface/envs/JAL_env.py ai_interface/envs/reward.py
conda run -n rcai python -m json.tool configs/ppo_jal_curriculum_config.json   # OK
# active stages: [('stage4r_intercept_gate_v1', 200000)]
```

Manual test harness (rcai has no pytest):

```text
tests.test_stage4_support_pass: 26 passed, 0 failed   (incl. 2 new intercept-gate tests)
tests.test_ball_action_recovery: 19 passed, 0 failed
tests.test_stage3_reward_rules:  11 passed, 0 failed
```

The decisive new test proves the fix catches what the old code missed: a defender at `(28,3)` —
**off** the ball→target line so static `lane_clear=0.67` (gate-OFF classifies the pass "ok") — is
reachable in flight, so gate-ON correctly rejects it as `"pass_interceptable"`.

### Training Watchpoints

Success signal: `Pass RESOLVED` per episode rises **and the interception share of non-resolved
passes falls** (the gate should mostly eliminate doomed passes pre-launch). If passing volume drops
to ~0, `pass_intercept_min_margin=3.0` is too strict — lower it toward `1.0–2.0` (this is the first
knob to tune, not the geometry).

---

## 2026-06-28 — Stage 4q Pass-Resolution Fix (config-only) — WITHDRAWN

> Superseded by Stage 4r above. This config-only attempt raised the **static** lane threshold to fix
> interceptions, but the static metric is time-blind so it only suppressed passing without changing
> intercept physics (run `20260628_174911_977658`, stopped ~55%: `12` passes fired, macro fired `0×`,
> goals up only because the policy dribbled more). Kept for history; the entry below describes what
> was tried.

## 2026-06-28 — Stage 4q Pass-Resolution Fix (config-only)

### Problem

Stage 4p (run `20260628_171015_717398`, TRAINING.md §41) did not improve scoring and the
receive-finish macro it was built for almost never executed:

- Goal rate flat at `199/826 = 24.1%` (all solo-dribble goals; `dribble_to` 56–83%, `kick` 0–2%,
  `pass_to_teammate` 0–2%, supporter `goto=100%`).
- **Passes do not resolve:** `Pass FIRED=61`, `Pass RESOLVED=9` over the whole run (~15%).
- The pass-align macro is requested heavily (`requested=27–46/episode`) but `fired≈0` — it spins
  `align_steps≈27–46` per episode and re-latches without launching, draining the clock.
- The receive-finish macro **started 9 times in 826 episodes and fired 2 times**; finish banks
  totalled `banked=9, consumed=2, expired=5`.
- Fired pass aims cluster at the clamp edges `x=31.50` (`support_target_max_x`) and `|y|=10.0`
  (`support_target_y_clip`) with low `passer_lane_q≈0.21–0.29`, so even resolved passes bank ~0.

Root cause is **upstream of the finish macro**: the pass launch/align gate lets the carrier latch a
pass and spin for tens of steps without firing, and the support receive wedge projects deep/wide
into low-lane-quality space. The finish macro is validated (it fired twice, proving the plumbing) but
is starved because passes rarely complete.

### Fix

Config-only (the macro code from Stage 4p is correct and kept). Added `stage4q_pass_resolve_v1`,
marked `stage4p_receive_finish_macro_v1` failed (`timesteps: 0`). Warm-starts from the Stage 4o 160k
checkpoint, **not** the failed 4p policy.

Changes vs Stage 4p, all in
[configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):

- Fire fast or abandon: `pass_macro_max_align_steps` `60 → 18`. A geometric turn-to-face needs only a
  few steps; a latch that hasn't fired in 18 is chasing a bad/drifting target. Stops the 27–46
  step/episode spin.
- Pull receive targets out of the clamp edges so passes are reachable and lane-clear:
  - `support_forward_min` `12 → 8`, `support_forward_max` `18 → 14` (shorter lead → `tx` stops
    pinning at the x-clamp when the ball is already advanced);
  - `support_lateral_max` `12 → 8` (less wide → `ty` stops pinning at the y-clip, lane stays clearer);
  - `support_target_y_clip` `10 → 7`, `support_target_y_clip_x_min` `29 → 27` (harder cap on
    deep-wide);
  - `support_pass_receiver_max_target_dist` `9 → 7` (receiver must actually be near the target).
- Require completable, meaningfully-better passes instead of forcing low-quality ones:
  - `support_pass_lane_min_quality` `0.35 → 0.45`;
  - `stage4_force_pass_min_target_quality` `0.22 → 0.32`;
  - `stage4_force_pass_min_quality_gain` `0.05 → 0.10`.

Everything else (receive-finish macro, finish-bank quality scaling, pass power, urgency, reward EV)
is unchanged from Stage 4p.

### Validation

```bash
conda run -n rcai python -m json.tool configs/ppo_jal_curriculum_config.json   # OK
# active stages: [('stage4q_pass_resolve_v1', 200000)]
# load_model: models/ppo_jal_expandable/stage4o_forced_pass_finish_v1_steps160000.pt
```

### Training Watchpoints

The intended first sign of improvement is **not** higher pass volume — it is `Pass RESOLVED` per
episode rising and pass aims leaving the `x=31.5`/`|y|=10` clamp edges. Stop early (~50k) if:

- `fired ≈ 0` while `requested` is still high → the fix is still in the launch/align legality gate,
  not reward;
- `Pass RESOLVED` per episode stays near 0 → receive geometry still wrong;
- goal rate stays at ~24% with `dribble_to` dominant → still solo-dribble, passing not contributing.

---

## 2026-06-28 — Stage 4p Post-Pass Receive-Finish Macro

### Problem

Stage 4o proved that the team can sometimes create and resolve passes, but the receiver still did not
turn those resolved passes into shots:

- pass-finish credit was banked after pass resolution;
- the receiver was still controlled by the normal policy/mask path on following frames;
- if the receiver was not the current claimant, supporter `goto` behavior could take over again;
- the mask-only post-pass finish scaffold did not reliably handle receive, settle, aim, and kick.

The result was the failure mode we saw in logs: `finish_banked > 0` but `finish_consumed == 0`.

### Fix

Added a command-level per-robot receive-finish macro in
[ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py).

- A successful pass resolution now starts `post_pass_finish_macro[receiver_id]`.
- The macro runs before normal action decoding/masking, so the receiver can finish even if it is
  not currently the sticky ball claimant.
- Non-owner attackers keep normal role behavior.
- The macro expires after `post_pass_finish_macro_window_steps` (`80` by default, i.e. `8s` at
  `0.1s/step`).

Macro phases:

- `acquire`: if the receiver is outside kickable distance, use
  `approach_ball(..., obstacle_avoidance=True)`.
- `settle`: if kickable but outside the front reception cone, run a deterministic `dribble_to`
  reacquire/settle step toward the live ball.
- `settle`/staging: if the ball is in the cone but current `Q3(ball)` is poor, allow one short
  staging carry to the best local sampled point.
- `kick`: once kickable and inside the reception cone, reuse the existing keeper-away kick macro,
  reception cone, retargeting, and `kick_fired` accounting.

Staging target math:

- Candidate points are sampled from the ball toward goal with forward distances `{2, 4, 6}` env
  units and lateral offsets `{-4, 0, +4}`.
- Candidates are filtered inside the field, outside wide opponent penalty-area entry, and under
  `post_pass_finish_macro_staging_max_carry = 6.0`.
- Since SSL excessive dribbling is `10` env units (`1m`) and our normal safe segment cap is `8.5`,
  the macro staging carry is deliberately shorter than the training dribble cap.

Reward handling:

- Existing finish banks are still created on pass resolution.
- With `post_pass_finish_macro_enabled=true`, a bank is consumed only when the receiver fires a real
  macro kick (`post_pass_finish_macro_fired=true`).
- `post_pass_finish_macro_scale_reward_by_shot_quality=true` scales the bank payout by the shot
  quality at the fire frame, so a forced low-quality kick cannot collect the full pass-finish reward.
- Example: quality-`0.5` pass bank is `(14 + 12*0.5)/2 = +10.0`; a macro kick with fire-time shot
  quality `0.25` pays `+2.5`.

### Code Changes

Updated [ai_interface/envs/reward.py](../ai_interface/envs/reward.py):

- Added macro config knobs:
  - `post_pass_finish_macro_enabled`;
  - `post_pass_finish_macro_window_steps`;
  - `post_pass_finish_macro_min_shot_quality`;
  - `post_pass_finish_macro_staging_max_carry`;
  - `post_pass_finish_macro_scale_reward_by_shot_quality`.
- Added `support_pass_select_best_receiver` for Stage 5+ receiver selection.

Updated [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py):

- Added per-robot `post_pass_finish_macro` state.
- Added helpers:
  - `_start_post_pass_finish_macro(...)`;
  - `_execute_post_pass_finish_macro(...)`;
  - `_expire_post_pass_finish_macro(...)`;
  - `_post_pass_finish_staging_target(...)`;
  - `_post_pass_finish_shot_quality(...)`.
- Started the macro immediately after successful pending-pass resolution.
- Executed macro payloads before normal `kick`/`dribble_to`/`pass`/supporter decode branches.
- Added debug counters for macro started/acquire/settle/kick-align/fired/expired/bad-cone frames
  and average fire-time shot quality.
- Made `_support_pass_candidate(...)` optionally choose the highest-quality receiver when multiple
  legal receiver targets exist.

Updated [configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):

- Preserved `stage4o_forced_pass_finish_v1` with `timesteps: 0`.
- Added active `stage4p_receive_finish_macro_v1` for `200000` steps.
- Warm-starts from `models/ppo_jal_expandable/stage4o_forced_pass_finish_v1_steps160000.pt`.
- Uses `pass_finish_window_steps: 80`, `pass_macro_max_align_steps: 60`, macro enabled, and
  finish-bank quality scaling enabled.

Updated [tests/test_stage4_support_pass.py](../tests/test_stage4_support_pass.py):

- Added tests for macro start after pass resolution.
- Added tests that the macro overrides supporter `goto`.
- Added acquire, bad-cone settle, in-cone fire, scaled-payout math, expiry, and Stage 5-style
  best-receiver selection coverage.
- Added an `env.step(...)` integration test proving macro-fired kicks consume finish banks through
  the same shot-quality-scaled path training uses.

### Validation

```text
conda run -n rcai python -m py_compile \
  ai_interface/envs/JAL_env.py \
  ai_interface/envs/reward.py \
  tests/test_stage4_support_pass.py

conda run -n rcai python -m json.tool configs/ppo_jal_curriculum_config.json

manual tests.test_stage4_support_pass runner: ran 24 tests, all passed
active curriculum stages: [('stage4p_receive_finish_macro_v1', 200000)]
load_model: models/ppo_jal_expandable/stage4o_forced_pass_finish_v1_steps160000.pt
```

---

## 2026-06-28 — Stage 4o Debug-Only Coordination Diagnostics

### Problem

Before starting Stage 4o training, the existing logs could still leave an ambiguous failure:
we could see pass counts, but not whether the new forced pass/finish scaffold was actually being
entered, whether pass macros were timing out, or whether post-pass finish chances were being banked
and then expiring unused.

### Fix

Added debug-only episode counters in [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py).
These go to `train_log.log` because the file handler is `DEBUG`, but they do not print to the terminal
because the console handler is `INFO`.

- Counts forced Stage 4o mask decisions:
  `forced(pass=..., open_shot=..., post_pass_finish=...)`.
- Counts pass macro lifecycle:
  `started`, `align_steps`, `timeouts`, and `fallback_reasons`.
- Counts post-pass finish-bank lifecycle:
  `banked`, `consumed`, and `expired`.
- Changed support/pass diagnostic summaries and pass summaries from `info` to `debug`.
- Changed the `Pass→kick finish bonus` line from `info` to `debug`.

Look for:

```text
stage4o diagnostics
support/pass diagnostics
Episode N passes
Pass→kick finish bonus
```

### Validation

```text
py_compile: passed
manual regression harness: OK 45 tests
```

---

## 2026-06-28 — Stage 4o Forced Pass/Finish Scaffold + Approach-Ball Obstacle Fix

### Problem

Stage 4n completed but still did not teach coordinated attacking:

- Final goal rate was `10/799 = 1.3%`; last-100 goal rate was `1.0%`.
- `max_steps` was `684/799 = 85.6%`, so most episodes timed out.
- Passes existed (`Pass FIRED=101`, `Pass RESOLVED=34`), but `Pass→kick finish bonus=0`.
- Late categorical entropy was very low, and the policy collapsed back into carrier `dribble_to`.

Root causes:

- The model was asked to discover a long sparse chain: supporter target, carrier pass, receiver
  control, receiver shot. Reward-only shaping did not provide enough examples of the whole chain.
- The carrier could still choose raw `goto`/`turn`, even though those primitives are not useful for
  an in-possession carrier. They created no-progress escape hatches while the supporter was the only
  robot that should learn field-position `goto`.
- The pass mask used the latched support target, but pass execution could rebuild from the current
  same-step supporter action. This made “pass is valid” and “pass target used” inconsistent.
- `approach_ball` needed obstacle avoidance for recovery, but a naive avoidance call would treat the
  destination ball itself as an obstacle and steer away from the ball.

### Fix

Added Stage 4o, `stage4o_forced_pass_finish_v1`, and preserved Stage 4n as failed
(`timesteps: 0`).

- Carrier learned primitives are now scaffolded by role:
  - supporter remains `goto` only;
  - carrier raw `goto` and standalone `turn` can be masked;
  - carrier still recovers with `approach_ball`, carries with `dribble_to`, shoots with `kick`, and
    passes with `pass_to_teammate`.
- Useful pass windows can now force `pass_to_teammate`:
  - direct shot quality must be below `stage4_force_pass_max_direct_shot_quality`;
  - pass target quality must exceed `stage4_force_pass_min_target_quality`;
  - pass target quality must beat direct shot quality by `stage4_force_pass_min_quality_gain`.
- Open-shot windows can force `kick` when the carrier has a high-quality direct shot.
- After a resolved pass banks finish credit, the receiver can be temporarily forced into
  recover-or-kick behavior:
  - if not kickable, only `approach_ball`;
  - if kickable and inside the reception cone, only `kick`;
  - otherwise, only `dribble_to` to reorient/re-acquire.
- Pass execution can now use the same latched support target that made the pass mask/observation
  launchable.
- `approach_ball(...)` now uses obstacle avoidance by default, but calls `goto(..., avoid_ball=False)`
  so robot obstacles are avoided while the destination ball is not treated as an obstacle.
- PPO exploration is raised for the active run:
  - `target_kl: 0.01`;
  - `ent_coef_initial: 0.02`;
  - `ent_coef_final: 0.004`.

Math/logic check:

- A resolved quality-`0.5` pass now pays `(4 + 6*0.5)/2 = +3.5` immediately.
- The receiver finish bank pays `(14 + 12*0.5)/2 = +10.0` only if a shot follows.
- The full pass→shot chain is therefore `+13.5`, while a dead-end pass remains only `+3.5`.
- Interception remains `-16/2 = -8`, so pass EV stays negative unless the pass is likely to complete
  and lead to a finish.
- Dribble target rewards are reduced (`progress=0.05`, `quality=0.5`, achieved-gap `1.0`) so they no
  longer compete with the pass→finish chain.

### Code Changes

Updated [ai_interface/utils/basic_commands.py](../ai_interface/utils/basic_commands.py):

- Added `include_ball` to `build_avoid_points(...)`.
- Added `avoid_ball` to `goto(...)`.
- Made `approach_ball(...)` obstacle-aware by default while excluding the ball from avoid points.

Updated [ai_interface/envs/reward.py](../ai_interface/envs/reward.py):

- Added Stage 4 coordination scaffold knobs:
  `stage4_mask_carrier_goto`, `stage4_mask_carrier_turn`, `stage4_force_pass_window`,
  `stage4_force_pass_max_direct_shot_quality`, `stage4_force_pass_min_target_quality`,
  `stage4_force_pass_min_quality_gain`, `stage4_force_open_shot_finish`,
  `stage4_force_open_shot_quality`, `stage4_force_post_pass_finish`, and
  `stage4_use_latched_pass_target`.

Updated [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py):

- `get_primitive_valid_mask(...)` now applies the new carrier role masks.
- Added forced pass, forced open-shot kick, and forced post-pass receiver finish mask branches.
- Added `_has_post_pass_finish_bank(...)`.
- `_action_to_commands(...)` now can use latched support targets for pass launch/execution, keeping
  observation, mask, and executed pass target aligned.

Updated [configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):

- Marked `stage4n_safer_pass_finish_v1` as failed and set `timesteps: 0`.
- Added active `stage4o_forced_pass_finish_v1` for `300000` steps.
- Lowered Stage 4o dense dribble rewards and moved most pass value into receiver finish credit.
- Raised PPO entropy/KL settings for the active recovery run.

Updated [tests/test_stage4_support_pass.py](../tests/test_stage4_support_pass.py):

- Added regression tests for ball exclusion in approach avoidance.
- Added tests for carrier `goto`/`turn` masking.
- Added tests for forced pass-window masking.
- Added tests for forced post-pass receiver finish masking.

### Validation

```bash
conda run -n rcai python -m py_compile \
  ai_interface/utils/basic_commands.py \
  ai_interface/envs/JAL_env.py \
  ai_interface/envs/reward.py \
  tests/test_stage4_support_pass.py

conda run -n rcai python -m json.tool configs/ppo_jal_curriculum_config.json
```

Manual regression harness result:

```text
OK 45 tests
```

---

## 2026-06-28 — Stage 4n Safer Pass + Finish-Gated Coordination

### Problem

Stage 4m did not improve scoring. It solved the previous sparse-pass issue, but created a new failure:
passes became common without becoming useful football.

- Latest Stage 4m run stayed at `0/631 = 0.0%` goals.
- `pass_available_steps` became healthy (`71819`, about `31%` of env steps), so observability and
  role-relative support worked.
- Passes fired (`186`, about `0.295/episode`) but only `35/186 = 18.8%` resolved.
- Direct shooting collapsed to `53` kicks over `631` episodes (`0.084/episode`).
- Many pass targets were too deep and wide, commonly clamped near `x=33.5` with large `|y|`, which
  sent the ball toward the opponent penalty area or a dead wide receiver.
- Resolved passes were still rewarded immediately even when the receiver did not shoot afterward.

Root cause: Stage 4m rewarded “make pass available / complete transfer” more than “create and finish a
better scoring chance.” The supporter target manifold was too aggressive near the opponent defense
area, and the carrier was allowed to pass even when its current direct shot was already as good as the
support target.

### Fix

Added Stage 4n, `stage4n_safer_pass_finish_v1`, and preserved Stage 4m as failed (`timesteps: 0`).
Stage 4n starts again from the Stage 3 v3 defender checkpoint, not from the failed Stage 4m policy.

- Support role-relative target decode now has optional safety clamps:
  - `support_target_max_x` caps receive targets before the penalty-area edge;
  - `support_target_y_clip` with `support_target_y_clip_x_min` prevents deep wide targets;
  - Stage 4n uses `support_target_max_x=31.5`, `support_target_y_clip=10.0`, x gate `29.0`.
- Pass launch now compares the pass target against the current direct shot:
  - if current `Q3(ball) >= support_pass_current_shot_lock_quality`;
  - then pass is masked unless target quality exceeds current shot quality by
    `support_pass_min_quality_gain_over_shot`.
  - Stage 4n uses `0.38` and `+0.12`.
- Pass power is configurable and lower in Stage 4n:
  - `pass_power = clip(base + per_unit * distance, min, max)`;
  - Stage 4n uses `min=35`, `base=30`, `per_unit=3`, `max=75`.
- Pass reward is now finish-gated:
  - immediate transfer credit is reduced to `possession_transfer_bonus=4`,
    `pass_quality_weight=6`;
  - a resolved pass banks finish credit for the receiver;
  - the larger `pass_finish_bonus + pass_finish_quality_weight * quality` pays only if the receiver
    kicks within `pass_finish_window_steps`;
  - Stage 4n uses a 20-step finish window and `10 + 10 * quality`.
- `pass_available_bonus` is reduced to `0.02` and capped at 6 steps, so availability remains visible
  but cannot dominate shooting.
- Open-shot urgency is slightly stronger and earlier (`quality=0.38`, grace `6`, step `0.14`) to
  restore direct-shot frequency.

Math check:

- Old Stage 4m quality-`0.5` resolved pass paid `(14 + 18*0.5)/2 = +11.5` team reward immediately,
  even if the receiver never shot.
- Stage 4n pays only `(4 + 6*0.5)/2 = +3.5` immediately.
- The finish bank pays `(10 + 10*0.5)/2 = +7.5` only when the receiver kicks within 20 steps.
- Total successful pass→shot chain remains `+11.0`, close to the old value, but a dead-end pass loses
  most of the reward.
- Interception stays `-16/2 = -8`, so pass EV is positive only when it has a realistic chance to lead
  to a finish.

### Code Changes

Updated [ai_interface/envs/reward.py](../ai_interface/envs/reward.py):

- Added Stage 4 support/pass knobs:
  `support_target_max_x`, `support_target_y_clip`, `support_target_y_clip_x_min`,
  `support_pass_current_shot_lock_quality`, `support_pass_min_quality_gain_over_shot`,
  `pass_power_min`, `pass_power_base`, `pass_power_per_unit`, `pass_power_max`,
  `pass_finish_window_steps`, `pass_finish_bonus`, and `pass_finish_quality_weight`.

Updated [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py):

- `_decode_support_goto_target(...)` now applies optional max-x and deep-y clamps.
- `_support_pass_candidate(...)` now rejects `direct_shot_better` when the carrier already has a
  comparable or better direct shot.
- `pass_to_teammate` uses configurable pass power instead of hardcoded `40 + 6*distance`.
- Added `_post_pass_finish_banks` and helpers:
  `_bank_post_pass_finish_reward(...)`, `_consume_post_pass_finish_reward(...)`, and
  `_age_post_pass_finish_banks(...)`.
- A resolved pass can now bank finish credit for the intended receiver, paid only on a later
  receiver kick inside the configured window.

Updated [configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):

- Marked `stage4m_role_relative_pass_v1` as preserved/failed and set `timesteps: 0`.
- Added active `stage4n_safer_pass_finish_v1` with the safer support target geometry, direct-shot pass
  lock, lower pass power, lower immediate pass reward, finish-gated pass reward, and stronger
  open-shot urgency.

Updated [tests/test_stage4_support_pass.py](../tests/test_stage4_support_pass.py):

- Added regression coverage for deep/wide support target clamps.
- Added regression coverage for `direct_shot_better` pass masking.
- Added regression coverage for post-pass finish bank payout.

### Validation

```bash
conda run -n rcai python -m py_compile \
  ai_interface/envs/JAL_env.py \
  ai_interface/envs/reward.py \
  ai_interface/envs/stage4_support.py \
  tests/test_stage4_support_pass.py

conda run -n rcai python -m json.tool configs/ppo_jal_curriculum_config.json
```

Manual regression harness result:

```text
tests.test_stage4_support_pass 11
tests.test_ball_action_recovery 19
tests.test_stage3_reward_rules 11
manual tests ok 41
```

### Training Watchpoints

Stop Stage 4n early around `50k` steps if:

- goal rate is still `0%`;
- kick rate stays below `0.4/episode`;
- `pass_fired` is above `0.3/episode` but `pass→kick` finish payout is rare;
- `ball_in_penalty_off_target` remains above about `10%` of episodes.

The intended first sign of improvement is not high pass volume. It is: direct kicks recover, pass
targets stop clustering near `x=33.5`, and resolved passes are followed by receiver shots.

---

## 2026-06-28 — Stage 4m Time-Crunch Passing Recovery

### Problem

The Stage 4k/4l recovery ladder still did not make the model pass. The latest completed run showed
that the policy was no longer crowding the ball, but the supporter almost never became a usable
receiver:

- Stage 4l goal rate stayed low: `33/817 = 4.0%`, last 100 episodes `7.0%`.
- Passes never actually entered play: `requested=1`, `fired=0`, `resolved=0`.
- `pass_available_steps` was only `236` over about `299,664` env steps:
  `236 / 299664 = 0.079%`.
- Support target modes were still almost entirely non-receive:
  `GENERAL_SUPPORT=98.19%`, `RECEIVE_TARGET=1.56%`, `RECEIVE_READY=0.25%`.

Root cause: the supporter was asked to discover valid pass locations from raw global `goto(Dx,Dy)`
coordinates over the full field. Most samples landed outside the small useful receive region, then
failed legality/quality gates (`opponent_defense_area`, `outside_field`, `quality_low`,
`receiver_far`, `bad_reception_cone`). Even when a target briefly became valid, the carrier had to
sample `pass_to_teammate` on that same frame and already be aligned well enough to fire. This made
coordinated passing a rare one-frame lottery, not a learnable behavior.

### Fix

Stage 4 now has a time-crunch recovery rung, `stage4m_role_relative_pass_v1`, focused on making pass
attempts common enough for PPO to learn from.

- Supporter `goto(Dx,Dy)` can now be decoded as a **role-relative receive wedge** instead of raw global
  field coordinates:
  - forward offset from ball/carrier direction to goal: `12..24` env units (`1.2m..2.4m`);
  - lateral offset: `±16` env units, with a minimum lateral separation of `6` units to avoid sitting
    directly on the carrier shot lane;
  - target clamped inside the field and just outside the opponent defense area.
- `pass_to_teammate` is now a committed macro:
  - first valid pass request latches receiver + target;
  - later policy frames cannot interrupt the pass alignment;
  - the carrier turns toward the latched target for up to `pass_macro_max_align_steps`;
  - once aligned and still physically legal, the pass fires and records the normal pending-pass event.
- Stage 4m relaxes pass discovery gates so fired passes can emerge:
  - `support_pass_lane_min_quality: 0.25`;
  - `support_pass_shot_min_quality: 0.10`;
  - `support_pass_receiver_max_target_dist: 12.0`;
  - `support_receive_ready_radius: 6.0`.
- Support shaping was strengthened but kept below idle-profit levels:
  - `support_receive_target_bonus: 0.12 * quality`;
  - `support_receive_ready_bonus: 0.20 * quality`;
  - `pass_available_bonus: 0.10 * quality`, capped to the first `10` launchable steps;
  - `support_bad_target_penalty` reduced to `0.03` so early exploration is not crushed.

Math check: with target quality `0.5`, a ready support point pays `0.20 * 0.5 = +0.10` to the
supporter, or about `+0.05` after two-robot team averaging. That offsets only half of the `-0.1`
step cost, so idle camping at a ready point is still not profitable by itself.

### Code Changes

Updated [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py):

- Added per-robot pass macro state:
  `pass_macro_active`, `pass_macro_receiver`, `pass_macro_target`,
  `pass_macro_support_info`, and `pass_macro_align_steps`.
- Added `_reset_pass_macro(...)`.
- Added `_decode_support_goto_target(...)`.
  When `support_role_relative_targets=true`, this maps the supporter's raw `Dx,Dy` into the tactical
  receive wedge ahead/lateral of the ball instead of absolute field coordinates.
- `_build_support_targets(...)` now classifies the role-relative target when enabled.
- `_action_to_commands(...)` now executes the same decoded role-relative target for supporter `goto`,
  so the location being rewarded is the same location the robot moves toward.
- `pass_to_teammate` now latches a valid support candidate and continues alignment across frames until
  it fires, times out, or loses physical legality.
- Pass diagnostics now include macro fields:
  `pass_macro_active`, `pass_macro_receiver`, `pass_macro_target`, and `pass_macro_align_steps`.
- Pass request counting now counts raw categorical `pass_to_teammate` selections only; macro
  continuation frames do not inflate `requested`.

Updated [ai_interface/envs/reward.py](../ai_interface/envs/reward.py):

- Added Stage 4 recovery knobs:
  `support_role_relative_targets`, `support_forward_min`, `support_forward_max`,
  `support_lateral_max`, `support_lateral_min_abs`, and `pass_macro_max_align_steps`.

Updated [configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):

- Marked `stage4k_support_receive_pretrain` as preserved/failed and set `timesteps: 0`.
- Marked `stage4l_pass_execute_v1` as preserved/failed and set `timesteps: 0`.
- Added active `stage4m_role_relative_pass_v1`:
  - `timesteps: 300000`;
  - warm-start remains
    `models/ppo_jal_expandable_wide_stage3_v3/stage3_defender_v2_finetune_complete.pt`;
  - `support_role_relative_targets: true`;
  - relaxed pass gates and strengthened setup rewards as listed above.

Updated [tests/test_stage4_support_pass.py](../tests/test_stage4_support_pass.py):

- Added regression coverage for role-relative support target decoding.
- Added regression coverage that one `pass_to_teammate` request latches the pass macro and later
  frames continue the pass alignment.

### Validation

```bash
conda run -n rcai python -m py_compile \
  ai_interface/envs/JAL_env.py \
  ai_interface/envs/reward.py \
  ai_interface/envs/stage4_support.py \
  tests/test_stage4_support_pass.py

conda run -n rcai python -m json.tool configs/ppo_jal_curriculum_config.json
```

Manual regression harness result:

```text
tests.test_stage4_support_pass 8
tests.test_ball_action_recovery 19
tests.test_stage3_reward_rules 11
manual tests ok 38
```

### Training Watchpoints

Stop the Stage 4m run early at about `50k` steps if either of these remains true:

- `pass_fired < 0.3/episode`;
- `pass_available_steps < 5%` of env steps.

If either fails, the next fix should come from the new pass macro/support diagnostics rather than
letting the full `300k` run complete.

---

## 2026-06-28 — Stage 4j Supporter Target + Passing Reward Upgrade

### Problem

Stage 4 training was not becoming a real `2 attackers vs defender + goalie` task. The latest logs
showed the carrier mostly behaving like a Stage 3 solo attacker while the second attacker became a
passive or poorly positioned teammate:

- The supporter had been masked away from ball actions, but there was no strong, explicit signal for
  *where* it should go.
- Passes could be attempted from carrier-side heuristics rather than from a supporter-selected receive
  coordinate, so the policy was not learning the supporter's `goto(Dx,Dy)` as the pass target.
- `pass_min_distance=3.0` meant a "pass" could be only `0.3m` in Division B units (`1m = 10 env
  units`), which is too short and rewards non-football possession shuffling.
- Successful pass credit was too weak and failed pass outcomes were mostly just event deletion, so
  interceptions and timeouts were not clearly negative.
- `dribble_achieved_gap_weight` was still directly payable, so the model could keep farming carry-angle
  reward without a finish.
- When the carrier had an open, high-quality shot/pass window, there was no urgency cost for waiting.

### Fix

Stage 4 now has an explicit Sumatra-inspired support-target system. The non-claimant still only gets
`goto`, but its decoded `goto(Dx,Dy)` is now treated as the proposed receive/support coordinate. The
environment classifies that point as:

- `RECEIVE_READY`: legal target, clear pass lane, useful continuation shot/pass quality, and receiver
  is within `2.5` env units.
- `RECEIVE_TARGET`: legal and useful receive point, but the receiver is still moving there.
- `GENERAL_SUPPORT`: not a valid receive target; still logged and shaped as general support, with bad
  targets penalized when they are illegal or block the carrier.

The carrier's `pass_to_teammate` primitive is now masked unless there is exactly one ready supporter
target and all physical/football gates pass:

- carrier has the ball in the front reception cone;
- support target lane clearance `>= 0.70`;
- support target quality `>= 0.35`;
- carrier-to-target and ball-to-target distance `>= 12.0` env units (`1.2m`);
- target is inside field, outside the opponent defense area, not too close to opponents, not too close
  to the carrier/ball, and not blocking the carrier's shot lane.

When pass fires, it uses the supporter's selected target, not a separate carrier-generated lead target.
The target is hysteresis-stabilized for `8` steps unless it becomes illegal, preventing rapid ready/not
ready toggling.

### Code Changes

Added [ai_interface/envs/stage4_support.py](../ai_interface/envs/stage4_support.py):

- Defines `SupportTargetInfo` and modes `RECEIVE_READY`, `RECEIVE_TARGET`, `GENERAL_SUPPORT`.
- Implements `classify_support_target(...)` with the Stage 4 legality checks:
  field bounds, opponent defense area, minimum SSL pass distance, opponent clearance, carrier shot-lane
  blocking, pass-lane clearance, and continuation quality.
- Uses existing `lane_clear_quality(...)` and `positional_shot_quality(...)` so pass/support scoring
  stays consistent with the Stage 3 defender-lane and goalie-gap math.

Updated [ai_interface/envs/JAL_env.py](../ai_interface/envs/JAL_env.py):

- Added support target state:
  `self._last_support_targets`, `self._support_target_hysteresis`,
  `self._prev_support_target_dist`, `_open_shot_ready_steps`, `_open_shot_penalty_total`, and
  `_banked_dribble_gap_rewards`.
- Added `_stage4_support_enabled()`, `_classify_support_target(...)`,
  `_build_support_targets(...)`, `_ready_support_pass_target(...)`, `_shot_quality_at_point(...)`,
  `_consume_banked_dribble_gap_reward(...)`, and `_age_banked_dribble_gap_rewards()`.
- `get_primitive_valid_mask()` now keeps supporters `goto`-only and masks claimant `pass_to_teammate`
  unless a current ready support target exists and the claimant can physically pass from the front cone.
- `_action_to_commands()` now:
  - builds support targets from the supporter's decoded `goto` params each step;
  - forces any non-claimant non-`goto` sample to execute its decoded `goto` target instead of falling
    back to `turn 0`;
  - routes `pass_to_teammate` through the ready support target when Stage 4j support gating is enabled;
  - records pass diagnostics: target, mode, quality, lane clearance, pass-ready flag, and failure
    reason.
- Pass resolution now applies configured failed-pass penalties:
  interception `-16 total`, timeout `-10 total`, pass-caused off-target penalty-area terminal `-18 total`.
  Since team reward is averaged over two robots, these are about `-8`, `-5`, and `-9` scalar training
  reward respectively.
- Successful pass reward now uses:
  `possession_transfer_bonus + pass_quality_weight * target_quality`, with Stage 4j values
  `12 + 16*q`. For `q=0.5`, this is `20 total`, or about `+10` scalar team reward.
- Positive achieved-gap dribble credit is banked for `20` steps and only paid if followed by a kick or
  resolved pass. Negative achieved-gap remains immediate, so bad carries are still punished.
- Added open-shot urgency: if the claimant has `Q3(ball) >= 0.45` and can legally kick/pass for more
  than `10` steps, it pays `-0.12` per extra step, capped at `4.0`.
- Fixed an existing achieved-gap bug while touching this path: the non-defender keeper-zone branch
  referenced undefined `ag_ball_point`; it now uses the actual current ball point.

Updated [ai_interface/envs/reward.py](../ai_interface/envs/reward.py):

- Added Stage 4 support/pass config fields:
  `support_pass_gate_enabled`, `support_target_progress_weight`,
  `support_target_progress_clip`, `support_receive_ready_bonus`,
  `support_bad_target_penalty`, `support_min_pass_distance`,
  `support_receive_ready_radius`, `support_pass_lane_min_quality`,
  `support_pass_shot_min_quality`, `support_target_hysteresis_steps`,
  `pass_interception_penalty`, `pass_timeout_penalty`, `pass_off_target_penalty`,
  `dribble_achieved_gap_finish_window`, and open-shot urgency fields.
- Added reward inputs/intermediates for support target, previous distance, mode, quality, and bad-target
  flag.
- Non-claimant reward now pays:
  `clip(prev_dist - current_dist, +/-1.0) * support_target_progress_weight`,
  plus `support_receive_ready_bonus * target_quality` while ready,
  and subtracts `support_bad_target_penalty` for illegal/bad targets.

Updated [configs/ppo_jal_curriculum_config.json](../configs/ppo_jal_curriculum_config.json):

- Preserved old `stage4i_2atk_1def` with `timesteps: 0`.
- Added active `stage4j_support_pass_v1` with:
  - `timesteps: 300000`;
  - attacking-half curriculum `ball x=[10,34]`, `y=[-15,15]`;
  - `support_pass_gate_enabled: true`;
  - `support_target_progress_weight: 0.4`;
  - `support_receive_ready_bonus: 0.08`;
  - `support_bad_target_penalty: 0.05`;
  - `support_min_pass_distance: 12.0`;
  - `pass_min_distance: 12.0`;
  - `possession_transfer_bonus: 12.0`;
  - `pass_quality_weight: 16.0`;
  - pass failure penalties;
  - `dribble_achieved_gap_weight: 2.0` and `dribble_achieved_gap_finish_window: 20`;
  - open-shot urgency settings.
- Updated top-level warm-start to:
  `models/ppo_jal_expandable_wide_stage3_v3/stage3_defender_v2_finetune_complete.pt`.

Updated tests:

- Added [tests/test_stage4_support_pass.py](../tests/test_stage4_support_pass.py):
  support target classification, blocked-lane/defense-area rejection, primitive mask gating, pass
  decode using the supporter's selected target, and reward math checks.
- Updated [tests/test_ball_action_recovery.py](../tests/test_ball_action_recovery.py):
  non-claimants are now expected to execute `goto` as supporters instead of falling back to `turn 0`;
  non-claimant `turn` remains masked.

### Validation

`rcai` contains `gymnasium`, but it does not currently include `pytest`, so the test functions were
run directly under `rcai` with a tiny `pytest.approx` stub for the Stage 3 rule tests:

```bash
conda run -n rcai python -m py_compile \
  ai_interface/envs/JAL_env.py \
  ai_interface/envs/reward.py \
  ai_interface/envs/stage4_support.py \
  tests/test_stage4_support_pass.py \
  tests/test_ball_action_recovery.py \
  tests/test_stage3_reward_rules.py

conda run -n rcai python -m json.tool configs/ppo_jal_curriculum_config.json
```

Manual assertion harness result:

```text
tests.test_stage4_support_pass 5
tests.test_ball_action_recovery 19
tests.test_stage3_reward_rules 11
manual tests ok
```

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

---

## 2026-06-26 IST — Stage 4i intermediate stage: 2-attacker coordination + passing

### Context

Stage 4 (`stage4_2v2`, 2 RL attackers vs GK + 2 defenders) failed (§38): ~0% goals, sim bricks,
and no coordination — the 1→2 jump and a 3-defender block were both new at once. `stage4i_2atk_1def`
is the documented-safe intermediate rung: **2 RL attackers (both warm-started from
`final_models/stage3_complete.pt`) vs ONE scripted defender + goalie**, adding a real passing
primitive and a coordination reward so only coordination is new. The first run of it reproduced the
§38 failure mode (0 goals, `kick=0%`, recurring `ball_teleport` bricks), which the brick + reward
fixes below address.

### Changes — new passing primitive + coordination

- Added `pass_to_teammate` as a 6th discrete primitive (`PRIMITIVE_NAMES` → index 5, `NUM_PRIMITIVES`
  6 ≤ `NUM_PRIMITIVES_MAX` 12, no network rebuild; warm-starts cleanly from the reserved logit row).
  It consumes **one** live param — the `Dx` slot reused as a `lead` for **through balls**
  (`PRIMITIVE_PARAM_DIMS["pass_to_teammate"] = (0,)`). Receiver choice stays env-side (scalable to
  N teammates for 6v6). `ai_interface/algorithms/ppo_jal.py`.
- Wired pass execution into `JALTeamEnv._action_to_commands`
  ([JAL_env.py:2589](ai_interface/envs/JAL_env.py#L2589)): gates on can-kick + front reception cone;
  `_select_pass_target` picks the best **open** teammate (lane clear of opponents) and a through-ball
  target `receiver + lead·dir(receiver→goal)` where `lead = (Dx_raw·0.5+0.5)·pass_lead_max`; turns to
  face the target geometrically across steps, then fires `pass_to_teammate(...)` with power scaled by
  pass distance. Records a pending-pass event for reward resolution.
- Replaced the **exploitable** carrier-flip `possession_transfer_bonus` (two robots could oscillate
  possession for unbounded bonus) with a **tracked pass-event resolution** (`self._pending_passes`).
  A pass resolves only when the intended receiver gains the ball, the ball travelled ≥
  `pass_min_distance`, within `pass_window_steps`, and no opponent touched it in between — through-ball
  run-ons still credit, jostling never creates an event. New `RewardConfig` fields
  ([reward.py:269](ai_interface/envs/reward.py#L269)): `pass_quality_weight` (× `clip(receiver_lane_q −
  passer_lane_q, 0, 1)`), `pass_min_distance`, `pass_window_steps`, `pass_lead_max`.
- Added obstacle avoidance to dribbling. `dribble_to(..., obstacle_avoidance=True,
  obstacle_radius=0.9, obstacle_detour_margin=1.0)` ([basic_commands.py:408](ai_interface/utils/basic_commands.py#L408)):
  the CARRY branch now builds opponent obstacles and dashes toward a `_select_detour` waypoint
  (body-relative, so the glued ball is not orbited) instead of straight at the target. Open-space
  carries unchanged. `goto`'s existing avoidance is left on.
- Added `team_config_2atk_1def.json` (TritonBots ×2 vs TeamB GK + 1 defender) and the
  `stage4i_2atk_1def` curriculum stage (`num_robots 2`, stage-3 defender + goalie as `aux_team_policies`,
  `disabled_actions: []`, warm-start `final_models/stage3_complete.pt`, `timesteps 300000`; all other
  stages `timesteps 0`).

### Changes — Fix A: brick recovery (engine rebuild on brick terminal)

`ball_teleport` / `frozen_state_stale_sim` bricks recurred (21 in the last 60 episodes of the first
run) because `EmbeddedSimulatorBackend.reset`'s two rebuild triggers (sticky referee playmode,
latched `_ball_caught_by`) miss some latch sources, so the cheap soft-reset ran and the next episode
bricked again (1-step episodes). Added a symptom-level backstop that forces a full engine rebuild
whenever the previous episode ended in a brick state:

- `JALTeamEnv` records `self._last_terminal_reason` each step; new class constant
  `_SIM_REBUILD_TERMINAL_REASONS = ("ball_teleport", "frozen_state_stale_sim")`; `reset()` computes
  `force_rebuild` and clears the flag after. `ai_interface/envs/JAL_env.py`.
- Threaded `force_rebuild` through `Networker.reset_sim` → `Commander.reset_sim` →
  `EmbeddedSimulatorBackend.reset`, where it is OR'd into the existing rebuild condition (reuses
  `_initialize_simulator()`). `networking/networker.py`, `networking/socket_utils.py`.

### Changes — Fix B: kick-collapse reward retune (stage4i overrides)

The first run hit `kick=0%` / 0 goals because the carrier was **paid to hold and dribble**
(`has_ball_bonus 0.08`/step + `dribble_active_bonus 0.03` + `dribble_target_progress_weight 1.0`,
clip ±0.5 → up to ~+0.2–0.5/step risk-free) while Stage 3's **shot-release pressure was dropped**
(`carry_urgency` all 0). "Dribble forever" beat the gated, penalty-risking kick. Restored Stage-3's
proven structure in `stage4i_2atk_1def.reward_config_overrides` (`configs/ppo_jal_curriculum_config.json`):

- `has_ball_bonus` 0.08 → **0.0**, `dribble_active_bonus` 0.03 → **0.0** (stop paying to camp on the ball).
- `dribble_target_progress_weight` 1.0 → **0.3** (per-step dribble cap drops 0.5→0.15).
- restored `dribble_achieved_gap_weight` **5.0** (one-shot at carry-close; credits only carries that
  actually open an angle).
- restored `carry_urgency_grace_steps 25` / `carry_urgency_penalty_per_step 0.03` /
  `carry_urgency_penalty_max 2.25` (per-`rid`, fires only while a dribble session is active; −0.03/step
  after grace, cumulative `min(2.25, 0.03·late_steps)`).
- restored `keeper_zone_radius 4.0` / `keeper_zone_floor 0.0` (scales gap-based kick/dribble/achieved-gap
  rewards floor→1 over 4 m from the keeper; stops over-dribble into the catch zone).
- `goal_progress_weight` left at 2.0 so legitimate goalward dribbling still pays — only the gratuitous
  hold/dribble camp bonuses are removed.

### Changes — Fix B (cont.): role-gating bug found during verification

`enable_role_gating` was at its default **False** for `stage4i` (only `stage5_3v3` set it). At
[reward.py:950](ai_interface/envs/reward.py#L950) that makes `is_chaser` always-true for both robots,
so `spread_bonus`/`support_position_bonus` (gated `if not is_chaser:`) **never fired** (supporter
positioning reward inert) and `near_ball_bonus` was paid to both robots (rewarding crowding). Added
`"enable_role_gating": true` to the stage4i overrides so the carrier/supporter role split actually
works. **Any multi-robot coordination stage must set this true.**

### Validation

```bash
/opt/anaconda3/envs/rcai/bin/python -c "import ai_interface.envs.JAL_env, networking.networker, networking.socket_utils; print('imports OK')"
/opt/anaconda3/envs/rcai/bin/python -c "import json,dataclasses; from ai_interface.envs.reward import RewardConfig; ov=json.load(open('configs/ppo_jal_curriculum_config.json'))['curriculum']['stage4i_2atk_1def']['reward_config_overrides']; RewardConfig(**ov); print('RewardConfig built OK')"
/opt/anaconda3/envs/rcai/bin/python train.py --trainer ppo_jal_curriculum --config configs/ppo_jal_curriculum_config.json --env sim-embedded --timesteps 3000   # 3k-step embedded smoke
```

Result:

```text
imports OK
RewardConfig built OK
Smoke: 401 episodes, no Traceback/NaN; carry_urgency firing per math
  ("Carry urgency penalty -0.03 (rid=1 age=34 grace=25 total=0.27/2.25)");
  Fix A "Forcing embedded engine rebuild" fired 4× and the run continued.
```

### Changes — Fix C: claimant-follow stale-sim false positives + debug logging

The first robot-agnostic hybrid smoke (`20260629_233341_053084`) was stopped at 6,711 / 200,000
steps because `frozen_state_stale_sim` dominated outcomes: 37 / 52 episodes (71.2%). The pattern was
not consistent with a true embedded-sim freeze: many exits were short episodes dominated by
`hardcoded_interlude_hold`, for example ep24 ended after 43 steps with `hardcoded_interlude_hold=76%`.

Root cause: claimant-follow remaps the single PPO slot to one physical attacker each step by mutating
`self.robot_ids = [active_robot_id]`. During a hardcoded-pass interlude the active PPO robot is
intentionally held (`turn 0`) while the auxiliary hardcoded provider drives the other attacker. The
frozen-state detector only checked `self.robot_ids`, so it could decide the sim was stale by observing
only the held PPO slot and ball, ignoring the other controlled attacker.

Fixes:

- Added `JALTeamEnv._controlled_team_robot_ids()`. It returns the full `_robot_pool` (`[1, 2]`) when
  claimant-follow is active, otherwise the normal `robot_ids`.
- Updated frozen-state detection to check the full controlled pool in claimant-follow mode.
- Updated robot out-of-bounds / teleport checks to use the same controlled pool.
- Updated `SSLRuleTracker.update(...)` to receive the controlled pool, so supporter collisions and
  touches are not missed when the PPO slot is remapped.
- Added DEBUG-only claimant-follow gate logs whenever mode / active robot / carrier / receiver changes.
- Added DEBUG-only frozen-terminal logs showing checked robot IDs, active robot, PPO-control state,
  gate info, ball position, and playmode.
- Changed `BaseTrainer` logging so `train_log.log` receives DEBUG records while the terminal remains
  INFO-only. This preserves detailed stage-4 diagnostics without flooding the console.
- Added regression coverage in `tests/test_hardcoded_supporter.py` for the full controlled-pool helper.

Validation:

```bash
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/envs/JAL_env.py ai_interface/trainers/base_trainer.py tests/test_hardcoded_supporter.py
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/test_hardcoded_supporter.py
```

Result: `tests/test_hardcoded_supporter.py` passed (`8 passed`).

### Changes — Fix D: claimant-follow command routing to physical robot IDs

The follow-up smoke (`20260629_234305_835646`) still ended mostly with
`frozen_state_stale_sim`: 80 / 110 episodes (72.7%). The new DEBUG logs showed almost every stale
terminal occurred in `mode=hardcoded_pass`, `ppo_control=false`, with `active_robot_id=2`.

Root cause: the previous fix made the frozen detector check both controlled attackers, but it did not
fix command delivery. In claimant-follow mode the env mutates `self.robot_ids` to the single physical
robot currently mapped to the PPO slot, for example `[2]`. `_action_to_commands()` therefore returned
a compact one-command list such as `["turn 0"]`. The simulator command path is positional by uniform
number: list index 0 commands robot 1, list index 1 commands robot 2. So when PPO was mapped to robot
2 during a hardcoded-pass interlude, the env's intended receiver hold command was sent to robot 1
instead. That could overwrite the auxiliary hardcoded carrier command, leaving both the ball and the
pass carrier effectively stalled until the stale detector ended the episode.

Fixes:

- Added `JALTeamEnv._commands_for_current_robot_ids(...)`.
- `_send_commands(...)` now pads compact env commands to simulator unum positions when
  `robot_ids` is non-contiguous or dynamically remapped. Example: `robot_ids=[2]` plus `["turn 0"]`
  becomes `[None, "turn 0"]`.
- `_preprocess_commands_for_send(...)` now preserves `None` gaps instead of converting them to
  `turn 0`, allowing the simulator sender to leave non-owned robots untouched.
- Added regression coverage that active PPO robot 2 serializes as `[None, "turn 0"]`, not
  `["turn 0"]`.

Validation:

```bash
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/envs/JAL_env.py tests/test_hardcoded_supporter.py
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/test_hardcoded_supporter.py
```

Result: `tests/test_hardcoded_supporter.py` passed (`9 passed`).

### Changes — Fix E: possession-gated hardcoded-pass takeover

The next embedded smoke (`20260630_001040_645855`) proved command routing was fixed but exposed a
higher-level gate bug. Stale endings were still 47 / 59 episodes (79.7%), and the DEBUG lines showed
they occurred in `mode=hardcoded_pass`, `ppo_control=false`, with `active_robot_id=2`. The hybrid gate
was switching to hardcoded-pass mode from geometry alone: carrier lane blocked, teammate lane better,
pass lane open. It did not require the carrier to actually have usable contact with the ball.

Problem this solves:

- If the carrier was still far from the ball, entering hardcoded-pass mode paused PPO recovery.
- The PPO slot was held while the auxiliary hardcoded logic did not necessarily recover the loose
  ball.
- Ball and controlled attackers could stay nearly static until `frozen_state_stale_sim`.

Fixes:

- Added `JALTeamEnv._claimant_follow_carrier_ready_to_pass(...)`.
- `_update_claimant_follow_mapping(...)` now enters `hardcoded_pass` only when the carrier is within
  `kickable_dist + 0.25` env units of the ball. This matches the hardcoded supporter's existing
  `_has_usable_ball(...)` tolerance.
- If the geometric pass gate is favorable but the carrier is not physically ready, PPO remains in
  `solo_finish`/recovery mode and the gate reason becomes `carrier_not_ready_to_pass`.
- Gate diagnostics now include:
  - `carrier_pass_ready`
  - `carrier_ball_dist`
  - `carrier_ball_in_reception_cone`
- The gate does not require front-cone alignment. That is intentional: once the carrier has usable
  ball contact, the hardcoded pass helper can settle/reorient before kicking.
- Added regression coverage:
  - blocked lane + carrier at ball still switches to `hardcoded_pass`;
  - blocked lane + carrier far from ball stays PPO-controlled.

Validation:

```bash
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/envs/JAL_env.py tests/test_hardcoded_supporter.py
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/test_hardcoded_supporter.py
```

Result: `tests/test_hardcoded_supporter.py` passed (`10 passed`).

Embedded smoke result after the fix (`20260630_001519_500935`):

- stale endings: 47 / 59 (79.7%) -> 0 / 15 (0.0%);
- goal rate: 2 / 15 (13.3%);
- max-steps: 10 / 15 (66.7%);
- action mix: `approach_ball=23%`, `dribble_to=70%`, `kick=3%`, `turn=2%`.

The stale-freeze issue is fixed for the smoke. The remaining issue is performance/tactics: most
episodes still time out, and the gate can flicker near contested ball contact. Next likely fix is
hardcoded-pass hysteresis/timeout plus pass-event diagnostics, not more stale-sim handling.

### Changes — Fix F: hardcoded pass lifecycle diagnostics

Problem this solves:

- `hardcoded_interlude_hold` proved that PPO was being paused for hardcoded control, but it did not
  prove that a pass was actually fired, received, or converted into a shot.
- During the improving Stage 4 hybrid run, goals often contained some hardcoded interlude time, but
  the logs could not distinguish pass-assisted goals from solo finishes after gate flicker.
- We need this diagnostic data without interrupting the active training run.

Fixes:

- Added passive event classification to `HardcodedSupporter`:
  - `hc_support_move`
  - `hc_receive_intercept`
  - `hc_clearout_move`
  - `hc_pass_settle`
  - `hc_pass_align`
  - `hc_pass_fired`
  - `hc_shot_settle`
  - `hc_shot_align`
  - `hc_shot_fired`
  - `hc_staging_settle`
  - `hc_staging_align`
  - `hc_staging_fired`
- `HardcodedSupporterCommandProvider.last_event()` exposes the latest event label and details
  (`robot_id`, `main_attacker_robot_id`, target, quality, power, command, ball).
- Added `JALTeamEnv.record_external_action(...)`, which appends scripted-controller labels into
  the existing per-episode `action distribution` log. This means future episode lines can include
  hardcoded percentages like `hc_pass_fired=...%` and `hc_shot_fired=...%`.
- Terminal `info` now includes `external_action_counts` and an `external_action_events_tail`, so
  debug inference can summarize pass/shot lifecycle without scraping text logs.
- PPO curriculum training now records auxiliary hardcoded events into both:
  - the env episode action distribution;
  - the trainer rolling action summary.
- PPO inference now records the same auxiliary events into window/all action summaries and emits
  per-episode hardcoded action counts at DEBUG level.

This is instrumentation only. It does not change simulator commands, reward, masks, or control
decisions. The active training process that was already running will not pick this up; the next
training/inference process will.

Validation:

```bash
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/hardcoded_supporter.py ai_interface/trainers/policy_control.py ai_interface/trainers/ppo_jal_curriculum_trainer.py ai_interface/envs/JAL_env.py infer.py
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/test_hardcoded_supporter.py
```

Result: `tests/test_hardcoded_supporter.py` passed (`11 passed`).

### Changes — Fix G: env-owned hardcoded pass lifecycle

Problem this solves:

- The claimant-follow gate entered `hardcoded_pass`, but the same-team hardcoded supporter was still
  running the generic `_finish_or_return()` behavior. That routine shoots first and only returns a
  pass if the shot is poor, so forced-pass phases often produced `hc_shot_*`, `hc_support_move`, or
  `hardcoded_interlude_hold` instead of actual `hc_pass_*` events.
- Aux same-team commands were sent before `env.step()`, then the env sent the PPO command batch
  afterward. During hardcoded pass interludes, this made command ownership ambiguous and could
  overwrite the wrong robot.
- The receiver could be near the ball but still not get PPO finishing control, because there was no
  explicit stable-possession handoff after the pass.
- Receive/recover movement needed robot obstacle avoidance, but treating the ball as an obstacle
  makes the receiver detour around the ball it must collect.

Fixes:

- Added `ai_interface/hybrid_pass.py` with `HardcodedPassCoordinator`.
  - Owns the full lifecycle: `prepare/fire -> receive -> stable handoff -> miss/intercept/timeout`.
  - Emits carrier labels: `hc_pass_recover`, `hc_pass_settle`, `hc_pass_align`,
    `hc_pass_fired`, `hc_pass_clearout`.
  - Emits receiver labels: `hc_receive_hold`, `hc_receive_line`,
    `hc_receive_reposition`, `hc_receive_settle`.
  - Chooses pass power from a decaying-ball model using `BALL_DECAY=0.94` and the
    `4 m/s = 4 env units/step` ball-speed cap.
- Integrated the coordinator into `JALTeamEnv`.
  - While a hardcoded pass lifecycle is active, the env emits a positional command batch for both
    attackers, so carrier and receiver act in the same simulator cycle.
  - PPO transitions remain skipped during the interlude via the existing
    `ppo_control_active=false` path.
  - PPO is handed to the receiver only after stable front-cone possession for multiple frames.
  - Miss, intercept, missing-pose, or timeout returns control to PPO recovery/solo-finish mode.
- Suspended `HardcodedSupporterCommandProvider` while env-owned hardcoded pass is active.
  - Prevents pre-step same-team aux commands from fighting the env pass coordinator.
  - Prevents misleading `hc_support_move` counts during a pass lifecycle.
- Extended `goto()` / `approach_ball()` obstacle controls.
  - Movement can avoid players while ignoring the ball obstacle.
  - The pass settle and receive paths use robot avoidance without detouring around the ball.
- Added `docs/PASS_TO.md` documenting the complete hardcoded pass implementation.
- Added regression coverage:
  - hardcoded pass interlude emits `hc_pass_*` and `hc_receive_*`, not `hc_shot_*`;
  - successful receive maps PPO to the receiving robot;
  - `goto()` can avoid player obstacles without treating the ball as an obstacle.

Validation:

```bash
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/hybrid_pass.py ai_interface/envs/JAL_env.py ai_interface/utils/basic_commands.py ai_interface/trainers/policy_control.py infer.py
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/test_hardcoded_supporter.py
```

Result: `tests/test_hardcoded_supporter.py` passed (`14 passed`).

### Changes — Fix H: Stage-3-gated hardcoded attack bridge

Problem this solves:

- The hybrid 2v2 setup was handing the pass receiver to the Stage-3 PPO finisher immediately after
  a stable receive, even when the ball was outside the exact Stage-3 training envelope
  (`x=[25,34]`, `|y|<=10`).
- Latest inference showed many hardcoded passes were received in midfield or wide lanes
  (`target_x < 25` or `|target_y| > 10`), after which PPO wandered, over-dribbled, or aimed badly.
  That is not surprising: the Stage-3 model was trained as a close-range solo finisher against a
  defender + goalie, not as a full-field receiver/attacker.
- The old strategic pass gate still behaved like "pass only when the carrier lane is blocked",
  instead of "pass whenever the teammate option is materially better and the pass lane is open."

Fixes:

- Added `ai_interface/hardcoded_attack.py` with `HardcodedAttackCoordinator`.
  - Checks whether the current carrier state is truly Stage-3-ready before PPO takes over:
    `25 <= ball_x <= 34`, `|ball_y| <= 10`, controlled/kickable ball in the front reception cone,
    and minimum shot quality.
  - If not Stage-3-ready, the hardcoded bridge owns the carrier instead of PPO.
  - If a high-quality hardcoded shot is available, it shoots through the same physical front-cone
    checks.
  - Otherwise it stages the ball using short controlled moves: forward `{4,6,8}` and lateral
    `{-4,0,+4}`, with each candidate vector capped to the `8.5` env-unit SSL dribble safety limit.
  - Candidate staging points are scored by shot quality, Stage-3-envelope progress, lane clearance,
    goalward progress, and center bias.
- Integrated the bridge into `JALTeamEnv`.
  - Post-pass finish lock now chooses:
    - `solo_finish`/PPO only when Stage-3-ready;
    - `hardcoded_attack` when the receiver has the ball but is outside Stage-3 conditions.
  - Normal carrier mapping now also uses `hardcoded_attack` when a controlled carrier is outside
    Stage-3 conditions and no better hardcoded pass is active.
  - Hardcoded pass gating no longer requires the carrier shot lane to be below a fixed blocked-lane
    threshold; it passes when the teammate option is better by the configured margin and the pass
    lane is clear.
  - `hardcoded_attack` events are recorded into the existing action distribution/debug pipeline
    (`hc_attack_recover`, `hc_attack_stage`, `hc_attack_shot_*`, `hc_attack_clearout`).
- Updated `HardcodedSupporterCommandProvider` suspension so the aux supporter does not fight the
  env-owned hardcoded attack interlude.
- Added regression tests for:
  - outside-Stage-3 post-receive states using `hardcoded_attack`;
  - inside-Stage-3 post-receive states handing to PPO;
  - short staging targets staying under the dribble cap;
  - the existing pass/receive handoff still working under the new Stage-3 gate.

Validation:

```bash
/opt/anaconda3/envs/rcai/bin/python -m py_compile ai_interface/hardcoded_attack.py ai_interface/envs/JAL_env.py ai_interface/trainers/policy_control.py tests/test_hardcoded_supporter.py
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/test_hardcoded_supporter.py
```

Result: `tests/test_hardcoded_supporter.py` passed (`19 passed`).

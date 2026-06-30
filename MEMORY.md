# MEMORY.md — project memory (checked-in, agent-readable)

Hard-won, durable facts about this project: failure modes already diagnosed, what *not* to
re-try, and decisions that are settled. Both Claude (via `CLAUDE.md` instructions) and Codex
(via `project_doc_fallback_filenames` in `.codex/config.toml`) read this. Treat it as
always-applied context, alongside `CLAUDE.md`.

> Append a new bullet when a non-obvious fact is established (a diagnosed failure, a settled
> decision, a "don't re-try this"). Convert relative dates to absolute. Keep each entry one
> compact fact. Claude's richer, evolving notes live in
> `~/.claude/projects/.../memory/` — this file is the checked-in, Codex-readable subset.

## Settled architecture decisions

- **Centralized JAL is the design** — one team-level policy emits, per robot, a discrete
  primitive + continuous params, on a weight-shared per-robot encoder (robots interchangeable;
  goalie is hardcoded, not RL). Do **not** propose MAPPO/decentralized refactors.
- **PPO head is 12-wide** (12-primitive head + tanh-squashed bounded params) — this is the
  canonical "expandable-wide" arch. `origin/ai-train` forked an older 8-wide unbounded arch;
  checkpoints are cross-incompatible. 12-wide wins on merge.
- **Target league is RoboCup SSL** (real robots with dribbler bars). The 2D rcssserver is only
  a training proxy. The catch-glue "dribble" is deliberate dribbler-bar emulation, **not a bug** —
  evaluate dribble on `sim-embedded`, not `sim-only`. Do not rewrite toward 2D kick-and-chase.

## Environment / simulator pitfalls

- **Two sim backends behave differently.** `sim-embedded` is in-process and carries local engine
  patches (field-player catch/dribble glue, native `drop`); the external `rcssserver` does not, and
  only it honors `server.conf` overrides. Behavior depending on embedded patches (esp. catch-based
  dribble) won't reproduce externally. Watch train(embedded)/infer(external) mismatch.
- **Embedded sim "brick" failure** — referee set-piece modes (free_kick etc.) latch held-ball state
  and freeze the embedded sim; a mid-carry episode end can re-snap the ball to the holder
  (`ball_teleport`). Recovery requires rebuilding the engine on reset. Multiple variants have been
  found and patched (tracked `_ball_caught_by`, cone-rejected-kick catcher clear, force-rebuild on
  `ball_teleport`/`frozen_state_stale_sim` terminals). If bricks recur, harden
  `EmbeddedSimulatorBackend.reset`.
- **PPO JAL inference must not run deterministic-mean** (bang-bang turn stall). Keep
  `--ppo_param_noise_std > 0` (0.3 default) and prefer `--ppo_stochastic`. Argmax primitive is a
  separate failure (dribble-lock freeze). Inference reuses `JALTeamEnv`, so env-resident changes
  affect both training and inference.

## Curriculum / training mechanics

- The curriculum trainer runs **every** stage with `timesteps > 0` in sorted order — the `ACTIVE`
  tag only labels the log parser. To run a single stage, zero out all others. New `model_params`
  keys must be added to the trainer's hparam whitelist or they're silently dropped.
- **Multi-robot coordination stages must set `enable_role_gating: true`** — otherwise both robots
  read as chaser, supporter-positioning rewards never fire, and crowding is rewarded.
- **`ball_action_recovery` enables the anti-crowd machinery** (`_ball_claimant` sticky claimant +
  `get_primitive_valid_mask` + decode enforcement → non-claimant is goto-only). Any 2-attacker
  stage that wants the carrier/supporter split must set it (`ball_claimant_robot_ids`, switch
  margin, etc.), or both robots run the same selfish go-to-ball reflex and crowd.
- **2v2 hybrid stage is `stage4_hardcoded_support_2v2`** (renamed from `_2v3`; live matchup = 1 RL
  attacker + 1 hardcoded supporter vs goalie + 1 defender). The frozen Stage-3 solo finisher
  (`final_models/stage3_complete.pt`) dribbles-until-timeout here (11% goals, 37 kicks, 48%
  `max_steps`); **fine-tuning it 200k steps in this env with passing disabled
  (`claimant_follow_solo_lane_blocked_max: 0.0`) fixed it → 15.2% goals, 273 kicks, aim 0.18→0.33**
  (`models/ppo_jal_hybrid_support_2v2/...`, TRAINING.md §67). The fine-tune scores via more
  well-aimed 10–15m shots, not closer ones. The trainer's semi-MDP `ppo_control_active` gate
  (`ppo_jal_curriculum_trainer.py` ~L843) only stores a transition when PPO actually drove robot 1,
  so a hardcoded takeover never corrupts the gradient (reward is deferred to the next PPO step).

## Diagnosed failure modes (don't re-try)

- **Turn cap broke aiming** — a 20°/s turn cap (MAXMOMENT ±2) made incremental head-aiming
  diverge, kick atrophied, policy collapsed onto dribble_to. Fix is a geometric turn-to-heading
  macro (`face_then_kick`), **not** reward tuning.
- **Stage 4 carrier-won't-shoot** — across stages 4j–4n the claimant stays ~80% dribble / ~0%
  kick; passing can't work without a shooter. The fix direction is finish-gated pass reward +
  open-shot urgency that forces direct kicks, **not** more pass plumbing. First sign of recovery
  is direct kicks returning, not pass volume.
- **Dribble-target parameterization, not the keeper,** was the real dribble constraint
  (reaction-lag keeper made 0% difference). Pivot was goal-relative dribble target + end-of-carry
  achieved-gap credit.
- **Pass exploration starved by an over-strict gate** (6/28) — "model won't explore
  `pass_to_teammate`" was NOT a reward/value problem: the Stage 4r intercept gate
  (`stage4_support.pass_intercept_margin`) over-rejected, so a legal pass target existed only ~1.1%
  of steps and the mask zeroed pass ~99% of the time → the policy could not even sample it (live:
  485 eps, requested=7 fired=1). Two gate bugs: constant ball speed (real ball decays), and no
  directional filter (a defender *behind* the ball or *behind* the receiver still vetoed). Fix =
  decaying-speed model + directional eligibility filter + relax `min_intercept_margin` 3→1 /
  `min_pass_distance` 12→8. **Debugging order: confirm the action is being SAMPLED (read aggregated
  `pass_mask_reasons` / supporter `target_reasons` in the train log) before blaming reward** — if
  `target_reasons ok`≈1% and `requested`≈0, the env is forbidding the action, not the policy
  avoiding it.
- **Stage 4 full-defender passing is too sparse as a first pass curriculum** (2026-06-28) — the
  Stage 4r defender run after receiver-claim handoff still produced only 301 requested / 14 fired /
  3 resolved passes in 843 episodes and `pass_available_steps=0`; learn pass mechanics first in a
  goalie-only rung, then reintroduce defender pressure.
- **Stage 4 pass-to-finish failure became a low-level macro execution bug** (2026-06-29) — goalie-only
  Stage 4s produced many pass requests but almost no fired passes (`22,840` requested / `62` fired)
  and zero receiver finish shots because the pass macro's 10° align gate did not match the physical
  5° kick helper and receiver finish loop stayed in bad-cone/kick-align. Fix direction is
  deterministic contact-pose pass/finish macros, not more pass reward.
- **Stage 4t fake-pass regression** (2026-06-29) — carrying `dribble_to(target=receiver_target)` in
  pass `settle_contact` solved goalie-only alignment but broke 2v2 by moving the ball nearly to the
  receive target before firing (`release≈aim`), producing short crowding "passes" and zero resolved
  passes in valid embedded inference. Keep pass settle as a short front-cone touch and guard
  remaining flight distance with `pass_min_distance`.
- **Carrier pass fired ≈0% was a missing catch-glue, FIXED** (2026-06-30) — `hc_pass_fired` was ~0
  (2328 req / 0 fired in smoke) because `_carrier_pass_command` issued a bare `turn` to aim, spinning
  the body off the **un-glued** ball so it fell out of the mouth and the kick gate never passed. Fix:
  catch-glue align — `dribble()` face+`catch 0`, sticky `HybridPassState.carrier_caught`, geometric
  full-error `turn` while the glued ball revolves in the mouth, then real `kick` on alignment
  (0→~30 fires/6k steps). **Gotcha: once caught, do NOT re-grab** — a glued ball orbits with the
  body, so calling `dribble()` again chases that orbit forever (perpetual spin). `carrier_caught` is
  sticky until the recover branch (ball gone) resets it; do not add a transient `in_mouth`-False reset.

## SSL rule constraints (enforced in code, not reward)

- **Collisions are constrained at the movement-primitive layer**, not via reward penalties:
  `basic_commands.goto` has `_robot_proximity_speed_factor` ramping dash speed down when heading at
  another robot, covering SSL Crashing (8.4.2, >1.5 m/s closing) + Pushing (8.4.1). 6/30: the cap is
  enforced on **every** player-aware dash via `goto(crash_speed_cap=True)` — gated only on
  `include_player_obstacles`, NOT on `obstacle_avoidance` — so the defender's **full-speed INTERCEPT**
  (`defender._move_to(full_speed=True)` → `obstacle_avoidance=False`) is capped too; previously it
  bypassed the cap and banged in at speed 100. The **keeper is the only opt-out** (`goalie._goto`
  passes `crash_speed_cap=False`) so dives across the mouth aren't slowed. The cap **distinguishes
  opponents from teammates** (self team found by matching `self_pose` in `robot_poses`): opponents
  strict (`PROXIMITY_SLOW_DIST=3.8·PLAYER_SIZE`, floor `0.10`); teammates gentle
  (`PROXIMITY_TEAMMATE_SLOW_DIST=2.6·PLAYER_SIZE`, floor `0.45`). **DON'T use a 0.0 floor or apply the
  wide opponent cone to teammates** — that froze robots (deadlock) and pinned the supporter behind the
  main attacker (the floor/cone variant of the "second attacker waits" bug, 6/30). Floors are non-zero
  so robots always creep and never deadlock. `PROXIMITY_SLOW_DIST` was pulled 5.0→3.8·PLAYER_SIZE so a
  defender stays full-speed until ~3.4u and only decelerates in the final approach (the 4.5u start made
  intercepts feel sluggish); ramp+floor still keep contact closing speed low. The RL attacker's
  `approach_ball` sets `obstacle_avoidance=True`. Note `ssl_rule_events.py` ALSO has reward-side
  `detect_crash`/`detect_pushing` + `teammate_crash_penalty` — code constraint is primary; don't duplicate.
- **Defender "handed the ball to the attacker after catching it" = the `drop` fallback in
  `defender._clear()` (6/30).** The defender catch-glues then rotates (capped ~2°/cycle) to aim a
  clear; with `CLEAR_HOLD_CYCLE_LIMIT=8` / `CLEAR_CARRY_LIMIT=0.75` it couldn't point upfield in time
  and fell through to `drop`, releasing the glued ball AT ITS OWN FEET → free ball to the pressing
  attacker. FIX: replaced `drop` with a forward clear kick (`kick 70/85 0`, units-safe — pipeline does
  no rad/deg convert, only ever kick straight ahead); raised `CLEAR_HOLD_CYCLE_LIMIT 8→24` and
  `CLEAR_CARRY_LIMIT 0.75→1.4` so the geometric turn swings upfield before release. **A `drop` under
  pressure is never right — boot it away.** Verified: 6k smoke 29.2% goals, `hc_clearout_move=10%`, no drop.
- **Robot-robot touching/pushing in the 2D proxy is expected and not necessarily a foul.** SSL Crashing
  (8.4.2) is about CLOSING SPEED >1.5 m/s at collision; the proximity cap decelerates to a low contact
  speed (floor 0.10 opp / 0.45 teammate) so gentle contact while contesting the ball is fine. Zero
  contact is unachievable (can't press/contest without closing) and trades directly against
  intercept/press speed. Tune the cap only if the user explicitly wants less contact at the cost of slower pressing.
- **"Second attacker waits for the first to get the ball" was the claimant-follow interlude, NOT
  `goto` (6/30).** While the gate is in `hardcoded_attack` mode (carrier soloing, PPO not in control)
  the env interlude drives only the carrier and defaulted other pool robots to `turn 0`
  (`JAL_env.py` ~L2810), and `HardcodedSupporterCommandProvider` suspends itself for
  `mode in {hardcoded_pass, hardcoded_attack}`. So robot 2 idled until PPO retook control
  (`solo_finish`). Fix: the env now owns `self.hardcoded_supporter` and drives the **non-carrier** pool
  robot with it during a `hardcoded_attack` interlude (aux provider stays suspended → no double-drive;
  env drives robot 2 during the interlude, aux drives it during PPO control). `hardcoded_pass` already
  drives the receiver as robot 2, so its suspension MUST stay. Shared `JALTeamEnv` path → fixes train
  and infer. Verified: `hc_support_move` now fires from env_step 1 (was ~54).
- **Loose-ball approach vibration fix (6/30):** the attacker vibrated back-and-forth before
  reacquiring a contested loose ball because `goto`'s left/right detour side flipped every cycle
  (near-tie + FP noise). Fix: `DETOUR_SKIP_DIST = 3·PLAYER_SIZE` skips detour planning on the final
  approach (go straight, let the crash cap decelerate), and `_select_detour` biases a near-tie
  toward the robot's current `heading`. Don't reintroduce per-cycle unbiased detour near the target.
- **Lead pass around a blocked defender**: `HardcodedPassCoordinator._lead_target_around_defender`
  (called once in `start()`) shifts the pass target perpendicular away from an opponent sitting on
  the ball→receiver lane; the receiver follows via `_receive_pose`. Computed once at pass start
  (not re-evaluated as the defender moves) to keep the carrier's aim target stable.

## Conventions

- Run everything with `/opt/anaconda3/envs/rcai/bin/python` (the `rcai` conda env).
- Don't commit logs/checkpoints/non-code artifacts unless told. Commit only when asked. `git add`
  by name, never `git add .`. Never `git push --force`. Don't push under `robocup_downloads/` or
  `tests/`.
- For codebase questions, run `graphify query "<question>"` before grepping; `graphify update .`
  after editing code.
- Explain RL/reward/architecture simply (audience is an RL newcomer); define referenced symbols
  briefly; stay concise.

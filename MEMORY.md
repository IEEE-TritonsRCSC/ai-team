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

## Conventions

- Run everything with `/opt/anaconda3/envs/rcai/bin/python` (the `rcai` conda env).
- Don't commit logs/checkpoints/non-code artifacts unless told. Commit only when asked. `git add`
  by name, never `git add .`. Never `git push --force`. Don't push under `robocup_downloads/` or
  `tests/`.
- For codebase questions, run `graphify query "<question>"` before grepping; `graphify update .`
  after editing code.
- Explain RL/reward/architecture simply (audience is an RL newcomer); define referenced symbols
  briefly; stay concise.

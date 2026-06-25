# Plan: Stage 2 reward + curriculum to teach *selective* dribbling vs the keeper

## Context

Stage 1 produced a policy that approaches, turns, and kicks well (≈89% goal, `bad_aim`≈0)
but has **never used `dribble_to`** — it was `disabled_actions` for all of Stage 1. The
`dribble_to` mechanic itself is now hardened (P0/P1/P2 done: target latching, stall
watchdog, mid-carry abandon release). The remaining problem is *training*: teach the
policy to **dribble only when it buys a better shot at the keeper**, not every possession.

Desired behavior chain: approach ball → (turn toward a dribble direction if useful) →
dribble **only if it improves the shot** → turn to align → kick.

Decisions locked with the user:
- **Enforce selectivity via the EV fade, not a hard gate/mask** (config-only; add a
  counterfactual penalty later *only* if 2C shows over-dribbling).
- **Keep the 3-substage curriculum** (2A → 2B → 2C) already scaffolded in
  `configs/ppo_jal_curriculum_config.json`; refine weights + add acceptance gates.

## Answer: one stage or sub-stages? → **Three sub-stages.**

A single fixed reward cannot teach this. The reward must *change shape* over training:
- **Early**, dribble must be **encouraged** or the policy never explores it — a direct
  kick already scores 89%, so under a sparse/outcome-only reward `dribble_to` is strictly
  dominated and never sampled.
- **Late**, dribble must pay **only through the goal/combo chain**, or the dense
  exploration rewards get **farmed** (dribble every episode → the "dribble every time"
  failure the user wants to avoid).

You cannot hold both at once, so we **fade** the dense dribble rewards across three
substages while the quality-scaled combo bonus + the goal stay dominant.

## How the 5 desired behaviors are already covered (no new code)

| Behavior | Mechanism (existing) |
|---|---|
| Approach the ball | `approach_ball` primitive + `approach_weight` (learned Stage 1) |
| Turn toward dribble direction | Internal to `dribble_to` phase machine (faces latched target before carry) — not a reward term |
| Dribble only if better shot | `dribble_target_quality_delta` (one-shot, gated on `carry_started`, clamped ≥0) + `use_goalie_aim_gate`/`kick_into_keeper_penalty` make a bad-angle direct kick low-EV → *creates the need*; the fade makes a *useless* dribble net-negative |
| Turn to align before kicking | `alignment_weight=3.0` + `kick_aim_bonus_weight=20` via the Gaussian Dθ head (learned Stage 1) |
| Kick | `kick` primitive + quality-scaled `post_dribble_kick_bonus` ties dribble→kick as a unit |

The turning steps need **no reward term** — kick-aim turning is the Stage-1 Dθ skill, and
dribble-direction turning is internal to `dribble_to`. So selective dribbling is purely a
reward-economics + curriculum problem.

## Why the EV math already favors selectivity (validated against code)

`goalie_gap_quality`/`positional_gap_quality` (reward.py:223-264) return 0 at the keeper's
y / off-mouth, →1 at max lateral gap. With `use_goalie_aim_gate=true`, the kick-aim bonus
and the combo `scale` both use **gap** quality (JAL_env.py:769-786). So, schematically:

- **Bad-angle direct kick** (gap≈0.2): tiny aim bonus, `kick_into_keeper_penalty`, keeper
  saves → P(goal) low ⇒ EV ≈ marginal.
- **Dribble → good angle (0.2→0.7) → kick**: `quality_delta` one-shot (≥0, gated on real
  carry), dense `progress`, big quality-scaled combo (10×~0.5 in 2C), high P(goal)×70,
  minus step-cost over the carry ⇒ EV strongly positive.
- **Useless dribble** (gap stays ≈0.2): `quality_delta`=0, combo `scale`≈0.2, no P(goal)
  gain, **minus the accumulated negative step_bonus over the carry** ⇒ EV < just kicking.

The selectivity signal is the **negative step cost on a dribble that didn't raise gap
quality**, sharpened each substage by raising `|step_bonus|` and fading the dense terms.

## The three substages (refine the existing config)

All keep: `goal_reward=70`, `kick_aim_bonus_weight=20`, `alignment_weight=3`,
`bad_aim_*`, `use_goalie_aim_gate=true`, scripted bisector goalie, `disabled_actions:[goto]`,
`max_steps=300`. **Lineage: 2A warm-starts the Stage-2 *wide* baseline
`models/ppo_jal_expandable_wide/stage2_0_baseline_complete.pt`** (the top-level `load_model`),
**not** the Stage-1 narrow final; 2B ← 2A; 2C ← 2B. Note: the current `stage2_goalie`
description in the config says "Load from …/stage1_9_complete.pt" — that's the stale narrow
lineage and must be repointed to the wide baseline as part of this refinement.

**2A `stage2_goalie` — DISCOVER dribble (dense rewards, easy).** ball_y ±10,
`goalie_gap_blend_center=0.3` (transfer), `step_bonus=-0.05`, `dribble_target_progress=1.0`,
`dribble_target_quality=2.0`, `dribble_active_bonus=0.05`, `post_dribble_kick_bonus=3.0`.
- **Refinement:** set `post_dribble_kick_quality_scale=true` here too (currently `false`) —
  cheap insurance against "dribble then wild kick" farming the 3.0 from day one.
- Goal: policy *tries* `dribble_to` and learns gap-shots beat center-shots. ~300k.

**2B `stage2b_goalie_dribble` — SHARPEN targeting (full width).** ball_y ±15, blend 0.0,
`step_bonus=-0.1`, `progress=1.5`, `quality=4.0`, `active=0.03`, `combo=6.0` quality-scaled.
- Goal: precise, gap-improving targets; dribble starts paying mainly through the kick.

**2C `stage2c_goalie_balance` — WEAN to selective (combo-only).** ball_y ±15,
`step_bonus=-0.15`, `progress=0.5/clip0.3`, `quality=2.0`, `active=0.0` (dribble pays *only*
through the chain), `combo=10.0` quality-scaled, window 6.
- Goal: dribble survives *only* when it beats a direct shot ⇒ **selective** behavior.

This is already a clean monotonic fade in the config; the only true edit is
`post_dribble_kick_quality_scale=true` in 2A — the rest is validate/confirm.

## Acceptance gates (per substage, from train log / `log-training`)

- **2A:** dribble_to sampled in a meaningful fraction of episodes (explored, not ignored);
  goal rate does **not** collapse below the Stage-1 carry-in (holds ≈≥70%); combo bonus
  fires (dribble→kick chain occurs); no `Error parsing`; ≥2 clean PPO updates, no NaNs.
- **2B:** goal rate climbing toward ~80%+; mean `dribble_target_quality_delta` of opened
  carries clearly >0 (targets actually improve the gap); `bad_aim` stays low.
- **2C (the real test):** goal rate ≥ Stage-1 baseline; **dribble usage moderate (~15–40%
  of possessions), not >60%** (selective, not compulsive); dribbles that fire show
  gap-improvement; no stall/`max_steps` blowup from wedged dribbles (P0 watchdog working).
- If 2C shows **over-dribbling** (>~60%), revisit the locked decision and add the
  counterfactual penalty (penalize dribbling when current gap quality already exceeds a
  "good shot" threshold) — mirror the existing `dribble_target_quality` block in
  `evaluate_reward`/JAL_env.

## Files to change

- `configs/ppo_jal_curriculum_config.json` — the three `reward_config_overrides` blocks
  above. Only `post_dribble_kick_quality_scale` in 2A is a true change; the rest is
  validate/tune. Set **exactly one** stage description to `ACTIVE` at a time (start 2A) so
  the `log-training` parser records the right stage; flip ACTIVE forward as each completes.
  Confirm the warm-start lineage (`load_model` / per-stage checkpoint paths) chains
  `stage2_0_baseline_complete.pt` (wide) → 2A → 2B → 2C, and repoint the stale
  `stage2_goalie` "Load from stage1_9" note to the wide baseline.
- No source changes (reward.py / JAL_env.py / basic_commands.py) under the EV-fade approach.

## Verification

1. **Config sanity:** load the JSON, assert exactly one ACTIVE stage, and that 2A→2B→2C
   show the intended monotonic fade (`step_bonus` ↓, `dribble_active_bonus` ↓ to 0,
   `post_dribble_kick_bonus` ↑, `progress` ↓) — run with `/opt/anaconda3/envs/rcai/bin/python`.
2. **Embedded smoke (2A, ~2k steps):** confirm `dribble_to` is selected, carries open/close
   cleanly, `Kick fired` + `Dribble→kick combo bonus` lines appear, episodes end on sane
   reasons (not `max_steps` from stuck dribbles), no `Error parsing` in stderr, no NaNs.
3. **Per-substage:** run to budget, then `/log-training` to append the run to `TRAINING.md`
   and check the acceptance gates above before promoting to the next substage.

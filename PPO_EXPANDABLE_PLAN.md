# PPO JAL — Expandable Centralized Backbone: Implementation Plan

**Status:** design locked, not yet implemented.
**Owner:** Adnan.
**Created:** 2026-06-03.
**Scope:** Convert the PPO JAL policy (`ai_interface/algorithms/ppo_jal.py`) from a
fixed-robot-count network into a **count-agnostic, permutation-symmetric, centralized**
network so that:

1. Adding controlled robots across curriculum stages **never** requires retraining from
   scratch — the new robot inherits the shared, already-trained weights.
2. The network is aware of **observed-but-not-controlled** entities (our goalie, the
   opponents) without producing actions for them.
3. Team sizes can change **mid-episode** (fouls / red cards remove robots from either side)
   without any architecture change — it is handled by runtime masking.

This document is the standing reference for the whole curriculum. Each stage should refer
back here. Treat the "Design principles" section as non-negotiable unless explicitly revised.

---

## Table of contents

1. [Design principles (the rules)](#1-design-principles-the-rules)
2. [What exists today and what blocks us](#2-what-exists-today-and-what-blocks-us)
3. [Target architecture](#3-target-architecture)
4. [Observation layout](#4-observation-layout)
5. [Network components (detailed specs)](#5-network-components-detailed-specs)
6. [Forward pass with masking](#6-forward-pass-with-masking)
7. [Action sampling & primitive masking](#7-action-sampling--primitive-masking)
8. [Rollout buffer changes](#8-rollout-buffer-changes)
9. [PPO update changes](#9-ppo-update-changes)
10. [Trainer changes](#10-trainer-changes)
11. [Environment changes](#11-environment-changes)
12. [Inference changes](#12-inference-changes)
13. [Curriculum stage progression](#13-curriculum-stage-progression)
14. [Training robustness: count randomization](#14-training-robustness-count-randomization)
15. [Checkpoint compatibility & the one-time Stage 1 retrain](#15-checkpoint-compatibility--the-one-time-stage-1-retrain)
16. [Implementation phases & checklist](#16-implementation-phases--checklist)
17. [Risks, gotchas & open questions](#17-risks-gotchas--open-questions)

---

## 1. Design principles (the rules)

These are decisions already made with the team. Do not silently revise them.

- **P1 — Centralized JAL.** One network sees the whole team state and emits the joint action
  in a single forward pass; one joint value head. This is **not** MAPPO / decentralized. (See
  memory `project_centralized_jal`.)
- **P2 — Weight-shared per-robot encoder.** There is exactly **one** `PerRobotEncoder` reused
  for every controlled robot. No per-slot encoders. Robots are physically identical and
  interchangeable; the RL is the **team strategist** that assigns roles from the game state,
  so no robot has specialized weights. (See memory `project_ppo_expandable_shared_encoder`.)
  Consequence: adding a robot = the shared encoder is applied one more time. Nothing to
  initialize, nothing to transfer.
- **P3 — Controlled vs context entities.** Two roles:
  - **Controlled agents** (the ≤5 outfield robots on our team): get an encoder token **and**
    an action head.
  - **Context entities** (ball, our goalie, opponents): get an encoder token (so the policy is
    *aware* of them) but **no action head**.
  Our goalie is a context entity — it is driven by a separate **hardcoded algorithm**, never by
  this network.
- **P4 — Variable counts via masking, not module growth.** Build the network **once at maximum
  size** and switch entities on/off per timestep with an attention `key_padding_mask`. This is
  the only design that survives mid-episode fouls (you cannot re-instantiate modules
  mid-episode). It also subsumes the curriculum scale-up: "add a robot" = "unmask a slot."
  **`expand_robots()` is therefore not used at all** in the PPO path.
- **P5 — Permutation symmetry, no positional encoding.** Because the encoder and heads are
  shared and the aggregator is attention, robot identity comes only from each robot's own
  observation (its position, heading, possession). Do **not** add per-slot positional
  embeddings — that would reintroduce "special" robots.

---

## 2. What exists today and what blocks us

### 2.1 The PPO network — `ai_interface/algorithms/ppo_jal.py`

`JALActorCritic` (lines 87–144) is the blocker:

```python
self.encoder = nn.Sequential(Linear(obs_dim, 256), Tanh, Linear(256, 256), Tanh)  # flat MLP
self.primitive_head = nn.Linear(encoder_out, NUM_PRIMITIVES * num_robots)   # FUSED, count-dependent
self.param_mean     = nn.Linear(encoder_out, PARAM_DIM   * num_robots)       # FUSED, count-dependent
self.param_log_std  = nn.Parameter(torch.zeros(num_robots * PARAM_DIM))      # count-dependent
self.value_head     = nn.Linear(encoder_out, 1)                              # joint scalar (good)
```

- The **flat MLP encoder** has no notion of "per robot" — it ingests the whole flat obs. It is
  not permutation-symmetric and cannot share per-robot features.
- The **fused heads** are sized `… * num_robots`. Their shape changes with robot count, so a
  checkpoint trained at N robots cannot `load_state_dict` into an N+1 net (the trainer rebuilds
  the agent from scratch on count change — see §2.3). **This is exactly what we are removing.**
- Constants: `NUM_PRIMITIVES = 6` (`goto, approach_ball, turn, kick, start_dribble,
  stop_dribble`), `PARAM_DIM = 3` (`Dx, Dy, Dtheta`), `LOG_STD_MIN = -3.0`, `LOG_STD_MAX = 0.5`
  (lines 41–58). These are preserved unchanged.
- The hybrid structure (Categorical primitive + Gaussian params + shared `log_std`) is correct
  and **survives the refactor untouched** — we only change *where* the heads attach and that
  they are *shared per robot* instead of fused.

### 2.2 The expandable encoder we are reusing — `ai_interface/algorithms/td3_jal_expandable.py`

This file already contains the encoder pieces we want, proven on the TD3+HER path:

- `GlobalEncoder` (line 35): `Linear(global_dim, 64) → ReLU → Linear(64,64) → ReLU`.
- `PerRobotEncoder` (line 58): same shape, `per_robot_dim=8 → 64`.
- `RobotAggregator` (line 81): `nn.MultiheadAttention(64, 4, batch_first=True)` + `LayerNorm`,
  with the global feature added as a context token; currently returns **only** the pooled
  global token (`attn_out[:, 0, :]`).

We will **reuse `GlobalEncoder` and `PerRobotEncoder` as-is** (they are the right shape). We
will **not** reuse `ExpandableJALEncoder.expand_robots()` (per-slot `ModuleList`, violates P2
and P4) nor the `RobotAggregator` verbatim (it discards per-agent tokens, which we need for
per-agent action heads — see §5.4). The decoder half of that file stays dead code.

### 2.3 The trainer — `ai_interface/trainers/ppo_jal_curriculum_trainer.py`

- `setup_model` (lines 233–270): builds `PPOJALAgent(obs_dim, num_robots, …)` and optionally
  `agent.load(load_model)`.
- `_run_stage_inner` (lines 317–343): **rebuilds the agent** whenever `num_robots` or `obs_dim`
  changes (`needs_new_agent`). Under the new design, obs_dim and the network are **constant
  across all stages**, so this rebuild path should essentially never fire for count changes —
  warm-start becomes a clean `load`.
- `_training_loop` (lines 374–544): on-policy collect→update loop; calls
  `agent.sample_action(obs, disabled_actions, deterministic=False)` (line 409), steps the env,
  normalizes reward (`_RewardNormalizer`), stores transition + reward, updates when the rollout
  buffer is full.

### 2.4 The environment — `ai_interface/envs/JAL_env.py`

- Obs layout (lines 145–160): `[ball_x, ball_y, ball_vx, ball_vy]` (`non_robot_obs_dim=4`) then
  per robot `[x, y, theta, vx, vy, is_dribbling, start_dribble_x, start_dribble_y]`
  (`obs_dim_per_robot=8`). `obs_dim = 8*num_robots + 4`. **Already global-first then per-robot
  blocks** — the layout the shared encoder expects. Good.
- Action space (lines 164–177): `Box(num_robots * 9)` per-robot 9-D, but the PPO path consumes
  the dict action `{primitive_idx, params}` produced by `sample_action`. The env applies the
  chosen primitive per robot.
- Today the env has **no concept of context entities** (goalie/opponents are not in the obs)
  and **no concept of inactive agents**. Both are added in §11.

---

## 3. Target architecture

```
            ┌─────────────────────────── observation (fixed max size) ───────────────────────────┐
            │  ball(4)   agent_0(8) … agent_{A-1}(8)   ctx_0(d) … ctx_{C-1}(d)   + active masks    │
            └───────────────────────────────────────────────────────────────────────────────────┘
                  │              │   (shared PerRobotEncoder)        │  (shared ContextEncoder)
        GlobalEncoder      ┌─────┴─────┐                       ┌─────┴─────┐
                  │        │           │                       │           │
              global_tok  ag_tok_0 … ag_tok_{A-1}          ctx_tok_0 … ctx_tok_{C-1}
                  │        └───────────┴───────────┬───────────┴───────────┘
                  └──────────────────┐             │
                                     ▼             ▼
                       sequence = [global_tok, agent_toks…, ctx_toks…]   (batch, 1+A+C, 64)
                                     │
                       MultiheadAttention(self-attention) with key_padding_mask
                       (masks out fouled-out agents and absent context entities)
                                     │
                       ┌─────────────┼─────────────────────────────┐
                       ▼             ▼                             ▼
                attn[:,0,:]    attn[:,1:1+A,:]                (ctx token outputs unused
                (global)       (per-agent contextual tokens)   for output; they only
                   │                 │                          informed attention)
              value_head     ┌───────┴───────┐
              (joint scalar)  │               │
                       primitive_head   param_mean_head   (+ shared param_log_std)
                       (shared, →6)     (shared, →3)
                       applied to EACH active agent token
```

Key points:
- **One** `GlobalEncoder`, **one** `PerRobotEncoder` (shared over all agents), **one**
  `ContextEncoder` (shared over all context entities). All map to `feature_dim = 64`.
- The aggregator returns the **full attention output sequence**, not just the pooled token, so
  each agent's post-attention token feeds its (shared) action heads, while the global token
  feeds the value head.
- Action heads (`primitive_head: Linear(64,6)`, `param_mean: Linear(64,3)`, `param_log_std:
  Parameter(3)`) are **shared** and applied per agent. No `* num_robots` anywhere.
- Everything is sized by `A_max` (max controlled agents = 5) and `C_max` (max context
  entities). The active counts are runtime masks; the parameter shapes never change.

---

## 4. Observation layout

**Decision: fixed-maximum-size observation + explicit active masks** (approach (a)). The obs is
always built for `A_max` agent slots and `C_max` context slots; absent entities are zero-filled
and flagged inactive. This makes `obs_dim` **constant across every stage**, so the network is
literally never rebuilt and warm-start is a plain `load_state_dict`.

```
obs = [ ball(4) ]
      [ agent_slot_0(8) … agent_slot_{A_max-1}(8) ]      # zero-filled if inactive
      [ ctx_slot_0(d_ctx) … ctx_slot_{C_max-1}(d_ctx) ]  # zero-filled if absent
```

Plus, supplied alongside the obs (in `info` and stored per-transition — see §8):
- `agent_active_mask`: `float32[A_max]`, 1 = this controlled robot is on the field, 0 = fouled
  out / not yet introduced by the curriculum.
- `context_active_mask`: `float32[C_max]`, 1 = this context entity exists, 0 = absent.

### 4.1 Sizes

- `A_max = 5` (max outfield controlled robots). Our goalie is **not** a controlled agent.
- Context entities (`C_max`): ball is handled by `GlobalEncoder` separately (the always-present
  global token), so it is **not** counted in `C_max`. The maskable context entities are:
  - our goalie ×1,
  - opponent goalie ×1,
  - opponent outfield robots ×5.
  → `C_max = 7`.
- `d_ctx` (context-entity obs dim): proposed `[x, y, theta, vx, vy] = 5`. We observe pose +
  velocity for goalies/opponents but **not** internal flags like `is_dribbling`
  (`start_dribble_*`), which are only meaningful for our own controlled robots. `d_ctx` is a
  tunable schema; lock it before the first context-aware stage and keep it fixed thereafter
  (changing it later forces a context-encoder retrain).

### 4.2 Resulting constant obs_dim

`obs_dim = 4 (ball) + 8 * A_max + d_ctx * C_max = 4 + 8*5 + 5*7 = 4 + 40 + 35 = 79`.

During early stages (1 controlled robot, no opponents) slots 1..4 (agents) and all context
slots are zeroed and masked. The network ignores them via masking; the obs vector size is still
79 from Stage 1 onward. **This is why Stage 1 must be trained once on the new backbone** (see
§15) — but after that, every count change is free.

> Alternative considered (approach (b), variable-length obs + infer count from length, like
> `ExpandableJALFeatureExtractor.forward` lines 429–434): rejected as the *primary* design
> because it complicates batching, can't represent "agent fouled out mid-episode" without
> reshuffling, and reintroduces the obs_dim-change rebuild path. Fixed-max is cheap here
> (79 floats) and strictly simpler.

---

## 5. Network components (detailed specs)

All new code lives in `ppo_jal.py` (the algorithm owns both encoder and heads — it is a custom,
non-SB3 implementation). Import `GlobalEncoder` and `PerRobotEncoder` from
`td3_jal_expandable.py` (or copy them locally if we want PPO to be import-independent — decide
in Phase 0). `feature_dim = 64`, `num_heads = 4` to match the proven TD3 config.

### 5.1 GlobalEncoder (reuse)
`GlobalEncoder(global_dim=4, hidden_dim=64)` → ball token `(batch, 64)`. Unchanged.

### 5.2 PerRobotEncoder — shared (reuse, single instance)
`PerRobotEncoder(per_robot_dim=8, hidden_dim=64)`. **One instance**, called in a loop over the
`A_max` agent slots. Produces `(batch, A_max, 64)`.

### 5.3 ContextEncoder — shared (new, single instance)
A `PerRobotEncoder`-shaped MLP `Linear(d_ctx, 64) → ReLU → Linear(64,64) → ReLU`. **One
instance**, called over the `C_max` context slots. Produces `(batch, C_max, 64)`. Shared across
both goalies and all opponents — they are all "robots we observe but don't control," and we want
the same symmetry argument (no special opponent).

> If we later want to distinguish "opponent" from "our goalie" semantically, add a small
> learned **type embedding** (a 64-vector per entity type) added to the context token, rather
> than separate encoders. Out of scope for the first cut.

### 5.4 Aggregator — new (returns full sequence)
Replace `RobotAggregator` (which returns only the pooled token) with a version that returns the
**entire post-attention sequence**, because per-agent action heads need each agent's contextual
token.

```python
class TeamAttention(nn.Module):
    def __init__(self, feature_dim=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, tokens, key_padding_mask):
        # tokens: (batch, S, 64);  key_padding_mask: (batch, S) bool, True = ignore
        attn_out, _ = self.attn(tokens, tokens, tokens,
                                key_padding_mask=key_padding_mask)
        return self.norm(attn_out + tokens)   # residual + norm over the whole sequence
```

`S = 1 + A_max + C_max = 1 + 5 + 7 = 13`. Token 0 is the ball/global token (always unmasked),
tokens `1 .. A_max` are agents, tokens `1+A_max ..` are context entities.

### 5.5 Heads — shared
```python
self.primitive_head = nn.Linear(64, NUM_PRIMITIVES)   # 6  — shared across agents
self.param_mean     = nn.Linear(64, PARAM_DIM)        # 3  — shared across agents
self.param_log_std  = nn.Parameter(torch.zeros(PARAM_DIM))   # 3 — shared
self.value_head     = nn.Linear(64, 1)                # joint scalar, reads global token
```
Note the **dropped `* num_robots`** everywhere — this is the whole point. `param_log_std` is now
shape `(3,)` (per-parameter), broadcast across agents and batch.

---

## 6. Forward pass with masking

New `JALActorCritic.forward` signature:
`forward(obs, agent_active_mask, context_active_mask)`.

```python
def forward(self, obs, agent_active_mask, context_active_mask):
    # obs: (B, 79); masks: (B, A_max), (B, C_max), float 1/0
    B = obs.shape[0]
    # --- slice ---
    ball = obs[:, :4]
    ag   = obs[:, 4 : 4 + 8*A_max].view(B, A_max, 8)
    ctx  = obs[:, 4 + 8*A_max :].view(B, C_max, d_ctx)
    # --- encode (shared modules, applied per slot) ---
    g_tok  = self.global_encoder(ball).unsqueeze(1)            # (B,1,64)
    ag_tok = self.per_robot_encoder(ag.reshape(B*A_max, 8)).view(B, A_max, 64)
    cx_tok = self.context_encoder(ctx.reshape(B*C_max, d_ctx)).view(B, C_max, 64)
    tokens = torch.cat([g_tok, ag_tok, cx_tok], dim=1)         # (B, 13, 64)
    # --- key_padding_mask: True = ignore ---
    glob_keep = torch.ones(B, 1, device=obs.device)           # ball always present
    keep = torch.cat([glob_keep, agent_active_mask, context_active_mask], dim=1)
    key_padding_mask = (keep < 0.5)                            # (B,13) bool
    # --- attention ---
    attn = self.team_attention(tokens, key_padding_mask)      # (B,13,64)
    # --- outputs ---
    value = self.value_head(attn[:, 0, :])                    # (B,1) joint
    agent_ctx = attn[:, 1:1+A_max, :]                          # (B,A_max,64)
    primitive_logits = self.primitive_head(agent_ctx)          # (B,A_max,6)
    param_mean       = self.param_mean(agent_ctx)              # (B,A_max,3)
    param_std = torch.exp(torch.clamp(self.param_log_std, LOG_STD_MIN, LOG_STD_MAX))  # (3,)
    return primitive_logits, param_mean, param_std, value
```

**Masking caveat (important):** `nn.MultiheadAttention` with a fully-masked query row can emit
NaNs. Our query token 0 (ball) is always kept, and at least one agent is always active, so the
global/agent rows always attend to ≥1 key. But a *fully inactive* agent slot is still a query
row whose output we simply **discard** (its action is never used and it is excluded from the
loss via the agent-active mask — §9). To be safe, after attention, zero out inactive agent
tokens before the heads, or just never read them. Add a unit test asserting no NaNs when only
1 of 5 agents is active and 0 context entities exist.

---

## 7. Action sampling & primitive masking

`sample_action` (currently lines 247–318) changes minimally:

- Inputs gain `agent_active_mask`, `context_active_mask` (from env `info`).
- Call `model(obs, agent_active_mask, context_active_mask)`.
- The existing **disabled-primitive masking** (lines 277–287: `logits + log(mask+1e-45)` for
  primitives in `disabled_actions`) is unchanged and is applied to all `A_max` agent rows.
- Sample primitive (`Categorical`) and params (`Normal`), clamp params to `[-1,1]` (lines
  289–303). Unchanged.
- **Return actions for all `A_max` slots**, but the env applies only to active agents (the
  inactive ones' actions are ignored). The buffer also stores `agent_active_mask` so the loss
  ignores inactive slots.

Inference param-noise rule (memory `project_ppo_inference_param_noise`) is **unchanged**: never
run pure-deterministic mean; `--ppo_param_noise_std=0.3` (argmax primitive + Gaussian noise on
the param mean). Applies per active agent.

---

## 8. Rollout buffer changes

`RolloutBuffer` (lines 151–191) stores fixed-`A_max`-shaped per-agent arrays plus the new masks:

Add to each transition:
- `agent_active_mask`: `float32[A_max]`.
- `context_active_mask`: `float32[C_max]`.

Everything else (`primitive_actions[A_max]`, `param_actions[A_max,3]`,
`primitive_logprobs[A_max]`, `param_logprobs[A_max]`, `disabled_mask[A_max,6]`) keeps a fixed
`A_max` leading dimension, so `np.stack` over a rollout still works without ragged arrays — this
is the batching payoff of the fixed-max obs design.

Inactive agent slots store dummy zeros; they are zeroed out of every loss term by
`agent_active_mask` (§9).

---

## 9. PPO update changes

`update` (lines 361–517). The math is unchanged except every per-agent quantity is **masked by
`agent_active_mask` before summing across agents** so fouled-out / not-yet-introduced agents
contribute nothing.

Current (fixed N, all agents active):
```python
logp_new = (prim_logp + param_logp).sum(dim=-1)     # sum over robots
entropy  = (prim_entropy + param_entropy).sum(dim=-1).mean()
```
New (mask-weighted sums):
```python
m = b_agent_active            # (bs, A_max)
logp_new = ((prim_logp + param_logp) * m).sum(dim=-1)
logp_old = ((b_old_prim_logp + b_old_param_logp) * m).sum(dim=-1)
entropy  = (((prim_entropy + param_entropy) * m).sum(dim=-1)).mean()
```
- Pass `agent_active_mask`, `context_active_mask` into `self.model(...)` in **all three** call
  sites: the minibatch forward (≈ line 439), the post-epoch KL recompute (≈ line 487), and the
  bootstrap-value forward (≈ line 370).
- The value head is a single joint scalar (unchanged) — it already reads the global token, which
  attended over only the active entities, so the joint value is automatically conditioned on the
  current team size.
- GAE, advantage normalize/clip, value clipping, KL-adaptive LR halving + early stop: all
  unchanged.

---

## 10. Trainer changes

`ppo_jal_curriculum_trainer.py`:

- **`setup_model`**: build `PPOJALAgent` with the new constructor (`A_max`, `C_max`, `d_ctx`,
  `feature_dim`, `num_heads`) instead of `num_robots`. `obs_dim` is the constant 79.
- **`_run_stage_inner` rebuild logic (lines 317–343):** with constant obs_dim and a
  count-agnostic net, the `needs_new_agent` condition should no longer trigger on count change.
  Keep a guard that rebuilds **only** if `obs_dim` or the structural hyperparameters (`A_max`,
  `C_max`, `d_ctx`, `feature_dim`) actually change — which across the planned curriculum they do
  not. Warm-start between stages is `agent.load(prev_stage_complete.pt)` (a clean
  `load_state_dict`, no shape surgery, no `_transfer_policy_weights`).
- **`_training_loop`:** thread `agent_active_mask` / `context_active_mask` from `env.step`'s
  `info` (or `reset`'s `info`) into `sample_action(...)` and into `store_transition(...)`.
- **Reward normalizer (`_RewardNormalizer`):** unchanged — it operates on the scalar joint
  reward, which is independent of team size.
- The per-stage `num_robots`/`robot_ids` config keys now control **how many agent slots the env
  marks active** (and which robot ids map to them), not the network shape.

---

## 11. Environment changes (`JAL_env.py`)

This is the largest chunk of new work. The env must:

1. **Emit the fixed-max obs (79-D)** described in §4: always `A_max` agent slots + `C_max`
   context slots, zero-filling inactive/absent entities. The current builder (around the obs
   design comment at lines 145–160 and `_game_state_to_obs`) writes `num_robots` agent blocks;
   change it to write `A_max` blocks (active ones filled from game state, inactive ones zeroed)
   and append `C_max` context blocks (our goalie, opponent goalie, opponent outfielders).
2. **Produce the active masks** and return them in the `info` dict from both `reset` and
   `step` (`agent_active_mask[A_max]`, `context_active_mask[C_max]`).
3. **Track which controlled agents are active** (fouled-out handling — see §14) and **which
   context entities exist** (opponents present? opponent count after their fouls?).
4. **Apply actions only to active agents.** `step` consumes the dict action for all `A_max`
   slots but only sends commands for active robots; masked slots are no-ops.
5. **Keep `obs_dim` constant** = 79 (set `observation_space` accordingly). `obs_dim_per_robot`
   stays 8; add config for `A_max`, `C_max`, `d_ctx`.
6. **Context observation source:** the goalie and opponents come from the same `GameState` the
   env already parses for our robots (`_game_state_to_obs`). Extract their `[x, y, theta, vx,
   vy]`. Our goalie is identified by the team's `goalie_id` (already in `TeamInfo` /
   `team_config`). Opponents are the other team's players.

> **Goalie exclusion:** the controlled-agent list must exclude our goalie id. The hardcoded
> goalie algorithm is separate (not in this repo path yet); for training stages where the goalie
> matters, either (a) run the hardcoded goalie in the sim alongside, or (b) treat the goalie as a
> static/scripted context entity. Decide per stage. Until the goalie is introduced, its context
> slot is simply masked inactive.

---

## 12. Inference changes (`infer.py` / `launch_infer.py`)

- The `ppo_jal` inference runner must build the obs the same fixed-max way and pass the active
  masks into `sample_action`.
- Keep `--ppo_param_noise_std=0.3` default and the "never pure-deterministic mean" rule
  (memory `project_ppo_inference_param_noise`).
- For competition, the active masks come from live game state (who's been carded). The same
  masking path that we train with handles it — provided we **trained with count randomization**
  (§14), the policy will be robust to it.

---

## 13. Curriculum stage progression

Under this design a "stage" is defined by **(active agent count, active context entities,
disabled primitives, spawn/reward config)** — never by a network rebuild. Indicative ladder
(exact timesteps/rewards per existing curriculum docs):

| Stage | Active agents | Context entities | Notes |
|------|---------------|------------------|-------|
| 1.x (current) | 1 | none | reproduce existing Stage 1 skills on the new backbone (one-time retrain, §15) |
| 2 | 1 | ball only (already in global) | goto + dribbles unlock; still single agent |
| 3 | 2 | none/our goalie | first multi-agent; **new robot inherits shared weights**, just unmasked |
| 4 | 3–5 | our goalie | scale controlled agents up; unmask slots one/two at a time |
| 5 | 5 | + opponents (context) | introduce opponent context tokens; opponents scripted/hardcoded first |
| 6 | 5 | + opponents + foul randomization | train robustness to mid-game removals (§14) |

At each step up, the *only* change is config: which slots are active + scenario. No code change,
no retrain-from-scratch. Warm-start from the previous stage's `_complete.pt`.

---

## 14. Training robustness: count randomization

The architecture *supports* variable counts natively, but the policy only becomes *robust* to
them if it **sees** them in training. Add (env + stage config):

- `randomize_active_agents`: with some probability per episode, deactivate a random subset of our
  controlled agents (simulating a red card on our side) — set their `agent_active_mask = 0`,
  zero their obs block, and skip their actions.
- `randomize_opponent_count`: similarly vary how many opponent context entities are active.
- Recommended: introduce this **only at the final robustness stage** (Stage 6), after the policy
  is solid at full strength — same philosophy as the rest of the curriculum (don't show the hard
  distribution before the base skill exists).

This is what makes a mid-match foul a no-op at inference instead of a brittle out-of-distribution
event.

---

## 15. Checkpoint compatibility & the one-time Stage 1 retrain

- The existing `models/ppo_jal_curriculum/stage1_9_complete.pt` is a **flat-MLP** checkpoint
  (old `JALActorCritic`). It **cannot** load into the attention backbone — different modules,
  different parameter names/shapes. There is no automatic conversion.
- Therefore: **train Stage 1 once on the new backbone** to reproduce the turn→approach→kick
  skill, then save `stage1_*_complete.pt` as the new lineage root. Every later stage warm-starts
  cleanly from it via `load_state_dict`.
- Treat this as a **regression run**: target the same goal-rate / aim-quality bar Stage 1 hit on
  the flat MLP (per TRAINING.md). Expect a real (not instant) warm-up since the encoder is new.
- After Stage 1 is re-established on the new backbone, the "no retrain when adding robots"
  guarantee holds for the rest of the curriculum — that is the entire payoff.
- **Light hyperparameter retune may be needed** for the attention net (it has different
  optimization dynamics than the flat MLP). Start from the current `model_params`
  (`ent_coef_initial=0.05`, `lr` schedule, `rollout_size=4096`, etc.) and adjust only if the
  Stage 1 regression underperforms.

---

## 16. Implementation phases & checklist

**Phase 0 — scaffolding decisions**
- [ ] Lock `A_max=5`, `C_max=7`, `d_ctx=5`, `feature_dim=64`, `num_heads=4`.
- [ ] Decide: import `GlobalEncoder`/`PerRobotEncoder` from `td3_jal_expandable.py`, or copy
      into `ppo_jal.py` for independence. (Recommend copy — PPO shouldn't depend on the TD3
      module's lifecycle.)

**Phase 1 — network (`ppo_jal.py`), pure unit-testable**
- [ ] New `JALActorCritic` with shared `GlobalEncoder` / `PerRobotEncoder` / `ContextEncoder` /
      `TeamAttention` / shared heads (§5).
- [ ] New `forward(obs, agent_active_mask, context_active_mask)` (§6).
- [ ] Unit test: forward with (A active = 1, C = 0) and (A = 5, C = 7) → correct shapes, **no
      NaNs** with partial masks; permutation test (reorder agent slots ⇒ outputs reorder
      consistently, value unchanged).

**Phase 2 — agent plumbing (`ppo_jal.py`)**
- [ ] `sample_action` threads masks; returns `A_max`-slot actions (§7).
- [ ] `RolloutBuffer` stores the two masks (§8).
- [ ] `update` mask-weights per-agent logprob/entropy sums; passes masks to all 3 forward sites
      (§9).
- [ ] `save`/`load` store the structural hyperparameters; `load` is plain `load_state_dict`.

**Phase 3 — env (`JAL_env.py`)**
- [ ] Fixed-max 79-D obs builder; constant `observation_space` (§4, §11).
- [ ] `agent_active_mask` / `context_active_mask` in `reset`/`step` `info`.
- [ ] Context-entity extraction (our goalie, opponent goalie, opponents) from `GameState`.
- [ ] Apply actions only to active agents.
- [ ] (Stage 6) `randomize_active_agents` / `randomize_opponent_count` (§14).

**Phase 4 — trainer (`ppo_jal_curriculum_trainer.py`)**
- [ ] Build agent with structural hyperparams; remove count-driven rebuild.
- [ ] Thread masks from env → `sample_action` → `store_transition`.

**Phase 5 — inference (`infer.py`)**
- [ ] Fixed-max obs + masks; keep `--ppo_param_noise_std=0.3`.

**Phase 6 — Stage 1 regression**
- [ ] Retrain Stage 1 on the new backbone to the old bar (§15); log via `/log-training`.

**Phase 7 — scale-up validation**
- [ ] Stage with 1→2 agents: confirm warm-start loads cleanly and the 2nd robot is immediately
      competent (inherits shared weights), not random.

---

## 17. Risks, gotchas & open questions

- **NaN from fully-masked attention rows.** Mitigate as in §6 (ball + ≥1 agent always active;
  discard inactive query outputs; unit-test partial masks).
- **`d_ctx` is a one-way door.** Once a context-aware stage trains, changing `d_ctx` or the
  context schema forces a context-encoder retrain. Lock it in Phase 0.
- **Value scale across team sizes.** The joint return's magnitude may differ at 1 vs 5 agents;
  `_RewardNormalizer` (running discounted-return std) absorbs much of this, but watch value-loss
  conditioning when first scaling up.
- **Opponent realism.** Early opponent stages should use scripted/hardcoded opponents (context
  tokens with predictable motion) before any learned/self-play opponent — otherwise the context
  channel is noise. (See memory `project_stage2_ball_velocity_intercept` re: ball-velocity
  intercept once opponents start kicking.)
- **Goalie integration.** The hardcoded goalie algorithm is out of this repo path. Decide per
  stage whether to (a) run it live in sim, or (b) script the goalie context token. Until then,
  mask the goalie slot inactive.
- **Permutation symmetry vs role assignment.** With no positional encoding, the policy assigns
  roles purely from observed state — intended (P5). If role assignment turns out unstable, the
  fix is a *type* embedding for context entities (§5.3), **not** per-slot agent weights.
- **Open question:** do we want a single shared `ContextEncoder` for goalies + opponents, or a
  type embedding to let the net treat "our goalie" differently from "their players"? First cut:
  single shared encoder, no type embedding. Revisit if context-awareness underperforms.
- **Open question:** keep `GlobalEncoder`/`PerRobotEncoder` imported from `td3_jal_expandable.py`
  vs copied into `ppo_jal.py`. Leaning copy (decoupling).

---

*End of plan. Update this file as decisions change; it is the curriculum's standing reference
for the expandable PPO backbone.*

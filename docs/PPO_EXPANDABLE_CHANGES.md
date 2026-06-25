# PPO Expandable Architecture — Implementation Changelog

**Date:** 2026-06-03
**Goal:** Convert the PPO JAL policy from a fixed-robot-count flat MLP into a
**count-agnostic, permutation-symmetric, centralized attention backbone** so that adding
teammates / opponents / observation features / primitives later never forces a retrain. This
document records every code change made to get the new architecture training (rung 1 of the
Stage-1 retrain).

Design reference: [PPO_EXPANDABLE_PLAN.md](PPO_EXPANDABLE_PLAN.md) and
`~/.claude/plans/sprightly-singing-kite.md`.

## The core idea (why all these changes)

- **One network sees the whole team and emits the joint action in one forward pass** (centralized
  JAL), with **one shared per-robot encoder** so robots are interchangeable.
- The observation is a **fixed-maximum 105-D vector** built once at max size; absent entities are
  zero-filled and switched off with attention masks. `obs_dim` is therefore **constant across the
  whole curriculum** → the network is never rebuilt and warm-start is a plain `load_state_dict`.
- The **weight-baked widths** (per-entity input dims, primitive/param output dims) carry **reserved
  spare slots** that are zero/masked now → zero gradient → preserved for later activation. Entity
  *counts* (A_MAX/C_MAX) aren't weight-baked at all (attention is sequence-length-agnostic).

### Locked dimensions

| name | live | MAX (reserved) | meaning |
|---|---|---|---|
| `A_MAX` | — | 5 | max controlled agents (count, not weight-baked) |
| `C_MAX` | — | 7 | max context entities (count) |
| `GLOBAL_DIM` | 4 | 6 | ball `[x,y,vx,vy]` + 2 reserved |
| `PER_AGENT_DIM` | 8 | 10 | agent `[x,y,θ,vx,vy,is_dribbling,sd_x,sd_y]` + 2 reserved |
| `D_CTX` | 5 | 7 | context `[x,y,θ,vx,vy]` + 2 reserved |
| `NUM_PRIMITIVES` | 6 | 8 | goto/approach/turn/kick/start_dribble/stop_dribble + 2 reserved |
| `PARAM_DIM` | 3 | 5 | `[Dx,Dy,Dθ]` + 2 reserved |
| `FEATURE_DIM` | — | 64 | token width |
| `NUM_HEADS` | — | 4 | attention heads |
| **`obs_dim`** | — | **105** | `6 + 10·5 + 7·7` — constant for the whole curriculum |

---

## 1. `ai_interface/algorithms/ppo_jal.py` — FULL REWRITE

The old file had a flat-MLP `JALActorCritic` with **fused, count-dependent heads**
(`Linear(encoder_out, NUM_PRIMITIVES * num_robots)`, etc.) — a checkpoint trained at N robots
could not load into an N+1 network, which is exactly what blocked expansion. The whole file was
rewritten. Key pieces:

### 1.1 Constants — LIVE vs MAX widths (lines ~62–88)

```python
PRIMITIVE_NAMES = ("goto","approach_ball","turn","kick","start_dribble","stop_dribble")
NUM_PRIMITIVES = len(PRIMITIVE_NAMES)   # 6 live
NUM_PRIMITIVES_MAX = 8                  # primitive head width (6 live + 2 reserved)
PARAM_DIM = 3                           # Dx, Dy, Dtheta (live)
PARAM_DIM_MAX = 5                       # param head width (3 live + 2 reserved)
A_MAX = 5; C_MAX = 7
GLOBAL_DIM_MAX = 6                      # ball (4 live) + 2 reserved
PER_AGENT_DIM_MAX = 10                  # agent (8 live) + 2 reserved
D_CTX_MAX = 7                           # context (5 live) + 2 reserved
FEATURE_DIM = 64; NUM_HEADS = 4
LOG_STD_MIN = -3.0; LOG_STD_MAX = 0.5
```
**Why:** the MAX values are the weight-shape "one-way doors". Splitting live vs MAX lets the env
fill only the live dims while the network reserves headroom that stays inert until activated.
`feature_dim`/`num_heads` were also added to `DEFAULT_HPARAMS`; the old `encoder_hidden` is gone.

### 1.2 Encoders + attention (lines ~120–197)

Copied `GlobalEncoder` and `PerRobotEncoder` (2-layer `Linear→ReLU→Linear→ReLU`) **verbatim from
`td3_jal_expandable.py`** so PPO doesn't depend on the TD3 module's lifecycle, sized to the MAX
widths. Added a matching `ContextEncoder` (`Linear(D_CTX_MAX,64)→…`) for observed-but-not-controlled
entities, and a new **`TeamAttention`** that returns the *full* post-attention sequence (the TD3
`RobotAggregator` returned only the pooled token; we need each agent's token for its action head):

```python
class TeamAttention(nn.Module):
    def __init__(self, feature_dim=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
    def forward(self, tokens, key_padding_mask):   # tokens (B,S,F); mask (B,S) True=ignore
        attn_out, _ = self.attn(tokens, tokens, tokens, key_padding_mask=key_padding_mask)
        return self.norm(attn_out + tokens)
```
**Why:** attention parameters depend only on `feature_dim`, not sequence length, so growing entity
counts later changes **no** weight shape.

### 1.3 `JALActorCritic` (lines ~200–312)

- `__init__(obs_dim, a_max, c_max, global_dim, per_agent_dim, d_ctx, num_primitives, param_dim,
  feature_dim, num_heads)` — **`num_robots` dropped as a shape driver.** Builds one shared
  GlobalEncoder / PerRobotEncoder / ContextEncoder / TeamAttention and **shared** heads sized to MAX
  (no `* num_robots`): `primitive_head=Linear(64,8)`, `param_mean=Linear(64,5)`,
  `param_log_std=Parameter(zeros(5))`, `value_head=Linear(64,1)`. It also asserts
  `obs_dim == global_dim + per_agent_dim*a_max + d_ctx*c_max`.
- New `forward(obs, agent_active_mask, context_active_mask)` (line 253). Slices ball / `a_max`
  agent blocks / `c_max` context blocks, encodes each with its shared encoder, concatenates to
  `(B, 1+A+C, 64)`, builds the key-padding mask (ball token always kept), attends, then:

```python
glob_keep = torch.ones(B, 1, device=obs.device, dtype=obs.dtype)
keep = torch.cat([glob_keep, agent_active_mask, context_active_mask], dim=1)
key_padding_mask = keep < 0.5                                 # True = ignore
attn = self.team_attention(tokens, key_padding_mask)
value = self.value_head(attn[:, 0, :])                        # joint scalar from ball token
agent_ctx = attn[:, 1:1+A, :]
agent_ctx = agent_ctx * agent_active_mask.unsqueeze(-1)       # zero inactive agent tokens
primitive_logits = self.primitive_head(agent_ctx)            # (B,A,8)
param_mean       = self.param_mean(agent_ctx)                # (B,A,5)
param_std = torch.exp(torch.clamp(self.param_log_std, LOG_STD_MIN, LOG_STD_MAX))  # (5,)
```
**Why:** one joint value head (centralized JAL); per-agent shared heads applied to each token;
zeroing inactive agent tokens is a NaN/contamination guard (the ball key is always present so a
fully-masked query never occurs, but this makes inactive slots provably inert).

### 1.4 `RolloutBuffer` (lines ~314–372)

Added three fixed-shape per-transition fields so `np.stack` over a rollout stays rectangular:
`agent_active_masks (a_max,)`, `context_active_masks (c_max,)`, `param_active_masks (param_dim_max,)`.
**Why:** the update needs each transition's masks to reproduce the exact masked logprob it was
sampled with.

### 1.5 `PPOJALAgent` (lines ~374–809)

- Constructor takes the structural params (a_max … param_dim) and builds the new `JALActorCritic`;
  `num_robots` is kept only for a default-mask fallback and logging, **not** for shapes.
- **`_reserved_disabled_mask`** (line 441): builds the `(a_max, num_primitives)` enable mask and
  always disables reserved primitive slots `>= len(PRIMITIVE_NAMES)`:
  ```python
  mask = np.ones((self.a_max, self.num_primitives), dtype=np.float32)
  mask[:, NUM_PRIMITIVES:] = 0.0          # reserved slots always disabled
  for name in disabled_actions: mask[:, PRIMITIVE_NAMES.index(name)] = 0.0
  ```
- **`sample_action(obs, disabled_actions, agent_active_mask, context_active_mask,
  param_active_mask, deterministic)`** (line 472). Passes masks to the model; applies the disabled
  mask as a finite offset `logits + log(mask + 1e-45)` (≈ −103.6, **not** −inf, so entropy stays
  finite — preserved from the old code); samples `Categorical` primitive + `Normal` params; and the
  key new bit, the **param-active mask in the logprob**:
  ```python
  param_logprob = (param_dist.log_prob(param_action) * pa_mask_t).sum(dim=-1)  # (a_max,)
  ```
  Returns `action={primitive_idx[a_max], params[a_max*param_dim]}` and a `transition` that now also
  stores `agent_active_mask`, `context_active_mask`, `param_active_mask`.
  **Why param_active_mask:** only the param dims an enabled primitive actually uses contribute to
  the gradient. Reserved dims and stage-idle dims (e.g. Dx/Dy during a turn-only stage) get zero
  gradient → no drift, weights preserved for later activation.
- **`update(...)`** (line 601). All three model-forward sites (bootstrap, minibatch, KL recompute)
  now pass the masks. A helper `_joint_logp(...)` does the masked reduction consistently:
  ```python
  param_logp = (param_dist.log_prob(b_param) * b_param_active.unsqueeze(1)).sum(dim=-1)
  logp = ((prim_logp + param_logp) * b_agent).sum(dim=-1)   # param-mask, then agent-mask + sum
  ```
  Entropy is masked the same way. GAE, value clipping, KL-adaptive LR halving, advantage
  normalize/clip are unchanged. **Why:** fouled-out / not-yet-introduced agents and idle params must
  contribute nothing to the loss.
- **`save`/`load`** (lines ~787–809). `save` records a `structural` dict (all MAX widths + obs_dim).
  `load` **asserts the structural params match** before `load_state_dict`, then loads:
  ```python
  if mismatched: raise ValueError("Checkpoint structural params differ … one-way door …")
  self.model.load_state_dict(ckpt["model_state_dict"])
  ```
  **Why:** loading always succeeds across stages (shapes are constant), and a mistaken MAX-width
  change is caught loudly instead of silently corrupting a warm-start.

---

## 2. `ai_interface/envs/JAL_env.py`

### 2.1 Constructor params (line ~44)
Added `a_max=5, c_max=7, global_dim=6, per_agent_dim=10, d_ctx=7` to `JALTeamEnv.__init__`.

### 2.2 Fixed-max observation space + active masks (lines ~145–182)
Replaced the old `obs_dim = obs_dim_per_robot * num_robots + non_robot_obs_dim` with:
```python
self.obs_dim = self.global_dim + self.per_agent_dim * self.a_max + self.d_ctx * self.c_max  # 105
self.observation_space = spaces.Box(-inf, inf, (self.obs_dim,), float32)
self.agent_active_mask = np.zeros(self.a_max, dtype=np.float32)
self.agent_active_mask[: self.num_robots] = 1.0
self.context_active_mask = np.zeros(self.c_max, dtype=np.float32)   # TODO(opponent-stage)
```
**Why:** constant obs size across all stages; masks tell the policy which slots are real. Also
validates `num_robots <= a_max`.

### 2.3 Fixed analytic normalization constants (lines ~214–220)
```python
self._NORM_POS_X = self.field_half_width   # 45.0
self._NORM_POS_Y = self.field_half_height  # 30.0
self._NORM_THETA = float(np.pi)
self._NORM_VEL = 10.0
```
**Why:** a locked, constant (non-running) transform keeps PPO inputs well-scaled and stays
stage-invariant — chosen now because Stage 1 is being retrained from scratch anyway, and changing
it later would be a distribution shift = retrain.

### 2.4 Observation builder rewrite (`_game_state_to_obs`, lines ~806–863)
Replaced the variable-length `obs_values` list + pad/truncate fallback with a fixed 105-vector
filled block-by-block, normalized, with reserved/inactive dims left at zero:
```python
obs = np.zeros(self.obs_dim, dtype=np.float32)
obs[0] = ball_x / self._NORM_POS_X; obs[1] = ball_y / self._NORM_POS_Y
obs[2] = ball_vx / self._NORM_VEL;  obs[3] = ball_vy / self._NORM_VEL   # global, 4 live
for slot in range(self.a_max):
    if slot >= self.num_robots: continue          # inactive slot → zeros
    ... # off = global_dim + slot*per_agent_dim; write 8 normalized live dims
    obs[off+0] = robot_x/self._NORM_POS_X; ...; obs[off+5] = float(is_dribbling)
    obs[off+6] = sd_x/self._NORM_POS_X; obs[off+7] = sd_y/self._NORM_POS_Y
# context slots (c_max × d_ctx) left ALL ZERO — stub; TODO(opponent-stage)
return obs
```
**Why:** produces the fixed-max layout the network expects; the dribble/velocity bookkeeping
(keyed by `robot_id`) is preserved. Context extraction is deliberately stubbed (no opponents in
1v0); `extract_opponent_positions` in `reward.py` is the hook for the opponent stage.

### 2.5 Masks in `info` (lines ~411 and ~660)
Both `reset()` and `step()` info dicts now include
`"agent_active_mask"` and `"context_active_mask"` (copies). **Why:** the trainer/inference loop
threads them into `sample_action`.

### 2.6 Action decode for `a_max` slots (`_action_to_commands`, lines ~908–945)
The PPO-dict branch previously required `primitive_idx` length `== num_robots` and `params` of
`num_robots*3`. Now it accepts the `a_max`-slot action and reads only the active slots with a
slot-width inferred from the array:
```python
n_slots = primitive_idx_arr.shape[0]
if n_slots < self.num_robots: raise ValueError(...)
params_per_slot = params_arr.shape[0] // n_slots          # = param_dim_max (5)
for i in range(self.num_robots):                          # only active slots dispatched
    base = i * params_per_slot
    goto_x_raw, goto_y_raw, turn_theta_raw = params_arr[base:base+3]
```
**Why:** the policy emits all `a_max` slots; the env dispatches commands only for active controlled
robots (slot `i` ↔ `robot_ids[i]`), reading the first 3 params per slot (reserved params ignored
until a future primitive uses them). Inferring `params_per_slot` keeps the env agnostic to the MAX
widths.

---

## 3. `ai_interface/trainers/ppo_jal_curriculum_trainer.py`

### 3.1 `setup_environment` (line ~205)
Passes `a_max / c_max / global_dim / per_agent_dim / d_ctx` from config to `JALTeamEnv`.

### 3.2 `setup_model` (lines ~239–270)
- Added `feature_dim`, `num_heads` to the hparam pull list; removed `encoder_hidden`.
- Builds `PPOJALAgent` with the structural params (a_max … num_primitives … param_dim) from config.
- Fixed the setup log line that referenced the now-deleted `self.agent.model.encoder` →
  logs `a_max` and `feature_dim` instead.

### 3.3 Rebuild guard (lines ~330–360)
Changed `needs_new_agent` from "rebuild on num_robots **or** obs_dim change" to **obs_dim change
only**:
```python
needs_new_agent = (self.agent is None or self._current_obs_dim != obs_dim_now)
...
else:   # reuse — count-agnostic warm-start
    self.agent.num_robots = num_robots   # default-mask fallback only
    self.agent.buffer.clear()
```
**Why:** adding a teammate (num_robots 1→2) must **not** rebuild the network — the whole point is
that the shared weights run one more slot. Across this curriculum obs_dim is constant 105, so this
branch never rebuilds; warm-start is a clean weight reuse.

### 3.4 Training loop (`_training_loop`, lines ~400–450, ~520–530)
- Computes a per-stage **`param_active_mask`** from the enabled primitives:
  ```python
  param_active_mask = np.zeros(self.agent.param_dim, dtype=np.float32)
  if "goto" not in disabled_actions: param_active_mask[0] = param_active_mask[1] = 1.0  # Dx,Dy
  if "turn" not in disabled_actions: param_active_mask[2] = 1.0                          # Dtheta
  ```
- Captures `agent_active_mask` / `context_active_mask` from `reset()`/`step()` `info` and threads
  all three masks into `sample_action(...)`; refreshes them on reset and after each step.
**Why:** the masks must match the obs they're sampled against; the param mask gates idle params
(e.g. turn-only stage ⇒ `[0,0,1,0,0]`).

---

## 4. `configs/ppo_jal_curriculum_config.json`

### 4.1 Structural keys (top level)
Replaced the **stale** `"obs_dim_per_robot": 11` with `8` (the actual live count) and added:
```json
"a_max": 5, "c_max": 7, "global_dim": 6, "per_agent_dim": 10, "d_ctx": 7,
"num_primitives": 8, "param_dim": 5, "feature_dim": 64, "num_heads": 4
```
**Why:** the new env/agent read these; the stale `11` was a leftover from the removed cone channels
(the builder only ever wrote 8 + 3 zero-pad).

### 4.2 Fresh start + new lineage
```json
"save_path": "models/ppo_jal_expandable",   // was models/ppo_jal_curriculum
"load_model": null                          // was the flat-MLP stage1_180_y15 checkpoint
```
**Why:** flat-MLP checkpoints can't load into the attention net, so Stage 1 starts fresh; the new
checkpoint lineage is kept separate so the old ones are preserved.

### 4.3 Rung 1 activation
`stage1_090_mid`: `timesteps 0 → 300000`, description marked `ACTIVE` (it's the only ACTIVE stage
and the only one with `timesteps>0`, so `/log-training` won't warn). All other rungs stay
`timesteps:0` (skipped) until run rung-by-rung.

---

## 5. `infer.py` (`_run_ppo_jal`)

- Env build (line ~634): passes `a_max / c_max / global_dim / per_agent_dim / d_ctx` from config.
- hparams (line ~673): added `feature_dim`, `num_heads`; removed `encoder_hidden`.
- Agent build (line ~687): structural params (a_max … num_primitives … param_dim) from config.
- Loop: computes the same `param_active_mask` from `disabled_actions` (line ~702), captures masks
  from `reset()`/`step()` info, and threads `agent_active_mask` / `context_active_mask` /
  `param_active_mask` into `sample_action` (line ~739). Kept `--ppo_param_noise_std=0.3` and the
  "never pure-deterministic mean" rule.
**Why:** inference must build the identical 105-D obs and pass the identical masks the policy was
trained with.

---

## 6. `tests/test_ppo_jal_expandable.py` — NEW (no simulator required)

Nine unit tests covering the network/agent in isolation:
- `test_obs_dim_constant` — obs_dim == 105.
- `test_forward_shapes_minimal` / `_full` — correct output shapes at (1 agent, 0 ctx) and (5, 7).
- `test_no_nan_partial_masks` — **every** (agents 1–5 × context 0–7) mask combo is NaN-free.
- `test_inactive_agents_zeroed` — inactive agent slots produce identical (bias-only) logits.
- `test_permutation_equivariance` — reordering active agents reorders outputs consistently; the
  joint value is unchanged (shared weights, no positional encoding).
- `test_sample_action_shapes_and_reserved_disabled` — action shapes `(a_max,)` / `(a_max*5,)`;
  reserved primitive slots never selected; disabled mask correct.
- `test_train_update_runs_and_is_finite` — a tiny end-to-end PPO update over fabricated transitions
  produces finite metrics (exercises masked logprob/entropy + value/KL paths).
- `test_save_load_roundtrip_and_structural_guard` — load succeeds across a count change; a MAX-width
  mismatch is rejected.

Run: `python -m pytest tests/test_ppo_jal_expandable.py -q` (or `python tests/test_ppo_jal_expandable.py`).

---

## Verification performed

1. **Unit tests:** 9/9 pass.
2. **No-sim integration smoke:** constructed `JALTeamEnv`, fed a fabricated `GameState` → obs is
   105-D, correctly normalized, inactive/context slots zero; `sample_action` → env
   `_action_to_commands` round-trip dispatches exactly one command for the single active robot.
3. **Embedded-sim smoke (700 steps):** env obs_dim=105, agent built, `param_active_mask=[0,0,1,0,0]`,
   two clean PPO updates (value loss 4.59→1.63, KL 0.006/0.014, cat-entropy 0.68 > tripwire), **zero
   NaNs/errors**, checkpoint saved, goals scoring. Smoke artifacts were then removed.

## Files changed

| File | Nature |
|---|---|
| `ai_interface/algorithms/ppo_jal.py` | full rewrite (attention backbone + masked agent) |
| `ai_interface/envs/JAL_env.py` | fixed-max obs, normalization, masks, action dispatch |
| `ai_interface/trainers/ppo_jal_curriculum_trainer.py` | structural agent build, mask threading, rebuild guard |
| `configs/ppo_jal_curriculum_config.json` | structural dims, fresh-start, new lineage, rung-1 active |
| `infer.py` | structural agent/env build + mask threading |
| `tests/test_ppo_jal_expandable.py` | new unit-test suite |

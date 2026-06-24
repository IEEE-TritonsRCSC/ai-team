"""Hierarchical PPO for the JAL action space — expandable attention backbone.

Adapted from the original flat-MLP `ppo_jal.py` to a **count-agnostic,
permutation-symmetric, centralized** network (see PPO_EXPANDABLE_PLAN.md). The
hybrid action space is unchanged (one Categorical primitive choice + a per-robot
Gaussian over continuous params), and the anti-freeze safeguards (high entropy
coef with schedule, value-loss clipping, KL-adaptive LR, large rollout buffer)
are preserved.

Architecture (single forward pass, one joint value head):

    obs (fixed max size) = ball(GLOBAL_DIM) + A_MAX agent slots(PER_AGENT_DIM)
                                            + C_MAX context slots(D_CTX)
      ├─ GlobalEncoder(ball)                         → global/ball token
      ├─ PerRobotEncoder(each agent slot)  [SHARED]  → A_MAX agent tokens
      └─ ContextEncoder(each context slot) [SHARED]  → C_MAX context tokens
            └─ TeamAttention (self-attention, key_padding_mask)
                  ├─ token 0  → value_head (joint scalar)
                  └─ agent tokens → shared primitive_head + param heads

Why this shape: robots are physically identical and interchangeable, so ONE
shared per-robot encoder + shared heads make the policy permutation-symmetric —
a newly-added robot automatically runs the previously-trained weights. Entity
counts are switched on/off per timestep via the attention mask, which also
handles mid-episode fouls. Context entities (our goalie, opponents) get tokens
(the policy is *aware* of them) but no action head.

Reserved capacity (no-future-retrain): the per-entity input widths and the
action-output widths carry spare, zero/masked slots so new obs features /
primitives / continuous params can be added later without changing any weight
shape (reserved slots are inert — zero gradient — until activated).

Primitives (Categorical, live order):
    0: goto           — reads (Dx, Dy)
    1: approach_ball  — reads nothing (uses ball position)
    2: turn           — reads Dtheta
    3: kick           — reads nothing
    4: dribble_to     — reads (Dx, Dy) as target coordinate
    (5–11 reserved — always disabled until a future primitive is introduced)

Continuous params (Gaussian, live order): Dx, Dy, Dtheta. (3–7 reserved.)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Constants — LIVE vs MAX (reserved) widths.
#
# The env only ever *fills* the live dims; reserved dims/slots stay zero and
# masked, so they get zero gradient and keep their init until later activated.
# Changing a MAX width is the only thing that breaks warm-start, so treat them
# as one-way doors locked here.
# ----------------------------------------------------------------------

PRIMITIVE_NAMES: Tuple[str, ...] = (
    "goto",
    "approach_ball",
    "turn",
    "kick",
    "dribble_to",
)
NUM_PRIMITIVES = len(PRIMITIVE_NAMES)   # 5 live (== len(PRIMITIVE_NAMES))
NUM_PRIMITIVES_MAX = 12                 # primitive head width (5 live + 7 reserved)
PARAM_DIM = 3                           # Dx, Dy, Dtheta (live)
PARAM_DIM_MAX = 8                       # param head width (3 live + 5 reserved)

# Which continuous param dims each primitive actually consumes (Dx=0, Dy=1,
# Dtheta=2). Used to mask the param logprob/entropy to the SELECTED primitive on
# each step, so e.g. a `kick` or `approach_ball` step gives the dribble-target
# (Dx, Dy) head ZERO gradient. Previously the mask was stage-wide (every step
# trained Dx,Dy whenever dribble_to was merely enabled), which diluted/corrupted
# the dribble-target gradient with unrelated transitions.
PRIMITIVE_PARAM_DIMS: Dict[str, Tuple[int, ...]] = {
    "goto": (0, 1),
    "approach_ball": (),
    "turn": (2,),
    "kick": (),
    "dribble_to": (0, 1),
}


def _build_primitive_param_mask() -> np.ndarray:
    m = np.zeros((NUM_PRIMITIVES_MAX, PARAM_DIM_MAX), dtype=np.float32)
    for name, dims in PRIMITIVE_PARAM_DIMS.items():
        p = PRIMITIVE_NAMES.index(name)
        for d in dims:
            m[p, d] = 1.0
    return m


# (NUM_PRIMITIVES_MAX, PARAM_DIM_MAX) — row = primitive index, 1 = dim used.
PRIMITIVE_PARAM_MASK = _build_primitive_param_mask()

A_MAX = 5                               # max controlled outfield agents
C_MAX = 7                               # max context entities (our goalie + opp goalie + 5 opp)
GLOBAL_DIM_MAX = 6                      # ball [x,y,vx,vy] (4 live) + 2 reserved
PER_AGENT_DIM_MAX = 10                  # agent [x,y,theta,vx,vy,is_dribbling,sd_x,sd_y] (8 live) + 2 reserved
D_CTX_MAX = 7                           # context [x,y,theta,vx,vy] (5 live) + 2 reserved

FEATURE_DIM = 64                        # token width (matches proven TD3 encoder)
NUM_HEADS = 4                           # attention heads

# LOG_STD bounds (see TRAINING.md §-aim-floor): -3.0 lets the turn-angle std
# tighten to exp(-3)*pi ≈ 9°, inside the goal window.
# MAX was 0.5 (std=exp(0.5)=1.65) — but the entropy bonus drove param_log_std
# straight to that ceiling and the clamp then zeroed its gradient, pinning the
# continuous params at std=1.65 permanently (≈54% of clamp(-1,1) samples slam to
# a rail = near-random targets). Lowered to 0.0 (std cap = 1.0); the real cure is
# decoupling the param entropy (param_ent_coef) and resetting the contaminated
# value on warm-start (param_log_std_init) so the std can actually be LEARNED.
LOG_STD_MIN = -3.0
LOG_STD_MAX = 0.0

# Default hyperparameters — overridable via config.
DEFAULT_HPARAMS: Dict[str, Any] = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "target_kl": 0.015,
    "n_epochs": 10,
    "minibatch_size": 256,
    "rollout_size": 4096,           # transitions per PPO update (n_steps)
    "learning_rate_initial": 3e-4,
    "learning_rate_final": 1e-4,
    "ent_coef_initial": 0.05,
    "ent_coef_final": 0.005,
    # Separate entropy coef for the continuous Gaussian params. None = use the
    # (categorical) ent_coef, i.e. legacy behaviour. Set to 0.0 to stop the
    # entropy bonus from inflating param_log_std to the clamp ceiling — the
    # continuous std is then learned from returns while the categorical entropy
    # still keeps primitive selection from freezing.
    "param_ent_coef": None,
    # If not None, param_log_std is reset to this value on warm-start load (and
    # at init). Use to wipe a checkpoint whose std was pinned at the old ceiling.
    # log(0.5) ≈ -0.69 → std 0.5, comfortably below LOG_STD_MAX so gradient flows.
    "param_log_std_init": None,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "value_clip_range": 0.2,
    "advantage_clip": 5.0,           # clamp normalized advantages to [-c, c]
    "feature_dim": FEATURE_DIM,      # attention token width
    "num_heads": NUM_HEADS,          # attention heads
    "entropy_tripwire": 0.5,         # warn if per-categorical entropy < this
    "kl_lr_halve_factor": 2.0,       # halve LR if KL > factor*target_kl
}


# ----------------------------------------------------------------------
# Encoders (copied from td3_jal_expandable.py so PPO is import-independent)
# ----------------------------------------------------------------------

class GlobalEncoder(nn.Module):
    """Encodes the global state (ball position + velocity) into one token."""

    def __init__(self, global_dim: int = GLOBAL_DIM_MAX, hidden_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs)


class PerRobotEncoder(nn.Module):
    """Encodes a single controlled robot's state. Shared across all agent slots."""

    def __init__(self, per_robot_dim: int = PER_AGENT_DIM_MAX, hidden_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(per_robot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, robot_obs: torch.Tensor) -> torch.Tensor:
        return self.net(robot_obs)


class ContextEncoder(nn.Module):
    """Encodes a single observed-but-not-controlled entity (goalie / opponent).

    Shared across all context slots — same permutation-symmetry argument as the
    per-robot encoder (no special opponent). Stays at init until the first
    context-aware stage fills its slots (fully masked ⇒ zero gradient).
    """

    def __init__(self, d_ctx: int = D_CTX_MAX, hidden_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_ctx, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, ctx_obs: torch.Tensor) -> torch.Tensor:
        return self.net(ctx_obs)


class TeamAttention(nn.Module):
    """Self-attention over [global, agents…, context…] returning the FULL sequence.

    Differs from TD3's RobotAggregator (which returns only the pooled token):
    per-agent action heads need each agent's post-attention token. The
    attention parameters depend only on `feature_dim`, NOT on sequence length,
    so growing entity counts later does not change any weight shape.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, num_heads: int = NUM_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # tokens: (B, S, F);  key_padding_mask: (B, S) bool, True = ignore as a key.
        attn_out, _ = self.attn(tokens, tokens, tokens, key_padding_mask=key_padding_mask)
        return self.norm(attn_out + tokens)


# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------

class JALActorCritic(nn.Module):
    """Shared encoders + team self-attention + shared heads + joint value head.

    Forward consumes the fixed-max observation plus per-entity active masks and
    returns MAX-width primitive logits / param means; reserved primitive slots
    and param dims are gated downstream (in sample_action / update).
    """

    def __init__(
        self,
        obs_dim: int,
        a_max: int = A_MAX,
        c_max: int = C_MAX,
        global_dim: int = GLOBAL_DIM_MAX,
        per_agent_dim: int = PER_AGENT_DIM_MAX,
        d_ctx: int = D_CTX_MAX,
        num_primitives: int = NUM_PRIMITIVES_MAX,
        param_dim: int = PARAM_DIM_MAX,
        feature_dim: int = FEATURE_DIM,
        num_heads: int = NUM_HEADS,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.a_max = int(a_max)
        self.c_max = int(c_max)
        self.global_dim = int(global_dim)
        self.per_agent_dim = int(per_agent_dim)
        self.d_ctx = int(d_ctx)
        self.num_primitives = int(num_primitives)
        self.param_dim = int(param_dim)
        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)

        expected = self.global_dim + self.per_agent_dim * self.a_max + self.d_ctx * self.c_max
        if self.obs_dim != expected:
            raise ValueError(
                f"obs_dim {self.obs_dim} != global_dim + per_agent_dim*a_max + d_ctx*c_max "
                f"= {self.global_dim} + {self.per_agent_dim}*{self.a_max} + {self.d_ctx}*{self.c_max} "
                f"= {expected}"
            )

        # Shared encoders (one instance each).
        self.global_encoder = GlobalEncoder(self.global_dim, self.feature_dim)
        self.per_robot_encoder = PerRobotEncoder(self.per_agent_dim, self.feature_dim)
        self.context_encoder = ContextEncoder(self.d_ctx, self.feature_dim)
        self.team_attention = TeamAttention(self.feature_dim, self.num_heads)

        # Shared heads — no `* num_robots`; applied per agent token.
        self.primitive_head = nn.Linear(self.feature_dim, self.num_primitives)
        self.param_mean = nn.Linear(self.feature_dim, self.param_dim)
        self.param_log_std = nn.Parameter(torch.zeros(self.param_dim))
        self.value_head = nn.Linear(self.feature_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        agent_active_mask: torch.Tensor,
        context_active_mask: torch.Tensor,
    ):
        """Returns (primitive_logits, param_mean, param_std, value).

        primitive_logits: (batch, a_max, num_primitives)
        param_mean:       (batch, a_max, param_dim)
        param_std:        (param_dim,)  — broadcast over batch & agents
        value:            (batch, 1)    — joint scalar (reads the global token)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]
        g, pa, dc = self.global_dim, self.per_agent_dim, self.d_ctx
        A, C = self.a_max, self.c_max

        agent_active_mask = agent_active_mask.to(obs.dtype)
        context_active_mask = context_active_mask.to(obs.dtype)
        if agent_active_mask.dim() == 1:
            agent_active_mask = agent_active_mask.unsqueeze(0).expand(B, -1)
        if context_active_mask.dim() == 1:
            context_active_mask = context_active_mask.unsqueeze(0).expand(B, -1)

        # --- slice ---
        ball = obs[:, :g]
        ag = obs[:, g : g + pa * A].reshape(B, A, pa)
        cx = obs[:, g + pa * A :].reshape(B, C, dc)

        # --- encode (shared modules applied per slot) ---
        g_tok = self.global_encoder(ball).unsqueeze(1)                       # (B, 1, F)
        ag_tok = self.per_robot_encoder(ag.reshape(B * A, pa)).reshape(B, A, -1)
        cx_tok = self.context_encoder(cx.reshape(B * C, dc)).reshape(B, C, -1)
        tokens = torch.cat([g_tok, ag_tok, cx_tok], dim=1)                   # (B, 1+A+C, F)

        # --- key_padding_mask: True = ignore. Ball (token 0) is always kept. ---
        glob_keep = torch.ones(B, 1, device=obs.device, dtype=obs.dtype)
        keep = torch.cat([glob_keep, agent_active_mask, context_active_mask], dim=1)
        key_padding_mask = keep < 0.5                                        # (B, S) bool

        # --- attention ---
        attn = self.team_attention(tokens, key_padding_mask)                 # (B, S, F)

        # --- outputs ---
        value = self.value_head(attn[:, 0, :])                               # (B, 1)
        agent_ctx = attn[:, 1 : 1 + A, :]                                    # (B, A, F)
        # Zero inactive agent tokens so their (ignored) head outputs can't
        # contaminate anything — belt-and-suspenders NaN/contamination guard.
        agent_ctx = agent_ctx * agent_active_mask.unsqueeze(-1)
        primitive_logits = self.primitive_head(agent_ctx)                    # (B, A, P_max)
        param_mean = self.param_mean(agent_ctx)                              # (B, A, PARAM_max)
        param_std = torch.exp(torch.clamp(self.param_log_std, LOG_STD_MIN, LOG_STD_MAX))  # (PARAM_max,)
        return primitive_logits, param_mean, param_std, value


# ----------------------------------------------------------------------
# Rollout buffer
# ----------------------------------------------------------------------

class RolloutBuffer:
    """Simple list-based rollout buffer; cleared after each PPO update.

    Every per-agent / per-param array keeps a fixed leading dimension (a_max,
    param_dim_max) so np.stack over a rollout stays rectangular.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.observations: List[np.ndarray] = []
        self.primitive_actions: List[np.ndarray] = []   # (a_max,) per-slot primitive index
        self.param_actions: List[np.ndarray] = []       # (a_max, param_dim_max)
        self.primitive_logprobs: List[np.ndarray] = []  # (a_max,)
        self.param_logprobs: List[np.ndarray] = []       # (a_max,) summed over active params
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.masks: List[float] = []                    # 1.0 if not terminal
        self.disabled_masks: List[np.ndarray] = []      # (a_max, num_primitives) 1=enabled
        self.agent_active_masks: List[np.ndarray] = []  # (a_max,) 1=active
        self.context_active_masks: List[np.ndarray] = []  # (c_max,) 1=present
        self.param_active_masks: List[np.ndarray] = []  # (a_max, param_dim_max) 1=active, per selected primitive

    def __len__(self) -> int:
        return len(self.observations)

    def add(
        self,
        obs: np.ndarray,
        primitive_action: np.ndarray,
        param_action: np.ndarray,
        primitive_logprob: np.ndarray,
        param_logprob: np.ndarray,
        value: float,
        disabled_mask: np.ndarray,
        agent_active_mask: np.ndarray,
        context_active_mask: np.ndarray,
        param_active_mask: np.ndarray,
    ):
        self.observations.append(np.asarray(obs, dtype=np.float32))
        self.primitive_actions.append(np.asarray(primitive_action, dtype=np.int64))
        self.param_actions.append(np.asarray(param_action, dtype=np.float32))
        self.primitive_logprobs.append(np.asarray(primitive_logprob, dtype=np.float32))
        self.param_logprobs.append(np.asarray(param_logprob, dtype=np.float32))
        self.values.append(float(value))
        self.disabled_masks.append(np.asarray(disabled_mask, dtype=np.float32))
        self.agent_active_masks.append(np.asarray(agent_active_mask, dtype=np.float32))
        self.context_active_masks.append(np.asarray(context_active_mask, dtype=np.float32))
        self.param_active_masks.append(np.asarray(param_active_mask, dtype=np.float32))

    def add_reward(self, reward: float, mask: float):
        self.rewards.append(float(reward))
        self.masks.append(float(mask))


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------

class PPOJALAgent:
    """PPO agent for the JAL hybrid action space on the expandable backbone."""

    def __init__(
        self,
        obs_dim: int,
        num_robots: int = 1,
        a_max: int = A_MAX,
        c_max: int = C_MAX,
        global_dim: int = GLOBAL_DIM_MAX,
        per_agent_dim: int = PER_AGENT_DIM_MAX,
        d_ctx: int = D_CTX_MAX,
        num_primitives: int = NUM_PRIMITIVES_MAX,
        param_dim: int = PARAM_DIM_MAX,
        device: Optional[torch.device] = None,
        hparams: Optional[Dict[str, Any]] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        merged = dict(DEFAULT_HPARAMS)
        if hparams:
            merged.update(hparams)
        self.hparams = merged

        # Structural params (the weight-shape one-way doors).
        self.a_max = int(a_max)
        self.c_max = int(c_max)
        self.global_dim = int(global_dim)
        self.per_agent_dim = int(per_agent_dim)
        self.d_ctx = int(d_ctx)
        self.num_primitives = int(num_primitives)
        self.param_dim = int(param_dim)
        # Retained for back-compat / logging only — NOT a shape driver.
        self.num_robots = int(num_robots)

        self.model = JALActorCritic(
            obs_dim=obs_dim,
            a_max=self.a_max,
            c_max=self.c_max,
            global_dim=self.global_dim,
            per_agent_dim=self.per_agent_dim,
            d_ctx=self.d_ctx,
            num_primitives=self.num_primitives,
            param_dim=self.param_dim,
            feature_dim=int(self.hparams["feature_dim"]),
            num_heads=int(self.hparams["num_heads"]),
        ).to(self.device)
        self._apply_param_log_std_init("init")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate_initial"])

        self.buffer = RolloutBuffer()
        # Progress fraction in [0, 1] for LR/entropy schedules — trainer sets this.
        self._progress_remaining: float = 1.0
        self.last_metrics: Dict[str, float] = {}

    # -- helpers ----------------------------------------------------------

    def _default_agent_mask(self) -> np.ndarray:
        """Fallback when the env doesn't supply a mask: first `num_robots` active."""
        m = np.zeros(self.a_max, dtype=np.float32)
        m[: max(1, min(self.num_robots, self.a_max))] = 1.0
        return m

    def _default_param_mask(self) -> np.ndarray:
        """Live params (first PARAM_DIM) active, reserved params off."""
        m = np.zeros(self.param_dim, dtype=np.float32)
        m[:PARAM_DIM] = 1.0
        return m

    def _apply_param_log_std_init(self, where: str) -> None:
        """Reset param_log_std to hparams['param_log_std_init'] if set.

        Used to wipe a warm-start checkpoint whose std drifted to the clamp
        ceiling (zero-gradient pin). No-op when the hparam is None (legacy)."""
        init = self.hparams.get("param_log_std_init")
        if init is None:
            return
        with torch.no_grad():
            self.model.param_log_std.data.fill_(float(init))
        logger.info("param_log_std reset to %.3f (std=%.3f) on %s",
                    float(init), float(np.exp(float(init))), where)

    def _reserved_disabled_mask(self, disabled_actions: Optional[Sequence[str]]) -> np.ndarray:
        """(a_max, num_primitives) with 1=enabled. Named-disabled and reserved
        primitive slots (index >= len(PRIMITIVE_NAMES)) are set to 0."""
        mask = np.ones((self.a_max, self.num_primitives), dtype=np.float32)
        mask[:, NUM_PRIMITIVES:] = 0.0  # reserved slots always disabled
        if disabled_actions:
            for name in disabled_actions:
                if name in PRIMITIVE_NAMES:
                    mask[:, PRIMITIVE_NAMES.index(name)] = 0.0
        return mask

    # -- scheduling -------------------------------------------------------

    def set_progress_remaining(self, fraction: float):
        """fraction = 1 - (timesteps_done / total_timesteps); used for schedules."""
        self._progress_remaining = float(max(0.0, min(1.0, fraction)))

    def _current_lr(self) -> float:
        f = self._progress_remaining
        lr_i = self.hparams["learning_rate_initial"]
        lr_f = self.hparams["learning_rate_final"]
        return lr_f + (lr_i - lr_f) * f

    def _current_ent_coef(self) -> float:
        f = self._progress_remaining
        ei = self.hparams["ent_coef_initial"]
        ef = self.hparams["ent_coef_final"]
        return ef + (ei - ef) * f

    # -- action sampling --------------------------------------------------

    def sample_action(
        self,
        obs: np.ndarray,
        disabled_actions: Optional[Sequence[str]] = None,
        agent_active_mask: Optional[np.ndarray] = None,
        context_active_mask: Optional[np.ndarray] = None,
        param_active_mask: Optional[np.ndarray] = None,
        primitive_valid_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Sample a joint action over all a_max agent slots.

        Returns (action_dict, transition_dict) where:
          action_dict = {
            "primitive_idx": np.ndarray[a_max] (int),
            "params":        np.ndarray[a_max * param_dim] (float, flattened),
          }
        Inactive agent slots still produce (ignored) actions; the env applies
        only active slots and the masks zero them out of the loss.
        """
        if agent_active_mask is None:
            agent_active_mask = self._default_agent_mask()
        else:
            agent_active_mask = np.asarray(agent_active_mask, dtype=np.float32)
        if context_active_mask is None:
            context_active_mask = np.zeros(self.c_max, dtype=np.float32)
        else:
            context_active_mask = np.asarray(context_active_mask, dtype=np.float32)
        if param_active_mask is None:
            param_active_mask = self._default_param_mask()
        else:
            param_active_mask = np.asarray(param_active_mask, dtype=np.float32)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        ag_mask_t = torch.as_tensor(agent_active_mask, dtype=torch.float32, device=self.device)
        cx_mask_t = torch.as_tensor(context_active_mask, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            primitive_logits, param_mean, param_std, value = self.model(obs_t, ag_mask_t, cx_mask_t)
        primitive_logits = primitive_logits.squeeze(0)  # (a_max, num_primitives)
        param_mean = param_mean.squeeze(0)              # (a_max, param_dim)

        # Disabled / reserved primitive masking — finite large-negative offset
        # (log(1e-45) ≈ -103.6) rather than -inf, so entropy stays finite.
        disabled_mask = self._reserved_disabled_mask(disabled_actions)
        if primitive_valid_mask is not None:
            runtime_mask = np.asarray(primitive_valid_mask, dtype=np.float32)
            expected = (self.a_max, self.num_primitives)
            if runtime_mask.shape != expected:
                raise ValueError(
                    f"primitive_valid_mask shape {runtime_mask.shape} != {expected}"
                )
            disabled_mask = disabled_mask * np.clip(runtime_mask, 0.0, 1.0)
            if np.any(disabled_mask.sum(axis=1) <= 0.0):
                raise ValueError("primitive_valid_mask disabled every primitive for a slot")
        mask_t = torch.as_tensor(disabled_mask, dtype=torch.float32, device=self.device)
        masked_logits = primitive_logits + torch.log(mask_t + 1e-45)

        prim_dist = Categorical(logits=masked_logits)
        if deterministic:
            primitive_action = torch.argmax(masked_logits, dim=-1)
        else:
            primitive_action = prim_dist.sample()
        primitive_logprob = prim_dist.log_prob(primitive_action)  # (a_max,)

        param_dist = Normal(param_mean, param_std)
        if deterministic:
            param_action = param_mean
        else:
            param_action = param_dist.sample()
        # Bounded action with a correct density. Hard clipping maps an interval
        # of Gaussian samples to each rail while retaining different Gaussian
        # log-probabilities, violating PPO's action/likelihood correspondence.
        param_action_squashed = torch.tanh(param_action)
        # Per-slot param mask: only the SELECTED primitive's dims count toward the
        # logprob (and therefore its gradient), AND'd with the stage mask so a
        # stage-disabled dim stays off. A `kick`/`approach_ball` step contributes
        # zero param logprob ⇒ the dribble-target (Dx,Dy) head gets no gradient
        # from it. (Was a stage-wide mask applied to every step.)
        prim_idx_np = primitive_action.cpu().numpy().astype(np.int64)  # (a_max,)
        per_slot_param_mask = (
            PRIMITIVE_PARAM_MASK[prim_idx_np] * param_active_mask[None, :]
        ).astype(np.float32)  # (a_max, param_dim)
        pa_mask_t = torch.as_tensor(per_slot_param_mask, dtype=torch.float32, device=self.device)
        squash_log_jac = torch.log(1.0 - param_action_squashed.pow(2) + 1e-6)
        param_logprob = (
            (param_dist.log_prob(param_action) - squash_log_jac) * pa_mask_t
        ).sum(dim=-1)  # (a_max,)

        action = {
            "primitive_idx": primitive_action.cpu().numpy().astype(np.int64),
            "params": param_action_squashed.cpu().numpy().reshape(-1).astype(np.float32),
            "runtime_mask_applied": primitive_valid_mask is not None,
        }
        transition = {
            "obs": obs_t.squeeze(0).cpu().numpy(),
            "primitive_action": primitive_action.cpu().numpy(),
            "param_action": param_action.cpu().numpy(),  # unclamped sample for logprob consistency
            "primitive_logprob": primitive_logprob.cpu().numpy(),
            "param_logprob": param_logprob.cpu().numpy(),
            "value": float(value.squeeze().item()),
            "disabled_mask": disabled_mask,
            "agent_active_mask": agent_active_mask,
            "context_active_mask": context_active_mask,
            "param_active_mask": per_slot_param_mask,  # (a_max, param_dim) per selected primitive
        }
        return action, transition

    def store_transition(self, transition: Dict[str, Any]):
        self.buffer.add(
            obs=transition["obs"],
            primitive_action=transition["primitive_action"],
            param_action=transition["param_action"],
            primitive_logprob=transition["primitive_logprob"],
            param_logprob=transition["param_logprob"],
            value=transition["value"],
            disabled_mask=transition["disabled_mask"],
            agent_active_mask=transition["agent_active_mask"],
            context_active_mask=transition["context_active_mask"],
            param_active_mask=transition["param_active_mask"],
        )

    def store_reward(self, reward: float, mask: float):
        self.buffer.add_reward(reward, mask)

    @property
    def rollout_full(self) -> bool:
        return len(self.buffer) >= int(self.hparams["rollout_size"])

    # -- update -----------------------------------------------------------

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        masks: List[float],
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        gamma = float(self.hparams["gamma"])
        lam = float(self.hparams["gae_lambda"])
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            delta = rewards[t] + gamma * next_value * masks[t] - values[t]
            gae = delta + gamma * lam * masks[t] * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + np.asarray(values, dtype=np.float32)
        return advantages, returns

    def _bootstrap_value(
        self,
        last_obs: Optional[np.ndarray],
        last_agent_mask: Optional[np.ndarray],
        last_context_mask: Optional[np.ndarray],
    ) -> float:
        """Compute bootstrap value for a non-terminal last observation."""
        if last_obs is None:
            return 0.0
        if last_agent_mask is None:
            last_agent_mask = np.zeros(self.a_max, dtype=np.float32)
        if last_context_mask is None:
            last_context_mask = np.zeros(self.c_max, dtype=np.float32)
        with torch.no_grad():
            last_t = torch.as_tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            la = torch.as_tensor(last_agent_mask, dtype=torch.float32, device=self.device)
            lc = torch.as_tensor(last_context_mask, dtype=torch.float32, device=self.device)
            _, _, _, lv = self.model(last_t, la, lc)
        return float(lv.squeeze().item())

    def _ppo_update_from_tensors(
        self,
        obs: torch.Tensor,
        prim_actions: torch.Tensor,
        param_actions: torch.Tensor,
        old_prim_logp: torch.Tensor,
        old_param_logp: torch.Tensor,
        old_values: torch.Tensor,
        disabled_masks: torch.Tensor,
        agent_masks: torch.Tensor,
        context_masks: torch.Tensor,
        param_masks: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Run the PPO minibatch update loop on pre-assembled tensors with pre-computed GAE."""
        # Normalize + clip advantages.
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -float(self.hparams["advantage_clip"]), float(self.hparams["advantage_clip"]))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self._current_lr()
        ent_coef = self._current_ent_coef()
        # Separate entropy coef for the continuous params; None ⇒ legacy (== ent_coef).
        # 0.0 stops the bonus inflating param_log_std to the clamp ceiling.
        _pec = self.hparams.get("param_ent_coef")
        param_ent_coef = ent_coef if _pec is None else float(_pec)

        clip_range = float(self.hparams["clip_range"])
        vclip_range = float(self.hparams["value_clip_range"])
        target_kl = float(self.hparams["target_kl"])
        n_epochs = int(self.hparams["n_epochs"])
        minibatch = int(self.hparams["minibatch_size"])
        max_grad_norm = float(self.hparams["max_grad_norm"])
        vf_coef = float(self.hparams["vf_coef"])
        kl_halve = float(self.hparams["kl_lr_halve_factor"])

        n = obs.shape[0]
        indices = np.arange(n)

        last_kl = 0.0
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        last_prim_entropy = 0.0
        early_stop = False

        def _joint_logp(prim_logits, param_mean_b, param_std_b1, b_prim, b_param,
                        b_disabled, b_agent, b_param_active, want_entropy=False):
            prim_logits = prim_logits + torch.log(b_disabled + 1e-45)
            prim_dist = Categorical(logits=prim_logits)
            prim_logp = prim_dist.log_prob(b_prim)
            std_b = param_std_b1.unsqueeze(0).expand_as(param_mean_b)
            param_dist = Normal(param_mean_b, std_b)
            # b_param_active is per-slot (B, A, param_dim): only the SELECTED
            # primitive's dims contribute, so non-param primitives add zero.
            squashed = torch.tanh(b_param)
            squash_log_jac = torch.log(1.0 - squashed.pow(2) + 1e-6)
            param_logp = (
                (param_dist.log_prob(b_param) - squash_log_jac) * b_param_active
            ).sum(dim=-1)
            logp = ((prim_logp + param_logp) * b_agent).sum(dim=-1)
            if not want_entropy:
                return logp
            # Categorical (primitive) and Gaussian (param) entropies kept separate
            # so they can carry different coefficients in the loss.
            prim_entropy = ((prim_dist.entropy() * b_agent).sum(dim=-1)).mean()
            # Monte-Carlo entropy of the tanh-transformed distribution using
            # the rollout latent sample. param_ent_coef is 0 in Stage 2g, but
            # retaining the correct statistic keeps future stages coherent.
            param_ent = (
                -(param_dist.log_prob(b_param) - squash_log_jac) * b_param_active
            ).sum(dim=-1)
            param_entropy = ((param_ent * b_agent).sum(dim=-1)).mean()
            return logp, prim_entropy, param_entropy

        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, minibatch):
                idx = indices[start : start + minibatch]
                if len(idx) == 0:
                    continue
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.device)
                b_obs = obs[idx_t]
                b_prim = prim_actions[idx_t]
                b_param = param_actions[idx_t]
                b_old_prim_logp = old_prim_logp[idx_t]
                b_old_param_logp = old_param_logp[idx_t]
                b_old_values = old_values[idx_t]
                b_returns = returns[idx_t]
                b_adv = advantages[idx_t]
                b_disabled = disabled_masks[idx_t]
                b_agent = agent_masks[idx_t]
                b_context = context_masks[idx_t]
                b_param_active = param_masks[idx_t]

                prim_logits, param_mean, param_std, value = self.model(b_obs, b_agent, b_context)
                logp_new, prim_entropy, param_entropy = _joint_logp(
                    prim_logits, param_mean, param_std, b_prim, b_param,
                    b_disabled, b_agent, b_param_active, want_entropy=True,
                )
                logp_old = ((b_old_prim_logp + b_old_param_logp) * b_agent).sum(dim=-1)
                ratio = torch.exp(logp_new - logp_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value = value.squeeze(-1)
                v_unclipped = (value - b_returns).pow(2)
                v_clipped_pred = b_old_values + torch.clamp(value - b_old_values, -vclip_range, vclip_range)
                v_clipped = (v_clipped_pred - b_returns).pow(2)
                value_loss = torch.max(v_unclipped, v_clipped).mean()

                total_loss = (policy_loss + vf_coef * value_loss
                              - ent_coef * prim_entropy - param_ent_coef * param_entropy)

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float((prim_entropy + param_entropy).item())
                last_prim_entropy = float(prim_entropy.item())

            with torch.no_grad():
                prim_logits_a, param_mean_a, param_std_a, _ = self.model(obs, agent_masks, context_masks)
                logp_new_a = _joint_logp(
                    prim_logits_a, param_mean_a, param_std_a, prim_actions, param_actions,
                    disabled_masks, agent_masks, param_masks, want_entropy=False,
                )
                logp_old_a = ((old_prim_logp + old_param_logp) * agent_masks).sum(dim=-1)
                approx_kl = ((logp_old_a - logp_new_a) ** 2).mean().item() / 2.0
            last_kl = float(approx_kl)
            if approx_kl > kl_halve * target_kl:
                for pg in self.optimizer.param_groups:
                    pg["lr"] *= 0.5
            if approx_kl > 1.5 * target_kl:
                early_stop = True
                break

        self.last_metrics = {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "categorical_entropy": last_prim_entropy,
            "approx_kl": last_kl,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "ent_coef": float(ent_coef),
            "early_stop": float(1.0 if early_stop else 0.0),
        }
        return self.last_metrics

    def update_from_rollout_segments(
        self,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """PPO update from N independent env rollouts, computing GAE per segment.

        Each segment dict must contain:
            obs, primitive_actions, param_actions, primitive_logprobs,
            param_logprobs, values (List[float]), rewards (List[float]),
            masks (List[float]), disabled_masks, agent_active_masks,
            context_active_masks, param_active_masks
        Optional keys:
            last_obs (np.ndarray or None) — current obs if mid-episode at segment end
            last_agent_mask, last_context_mask — active masks for last_obs
        """
        if not segments:
            return {}

        all_obs, all_prim, all_param = [], [], []
        all_old_prim_logp, all_old_param_logp = [], []
        all_old_values: List[float] = []
        all_advantages: List[float] = []
        all_returns: List[float] = []
        all_disabled, all_agent, all_context, all_param_active = [], [], [], []

        for seg in segments:
            last_value = self._bootstrap_value(
                seg.get("last_obs"),
                seg.get("last_agent_mask"),
                seg.get("last_context_mask"),
            )
            adv_np, ret_np = self._compute_gae(
                rewards=seg["rewards"],
                values=seg["values"],
                masks=seg["masks"],
                last_value=last_value,
            )
            all_advantages.extend(adv_np.tolist())
            all_returns.extend(ret_np.tolist())
            all_obs.extend(seg["obs"])
            all_prim.extend(seg["primitive_actions"])
            all_param.extend(seg["param_actions"])
            all_old_prim_logp.extend(seg["primitive_logprobs"])
            all_old_param_logp.extend(seg["param_logprobs"])
            all_old_values.extend(seg["values"])
            all_disabled.extend(seg["disabled_masks"])
            all_agent.extend(seg["agent_active_masks"])
            all_context.extend(seg["context_active_masks"])
            all_param_active.extend(seg["param_active_masks"])

        obs_t = torch.as_tensor(np.stack(all_obs), dtype=torch.float32, device=self.device)
        prim_t = torch.as_tensor(np.stack(all_prim), dtype=torch.long, device=self.device)
        param_t = torch.as_tensor(np.stack(all_param), dtype=torch.float32, device=self.device)
        old_prim_logp_t = torch.as_tensor(np.stack(all_old_prim_logp), dtype=torch.float32, device=self.device)
        old_param_logp_t = torch.as_tensor(np.stack(all_old_param_logp), dtype=torch.float32, device=self.device)
        old_val_t = torch.as_tensor(np.asarray(all_old_values, dtype=np.float32), device=self.device)
        disabled_t = torch.as_tensor(np.stack(all_disabled), dtype=torch.float32, device=self.device)
        agent_t = torch.as_tensor(np.stack(all_agent), dtype=torch.float32, device=self.device)
        context_t = torch.as_tensor(np.stack(all_context), dtype=torch.float32, device=self.device)
        param_active_t = torch.as_tensor(np.stack(all_param_active), dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(np.asarray(all_advantages, dtype=np.float32), device=self.device)
        ret_t = torch.as_tensor(np.asarray(all_returns, dtype=np.float32), device=self.device)

        return self._ppo_update_from_tensors(
            obs_t, prim_t, param_t, old_prim_logp_t, old_param_logp_t, old_val_t,
            disabled_t, agent_t, context_t, param_active_t, adv_t, ret_t,
        )

    def update(self, last_obs: Optional[np.ndarray] = None,
               last_agent_active_mask: Optional[np.ndarray] = None,
               last_context_active_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Run a PPO update using buffered transitions."""
        if len(self.buffer) == 0:
            return {}

        if last_agent_active_mask is None and self.buffer.agent_active_masks:
            last_agent_active_mask = self.buffer.agent_active_masks[-1]
        if last_context_active_mask is None and self.buffer.context_active_masks:
            last_context_active_mask = self.buffer.context_active_masks[-1]
        last_value = self._bootstrap_value(last_obs, last_agent_active_mask, last_context_active_mask)

        # Pull tensors from buffer.
        obs = torch.as_tensor(np.stack(self.buffer.observations), dtype=torch.float32, device=self.device)
        prim_actions = torch.as_tensor(np.stack(self.buffer.primitive_actions), dtype=torch.long, device=self.device)
        param_actions = torch.as_tensor(np.stack(self.buffer.param_actions), dtype=torch.float32, device=self.device)
        old_prim_logp = torch.as_tensor(np.stack(self.buffer.primitive_logprobs), dtype=torch.float32, device=self.device)
        old_param_logp = torch.as_tensor(np.stack(self.buffer.param_logprobs), dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(np.asarray(self.buffer.values, dtype=np.float32), device=self.device)
        disabled_masks = torch.as_tensor(np.stack(self.buffer.disabled_masks), dtype=torch.float32, device=self.device)
        agent_masks = torch.as_tensor(np.stack(self.buffer.agent_active_masks), dtype=torch.float32, device=self.device)
        context_masks = torch.as_tensor(np.stack(self.buffer.context_active_masks), dtype=torch.float32, device=self.device)
        param_masks = torch.as_tensor(np.stack(self.buffer.param_active_masks), dtype=torch.float32, device=self.device)

        advantages_np, returns_np = self._compute_gae(
            rewards=self.buffer.rewards,
            values=self.buffer.values,
            masks=self.buffer.masks,
            last_value=last_value,
        )
        advantages = torch.as_tensor(advantages_np, device=self.device)
        returns = torch.as_tensor(returns_np, device=self.device)

        metrics = self._ppo_update_from_tensors(
            obs, prim_actions, param_actions,
            old_prim_logp, old_param_logp, old_values,
            disabled_masks, agent_masks, context_masks, param_masks,
            advantages, returns,
        )
        self.buffer.clear()
        return metrics

    # -- persistence ------------------------------------------------------

    def _structural(self) -> Dict[str, int]:
        return {
            "a_max": self.a_max,
            "c_max": self.c_max,
            "global_dim": self.global_dim,
            "per_agent_dim": self.per_agent_dim,
            "d_ctx": self.d_ctx,
            "num_primitives": self.num_primitives,
            "param_dim": self.param_dim,
            "feature_dim": int(self.hparams["feature_dim"]),
            "num_heads": int(self.hparams["num_heads"]),
            "obs_dim": self.model.obs_dim,
        }

    def save(self, filepath: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "structural": self._structural(),
                "num_robots": self.num_robots,
                "obs_dim": self.model.obs_dim,
                "hparams": self.hparams,
            },
            filepath,
        )

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        struct = ckpt.get("structural")
        if struct is not None:
            mine = self._structural()
            mismatched = {k: (struct.get(k), mine.get(k)) for k in mine if struct.get(k) != mine.get(k)}
            if mismatched:
                raise ValueError(
                    f"Checkpoint structural params differ from this agent — warm-start "
                    f"would change weight shapes: {mismatched}. This is a one-way door; "
                    f"do not change MAX widths mid-curriculum."
                )
        self.model.load_state_dict(ckpt["model_state_dict"])
        # Wipe a checkpoint whose param_log_std was pinned at the old ceiling, so
        # the continuous std starts below LOG_STD_MAX with live gradient again.
        self._apply_param_log_std_init("warm-start")
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            # New optimizer structure won't always match — restart optimizer state.
            pass

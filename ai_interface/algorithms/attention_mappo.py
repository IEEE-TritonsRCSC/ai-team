"""Attention-based MAPPO for 6-agent RoboCup SSL team coordination.

AttentionMAPPOAgent extends HSMMAPPOAgent with:
- Multi-head self-attention over team embeddings (permutation-equivariant)
- Mean-pooled centralized critic (N-agnostic; warm-starts across curriculum stages)
- Minibatch PPO updates (scales to 6-agent rollouts)
- Three separate optimizers: encoder, actor, critic
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from ai_interface.algorithms.robot_attention import RobotAttentionBlock
from ai_interface.hsm.state_machine import Role


@dataclass
class AttentionMAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_encoder: float = 5e-4
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    rollout_length: int = 2048
    minibatch_size: int = 256
    min_std: float = 0.1
    agent_embed_dim: int = 128
    num_attn_heads: int = 4
    num_attn_layers: int = 2
    actor_hidden: Tuple[int, int] = (256, 256)


class AgentEncoder(nn.Module):
    """Project per-agent (obs + role one-hot) to a fixed-size embedding."""

    def __init__(self, obs_dim: int, role_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + role_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, obs_with_role: torch.Tensor) -> torch.Tensor:
        return self.net(obs_with_role)


class TeamAttentionEncoder(nn.Module):
    """Permutation-equivariant attention stack over N agent embeddings.

    Reuses RobotAttentionBlock from robot_attention.py (no code duplication).
    Input/output: (B, N, embed_dim).
    """

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.blocks = nn.Sequential(*[
            RobotAttentionBlock(
                hidden_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=embed_dim * 2,
            )
            for _ in range(num_layers)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.blocks(h)


class AttentionRoleActor(nn.Module):
    """Per-role actor: embedding -> Normal(action_dim)."""

    def __init__(self, embed_dim: int, action_dim: int, hidden: Tuple[int, int]):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(embed_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(h2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(z)
        return self.mean_head(x), self.log_std


class CentralizedPoolingCritic(nn.Module):
    """Centralized value function via mean-pool over N agent embeddings.

    Mean-pooling is N-agnostic: the same weights work for any team size,
    allowing warm-starting from a 1-agent curriculum stage to 6 agents.
    """

    def __init__(self, embed_dim: int, hidden: Tuple[int, int] = (256, 256)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(embed_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, team_embeddings: torch.Tensor) -> torch.Tensor:
        # team_embeddings: (B, N, embed_dim) or (N, embed_dim)
        if team_embeddings.dim() == 2:
            team_embeddings = team_embeddings.unsqueeze(0)
        pooled = team_embeddings.mean(dim=1)  # (B, embed_dim)
        return self.net(pooled)              # (B, 1)


class AttentionMAPPOAgent:
    """MAPPO with attention-based team encoding and mean-pooled centralized critic.

    Designed for 6-agent RoboCup SSL. Matches the HSMMAPPOAgent API so existing
    trainer infrastructure (BaseTrainer, checkpointing, logging) can be reused.

    Key improvements over HSMMAPPOAgent:
    - Attention contextualizes each agent's embedding with all teammates
    - Mean-pooled critic is N-agnostic (warm-starts across curriculum stages)
    - Minibatch PPO (not full-batch) for stability with long rollouts
    - Three separate optimizers (encoder/actor/critic) for independent LR tuning
    """

    ROLES = [Role.STRIKER, Role.SUPPORT, Role.DEFENDER, Role.GOALIE]
    ROLE_DIM = 4
    _ACTION_LOW = [-2.0, -2.0, -2.0, 0.0]
    _ACTION_HIGH = [2.0, 2.0, 2.0, 1.0]

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int = 4,
        config: Optional[AttentionMAPPOConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or AttentionMAPPOConfig()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cpu")

        cfg = self.config
        embed_dim = cfg.agent_embed_dim

        self.role_to_idx = {r: i for i, r in enumerate(self.ROLES)}

        self.agent_encoder = AgentEncoder(obs_dim, self.ROLE_DIM, embed_dim).to(self.device)
        self.team_attention = TeamAttentionEncoder(
            embed_dim, cfg.num_attn_heads, cfg.num_attn_layers
        ).to(self.device)

        # Four role actors, weight-shared within each role (2 strikers → same actor)
        self.actors = nn.ModuleDict({
            role.value: AttentionRoleActor(embed_dim, action_dim, cfg.actor_hidden).to(self.device)
            for role in self.ROLES
        })

        self.critic = CentralizedPoolingCritic(embed_dim).to(self.device)

        # Separate optimizers: encoder is shared between actor and critic loss paths
        encoder_params = (
            list(self.agent_encoder.parameters())
            + list(self.team_attention.parameters())
        )
        self.encoder_optimizer = optim.Adam(encoder_params, lr=cfg.lr_encoder)
        self.actor_optimizer = optim.Adam(self.actors.parameters(), lr=cfg.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self._action_lo = torch.tensor(self._ACTION_LOW, device=self.device)
        self._action_hi = torch.tensor(self._ACTION_HIGH, device=self.device)

        self.memory: List[Dict] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def batch_ready(self) -> bool:
        return len(self.memory) >= self.config.rollout_length

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _role_one_hot(self, role: Role) -> torch.Tensor:
        out = torch.zeros(self.ROLE_DIM, dtype=torch.float32, device=self.device)
        out[self.role_to_idx[role]] = 1.0
        return out

    def _encode_team(
        self,
        observations: List[torch.Tensor],
        roles: List[Role],
    ) -> torch.Tensor:
        """Encode N agents: (obs + role) → AgentEncoder → TeamAttention → (N, embed_dim)."""
        obs_stack = torch.stack(observations, dim=0)           # (N, obs_dim)
        role_oh = torch.stack([self._role_one_hot(r) for r in roles], dim=0)  # (N, ROLE_DIM)
        obs_with_role = torch.cat([obs_stack, role_oh], dim=-1)  # (N, obs_dim+ROLE_DIM)

        embeddings = self.agent_encoder(obs_with_role)         # (N, embed_dim)
        embeddings = embeddings.unsqueeze(0)                   # (1, N, embed_dim)
        embeddings = self.team_attention(embeddings)           # (1, N, embed_dim)
        return embeddings.squeeze(0)                           # (N, embed_dim)

    # ------------------------------------------------------------------
    # Public API (mirrors HSMMAPPOAgent)
    # ------------------------------------------------------------------

    def select_actions(
        self,
        observations: List[np.ndarray],
        roles: List[Role],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, List[torch.Tensor], torch.Tensor]:
        """Select continuous actions for all agents.

        Returns:
            actions: np.ndarray (N, 4)
            logprobs: List[Tensor], one scalar per agent
            value: Tensor scalar
        """
        obs_tensors = [
            torch.as_tensor(o, dtype=torch.float32, device=self.device) for o in observations
        ]
        with torch.no_grad():
            embeddings = self._encode_team(obs_tensors, roles)       # (N, embed_dim)
            value = self.critic(embeddings.unsqueeze(0)).squeeze()   # scalar

            actions: List[np.ndarray] = []
            logprobs: List[torch.Tensor] = []

            for i in range(self.num_agents):
                role_key = roles[i].value
                mean, log_std = self.actors[role_key](embeddings[i])
                std = torch.clamp(log_std.exp(), min=self.config.min_std)
                dist = Normal(mean, std)
                action = mean if deterministic else dist.sample()
                action = torch.clamp(action, self._action_lo, self._action_hi)
                logprobs.append(dist.log_prob(action).sum().detach())
                actions.append(action.detach().cpu().numpy())

        return np.asarray(actions, dtype=np.float32), logprobs, value.detach()

    def store_transition(
        self,
        observations: List[np.ndarray],
        roles: List[Role],
        actions: np.ndarray,
        logprobs: List[torch.Tensor],
        value: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        self.memory.append({
            "obs": [
                torch.as_tensor(o, dtype=torch.float32, device=self.device) for o in observations
            ],
            "roles": list(roles),
            "actions": torch.as_tensor(actions, dtype=torch.float32, device=self.device),
            "logprobs": torch.stack(logprobs).to(self.device),  # (N,)
            "value": value.to(self.device),
            "reward": float(reward),
            "done": float(done),
        })

    def update(self) -> Dict[str, float]:
        """Minibatch PPO update over the stored rollout. Clears memory afterward."""
        if not self.memory:
            return {}

        T = len(self.memory)
        values = torch.stack([m["value"] for m in self.memory])    # (T,)
        rewards = [m["reward"] for m in self.memory]
        masks = [1.0 - m["done"] for m in self.memory]

        advantages = self._compute_gae(rewards, masks, values.detach().cpu().tolist())
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = adv_t + values

        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        old_logprobs = torch.stack([m["logprobs"] for m in self.memory])  # (T, N)
        all_actions = torch.stack([m["actions"] for m in self.memory])    # (T, N, A)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        indices = list(range(T))
        for _ in range(self.config.ppo_epochs):
            random.shuffle(indices)
            for start in range(0, T, self.config.minibatch_size):
                batch_idx = indices[start: start + self.config.minibatch_size]
                if not batch_idx:
                    continue

                mb_new_logprobs: List[torch.Tensor] = []
                mb_entropies: List[torch.Tensor] = []
                mb_team_embs: List[torch.Tensor] = []

                for t in batch_idx:
                    m = self.memory[t]
                    embs = self._encode_team(m["obs"], m["roles"])  # (N, embed_dim)
                    mb_team_embs.append(embs.unsqueeze(0))          # (1, N, embed_dim)

                    step_lp: List[torch.Tensor] = []
                    step_ent: List[torch.Tensor] = []
                    for i in range(self.num_agents):
                        mean, log_std = self.actors[m["roles"][i].value](embs[i])
                        std = torch.clamp(log_std.exp(), min=self.config.min_std)
                        dist = Normal(mean, std)
                        step_lp.append(dist.log_prob(all_actions[t, i]).sum())
                        step_ent.append(dist.entropy().sum())
                    mb_new_logprobs.append(torch.stack(step_lp))    # (N,)
                    mb_entropies.append(torch.stack(step_ent).mean())

                new_lp_t = torch.stack(mb_new_logprobs)            # (B, N)
                old_lp_t = old_logprobs[batch_idx]                 # (B, N)
                mb_adv = adv_t[batch_idx]                           # (B,)
                mb_ret = returns[batch_idx]                         # (B,)

                ratio = torch.exp(new_lp_t - old_lp_t)
                adv_expand = mb_adv.unsqueeze(-1).expand_as(ratio)
                surr1 = ratio * adv_expand
                surr2 = torch.clamp(
                    ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
                ) * adv_expand
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy = torch.stack(mb_entropies).mean()

                team_embs_t = torch.cat(mb_team_embs, dim=0)       # (B, N, embed_dim)
                predicted_v = self.critic(team_embs_t).squeeze(-1) # (B,)
                critic_loss = (mb_ret - predicted_v).pow(2).mean()

                total_loss = (
                    actor_loss
                    + self.config.value_coef * critic_loss
                    - self.config.entropy_coef * entropy
                )

                self.encoder_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                all_params = (
                    list(self.agent_encoder.parameters())
                    + list(self.team_attention.parameters())
                    + list(self.actors.parameters())
                    + list(self.critic.parameters())
                )
                nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)

                self.encoder_optimizer.step()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_actor_loss += float(actor_loss.item())
                total_critic_loss += float(critic_loss.item())
                total_entropy += float(entropy.item())
                n_updates += 1

        self.memory = []
        denom = max(1, n_updates)
        return {
            "actor_loss": total_actor_loss / denom,
            "critic_loss": total_critic_loss / denom,
            "entropy": total_entropy / denom,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "num_agents": self.num_agents,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "config": self.config,
                "agent_encoder": self.agent_encoder.state_dict(),
                "team_attention": self.team_attention.state_dict(),
                "actors": {k: v.state_dict() for k, v in self.actors.items()},
                "critic": self.critic.state_dict(),
                "encoder_opt": self.encoder_optimizer.state_dict(),
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str, weights_only: bool = False) -> None:
        """Load checkpoint. Set weights_only=True to skip optimizer state (warm-start)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.agent_encoder.load_state_dict(ckpt["agent_encoder"])
        self.team_attention.load_state_dict(ckpt["team_attention"])
        for key, model in self.actors.items():
            if key in ckpt.get("actors", {}):
                model.load_state_dict(ckpt["actors"][key])
        self.critic.load_state_dict(ckpt["critic"])
        if not weights_only:
            if "encoder_opt" in ckpt:
                self.encoder_optimizer.load_state_dict(ckpt["encoder_opt"])
            if "actor_opt" in ckpt:
                self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
            if "critic_opt" in ckpt:
                self.critic_optimizer.load_state_dict(ckpt["critic_opt"])

    # ------------------------------------------------------------------
    # Internal: GAE
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        rewards: List[float],
        masks: List[float],
        values: List[float],
    ) -> List[float]:
        advantages: List[float] = []
        gae = 0.0
        values_boot = values + [0.0]
        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.config.gamma * values_boot[t + 1] * masks[t]
                - values_boot[t]
            )
            gae = delta + self.config.gamma * self.config.gae_lambda * masks[t] * gae
            advantages.insert(0, gae)
        return advantages

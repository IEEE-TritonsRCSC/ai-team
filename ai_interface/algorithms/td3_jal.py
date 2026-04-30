"""TD3 algorithm wrapper for centralized JAL training.

This module keeps TD3 implementation details in algorithms/ so trainers can
swap algorithm/network settings without embedding SB3 logic in trainer code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ai_interface.algorithms.base import AlgorithmBase


class JALMLPBackbone(nn.Module):
    """Plain MLP backbone over flattened observation."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JALConvBackbone(nn.Module):
    """Conv1D backbone over robot-wise token sequence."""

    def __init__(
        self,
        num_robots: int,
        per_robot_dim: int,
        global_dim: int,
        conv_channels: int,
        output_dim: int,
    ):
        super().__init__()
        self.num_robots = num_robots
        self.per_robot_dim = per_robot_dim
        self.global_dim = global_dim

        self.conv = nn.Sequential(
            nn.Conv1d(per_robot_dim, conv_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(conv_channels + global_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_part = x[:, : self.global_dim]
        robot_part = x[:, self.global_dim :]
        robot_part = robot_part.view(-1, self.num_robots, self.per_robot_dim).transpose(1, 2)
        conv_feats = self.conv(robot_part).mean(dim=-1)
        return self.proj(torch.cat([global_part, conv_feats], dim=1))


class JALAttentionBackbone(nn.Module):
    """Self-attention backbone over robot tokens, fused with global context."""

    def __init__(
        self,
        num_robots: int,
        per_robot_dim: int,
        global_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.num_robots = num_robots
        self.per_robot_dim = per_robot_dim
        self.global_dim = global_dim

        self.robot_embed = nn.Linear(per_robot_dim, embed_dim)
        self.global_embed = nn.Linear(global_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_part = x[:, : self.global_dim]
        robot_part = x[:, self.global_dim :]
        robot_tokens = robot_part.view(-1, self.num_robots, self.per_robot_dim)

        robot_tokens = self.robot_embed(robot_tokens)
        global_token = self.global_embed(global_part).unsqueeze(1)
        tokens = torch.cat([global_token, robot_tokens], dim=1)

        attended, _ = self.attn(tokens, tokens, tokens)
        global_att = attended[:, 0, :]
        robot_att = attended[:, 1:, :].mean(dim=1)
        return self.fuse(torch.cat([global_att, robot_att], dim=1))


class JALFeatureExtractor(BaseFeaturesExtractor):
    """SB3-compatible extractor that wraps editable PyTorch backbones."""

    def __init__(
        self,
        observation_space: spaces.Box,
        network_type: str = "mlp",
        features_dim: int = 128,
        num_robots: int = 1,
        per_robot_dim: int = 5,
        global_dim: int = 4,
        hidden_dims: list[int] | None = None,
        conv_channels: int = 64,
        embed_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__(observation_space, features_dim)
        input_dim = int(observation_space.shape[0])
        hidden_dims = hidden_dims or [256, 256]

        network_type = str(network_type).lower()
        if network_type == "mlp":
            self.backbone = JALMLPBackbone(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=features_dim,
            )
        elif network_type == "conv":
            self.backbone = JALConvBackbone(
                num_robots=num_robots,
                per_robot_dim=per_robot_dim,
                global_dim=global_dim,
                conv_channels=conv_channels,
                output_dim=features_dim,
            )
        elif network_type == "attention":
            self.backbone = JALAttentionBackbone(
                num_robots=num_robots,
                per_robot_dim=per_robot_dim,
                global_dim=global_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                output_dim=features_dim,
            )
        else:
            raise ValueError(
                f"Unsupported network_type='{network_type}'. Use one of: mlp, conv, attention"
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.backbone(observations)


class TD3JALAlgorithm(AlgorithmBase):
    """Thin wrapper around SB3 TD3 for JAL environments."""

    def __init__(
        self,
        env,
        policy: str = "MlpPolicy",
        learning_rate: float = 1e-3,
        buffer_size: int = 100_000,
        learning_starts: int = 1000,
        batch_size: int = 64,
        tau: float = 0.01,
        gamma: float = 0.9,
        train_freq: int = 1,
        gradient_steps: int = 1,
        action_noise_std: float = 0.05,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        network_type: str = "mlp",
        network_kwargs: dict[str, Any] | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 1,
        device: str = "auto",
        tensorboard_log: str | None = None,
    ):
        if env is None:
            raise ValueError("env must not be None")

        action_shape = getattr(env.action_space, "shape", None)
        if not action_shape:
            raise ValueError("TD3JALAlgorithm requires a continuous Box action space")

        n_actions = int(action_shape[0])
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=float(action_noise_std) * np.ones(n_actions),
        )

        user_policy_kwargs = dict(policy_kwargs or {})
        user_network_kwargs = dict(network_kwargs or {})
        feature_extractor_kwargs = {
            "network_type": network_type,
            **user_network_kwargs,
        }

        # Keep policy heads configurable while making the feature backbone fully editable as PyTorch code.
        merged_policy_kwargs = {
            "features_extractor_class": JALFeatureExtractor,
            "features_extractor_kwargs": feature_extractor_kwargs,
            "net_arch": [256, 256],
            **user_policy_kwargs,
        }

        self.model = TD3(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            policy_kwargs=merged_policy_kwargs,
            verbose=verbose,
            device=device,
            tensorboard_log=tensorboard_log,
        )

    def predict(self, observation, deterministic: bool = True):
        action, _state = self.model.predict(observation, deterministic=deterministic)
        return action

    def learn(self, total_timesteps: int, reset_num_timesteps: bool = False):
        return self.model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=reset_num_timesteps,
        )

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env=None, device: str = "auto"):
        if env is None:
            raise ValueError("env must be provided when loading TD3JALAlgorithm")

        inst = cls.__new__(cls)
        inst.model = TD3.load(path, env=env, device=device)
        return inst

"""Expandable TD3-JAL with modular encoder/decoder for curriculum learning.

This module provides:
- Modular global and per-robot encoders
- Attention-based aggregation for scalable multi-robot handling
- Per-robot decoders with weight transfer on robot addition
- Curriculum learning support (1v0 -> 2v0 -> ...)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GlobalEncoder(nn.Module):
    """Encodes global state (ball position and velocity)."""

    def __init__(self, global_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_obs: (batch, global_dim) - ball state
        Returns:
            (batch, hidden_dim)
        """
        return self.net(global_obs)


class PerRobotEncoder(nn.Module):
    """Encodes a single robot's state (position, velocity, etc)."""

    def __init__(self, per_robot_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(per_robot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, robot_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            robot_obs: (batch, per_robot_dim)
        Returns:
            (batch, hidden_dim)
        """
        return self.net(robot_obs)


class RobotAggregator(nn.Module):
    """Aggregates per-robot features using multi-head attention."""

    def __init__(self, feature_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        robot_features: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            robot_features: (batch, num_robots, feature_dim)
            global_feature: (batch, feature_dim)
        Returns:
            aggregated: (batch, feature_dim)
        """
        # Add global feature as a context token
        batch_size = robot_features.shape[0]
        global_token = global_feature.unsqueeze(1)  # (batch, 1, feature_dim)
        query = torch.cat([global_token, robot_features], dim=1)  # (batch, 1+num_robots, feature_dim)

        attn_out, _ = self.attn(query, query, query)
        # Take the global token output
        aggregated = attn_out[:, 0, :]
        aggregated = self.norm(aggregated + global_feature)
        return aggregated


class PerRobotDecoder(nn.Module):
    """Decodes actions for a single robot."""

    def __init__(self, feature_dim: int = 64, action_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, feature_dim)
        Returns:
            (batch, action_dim)
        """
        return self.net(features)


class ExpandableJALEncoder(nn.Module):
    """
    Expandable encoder that processes variable numbers of robots.
    
    Expands when robots are added without losing previously learned weights.
    """

    def __init__(
        self,
        global_dim: int = 4,
        per_robot_dim: int = 8,
        max_robots: int = 1,
        feature_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.global_dim = global_dim
        self.per_robot_dim = per_robot_dim
        self.max_robots = max_robots
        self.feature_dim = feature_dim

        self.global_encoder = GlobalEncoder(global_dim, feature_dim)
        self.per_robot_encoders = nn.ModuleList(
            [PerRobotEncoder(per_robot_dim, feature_dim) for _ in range(max_robots)]
        )
        self.aggregator = RobotAggregator(feature_dim, num_heads)

    def forward(
        self,
        obs: torch.Tensor,
        num_active_robots: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, global_dim + num_active_robots * per_robot_dim)
            num_active_robots: Number of robots to process
        Returns:
            global_feat: (batch, feature_dim)
            aggregated_feat: (batch, feature_dim)
            per_robot_feats: (batch, num_active_robots, feature_dim)
        """
        batch_size = obs.shape[0]

        # Extract global part
        global_obs = obs[:, : self.global_dim]  # (batch, global_dim)
        global_feat = self.global_encoder(global_obs)

        # Extract and encode per-robot parts
        per_robot_feats = []
        for i in range(num_active_robots):
            start = self.global_dim + i * self.per_robot_dim
            end = start + self.per_robot_dim
            robot_obs = obs[:, start:end]
            robot_feat = self.per_robot_encoders[i](robot_obs)
            per_robot_feats.append(robot_feat)

        per_robot_feats = torch.stack(per_robot_feats, dim=1)  # (batch, num_active_robots, feature_dim)
        aggregated_feat = self.aggregator(per_robot_feats, global_feat)

        return global_feat, aggregated_feat, per_robot_feats

    def expand_robots(self, new_num_robots: int):
        """Expand to support more robots."""
        if new_num_robots > self.max_robots:
            current_size = len(self.per_robot_encoders)
            for _ in range(new_num_robots - current_size):
                self.per_robot_encoders.append(
                    PerRobotEncoder(self.per_robot_dim, self.feature_dim)
                )


class ExpandableJALDecoder(nn.Module):
    """
    Expandable decoder that outputs actions for variable numbers of robots.
    
    Expands when robots are added, initializing new robot decoders randomly.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        action_dim_per_robot: int = 8,
        max_robots: int = 1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim_per_robot = action_dim_per_robot
        self.max_robots = max_robots

        self.per_robot_decoders = nn.ModuleList(
            [PerRobotDecoder(feature_dim, action_dim_per_robot) for _ in range(max_robots)]
        )

    def forward(
        self,
        aggregated_feat: torch.Tensor,
        per_robot_feats: torch.Tensor,
        num_active_robots: int,
    ) -> torch.Tensor:
        """
        Args:
            aggregated_feat: (batch, feature_dim) - global aggregated features
            per_robot_feats: (batch, num_active_robots, feature_dim)
            num_active_robots: Number of robots to decode actions for
        Returns:
            actions: (batch, num_active_robots * action_dim_per_robot)
        """
        actions = []
        for i in range(num_active_robots):
            robot_feat = per_robot_feats[:, i, :]
            robot_action = self.per_robot_decoders[i](robot_feat)
            actions.append(robot_action)

        actions = torch.cat(actions, dim=1)  # (batch, num_active_robots * action_dim_per_robot)
        return actions

    def expand_robots(self, new_num_robots: int):
        """Expand to support more robots."""
        if new_num_robots > self.max_robots:
            current_size = len(self.per_robot_decoders)
            for _ in range(new_num_robots - current_size):
                self.per_robot_decoders.append(
                    PerRobotDecoder(self.feature_dim, self.action_dim_per_robot)
                )


class ExpandableJALBackbone(nn.Module):
    """
    Complete expandable backbone for TD3-JAL.
    
    Handles variable numbers of robots by expanding encoders/decoders.
    """

    def __init__(
        self,
        global_dim: int = 4,
        per_robot_dim: int = 8,
        action_dim_per_robot: int = 8,
        max_robots: int = 1,
        feature_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.global_dim = global_dim
        self.per_robot_dim = per_robot_dim
        self.action_dim_per_robot = action_dim_per_robot
        self.max_robots = max_robots
        self.feature_dim = feature_dim

        self.encoder = ExpandableJALEncoder(
            global_dim=global_dim,
            per_robot_dim=per_robot_dim,
            max_robots=max_robots,
            feature_dim=feature_dim,
            num_heads=num_heads,
        )
        self.decoder = ExpandableJALDecoder(
            feature_dim=feature_dim,
            action_dim_per_robot=action_dim_per_robot,
            max_robots=max_robots,
        )

    def forward(
        self,
        obs: torch.Tensor,
        num_active_robots: int,
    ) -> torch.Tensor:
        """
        Args:
            obs: (batch, global_dim + num_active_robots * per_robot_dim)
            num_active_robots: Number of active robots
        Returns:
            actions: (batch, num_active_robots * action_dim_per_robot)
        """
        global_feat, aggregated_feat, per_robot_feats = self.encoder(obs, num_active_robots)
        actions = self.decoder(aggregated_feat, per_robot_feats, num_active_robots)
        return actions

    def expand_robots(self, new_num_robots: int):
        """Expand to support more robots."""
        self.encoder.expand_robots(new_num_robots)
        self.decoder.expand_robots(new_num_robots)
        self.max_robots = max(self.max_robots, new_num_robots)


class ExpandableJALFeatureExtractor(BaseFeaturesExtractor):
    """SB3-compatible features extractor wrapping the expandable encoder.

    Expects an observation Space that is a Dict with key "observation" mapping
    to a flat Box of shape (global_dim + num_robots * per_robot_dim,).
    The extractor returns a fixed-size feature vector of `feature_dim`.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        global_dim: int = 4,
        per_robot_dim: int = 8,
        max_robots: int = 1,
        feature_dim: int = 64,
        num_heads: int = 4,
        **kwargs,
    ):
        # features_dim is the output dimension expected by SB3 policies
        super().__init__(observation_space, features_dim=feature_dim)

        # Accept either a Dict observation_space or a Box directly.
        if isinstance(observation_space, spaces.Dict):
            obs_space = observation_space.spaces.get("observation")
            if obs_space is None:
                raise ValueError("Dict observation_space must contain 'observation' key")
        else:
            obs_space = observation_space

        input_dim = int(obs_space.shape[0])

        self.global_dim = int(global_dim)
        self.per_robot_dim = int(per_robot_dim)
        self.max_robots = int(max_robots)
        self.feature_dim = int(feature_dim)

        # Build the expandable encoder only — the policy heads (actor/critic)
        # remain the SB3 default MLPs fed by this extractor's output.
        self.encoder = ExpandableJALEncoder(
            global_dim=self.global_dim,
            per_robot_dim=self.per_robot_dim,
            max_robots=self.max_robots,
            feature_dim=self.feature_dim,
            num_heads=int(num_heads),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations passed here are the flattened 'observation' tensor when
        # MultiInputPolicy/CombinedExtractor delegates to this extractor.
        # If SB3 passes a dict, ensure we extract the 'observation' key.
        if isinstance(observations, dict):
            obs = observations["observation"]
        else:
            obs = observations

        # SB3 will provide batch-first tensors.
        # Determine active robots from input size at runtime if possible.
        # Fall back to max_robots if ambiguous.
        try:
            batch_dim = obs.shape[0]
            total_dim = obs.shape[1]
            num_active = max(1, (total_dim - self.global_dim) // self.per_robot_dim)
        except Exception:
            num_active = self.max_robots

        # Encoder returns (global_feat, aggregated_feat, per_robot_feats)
        _g, aggregated, _per = self.encoder(obs, num_active)

        return aggregated

    def expand_robots(self, new_num_robots: int):
        """Expand internal encoder to handle additional robots."""
        self.encoder.expand_robots(new_num_robots)
        self.max_robots = max(self.max_robots, new_num_robots)

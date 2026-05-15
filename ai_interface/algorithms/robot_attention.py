from __future__ import annotations

from typing import Optional

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


class TemporalCNNEncoder(nn.Module):
    """
    Shared per-robot multi-scale temporal CNN.

    B: batch size
    N: number of robots
    M: observation dimension
    T: time window
    D: hidden dimension

    Input:
        x: (B, N, M, T)

    Output:
        h: (B, N, D)
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv1d(obs_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=8, dilation=8),
                nn.ReLU(),
            ),
        ]) # Receptive field = 1 + (kernel_size - 1) * dilation = 1 + 2 * dilation

        self.out_proj = nn.Sequential(
            nn.Conv1d(4 * hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.pool_proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, M, T = x.shape

        x = x.reshape(B * N, M, T)       # (B*N, M, T)

        x = self.in_proj(x)              # (B*N, D, T)

        branch_outs = [branch(x) for branch in self.branches]
        h = torch.cat(branch_outs, dim=1)  # (B*N, 4D, T)

        h = self.out_proj(h)             # (B*N, D, T)
        h_mean = h.mean(dim=-1)
        h_max = h.max(dim=-1).values
        h = torch.cat([h_mean, h_max], dim=-1)
        h = self.pool_proj(h)            # (B*N, D)
        h = h.reshape(B, N, -1)          # (B, N, D)

        return self.norm(h)


class RobotAttentionBlock(nn.Module):
    """
    Permutation-equivariant attention across robots.

    B: batch size
    N: number of robots
    D: hidden dimension

    Input:
        h: (B, N, D)

    Output:
        z: (B, N, D)
    """

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, ff_dim: int = 128):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(h, h, h)      # (B, N, D)
        h = self.norm1(h + attn_out)

        ff_out = self.ff(h)
        h = self.norm2(h + ff_out)

        return h


class MultiRobotFeatureExtractor(nn.Module):
    """
    Full feature extractor:

        per-robot temporal CNN
        -> robot attention
        -> per-robot features

    B: batch size
    N: number of robots
    M: observation dimension
    T: time window
    D: hidden dimension

    Input:
        x: (B, N, M, T)

    Output:
        features: (B, N, D)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_attention_layers: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.encoder = TemporalCNNEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
        )

        self.attention = nn.Sequential(
            *[
                RobotAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=2 * hidden_dim,
                )
                for _ in range(num_attention_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)          # (B, N, D)
        z = self.attention(h)        # (B, N, D)
        return z
    
class ActionHead(nn.Module):
    """
    Shared decoder applied independently to each robot.

    B: batch size
    N: number of robots
    D: hidden dimension
    A: action dimension (per robot)

    Input:
        z: (B, N, D)

    Output:
        action: (B, N * A)
    """

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.head(z)
        B, N, A = z.shape
        return z.reshape(B, N * A)


class SB3MultiRobotFeatureExtractor(BaseFeaturesExtractor):
    """SB3 wrapper around ``MultiRobotFeatureExtractor``."""

    def __init__(
        self,
        observation_space: spaces.Box,
        feature_extractor: MultiRobotFeatureExtractor,
        num_robots: Optional[int] = None,
        obs_dim: Optional[int] = None,
        time_window: Optional[int] = None,
    ):
        self.num_robots, self.obs_dim, self.time_window = self._resolve_input_shape(
            observation_space=observation_space,
            num_robots=num_robots,
            obs_dim=obs_dim,
            time_window=time_window,
        )
        if feature_extractor.obs_dim != self.obs_dim:
            raise ValueError(
                f"feature_extractor.obs_dim={feature_extractor.obs_dim} does not match "
                f"resolved obs_dim={self.obs_dim}"
            )

        super().__init__(
            observation_space,
            features_dim=self.num_robots * feature_extractor.hidden_dim,
        )

        self.backbone = feature_extractor

    @staticmethod
    def _resolve_input_shape(
        observation_space: spaces.Box,
        num_robots: Optional[int],
        obs_dim: Optional[int],
        time_window: Optional[int],
    ) -> tuple[int, int, int]:
        shape = tuple(int(dim) for dim in observation_space.shape)

        if len(shape) == 3:
            return shape[0], shape[1], shape[2]

        if len(shape) != 1:
            raise ValueError(
                "SB3MultiRobotFeatureExtractor expects observation_space.shape to be "
                "(N, M, T) or flattened (N*M*T,)"
            )

        if num_robots is None or obs_dim is None or time_window is None:
            raise ValueError(
                "num_robots, obs_dim, and time_window are required for flattened observations"
            )

        expected_dim = int(num_robots) * int(obs_dim) * int(time_window)
        if shape[0] != expected_dim:
            raise ValueError(
                f"Flattened observation dim {shape[0]} does not match "
                f"num_robots*obs_dim*time_window={expected_dim}"
            )

        return int(num_robots), int(obs_dim), int(time_window)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 2:
            batch_size = observations.shape[0]
            x = observations.reshape(
                batch_size, self.num_robots, self.obs_dim, self.time_window
            )
        elif observations.dim() == 4:
            x = observations
        else:
            raise ValueError(
                "Expected observations with shape (B, N*M*T) or (B, N, M, T)"
            )

        z = self.backbone(x.float())
        return z.reshape(z.shape[0], -1)

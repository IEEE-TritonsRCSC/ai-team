"""Utilities for converting logged robot-attention CSV rows into datapoints."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ACTION_INDEX_BY_TYPE = {
    "dash": 0,
    "turn": 1,
    "kick": 2,
    "start_dribble": 3,
    "catch": 3,
    "stop_dribble": 4,
    "drop": 4,
}


@dataclass(frozen=True)
class RobotAttentionDatapoint:
    """One behavior-cloning sample for RobotAttentionEnv."""

    observation: np.ndarray
    action: np.ndarray
    cycle: int
    timestamp: float

def _world_to_ego(dx: float, dy: float, ego_theta_rad: float) -> tuple[float, float]:
    """Rotate a world-frame offset into the ego robot's frame."""

    cos_theta = float(np.cos(ego_theta_rad))
    sin_theta = float(np.sin(ego_theta_rad))
    rel_x = cos_theta * dx + sin_theta * dy
    rel_y = -sin_theta * dx + cos_theta * dy
    return rel_x, rel_y


def _clockwise_deg_to_math_rad(angle_deg: float) -> float:
    """
    Convert game angles into standard math radians.

    Game convention:
    - x+ is right
    - y+ is up
    - 0 degrees points right
    - positive angles are clockwise

    Internal convention here:
    - 0 radians points right
    - positive angles are counterclockwise
    """

    return float(-np.deg2rad(angle_deg))


def _clockwise_deg_to_cos_sin(angle_deg: float) -> tuple[float, float]:
    """Encode a clockwise game angle in cosine/sine using math-angle convention."""

    angle_rad = _clockwise_deg_to_math_rad(angle_deg)
    return float(np.cos(angle_rad)), float(np.sin(angle_rad))


def row_to_robot_attention_datapoint(
    row: Mapping[str, Any],
    *,
    team_prefix: str = "t1",
    robot_ids: Sequence[int] = (1, 2, 3, 4, 5, 6),
) -> RobotAttentionDatapoint:
    """
    Convert one CSV row into the observation/action pair expected by RobotAttentionEnv.

    Observation layout matches RobotAttentionEnv:
    per ego robot -> [ball_rel_x, ball_rel_y] +
    [other_rel_x, other_rel_y, cos(other_rel_theta), sin(other_rel_theta)] for
    each other robot in ``robot_ids`` order.

    Action layout matches RobotAttentionEnv:
    per robot -> [dash, turn, kick, start_dribble, stop_dribble,
    dash_power, dash_cos, dash_sin, turn_cos, turn_sin]

    Assumptions about the log file:
    - headings are stored in degrees using the game convention:
      0 points right and positive angles rotate clockwise
    - dash ``action_val1`` is power in [0, 100]
    - dash ``action_val2`` is relative dash direction in game-angle degrees
    - turn ``action_val1`` is relative turn angle in game-angle degrees
    """

    robot_ids = tuple(robot_ids)
    ball_x = float(row["ball_pos_x"])
    ball_y = float(row["ball_pos_y"])

    pose_by_robot_id: dict[int, tuple[float, float, float]] = {}
    for robot_id in robot_ids:
        prefix = f"{team_prefix}_{robot_id}"
        pose_by_robot_id[robot_id] = (
            float(row[f"{prefix}_x"]),
            float(row[f"{prefix}_y"]),
            _clockwise_deg_to_math_rad(float(row[f"{prefix}_theta"])),
        )

    obs_values: list[float] = []
    for ego_robot_id in robot_ids:
        ego_x, ego_y, ego_theta = pose_by_robot_id[ego_robot_id]

        ball_rel_x, ball_rel_y = _world_to_ego(
            dx=ball_x - ego_x,
            dy=ball_y - ego_y,
            ego_theta_rad=ego_theta,
        )
        obs_values.extend([ball_rel_x, ball_rel_y])

        for other_robot_id in robot_ids:
            if other_robot_id == ego_robot_id:
                continue

            other_x, other_y, other_theta = pose_by_robot_id[other_robot_id]
            other_rel_x, other_rel_y = _world_to_ego(
                dx=other_x - ego_x,
                dy=other_y - ego_y,
                ego_theta_rad=ego_theta,
            )
            rel_theta = other_theta - ego_theta
            obs_values.extend(
                [other_rel_x, other_rel_y, float(np.cos(rel_theta)), float(np.sin(rel_theta))]
            )

    action_values: list[float] = []
    for robot_id in robot_ids:
        prefix = f"{team_prefix}_{robot_id}"
        action_type = str(row[f"{prefix}_action_type"]).strip().lower()
        action_val1 = float(row[f"{prefix}_action_val1"])
        action_val2 = float(row[f"{prefix}_action_val2"])

        if action_type not in ACTION_INDEX_BY_TYPE:
            action_type = "turn"
            action_val1 = 0.0
            action_val2 = 0.0

        robot_action = np.zeros(10, dtype=np.float32)
        robot_action[ACTION_INDEX_BY_TYPE[action_type]] = 1.0

        if action_type == "dash":
            robot_action[5] = float(np.clip(action_val1 / 100.0, 0.0, 1.0))
            robot_action[6], robot_action[7] = _clockwise_deg_to_cos_sin(action_val2)
        elif action_type == "turn":
            robot_action[8], robot_action[9] = _clockwise_deg_to_cos_sin(action_val1)

        action_values.extend(robot_action.tolist())

    return RobotAttentionDatapoint(
        observation=np.asarray(obs_values, dtype=np.float32),
        action=np.asarray(action_values, dtype=np.float32),
        cycle=int(float(row["cycle"])),
        timestamp=float(row["timestamp"]),
    )


def reshape_robot_attention_frame(
    observation: np.ndarray,
    *,
    num_robots: int,
) -> np.ndarray:
    """
    Reshape one per-timestep flattened observation into env frame layout.

    Input shape:
    - ``(num_robots * obs_dim_per_robot,)``

    Output shape:
    - ``(num_robots, obs_dim_per_robot)``
    """

    frame = np.asarray(observation, dtype=np.float32)
    if frame.ndim != 1:
        raise ValueError(f"Expected flattened observation, got shape={frame.shape}")
    if num_robots <= 0:
        raise ValueError("num_robots must be positive")
    if frame.shape[0] % num_robots != 0:
        raise ValueError(
            f"Observation dim {frame.shape[0]} is not divisible by num_robots={num_robots}"
        )

    obs_dim_per_robot = frame.shape[0] // num_robots
    return frame.reshape(num_robots, obs_dim_per_robot)


def stack_robot_attention_frames(
    frames: Sequence[np.ndarray],
    *,
    time_window: int,
) -> np.ndarray:
    """
    Stack per-frame observations into the env history layout.

    This matches ``RobotAttentionEnv._stack_obs_frames``:
    - output shape is ``(num_robots, obs_dim_per_robot, time_window)``
    - if fewer than ``time_window`` frames are available, the earliest frame is
      repeated on the left until the window is full
    """

    if time_window <= 0:
        raise ValueError("time_window must be positive")
    if not frames:
        raise ValueError("frames must contain at least one observation frame")

    normalized_frames = [np.asarray(frame, dtype=np.float32) for frame in frames]
    frame_shape = normalized_frames[0].shape
    for frame in normalized_frames:
        if frame.shape != frame_shape:
            raise ValueError(
                f"All frames must share the same shape, got {frame.shape} vs {frame_shape}"
            )

    usable_frames = normalized_frames[-time_window:]
    earliest_frame = usable_frames[0]
    if len(usable_frames) < time_window:
        usable_frames = [earliest_frame] * (time_window - len(usable_frames)) + usable_frames

    stacked = np.stack(usable_frames, axis=-1)
    return stacked.astype(np.float32, copy=False)


class RobotAttentionTorchDataset(Dataset):
    """
    Torch dataset for behavior cloning against ``RobotAttentionEnv``.

    Each sample returns:
    - observation tensor with shape ``(num_robots, obs_dim_per_robot, time_window)``
    - action tensor with shape ``(10 * num_robots,)``

    Time windows never cross CSV-file boundaries. All CSV contents are loaded at
    construction time so training does not repeatedly re-read files.
    """

    def __init__(
        self,
        *,
        data_dir: str | Path,
        csv_paths: Sequence[str | Path] | None = None,
        team_prefix: str = "t1",
        robot_ids: Sequence[int] = (1, 2, 3, 4, 5, 6),
        time_window: int = 1,
        return_metadata: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.team_prefix = team_prefix
        self.robot_ids = tuple(int(robot_id) for robot_id in robot_ids)
        self.time_window = int(time_window)
        self.return_metadata = bool(return_metadata)
        if self.time_window <= 0:
            raise ValueError("time_window must be positive")
        if csv_paths is None:
            self.csv_paths = sorted(self.data_dir.rglob("*.csv"))
        else:
            self.csv_paths = [Path(csv_path) for csv_path in csv_paths]
        if not self.csv_paths:
            raise ValueError(f"No CSV files found under data_dir={self.data_dir}")

        self.num_robots = len(self.robot_ids)
        self.samples: list[RobotAttentionDatapoint] = []
        self._observation_frames: list[np.ndarray] = []
        self._sequence_start_indices: list[int] = []
        self.csv_start_indices: list[int] = []
        valid_csv_paths: list[Path] = []

        for csv_path in tqdm(self.csv_paths, desc="Loading CSV files"):
            csv_start = len(self.samples)
            loaded_count = self._load_csv_file(csv_path, csv_start)
            if loaded_count > 0:
                valid_csv_paths.append(csv_path)
                self.csv_start_indices.append(csv_start)

        self.csv_paths = valid_csv_paths
        self.total_samples = len(self.samples)
        if self.total_samples == 0:
            raise ValueError("No datapoints were loaded")

        first_frame = self._observation_frames[0]
        self.obs_dim_per_robot = int(first_frame.shape[1])
        self.action_dim = int(self.samples[0].action.shape[0])

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int):
        if index < 0:
            index += self.total_samples
        if index < 0 or index >= self.total_samples:
            raise IndexError(f"Index {index} out of range for dataset of size {self.total_samples}")

        datapoint = self.samples[index]
        sequence_start = self._sequence_start_indices[index]
        history_start = max(sequence_start, index - self.time_window + 1)
        obs_tensor = torch.from_numpy(
            stack_robot_attention_frames(
                self._observation_frames[history_start : index + 1],
                time_window=self.time_window,
            )
        ).float()
        action_tensor = torch.from_numpy(np.asarray(datapoint.action, dtype=np.float32)).float()

        if not self.return_metadata:
            return obs_tensor, action_tensor

        return {
            "observation": obs_tensor,
            "action": action_tensor,
            "cycle": datapoint.cycle,
            "timestamp": datapoint.timestamp,
        }

    def _load_csv_file(self, csv_path: Path, csv_start_index: int) -> int:
        """Parse one CSV file and append its rows to the in-memory dataset."""

        loaded_count = 0
        sequence_start_index = csv_start_index
        previous_cycle: int | None = None
        previous_timestamp: float | None = None

        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                datapoint = row_to_robot_attention_datapoint(
                    row,
                    team_prefix=self.team_prefix,
                    robot_ids=self.robot_ids,
                )

                if (
                    previous_cycle is not None
                    and previous_timestamp is not None
                    and (
                        datapoint.cycle < previous_cycle
                        or datapoint.timestamp < previous_timestamp
                    )
                ):
                    sequence_start_index = len(self.samples)

                frame = reshape_robot_attention_frame(
                    datapoint.observation,
                    num_robots=self.num_robots,
                )

                self.samples.append(datapoint)
                self._observation_frames.append(frame)
                self._sequence_start_indices.append(sequence_start_index)
                loaded_count += 1
                previous_cycle = datapoint.cycle
                previous_timestamp = datapoint.timestamp

        return loaded_count


def make_robot_attention_dataloader(
    *,
    data_dir: str | Path,
    csv_paths: Sequence[str | Path] | None = None,
    team_prefix: str = "t1",
    robot_ids: Sequence[int] = (1, 2, 3, 4, 5, 6),
    time_window: int = 1,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    return_metadata: bool = False,
) -> DataLoader:
    """Build a torch ``DataLoader`` compatible with ``RobotAttentionEnv``."""

    dataset = RobotAttentionTorchDataset(
        data_dir=data_dir,
        csv_paths=csv_paths,
        team_prefix=team_prefix,
        robot_ids=robot_ids,
        time_window=time_window,
        return_metadata=return_metadata,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
    )

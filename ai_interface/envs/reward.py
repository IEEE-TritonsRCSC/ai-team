"""General reward helpers for single-agent environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple
import math

import numpy as np

from ai_interface.constants.field_constants import FIELD_X, FIELD_Y


@dataclass(frozen=True)
class RewardConfig:
    """Weights and thresholds used by the shared reward function."""

    invalid_context_reward: float = -0.1
    approach_clip: float = 0.5
    approach_weight: float = 1.8
    goal_progress_clip: float = 0.4
    goal_progress_weight: float = 3.0
    has_ball_bonus: float = 0.8
    near_ball_bonus: float = 0.2
    near_ball_scale: float = 1.5
    opponent_near_ball_penalty: float = -0.5
    opponent_near_ball_threshold: float = 5.0
    shoot_state_bonus: float = 0.25
    dribble_state_bonus: float = 0.15
    goal_reward: float = 70.0
    robot_out_of_bounds_penalty: float = 3.0
    ball_out_of_bounds_penalty: float = 1.5
    step_bonus: float = 0.03
    goal_half_height: float = 3.66


@dataclass(frozen=True)
class RewardInputs:
    """State needed to evaluate the shared reward independently of any env."""

    ball_pos: Tuple[float, float]
    self_pose_rad: Tuple[float, float, float]
    ball_dist: float
    has_ball: bool
    kickable_dist: float
    state: str
    opponent_positions: Tuple[Tuple[float, float], ...] = ()
    prev_ball_dist: Optional[float] = None
    prev_ball_to_goal_dist: Optional[float] = None


@dataclass(frozen=True)
class RewardIntermediates:
    """Intermediate values used to assemble the scalar reward."""

    ball_dist: float
    ball_to_goal_dist: float
    approach: Optional[float]
    goal_progress: Optional[float]
    has_ball: bool
    near_ball: bool
    nearest_opponent_ball_dist: Optional[float]
    opponent_near_ball: bool
    in_shoot_state: bool
    in_dribble_state: bool
    goal_scored: bool
    robot_out_of_bounds: bool
    ball_out_of_bounds: bool


@dataclass(frozen=True)
class RewardResult:
    """Reward output plus terminal flags and intermediate values."""

    reward: float
    goal_scored: bool
    robot_out_of_bounds: bool
    ball_out_of_bounds: bool
    intermediates: RewardIntermediates


def extract_opponent_positions(
    robot_poses: Mapping[str, Sequence[Mapping[int, Tuple[float, float, float]]]],
    team_name: str,
) -> Tuple[Tuple[float, float], ...]:
    """Extract 2D opponent positions from the `GameState.robot_poses` shape."""

    positions = []
    for other_team_name, team_poses in robot_poses.items():
        if other_team_name == team_name:
            continue
        for pose_dict in team_poses:
            for _unum, (x, y, _theta) in pose_dict.items():
                positions.append((float(x), float(y)))
    return tuple(positions)


def calculate_reward_intermediates(
    inputs: RewardInputs,
    config: Optional[RewardConfig] = None,
) -> RewardIntermediates:
    """Calculate the per-step intermediate reward terms."""

    if config is None:
        config = RewardConfig()

    bx, by = inputs.ball_pos
    rx, ry, _ = inputs.self_pose_rad

    ball_dist = float(inputs.ball_dist)
    ball_to_goal_dist = float(math.hypot(FIELD_X[1] - bx, -by))

    approach = None
    if inputs.prev_ball_dist is not None:
        approach = float(np.clip(inputs.prev_ball_dist - ball_dist, -config.approach_clip, config.approach_clip))

    goal_progress = None
    if inputs.prev_ball_to_goal_dist is not None:
        goal_progress = float(
            np.clip(
                inputs.prev_ball_to_goal_dist - ball_to_goal_dist,
                -config.goal_progress_clip,
                config.goal_progress_clip,
            )
        )

    nearest_opponent_ball_dist = None
    if inputs.opponent_positions:
        nearest_opponent_ball_dist = min(
            math.hypot(ox - bx, oy - by) for ox, oy in inputs.opponent_positions
        )

    goal_scored = bool(bx >= FIELD_X[1] and abs(by) < config.goal_half_height)
    robot_out_of_bounds = bool(abs(rx) > FIELD_X[1] or abs(ry) > FIELD_Y[1])
    ball_out_of_bounds = bool(abs(by) > FIELD_Y[1] or bx < FIELD_X[0])

    return RewardIntermediates(
        ball_dist=ball_dist,
        ball_to_goal_dist=ball_to_goal_dist,
        approach=approach,
        goal_progress=goal_progress,
        has_ball=bool(inputs.has_ball),
        near_ball=bool(ball_dist < inputs.kickable_dist * config.near_ball_scale),
        nearest_opponent_ball_dist=nearest_opponent_ball_dist,
        opponent_near_ball=bool(
            nearest_opponent_ball_dist is not None
            and nearest_opponent_ball_dist < config.opponent_near_ball_threshold
        ),
        in_shoot_state=bool(inputs.state == "shoot_on_goal"),
        in_dribble_state=bool(inputs.state == "dribble_to_goal"),
        goal_scored=goal_scored,
        robot_out_of_bounds=robot_out_of_bounds,
        ball_out_of_bounds=ball_out_of_bounds,
    )


def calculate_reward(
    intermediates: RewardIntermediates,
    config: Optional[RewardConfig] = None,
) -> float:
    """Convert intermediate reward terms into the final scalar reward."""

    if config is None:
        config = RewardConfig()

    reward = 0.0

    if intermediates.approach is not None:
        reward += intermediates.approach * config.approach_weight

    if intermediates.goal_progress is not None:
        reward += intermediates.goal_progress * config.goal_progress_weight

    if intermediates.has_ball:
        reward += config.has_ball_bonus
    if intermediates.near_ball:
        reward += config.near_ball_bonus
    if intermediates.opponent_near_ball:
        reward += config.opponent_near_ball_penalty

    if intermediates.in_shoot_state:
        reward += config.shoot_state_bonus
    if intermediates.in_dribble_state:
        reward += config.dribble_state_bonus

    if intermediates.goal_scored:
        reward += config.goal_reward
    if intermediates.robot_out_of_bounds:
        reward -= config.robot_out_of_bounds_penalty
    if intermediates.ball_out_of_bounds:
        reward -= config.ball_out_of_bounds_penalty

    reward += config.step_bonus
    return float(reward)


def evaluate_reward(
    inputs: RewardInputs,
    config: Optional[RewardConfig] = None,
) -> RewardResult:
    """Evaluate the shared reward from plain inputs."""

    if config is None:
        config = RewardConfig()

    intermediates = calculate_reward_intermediates(inputs=inputs, config=config)
    reward = calculate_reward(intermediates=intermediates, config=config)
    return RewardResult(
        reward=reward,
        goal_scored=intermediates.goal_scored,
        robot_out_of_bounds=intermediates.robot_out_of_bounds,
        ball_out_of_bounds=intermediates.ball_out_of_bounds,
        intermediates=intermediates,
    )

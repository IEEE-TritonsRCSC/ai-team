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
    goal_progress_clip: float = 2.0
    goal_progress_weight: float = 3.0
    has_ball_bonus: float = 0.1
    near_ball_bonus: float = 0.05
    near_ball_scale: float = 1.5
    opponent_near_ball_penalty: float = -0.5
    opponent_near_ball_threshold: float = 5.0
    shoot_state_bonus: float = 0.25
    dribble_state_bonus: float = 0.15
    goal_reward: float = 70.0
    robot_out_of_bounds_penalty: float = 3.0
    ball_out_of_bounds_penalty: float = 1.5
    step_bonus: float = 0.03
    goal_half_height: float = 5.0  # SSL Div B goal width=1000mm → 10 sim units → half=5.0
    ball_speed_bonus_weight: float = 2.0
    ball_speed_threshold: float = 0.2
    ball_speed_clip: float = 2.0
    alignment_weight: float = 3.0
    alignment_near_ball_only: bool = True
    # Stricter gate: when True, the alignment reward fires only while the
    # robot actually possesses the ball (kickable range), not merely while
    # it is "near" it. Overrides alignment_near_ball_only.
    alignment_has_ball_only: bool = False
    # Per-step clip on the cos-delta used for alignment reward. cos itself is
    # in [-1, 1] so the delta is in [-2, 2]; clipping at 1.0 prevents reward
    # spikes from large single-step rotations.
    alignment_clip: float = 1.0
    # One-shot bonus added by JAL_env when the model picks the `kick` action
    # while having the ball. Scaled by max(0, cos)^4 where cos is the angle
    # from the robot's facing direction to the goal. cos^4 (instead of cos^2)
    # sharpens aim discrimination — a kick at ~45° off goal earns only 24%
    # of the bonus instead of 49%. Cannot be exploited by spinning because
    # it only fires on a committed kick action.
    kick_aim_bonus_weight: float = 0.0
    # Hard penalty applied (one-shot, by JAL_env) when a kick fires but its
    # straight-line projection misses the goal mouth — i.e. |predicted_y| >
    # goal_half_height. Pushes the EV of unaimed kicks negative so the policy
    # cannot collect zero-reward "free kicks" by firing on step 1 of every
    # episode; forces it to delay-and-aim. Set per stage via reward_config_overrides.
    bad_aim_kick_penalty: float = 0.0


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
    prev_ball_pos: Optional[Tuple[float, float]] = None
    prev_facing_goal_cos: Optional[float] = None


@dataclass(frozen=True)
class RewardIntermediates:
    """Intermediate values used to assemble the scalar reward."""

    ball_dist: float
    ball_to_goal_dist: float
    approach: Optional[float]
    goal_progress: Optional[float]
    ball_speed: Optional[float]
    facing_goal_cos: Optional[float]
    facing_goal_cos_delta: Optional[float]
    # Predicted y-coordinate at which the ball would cross x=FIELD_X[1] if it
    # continued on its current linear velocity. None when ball isn't moving
    # forward (vx <= 0) or when prev_ball_pos is unavailable.
    predicted_y_at_goal_line: Optional[float]
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
    rx, ry, theta = inputs.self_pose_rad

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

    ball_speed = None
    predicted_y_at_goal_line: Optional[float] = None
    if inputs.prev_ball_pos is not None:
        pbx, pby = inputs.prev_ball_pos
        vx = bx - pbx
        vy = by - pby
        ball_speed = float(math.hypot(vx, vy))
        # Project ball's current velocity to the goal line (x=FIELD_X[1]).
        # Only meaningful when the ball is moving toward the goal (vx > 0).
        if vx > 1e-6:
            predicted_y_at_goal_line = float(by + (FIELD_X[1] - bx) * (vy / vx))

    facing_goal_cos = None
    to_goal_x = FIELD_X[1] - rx
    to_goal_y = -ry
    to_goal_norm = math.hypot(to_goal_x, to_goal_y)
    if to_goal_norm > 1e-6:
        facing_goal_cos = float(
            (math.cos(theta) * to_goal_x + math.sin(theta) * to_goal_y) / to_goal_norm
        )

    facing_goal_cos_delta = None
    if facing_goal_cos is not None and inputs.prev_facing_goal_cos is not None:
        facing_goal_cos_delta = float(
            np.clip(
                facing_goal_cos - inputs.prev_facing_goal_cos,
                -config.alignment_clip,
                config.alignment_clip,
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
        ball_speed=ball_speed,
        facing_goal_cos=facing_goal_cos,
        facing_goal_cos_delta=facing_goal_cos_delta,
        predicted_y_at_goal_line=predicted_y_at_goal_line,
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

    if intermediates.approach is not None and not intermediates.has_ball:
        reward += intermediates.approach * config.approach_weight

    if intermediates.goal_progress is not None:
        reward += intermediates.goal_progress * config.goal_progress_weight

    # Aim-gated ball-speed reward: a fast ball only counts toward reward to the
    # extent that its current velocity would carry it through the goal mouth.
    # `aim_quality` is 1.0 if the linear projection lands at y=0 (dead center),
    # decays linearly to 0 at |y|=goal_half_height, and is 0 beyond that. This
    # makes "kick hard but misaim" worth almost nothing, so the policy can't
    # collect intermediate speed reward without also aiming.
    if intermediates.ball_speed is not None and intermediates.ball_speed > config.ball_speed_threshold:
        if intermediates.predicted_y_at_goal_line is None:
            aim_quality = 0.0  # ball not moving toward goal
        else:
            aim_quality = max(
                0.0,
                1.0 - abs(intermediates.predicted_y_at_goal_line) / config.goal_half_height,
            )
        reward += (
            min(intermediates.ball_speed, config.ball_speed_clip)
            * config.ball_speed_bonus_weight
            * aim_quality
        )

    # Alignment reward is now delta-based: reward the *improvement* in cos
    # this step, not the absolute cos value. A robot that stands still next
    # to the ball facing the goal gets delta=0 (no harvest), while a robot
    # that actively turns toward the goal gets positive reward proportional
    # to the cos improvement. Total reward for ending up well-aligned is
    # unchanged, but idle-camp exploits are structurally eliminated.
    if intermediates.facing_goal_cos_delta is not None:
        if config.alignment_has_ball_only:
            fire_alignment = intermediates.has_ball
        elif config.alignment_near_ball_only:
            fire_alignment = intermediates.near_ball
        else:
            fire_alignment = True
        if fire_alignment:
            reward += intermediates.facing_goal_cos_delta * config.alignment_weight

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

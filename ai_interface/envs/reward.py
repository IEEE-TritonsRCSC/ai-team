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
    # Distance (in sim units of |predicted_y| BEYOND the goalpost) over which the
    # bad-aim penalty ramps from 0 (at the post) to its full value (capped). When
    # > 0, JAL_env applies a GRADED miss penalty instead of the flat cliff: a kick
    # that lands 12 wide is punished less than one 20 wide, giving the policy a
    # gradient toward better aim on EVERY kick — not just the ~12% that hit the
    # mouth. 0.0 keeps the legacy binary cliff (flat penalty for any miss). See
    # TRAINING.md §18.
    bad_aim_grad_scale: float = 0.0

    # ---- Stage 2: goalie-aware kick placement (1v1 vs scripted goalie) ----
    # When True, the aim quality used for the kick bonus (JAL_env.step) and the
    # dense ball_speed / goal_progress gates switches from center-mouth quality
    # to GOALIE-GAP quality: 0 when the shot projects at the keeper's y, 1 at
    # max lateral separation inside the mouth, and 0 for any off-mouth shot.
    # This breaks the stage-1 center-shot habit (the bisector goalie covers
    # center) and teaches corner placement. Falls back to center-mouth quality
    # whenever the goalie pose is unavailable.
    use_goalie_aim_gate: bool = False
    # On-target kicks whose goalie-gap quality is below this threshold ("shot
    # straight at the keeper") additionally pay kick_into_keeper_penalty.
    goalie_gap_min_quality: float = 0.3
    kick_into_keeper_penalty: float = 0.0
    # Early-stage-2 warmup blend: fraction of CENTER-mouth quality mixed into
    # the goalie-gap kick bonus (final = (1-b)*gap + b*center). Eases transfer
    # from the stage-1 checkpoint; fade to 0.0 (pure gap) as the stage matures.
    goalie_gap_blend_center: float = 0.0

    # ---- Stage 2: dribble session rules (enforced by JAL_env, read here) ----
    # A dribble session is anchored where start_dribble fired; the robot may
    # not dribble farther than this from the anchor. Beyond it the env forces
    # a release and start_dribble becomes invalid until re-approach.
    dribble_max_radius: float = 1.0
    # After stop_dribble / exhaustion, a NEW session requires visible
    # separation first: robot-ball distance must exceed kickable_dist + this
    # margin before start_dribble is valid again.
    dribble_redribble_gap_margin: float = 0.2
    # Per-step reward on the change of |ball_y - goalie_y| while inside a
    # valid dribble envelope — pays lateral feints away from the keeper and
    # penalizes dribbling back into its cover. 0.0 disables.
    dribble_lateral_progress_weight: float = 0.0
    dribble_lateral_progress_clip: float = 0.3
    # One-shot bonus (applied by JAL_env) for releasing the ball with
    # stop_dribble late in the session (anchor distance >= 70% of the radius),
    # encouraging a deliberate stop -> re-approach -> kick chain instead of
    # spamming start_dribble at the 1 m boundary. 0.0 disables.
    stop_dribble_release_bonus: float = 0.0

    # ---- Stage 2: goalie-possession tug-of-war suppression ----
    # One-shot penalty applied when a kick fires while the ball is within
    # `goalie_possession_dist` of the keeper. Penalizes the tug-of-war pattern
    # (robot kicking while the goalie already has possession), complementing the
    # early-termination check in JAL_env._check_terminal. 0.0 disables.
    kick_near_goalie_penalty: float = 0.0
    # Ball-to-goalie distance threshold for the above penalty. Also used by
    # JAL_env._check_terminal as the early possession-detection radius.
    goalie_possession_dist: float = 2.0

    # ---- Stage 2: dribble→kick combo bonus ----
    # One-shot bonus applied when a kick fires within `post_dribble_kick_combo_window`
    # steps of a stop_dribble event. Directly rewards the dribble-to-create-angle
    # → kick sequence as a single unit, preventing the robot from learning dribble
    # and kick as independent behaviours. 0.0 disables.
    post_dribble_kick_bonus: float = 0.0
    # How many steps after stop_dribble a kick still qualifies for the combo bonus.
    post_dribble_kick_combo_window: int = 5

    # ---- Stage 2: re-dribble cycle bonus ----
    # One-shot bonus for completing a full SSL re-dribble cycle: a legitimate
    # stop_dribble (session active, ball released) followed by a successful
    # start_dribble (gap cleared, new session opened). Teaches the multi-hop
    # dribble chain required by the SSL 1 m rule — drop → move away → re-catch.
    # 0.0 disables.
    redribble_cycle_bonus: float = 0.0


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
    # Stage 2: opponent goalie y (None when no goalie is on the field). Used
    # for goalie-gap aim gating and dribble lateral-progress shaping.
    goalie_y: Optional[float] = None
    # Stage 2: command-level dribble session state, maintained by JAL_env.
    # is_dribbling is True only while a start_dribble session is active;
    # dribble_anchor_dist is the robot's distance from the session anchor.
    is_dribbling: bool = False
    dribble_anchor_dist: Optional[float] = None


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
    # Stage 2 fields (defaults keep stage-1 callers untouched).
    goalie_y: Optional[float] = None
    # True while a command-level dribble session is active AND the robot is
    # still inside the dribble_max_radius envelope from its anchor.
    valid_dribble: bool = False
    # Per-step change of |ball_y - goalie_y| while in a valid dribble
    # envelope; positive = ball moving laterally away from the keeper.
    dribble_lateral_progress: Optional[float] = None


@dataclass(frozen=True)
class RewardResult:
    """Reward output plus terminal flags and intermediate values."""

    reward: float
    goal_scored: bool
    robot_out_of_bounds: bool
    ball_out_of_bounds: bool
    intermediates: RewardIntermediates


def aim_quality_from_prediction(
    predicted_y_at_goal_line: Optional[float],
    goal_half_height: float,
    target_y: float = 0.0,
) -> float:
    """Return 0..1 linear aim quality for a projected goal-line crossing."""

    if predicted_y_at_goal_line is None or goal_half_height <= 0.0:
        return 0.0
    miss_from_target = abs(float(predicted_y_at_goal_line) - float(target_y))
    return float(max(0.0, 1.0 - miss_from_target / float(goal_half_height)))


def goalie_gap_quality(
    predicted_y_at_goal_line: Optional[float],
    goalie_y: Optional[float],
    goal_half_height: float,
) -> float:
    """Return 0..1 goalie-gap quality for a projected goal-line crossing.

    0 when the shot is off the goal mouth entirely OR projects exactly at the
    keeper's y; grows linearly with lateral separation from the keeper, capped
    at 1 when the gap reaches goal_half_height. Mirrors the w_goalie_gap
    heuristic in ai_interface/attacker.py. Falls back to center-mouth
    aim quality when no goalie y is available.
    """

    if predicted_y_at_goal_line is None or goal_half_height <= 0.0:
        return 0.0
    if goalie_y is None:
        return aim_quality_from_prediction(predicted_y_at_goal_line, goal_half_height)
    predicted_y = float(predicted_y_at_goal_line)
    # Mouth validity is a hard gate: an off-target shot earns no gap credit.
    if abs(predicted_y) >= float(goal_half_height):
        return 0.0
    gap = min(abs(predicted_y - float(goalie_y)), float(goal_half_height))
    return float(gap / float(goal_half_height))


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

    # Stage 2: dribble envelope validity + lateral progress away from keeper.
    valid_dribble = bool(
        inputs.is_dribbling
        and inputs.dribble_anchor_dist is not None
        and inputs.dribble_anchor_dist < config.dribble_max_radius
    )
    dribble_lateral_progress = None
    if (
        valid_dribble
        and inputs.goalie_y is not None
        and inputs.prev_ball_pos is not None
    ):
        prev_by = float(inputs.prev_ball_pos[1])
        gk_y = float(inputs.goalie_y)
        dribble_lateral_progress = float(abs(by - gk_y) - abs(prev_by - gk_y))

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
        goalie_y=inputs.goalie_y,
        valid_dribble=valid_dribble,
        dribble_lateral_progress=dribble_lateral_progress,
    )


def _dense_aim_quality(
    intermediates: RewardIntermediates,
    config: RewardConfig,
) -> float:
    """Aim quality used by the dense gates (goal_progress, ball_speed).

    Center-mouth quality by default; goalie-gap quality in stage 2 (so a
    powerful shot straight at the keeper earns nothing). goalie_gap_quality
    itself falls back to center-mouth when no goalie y is available.
    """

    if config.use_goalie_aim_gate:
        return goalie_gap_quality(
            intermediates.predicted_y_at_goal_line,
            intermediates.goalie_y,
            config.goal_half_height,
        )
    return aim_quality_from_prediction(
        intermediates.predicted_y_at_goal_line,
        config.goal_half_height,
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
        goal_progress = float(intermediates.goal_progress)
        if (
            goal_progress > 0.0
            and intermediates.ball_speed is not None
            and intermediates.ball_speed > config.ball_speed_threshold
        ):
            goal_progress *= _dense_aim_quality(intermediates, config)
        reward += goal_progress * config.goal_progress_weight

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
            aim_quality = _dense_aim_quality(intermediates, config)
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
    elif intermediates.valid_dribble:
        # Stage 2: pay the dribble bonus only inside the valid envelope
        # (anchor distance < dribble_max_radius) and only while the ball is
        # not moving back toward the keeper's y. Dribbling beyond the
        # envelope earns nothing — the env additionally marks it invalid.
        if (
            intermediates.dribble_lateral_progress is None
            or intermediates.dribble_lateral_progress > 0.0
        ):
            reward += config.dribble_state_bonus

    if (
        intermediates.dribble_lateral_progress is not None
        and config.dribble_lateral_progress_weight > 0.0
    ):
        reward += (
            float(
                np.clip(
                    intermediates.dribble_lateral_progress,
                    -config.dribble_lateral_progress_clip,
                    config.dribble_lateral_progress_clip,
                )
            )
            * config.dribble_lateral_progress_weight
        )

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

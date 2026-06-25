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
    # Keep rewarded shots away from the posts. Quality is unchanged inside
    # H-margin, then decays linearly to zero at the post.
    goal_post_safety_margin: float = 0.0

    # ---- Stage 3: defender lane awareness (1 attacker vs 1 defender + GK) ----
    # When True, shot AND dribble-target quality are additionally multiplied by
    # how clear the straight lane to goal is of the non-goalie DEFENDER. A shot
    # whose lane passes within `defender_lane_block_dist` of the defender earns
    # proportionally less; a lane straight through the defender earns nothing.
    # This is the core signal that teaches the attacker to dribble to an angle
    # the defender doesn't cover before shooting. Falls back to a no-op (lane
    # clear = 1) whenever no defender pose is available, so it is independent of
    # use_goalie_aim_gate (the two qualities multiply together).
    use_defender_lane_gate: bool = False
    # Distance (sim units) from the defender to the shot segment below which the
    # lane is considered (partially) blocked. Mirrors AttackerConfig.lane_block_dist.
    defender_lane_block_dist: float = 2.6
    # One-shot penalty (applied in JAL_env) when a kick fires through a lane the
    # defender blocks — i.e. lane clearance below `defender_lane_min_quality`.
    # Parallels kick_into_keeper_penalty for the keeper. 0.0 disables.
    defender_lane_penalty: float = 0.0
    defender_lane_min_quality: float = 0.3

    # ---- Stage 2: dribble_to target quality and progress ----
    # Dense per-step reward for ball moving toward the dribble_to target.
    # Positive = ball got closer. Teaches WHERE to target by rewarding
    # progress toward the chosen coordinate. 0.0 disables.
    dribble_target_progress_weight: float = 0.0
    dribble_target_progress_clip: float = 0.5
    # One-shot bonus (fired in JAL_env on the first step of a new dribble_to
    # selection) scaled by how much the chosen target improves goalie-gap
    # quality vs the ball's current position. Teaches WHEN to dribble: the
    # policy only gets credit when the target genuinely improves the shooting
    # angle. quality_delta = gap(target) - gap(current), clamped to [0, 1].
    dribble_target_quality_weight: float = 0.0
    # Per-possession carry limit used for reward lookahead and the dribble_to
    # macro. The simulator field is 10 units per SSL meter, so 8.5 units keeps
    # a 1 m dribble safely under the 10-unit excessive-dribbling limit.
    dribble_segment_limit: float = 0.85
    # Small per-step reward while dribble_to is active AND the target has
    # positive quality delta. Sustains reward signal across multi-step
    # transport so the value function doesn't over-discount. 0.0 disables.
    dribble_active_bonus: float = 0.0
    # Clamp on the absolute y (field units) of a decoded dribble_to target.
    # The dribble-target head emits goto_y_raw in [-1, 1] which scales by
    # field_half_height (30) -> targets span +/-30. But a USEFUL dribble must
    # land near the goal mouth (|y| < goal_half_height ~ 5) to open the shooting
    # angle; with the full +/-30 range, gap-improving targets are a tiny ~17%
    # slice of action space the policy never finds by exploration (§34: 263/264
    # carries moved the ball to a WORSE angle). Clamping goto_y to +/-this value
    # concentrates the head's output on reachable, shot-improving targets.
    # 0.0 disables (no clamp, full +/-field_half_height range).
    dribble_target_y_clip: float = 0.0
    # Extra dribble safety near the opponent penalty area. If a decoded
    # dribble_to target is within `dribble_penalty_area_guard_margin` of the
    # opponent penalty-area x boundary, clamp its |y| to
    # `dribble_penalty_area_y_clip`. This prevents the macro from carrying a
    # wide ball into x>=35 outside the goal mouth, which terminates as
    # ball_in_penalty_off_target under the Stage-2 keeper duel rules.
    # Both >0 to enable.
    dribble_penalty_area_guard_margin: float = 0.0
    dribble_penalty_area_y_clip: float = 0.0
    # Goal-RELATIVE dribble-target parameterization (replaces the absolute
    # field-coordinate decode for dribble_to). The decoded target is
    # ball + fwd*(ball->goal unit) + lat*(perp), with
    #   fwd = (goto_x_raw*0.5 + 0.5) * dribble_fwd_max   (always forward, 0..max)
    #   lat = goto_y_raw * dribble_lat_max               (steer to the open side)
    # This makes the head's NEUTRAL output (~0,0) a forward carry toward the goal
    # centre — already gap-improving — so the quality one-shot fires from step 1
    # and the head gets a gradient (the absolute decode put the neutral target at
    # midfield (0,0), dx=45, gap~0.07 << current gap ~0.32, so every carry was a
    # backward dead target and the head never learned: §34/§35 + 6/20 lag runs all
    # showed 0% useful carries). Both >0 to enable; 0 keeps the legacy absolute
    # decode. dribble_target_y_clip still applies as a final |y| clamp.
    dribble_fwd_max: float = 0.0
    dribble_lat_max: float = 0.0

    # Achieved-gap reward (computed in JAL_env): credit the REALIZED change in
    # shooting-gap quality of a carry — positional_gap_quality at the ball's
    # position when the segment CLOSES minus when it OPENED — scaled by this
    # weight. Unlike dribble_target_quality_weight (which pays the latched
    # HYPOTHETICAL target's gap at carry-open), this pays only for the angle the
    # carry actually produced, so an in-place orbit that never moves the ball
    # nets ~0 and a carry that worsens the angle is penalized.
    dribble_achieved_gap_weight: float = 0.0

    # Deterministic keeper-away kick target y used by the env's committed kick
    # macro. 0.0 keeps the legacy auto target at min(goal_half_height -
    # safety_margin, 0.8*goal_half_height). Positive values are clipped inside
    # the post-safety edge. Stage 2h uses 3.5 so keeper-away shots retain
    # enough post margin under the 5° kick-fire tolerance.
    kick_keeper_away_target_y: float = 0.0
    # Optional one-time retarget while the committed kick macro is still
    # aligning. If the keeper moves onto the selected side, or the intended
    # target-vs-keeper gap quality falls below this threshold, the env may flip
    # to the better of +/-kick_keeper_away_target_y. Defaults disabled.
    kick_keeper_retarget_max_count: int = 0
    kick_keeper_retarget_min_gap_quality: float = 0.0
    kick_keeper_retarget_same_side_y: float = 0.0
    kick_keeper_retarget_min_improvement: float = 0.05

    # Terminal penalty for a shot entering the penalty area outside the mouth.
    ball_in_penalty_off_target_penalty: float = 0.0

    # ---- Deprecated dribble fields (kept for backward compat, no longer
    # referenced in reward calculation — internal to dribble_to phase machine) ----
    dribble_max_radius: float = 1.0
    dribble_redribble_gap_margin: float = 0.2
    dribble_lateral_progress_weight: float = 0.0
    dribble_lateral_progress_clip: float = 0.3
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
    # steps of a dribble release (CARRY→RELEASE transition). Directly rewards the
    # dribble-to-create-angle → kick sequence as a single unit. 0.0 disables.
    post_dribble_kick_bonus: float = 0.0
    # How many steps after a dribble release a kick still qualifies for the combo.
    post_dribble_kick_combo_window: int = 5
    # When True, the combo bonus is scaled by the kick's actual aim quality at
    # fire time. Prevents "dribble then wild kick" from farming the bonus.
    post_dribble_kick_quality_scale: bool = False
    # Stage 3 release-timing pressure. Once a verified carry is open, give the
    # policy a short grace window to finish the segment and shoot; after that,
    # subtract carry_urgency_penalty_per_step up to carry_urgency_penalty_max.
    # This specifically targets long post-carry stalls without changing lane
    # discovery reward.
    carry_urgency_grace_steps: int = 0
    carry_urgency_penalty_per_step: float = 0.0
    carry_urgency_penalty_max: float = 0.0
    # Repeated opponent defense-area touches are treated as a serious training
    # failure after this many attacker-caused touches in one episode. 0 disables.
    defense_area_touch_terminal_count: int = 0

    # ---- Stage 2: keeper catch-zone suppression ----
    # positional_gap_quality (which scales kick_aim, the combo, dribble target-
    # quality and achieved-gap) PEAKS inside the keeper's catch radius (~0.92 at
    # 2 units from goal vs ~0.46 at 10), so every gap-scaled dense reward is
    # maximised exactly where the keeper catches — the reward gradient points
    # INTO the keeper and the policy over-dribbles to point-blank (§stage2h:
    # 49.7% goalie_catch). These knobs multiply that gap-scaled credit by a
    # factor that ramps from `keeper_zone_floor` (at the keeper's position) to
    # 1.0 at `keeper_zone_radius` away, flattening/inverting the pull. The −70
    # goalie_catch terminal is unchanged. radius 0.0 disables (backwards-compat).
    keeper_zone_radius: float = 0.0
    keeper_zone_floor: float = 0.0
    # ---- Stage 4: own-goalie coordination ----
    own_goalie_clearance_dist: float = 5.0
    clearance_receive_weight: float = 0.0
    clearance_receive_clip: float = 0.5
    receive_positioning_bonus: float = 0.0
    receive_positioning_x_range: Tuple[float, float] = (-5.0, 15.0)
    own_half_loiter_x: float = -15.0
    own_half_loiter_penalty: float = 0.0
    ball_recovery_bonus: float = 0.0

    # ---- Stage 5+: multi-robot coordination (N-robot scalable) ----
    enable_role_gating: bool = False
    spread_bonus: float = 0.0
    spread_min_dist: float = 8.0
    redundant_chase_penalty: float = 0.0
    redundant_chase_dist: float = 3.0
    support_position_bonus: float = 0.0
    support_position_max_dist: float = 20.0
    support_position_min_angle_deg: float = 20.0
    possession_transfer_bonus: float = 0.0
    team_goal_multiplier: float = 1.0


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
    # Stage 2: opponent goalie y (None when no goalie is on the field).
    goalie_y: Optional[float] = None
    # Stage 3: (x, y) of the non-goalie defender whose lane coverage matters for
    # the current shot. None when there is no defender (e.g. stages 1-2).
    defender_pos: Optional[Tuple[float, float]] = None
    # True while the dribble_to phase machine is in GRAB or CARRY.
    is_dribbling: bool = False
    dribble_anchor_dist: Optional[float] = None
    # The (x, y) target the policy chose for dribble_to this step.
    # None when the active action is not dribble_to.
    dribble_target: Optional[Tuple[float, float]] = None
    # Ball-to-dribble-target distance from the previous step (for delta).
    prev_ball_to_dribble_target_dist: Optional[float] = None

    # ---- Stage 4: own-goalie coordination ----
    own_goalie_pos: Optional[Tuple[float, float]] = None
    own_goalie_has_ball: bool = False
    prev_clearance_zone_dist: Optional[float] = None
    prev_has_ball: bool = False
    prev_opponent_near_ball: bool = False

    # ---- Stage 5+: multi-robot coordination ----
    ally_positions: Tuple[Tuple[float, float], ...] = ()
    is_nearest_to_ball: bool = True
    ball_carrier_id: Optional[int] = None


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
    # Stage 3: clearance (0..1) of the actual shot lane (ball-velocity projection)
    # from the defender. 1.0 = clear / no defender; multiplies the dense aim gates.
    defender_lane_clear: float = 1.0
    # True when the current action is dribble_to (regardless of phase).
    dribble_to_active: bool = False
    # Per-step ball-to-dribble-target distance reduction (positive = closer).
    dribble_target_progress: Optional[float] = None
    # Gap quality improvement of target over current ball position [0, 1].
    dribble_target_quality_delta: Optional[float] = None
    # True only after catch ownership has been verified and the macro is carrying.
    is_dribbling: bool = False
    # Deprecated: kept for backward compat, always False / None.
    valid_dribble: bool = False
    dribble_lateral_progress: Optional[float] = None

    # ---- Stage 4: own-goalie coordination ----
    own_goalie_has_ball: bool = False
    clearance_receive_progress: Optional[float] = None
    in_receive_position: bool = False
    in_own_half_loiter: bool = False
    ball_recovered: bool = False

    # ---- Stage 5+: multi-robot coordination ----
    is_nearest_to_ball: bool = True
    nearest_ally_dist: Optional[float] = None
    is_well_spread: bool = False
    in_support_position: bool = False
    is_redundant_chaser: bool = False


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


def post_safe_goalie_gap_quality(
    predicted_y_at_goal_line: Optional[float],
    goalie_y: Optional[float],
    goal_half_height: float,
    safety_margin: float,
) -> float:
    """Goalie-gap quality with a continuous safety taper near each post."""
    base = goalie_gap_quality(predicted_y_at_goal_line, goalie_y, goal_half_height)
    if base <= 0.0 or predicted_y_at_goal_line is None or safety_margin <= 0.0:
        return base
    safe_edge = max(0.0, float(goal_half_height) - float(safety_margin))
    abs_y = abs(float(predicted_y_at_goal_line))
    if abs_y <= safe_edge:
        return base
    safety = max(0.0, (float(goal_half_height) - abs_y) / float(safety_margin))
    return float(base * safety)


# Half-width (sim units) of the goal-mouth segment the keeper is treated as
# covering, centred on its y. The goal mouth is 10 wide (±5); a value of 1.0
# means the keeper blocks ~20% of the mouth. Used only by positional_gap_quality
# to model how much of a shooter's goal-angle the keeper occludes.
GOALIE_BLOCK_HALF_WIDTH = 1.0

# Reference shot distance (sim units from the goal line) used to normalise the
# open-goal angle in positional_gap_quality. A clear full-mouth shot from this
# distance maps to quality 1.0; closer/more-central shots saturate at 1, and
# wider/farther/keeper-blocked shots fall below it. 12 sits inside the typical
# attacking-third kick distance (ball spawns x~25-34, goal line x=45 -> dx~11-20).
GOAL_ANGLE_REF_DX = 12.0


def positional_gap_quality(
    point: Tuple[float, float],
    goalie_y: float,
    goal_half_height: float,
    goalie_block_half_width: float = GOALIE_BLOCK_HALF_WIDTH,
) -> float:
    """Achievable shot quality from a field position, as the OPEN-GOAL angle.

    Returns the fraction of the goal-angle subtended from `point` that the
    keeper does NOT block — i.e. "if I shoot from here and aim at the open
    side, how much of my shooting angle is clear of the keeper?".

    Geometry (attacking the right goal): the goal line is x=FIELD_X[1], the
    mouth spans y in [-H, H] with H=goal_half_height, and the keeper occupies
    [ky-w, ky+w] within the mouth (ky=goalie_y clamped into the mouth,
    w=goalie_block_half_width). The keeper splits the mouth into a low segment
    [-H, ky-w] and a high segment [ky+w, H]; the shooter exploits whichever
    subtends the LARGER ABSOLUTE angle from `point`. That open angle is
    normalised by a fixed reference (a full mouth from GOAL_ANGLE_REF_DX),
    giving a value in [0, 1] that rises as the shot gets closer, more central,
    or as the keeper clears the shooter's lane — and falls for wide/sharp
    angles. (Normalising by the *absolute* reference, not the point's own total
    goal-angle, is deliberate: a fraction-of-goal measure perversely rewards
    sharper angles, since a fixed keeper blocks a smaller fraction from the
    side.)

    Unlike the previous straight-ahead model, this does NOT assume the shot
    crosses at the point's own y, and there is NO hard cliff at |y|>=H: a ball
    wide of the posts still has a real angled shot and scores accordingly.
    Returns 0 only at/behind the goal line, or when the keeper occludes the
    entire mouth from this angle.

    Caveat: `goalie_y` is the keeper's CURRENT y (it bisects the CURRENT ball
    angle). Evaluating a dribble *target* against the current keeper y is a
    one-step counterfactual — it rewards moving to where the keeper is not
    currently covering, which is the intended dribble instinct.
    """
    gx = float(FIELD_X[1])
    px = float(point[0])
    py = float(point[1])
    H = float(goal_half_height)
    dx = gx - px
    if dx <= 1e-6 or H <= 0.0:
        # At/behind the goal line, or degenerate mouth: no meaningful shot.
        return 0.0

    def _subtended(y0: float, y1: float) -> float:
        """Angle (rad) subtended from `point` by the mouth interval [y0, y1]."""
        return abs(math.atan2(y1 - py, dx) - math.atan2(y0 - py, dx))

    ky = float(np.clip(goalie_y, -H, H))
    w = max(0.0, float(goalie_block_half_width))
    blk_lo = float(np.clip(ky - w, -H, H))
    blk_hi = float(np.clip(ky + w, -H, H))

    # Open segments either side of the keeper; shooter takes the larger angle.
    open_low = _subtended(-H, blk_lo)
    open_high = _subtended(blk_hi, H)
    open_ang = max(open_low, open_high)

    # Normalise by a full mouth subtended from the reference shot distance.
    ref_ang = 2.0 * math.atan2(H, max(GOAL_ANGLE_REF_DX, 1e-6))
    if ref_ang <= 1e-9:
        return 0.0
    return float(min(max(open_ang / ref_ang, 0.0), 1.0))


def reachable_gap_delta(
    ball_pos: Tuple[float, float],
    target: Tuple[float, float],
    goalie_y: float,
    goal_half_height: float,
    segment_limit: float = 0.85,
) -> Tuple[float, Tuple[float, float]]:
    """Signed gap change at the endpoint reachable in one carry segment."""
    bx, by = float(ball_pos[0]), float(ball_pos[1])
    dx, dy = float(target[0]) - bx, float(target[1]) - by
    dist = float(math.hypot(dx, dy))
    if dist > segment_limit > 0.0:
        scale = float(segment_limit) / dist
        endpoint = (bx + dx * scale, by + dy * scale)
    else:
        endpoint = (float(target[0]), float(target[1]))
    current = positional_gap_quality((bx, by), goalie_y, goal_half_height)
    reachable = positional_gap_quality(endpoint, goalie_y, goal_half_height)
    return float(reachable - current), endpoint


def _distance_point_to_segment(
    point: Tuple[float, float],
    seg_a: Tuple[float, float],
    seg_b: Tuple[float, float],
) -> float:
    """Shortest distance from `point` to the segment seg_a→seg_b."""

    px, py = float(point[0]), float(point[1])
    ax, ay = float(seg_a[0]), float(seg_a[1])
    bx, by = float(seg_b[0]), float(seg_b[1])
    vx, vy = bx - ax, by - ay
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1e-9:
        return float(math.hypot(px - ax, py - ay))
    t = ((px - ax) * vx + (py - ay) * vy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * vx, ay + t * vy
    return float(math.hypot(px - cx, py - cy))


def lane_clear_quality(
    ball_pos: Tuple[float, float],
    aim_point: Tuple[float, float],
    defender_pos: Optional[Tuple[float, float]],
    block_dist: float,
) -> float:
    """Return 0..1 for how clear the ball→aim_point lane is of the defender.

    1.0 when the defender is at least `block_dist` from the shot segment, ramping
    linearly to 0.0 when the defender sits exactly on the line. No defender (or a
    non-positive block_dist) means a fully clear lane.
    """

    if defender_pos is None or block_dist <= 0.0:
        return 1.0
    d = _distance_point_to_segment(defender_pos, ball_pos, aim_point)
    return float(max(0.0, min(1.0, d / float(block_dist))))


def safe_goal_target_ys(
    goal_half_height: float,
    safety_margin: float = 0.0,
    target_y: float = 0.0,
) -> Tuple[float, ...]:
    """Return safe in-mouth target y candidates used for defender-lane scoring."""

    safe_edge = max(0.0, float(goal_half_height) - max(0.0, float(safety_margin)))
    target_mag = abs(float(target_y))
    if target_mag <= 1e-6:
        target_mag = min(safe_edge, float(goal_half_height) * 0.8)
    else:
        target_mag = min(target_mag, safe_edge)

    candidates = [0.0]
    if target_mag > 1e-6:
        candidates.extend([-target_mag, target_mag])
    unique = []
    for y in candidates:
        if not any(math.isclose(y, existing, abs_tol=1e-9) for existing in unique):
            unique.append(float(y))
    return tuple(unique)


def best_defender_lane_quality(
    point: Tuple[float, float],
    defender_pos: Optional[Tuple[float, float]],
    block_dist: float,
    goal_half_height: float,
    safety_margin: float = 0.0,
    target_y: float = 0.0,
) -> float:
    """Best lane clearance from point to any safe in-mouth shot target."""

    return float(max(
        lane_clear_quality(point, (FIELD_X[1], y), defender_pos, block_dist)
        for y in safe_goal_target_ys(goal_half_height, safety_margin, target_y)
    ))


def positional_shot_quality(
    point: Tuple[float, float],
    goalie_y: float,
    defender_pos: Optional[Tuple[float, float]],
    goal_half_height: float,
    lane_block_dist: float,
    goal_post_safety_margin: float = 0.0,
    target_y: float = 0.0,
) -> float:
    """Keeper-gap quality of a shot from `point`, discounted by defender lanes.

    Combines `positional_gap_quality` (lateral separation from the keeper) with
    the best clear lane from `point` to safe in-mouth target candidates.
    Used to score dribble_to targets in Stage 3: a good target is one that is
    BOTH off the keeper's cover AND off the defender's covered lane.
    """

    base = positional_gap_quality(point, goalie_y, goal_half_height)
    if base <= 0.0:
        return 0.0
    lane = best_defender_lane_quality(
        point,
        defender_pos,
        lane_block_dist,
        goal_half_height,
        goal_post_safety_margin,
        target_y,
    )
    return float(base * lane)


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

    # dribble_to target progress and quality delta.
    dribble_to_active = bool(inputs.dribble_target is not None)

    dribble_target_progress = None
    if (
        dribble_to_active
        and inputs.prev_ball_to_dribble_target_dist is not None
    ):
        ball_to_target_dist = float(math.hypot(
            bx - inputs.dribble_target[0],
            by - inputs.dribble_target[1],
        ))
        dribble_target_progress = float(
            inputs.prev_ball_to_dribble_target_dist - ball_to_target_dist
        )

    dribble_target_quality_delta = None
    if dribble_to_active and inputs.goalie_y is not None:
        if config.use_defender_lane_gate:
            # Stage 3: target quality also accounts for whether the dribble spot
            # opens a lane the defender doesn't cover, not just the keeper gap.
            current_q = positional_shot_quality(
                inputs.ball_pos, inputs.goalie_y, inputs.defender_pos,
                config.goal_half_height, config.defender_lane_block_dist,
                config.goal_post_safety_margin, config.kick_keeper_away_target_y,
            )
            target_q = positional_shot_quality(
                inputs.dribble_target, inputs.goalie_y, inputs.defender_pos,
                config.goal_half_height, config.defender_lane_block_dist,
                config.goal_post_safety_margin, config.kick_keeper_away_target_y,
            )
        else:
            current_q = positional_gap_quality(
                inputs.ball_pos, inputs.goalie_y, config.goal_half_height,
            )
            target_q = positional_gap_quality(
                inputs.dribble_target, inputs.goalie_y, config.goal_half_height,
            )
        dribble_target_quality_delta = float(max(0.0, target_q - current_q))

    # Stage 3: clearance of the ACTUAL shot lane (ball-velocity projection to the
    # goal line) from the defender. Multiplies into the dense aim gates so pushing
    # the ball straight at goal through the defender earns essentially nothing.
    defender_lane_clear = 1.0
    if (
        config.use_defender_lane_gate
        and inputs.defender_pos is not None
        and predicted_y_at_goal_line is not None
    ):
        defender_lane_clear = lane_clear_quality(
            (bx, by),
            (FIELD_X[1], float(predicted_y_at_goal_line)),
            inputs.defender_pos,
            config.defender_lane_block_dist,
        )

    # ---- Stage 4: own-goalie coordination ----
    own_goalie_has_ball = False
    clearance_receive_progress = None
    if inputs.own_goalie_pos is not None:
        gx, gy = inputs.own_goalie_pos
        ball_to_own_goalie = float(math.hypot(bx - gx, by - gy))
        own_goalie_has_ball = bool(ball_to_own_goalie < config.own_goalie_clearance_dist)

        if own_goalie_has_ball and inputs.prev_clearance_zone_dist is not None:
            clearance_zone_x = 0.0
            clearance_zone_y = gy * 0.3
            clearance_zone_dist = float(math.hypot(rx - clearance_zone_x, ry - clearance_zone_y))
            clearance_receive_progress = float(inputs.prev_clearance_zone_dist - clearance_zone_dist)

    in_receive_position = bool(
        own_goalie_has_ball
        and config.receive_positioning_x_range[0] <= rx <= config.receive_positioning_x_range[1]
    )

    in_own_half_loiter = bool(rx < config.own_half_loiter_x)

    ball_recovered = bool(
        inputs.has_ball
        and not inputs.prev_has_ball
        and inputs.prev_opponent_near_ball
    )

    # ---- Stage 5+: multi-robot coordination ----
    nearest_ally_dist = None
    if inputs.ally_positions:
        nearest_ally_dist = float(min(
            math.hypot(rx - ax, ry - ay) for ax, ay in inputs.ally_positions
        ))

    is_well_spread = bool(
        nearest_ally_dist is not None
        and nearest_ally_dist >= config.spread_min_dist
    )

    in_support_position = False
    if (
        not inputs.is_nearest_to_ball
        and inputs.ally_positions
        and ball_dist < config.support_position_max_dist
    ):
        to_goal_from_ball_x = FIELD_X[1] - bx
        to_goal_from_ball_y = -by
        to_robot_from_ball_x = rx - bx
        to_robot_from_ball_y = ry - by
        goal_norm = math.hypot(to_goal_from_ball_x, to_goal_from_ball_y)
        robot_norm = math.hypot(to_robot_from_ball_x, to_robot_from_ball_y)
        if goal_norm > 1e-6 and robot_norm > 1e-6:
            cos_angle = (
                to_goal_from_ball_x * to_robot_from_ball_x
                + to_goal_from_ball_y * to_robot_from_ball_y
            ) / (goal_norm * robot_norm)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle_deg = math.degrees(math.acos(cos_angle))
            is_forward = bool(rx > bx - 5.0)
            in_support_position = bool(
                is_forward and angle_deg >= config.support_position_min_angle_deg
            )

    is_redundant_chaser = bool(
        not inputs.is_nearest_to_ball
        and ball_dist < config.redundant_chase_dist
    )

    opponent_near_ball = bool(
        nearest_opponent_ball_dist is not None
        and nearest_opponent_ball_dist < config.opponent_near_ball_threshold
    )

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
        opponent_near_ball=opponent_near_ball,
        in_shoot_state=bool(inputs.state == "shoot_on_goal"),
        in_dribble_state=bool(inputs.state == "dribble_to_goal"),
        goal_scored=goal_scored,
        robot_out_of_bounds=robot_out_of_bounds,
        ball_out_of_bounds=ball_out_of_bounds,
        goalie_y=inputs.goalie_y,
        defender_lane_clear=defender_lane_clear,
        dribble_to_active=dribble_to_active,
        dribble_target_progress=dribble_target_progress,
        dribble_target_quality_delta=dribble_target_quality_delta,
        is_dribbling=bool(inputs.is_dribbling),
        # Stage 4
        own_goalie_has_ball=own_goalie_has_ball,
        clearance_receive_progress=clearance_receive_progress,
        in_receive_position=in_receive_position,
        in_own_half_loiter=in_own_half_loiter,
        ball_recovered=ball_recovered,
        # Stage 5+
        is_nearest_to_ball=inputs.is_nearest_to_ball,
        nearest_ally_dist=nearest_ally_dist,
        is_well_spread=is_well_spread,
        in_support_position=in_support_position,
        is_redundant_chaser=is_redundant_chaser,
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
        quality = post_safe_goalie_gap_quality(
            intermediates.predicted_y_at_goal_line,
            intermediates.goalie_y,
            config.goal_half_height,
            config.goal_post_safety_margin,
        )
    else:
        quality = aim_quality_from_prediction(
            intermediates.predicted_y_at_goal_line,
            config.goal_half_height,
        )
    # Stage 3: a fast/progressing ball aimed through the defender's lane is not a
    # real chance — discount the dense gates by how clear that lane is.
    if config.use_defender_lane_gate:
        quality *= intermediates.defender_lane_clear
    return quality


def calculate_reward(
    intermediates: RewardIntermediates,
    config: Optional[RewardConfig] = None,
) -> float:
    """Convert intermediate reward terms into the final scalar reward."""

    if config is None:
        config = RewardConfig()

    reward = 0.0

    # Role-gating: when enabled, only the nearest-to-ball robot gets
    # chase/possession rewards. Non-nearest robots get positioning rewards.
    is_chaser = not config.enable_role_gating or intermediates.is_nearest_to_ball

    if intermediates.approach is not None and not intermediates.has_ball and is_chaser:
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

    if intermediates.has_ball and is_chaser:
        reward += config.has_ball_bonus
    if intermediates.near_ball and is_chaser:
        reward += config.near_ball_bonus
    if intermediates.opponent_near_ball:
        reward += config.opponent_near_ball_penalty

    if intermediates.in_shoot_state:
        reward += config.shoot_state_bonus

    # dribble_to target progress: reward ball approaching the chosen target.
    if (
        intermediates.dribble_target_progress is not None
        and config.dribble_target_progress_weight > 0.0
    ):
        progress = float(np.clip(
            intermediates.dribble_target_progress,
            -config.dribble_target_progress_clip,
            config.dribble_target_progress_clip,
        ))
        reward += progress * config.dribble_target_progress_weight

    # dribble_to active bonus: small per-step reward while dribble_to is
    # active AND the target has positive quality delta.
    if (
        intermediates.dribble_to_active
        and intermediates.is_dribbling
        and config.dribble_active_bonus > 0.0
        and intermediates.dribble_target_quality_delta is not None
        and intermediates.dribble_target_quality_delta > 0.0
    ):
        reward += config.dribble_active_bonus

    # ---- Stage 4: own-goalie coordination ----
    if (
        intermediates.clearance_receive_progress is not None
        and config.clearance_receive_weight > 0.0
        and intermediates.own_goalie_has_ball
    ):
        progress = float(np.clip(
            intermediates.clearance_receive_progress,
            -config.clearance_receive_clip,
            config.clearance_receive_clip,
        ))
        reward += progress * config.clearance_receive_weight

    if intermediates.in_receive_position and config.receive_positioning_bonus > 0.0:
        reward += config.receive_positioning_bonus

    if intermediates.in_own_half_loiter and config.own_half_loiter_penalty > 0.0:
        reward -= config.own_half_loiter_penalty

    if intermediates.ball_recovered and config.ball_recovery_bonus > 0.0:
        reward += config.ball_recovery_bonus

    # ---- Stage 5+: multi-robot coordination ----
    if not is_chaser:
        if config.spread_bonus > 0.0 and intermediates.is_well_spread:
            reward += config.spread_bonus

        if config.support_position_bonus > 0.0 and intermediates.in_support_position:
            reward += config.support_position_bonus

    if intermediates.is_redundant_chaser and config.redundant_chase_penalty > 0.0:
        reward -= config.redundant_chase_penalty

    if intermediates.goal_scored:
        reward += config.goal_reward * config.team_goal_multiplier
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

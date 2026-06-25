"""Marker/cover defender for Stage 4+ multi-defender setups.

Complements the ball-challenging Defender (defender.py) by marking receivers,
covering pass lanes, and providing center-back depth. Follows TIGERs Mannheim
Sumatra principles: only one non-goalie defender challenges the ball at a time;
the marker covers the most dangerous remaining threat.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from ai_interface.constants.field_constants import FIELD_X, FIELD_Y, GOAL_L, GOAL_R
from ai_interface.constants.player_constants import BALL_SIZE, KICKABLE_MARGIN, PLAYER_SIZE
from ai_interface.defender import (
    KICKABLE,
    OpponentInfo,
    _defended_goal,
    _dist,
    _opponents,
    _point_to_segment_distance,
    _unit_from_to,
    clamp_legal_defender_target,
    detect_ball_handler,
    is_inside_own_defense_area,
    safe_clear_target,
)
from ai_interface.player import Player
from ai_interface.utils import basic_commands
from ai_interface.utils.algo_utils import normalize_angle


MARK_STANDOFF = 3.0
MARK_SPEED = 68.0
COVER_SPEED = 60.0
COVER_DEPTH = 7.0
COVER_LATERAL_WEIGHT = 0.6
INTERCEPT_PASS_SPEED = 90.0
PASS_SPEED_THRESHOLD = 0.55
PASS_LOOKAHEAD_STEPS = 14
PASS_LINE_MAX_DIST = 3.5
CLEAR_HOLD_CYCLE_LIMIT = 8
CLEAR_CARRY_LIMIT = 0.75
FIELD_MARGIN = 2.0

THREAT_WEIGHT_REDIRECT = 0.9
THREAT_WEIGHT_CENTRALIZATION = 1.0
THREAT_WEIGHT_GOAL_DIST = 1.0
THREAT_WEIGHT_PASS_DIST = 0.5


def _attacker_opponents(
    teamname: str,
    game_state,
) -> list[OpponentInfo]:
    """Return opponent (attacker) robots visible in game_state."""
    return _opponents(teamname, game_state)


def _ball_defender_pose(
    teamname: str,
    ball_defender_robot_id: int,
    game_state,
) -> Optional[Tuple[float, float]]:
    """Read Defender 1's position from game_state to avoid co-located challenging."""
    for entry in getattr(game_state, "robot_poses", {}).get(teamname, []):
        if not isinstance(entry, dict):
            continue
        pose = entry.get(ball_defender_robot_id)
        if pose is not None and len(pose) >= 3:
            return (float(pose[0]), float(pose[1]))
    return None


def threat_rating(
    opp: OpponentInfo,
    ball_pos: Tuple[float, float],
    goal: Tuple[float, float],
) -> float:
    """TIGERs-style multi-factor threat score for an opponent robot.

    Higher = more dangerous to leave unmarked.
    """
    opp_x, opp_y = opp.xy

    to_goal_dx = goal[0] - opp_x
    to_goal_dy = goal[1] - opp_y
    goal_dist = math.hypot(to_goal_dx, to_goal_dy)
    if goal_dist < 1e-6:
        goal_dist = 0.01

    redirect_angle = 0.0
    ball_dx = opp_x - ball_pos[0]
    ball_dy = opp_y - ball_pos[1]
    ball_dist = math.hypot(ball_dx, ball_dy)
    if ball_dist > 1e-6:
        cos_redirect = (ball_dx * to_goal_dx + ball_dy * to_goal_dy) / (ball_dist * goal_dist)
        redirect_angle = max(0.0, cos_redirect)

    centralization = 1.0 - min(1.0, abs(opp_y) / (abs(FIELD_Y[1]) * 0.8))

    max_goal_dist = math.hypot(FIELD_X[1] - FIELD_X[0], FIELD_Y[1] - FIELD_Y[0])
    goal_proximity = 1.0 - min(1.0, goal_dist / max_goal_dist)
    goal_proximity = goal_proximity ** 1.5

    pass_distance = min(1.0, ball_dist / 40.0)

    score = (
        THREAT_WEIGHT_REDIRECT * redirect_angle
        + THREAT_WEIGHT_CENTRALIZATION * centralization
        + THREAT_WEIGHT_GOAL_DIST * goal_proximity
        + THREAT_WEIGHT_PASS_DIST * (1.0 - pass_distance)
    )
    return score


def select_mark_target(
    teamname: str,
    ball_pos: Tuple[float, float],
    handler: Optional[OpponentInfo],
    goal: Tuple[float, float],
    game_state,
) -> Optional[OpponentInfo]:
    """Pick the most dangerous opponent attacker to mark (excluding ball handler)."""
    attackers = _attacker_opponents(teamname, game_state)
    if not attackers:
        return None

    candidates = []
    for opp in attackers:
        if handler is not None and opp.unum == handler.unum and opp.team == handler.team:
            continue
        rating = threat_rating(opp, ball_pos, goal)
        candidates.append((rating, opp))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def mark_position(
    target: OpponentInfo,
    goal: Tuple[float, float],
    side: str,
    standoff: float = MARK_STANDOFF,
) -> Tuple[float, float]:
    """Goal-side position on the receiver-to-goal line."""
    ux, uy = _unit_from_to(target.xy, goal)
    point = (target.xy[0] + ux * standoff, target.xy[1] + uy * standoff)
    return clamp_legal_defender_target(point, side)


def cover_position(
    ball_pos: Tuple[float, float],
    goal: Tuple[float, float],
    side: str,
    depth: float = COVER_DEPTH,
) -> Tuple[float, float]:
    """Center-back cover position in front of goal, shifted toward ball."""
    goal_x, goal_y = goal
    if side == "right":
        cover_x = goal_x - depth
    else:
        cover_x = goal_x + depth

    cover_y = ball_pos[1] * COVER_LATERAL_WEIGHT

    return clamp_legal_defender_target((cover_x, cover_y), side)


def pass_intercept_point(
    self_xy: Tuple[float, float],
    ball_pos: Tuple[float, float],
    ball_vel: Tuple[float, float],
    side: str,
) -> Optional[Tuple[float, float]]:
    """Find an intercept point on a fast-moving ball trajectory."""
    speed = math.hypot(ball_vel[0], ball_vel[1])
    if speed < PASS_SPEED_THRESHOLD:
        return None

    best = None
    best_margin = -math.inf
    reach_per_step = 0.45
    for step in range(2, PASS_LOOKAHEAD_STEPS + 1):
        point = (
            ball_pos[0] + ball_vel[0] * step,
            ball_pos[1] + ball_vel[1] * step,
        )
        if not _is_legal_field_point(point):
            continue
        if is_inside_own_defense_area(point, side):
            continue
        reach = reach_per_step * step + 0.5
        margin = reach - _dist(self_xy, point)
        if margin >= 0.0:
            return clamp_legal_defender_target(point, side)
        if margin > best_margin:
            best = point
            best_margin = margin

    if best is not None:
        return clamp_legal_defender_target(best, side)
    return None


def _is_legal_field_point(point: Sequence[float]) -> bool:
    x, y = float(point[0]), float(point[1])
    return (
        FIELD_X[0] + FIELD_MARGIN <= x <= FIELD_X[1] - FIELD_MARGIN
        and FIELD_Y[0] + FIELD_MARGIN <= y <= FIELD_Y[1] - FIELD_MARGIN
    )


class MarkerDefender(Player):
    """Receiver-marking / cover defender for multi-defender stages."""

    def __init__(
        self,
        teamname: str,
        unum: int,
        side: str = "right",
        ball_defender_robot_id: int = 2,
    ) -> None:
        super().__init__(teamname=teamname, unum=unum)
        self.side = side
        self.goal = _defended_goal(side)
        self.ball_defender_robot_id = ball_defender_robot_id
        self.mode = "COVER"
        self._last_ball_pos: Optional[Tuple[float, float]] = None
        self._last_count: Optional[int] = None
        self._ball_vel: Tuple[float, float] = (0.0, 0.0)
        self._clear_hold_cycles = 0
        self._clear_anchor: Optional[Tuple[float, float]] = None

    def _update_ball_velocity(
        self, ball_pos: Tuple[float, float], game_state
    ) -> Tuple[float, float]:
        count_raw = getattr(game_state, "count", None) if game_state is not None else None
        count = int(count_raw) if count_raw is not None else None
        if self._last_ball_pos is not None:
            dt_count = (
                1
                if count is None or self._last_count is None
                else max(1, count - self._last_count)
            )
            self._ball_vel = (
                (ball_pos[0] - self._last_ball_pos[0]) / dt_count,
                (ball_pos[1] - self._last_ball_pos[1]) / dt_count,
            )
        self._last_ball_pos = ball_pos
        self._last_count = count
        return self._ball_vel

    def _move_to(
        self,
        self_pose_rad: Tuple[float, float, float],
        tx: float,
        ty: float,
        game_state,
        margin: float,
        face: Optional[Tuple[float, float]],
        speed: float,
        full_speed: bool = False,
    ) -> str:
        theta = None
        if face is not None:
            theta = math.atan2(face[1] - self_pose_rad[1], face[0] - self_pose_rad[0])
        cmd = basic_commands.goto(
            self_pose_rad,
            tx,
            ty,
            game_state,
            margin=margin,
            theta=theta,
            speed=(100.0 if full_speed else speed),
            is_goalie=full_speed,
            obstacle_avoidance=not full_speed,
        )
        return cmd if cmd != "done" else "turn 0"

    def _clear(
        self,
        self_pose_rad: Tuple[float, float, float],
        ball_pos: Tuple[float, float],
        game_state,
    ) -> str:
        self.mode = "CLEAR"
        self_xy = (self_pose_rad[0], self_pose_rad[1])
        if self._clear_anchor is None:
            self._clear_anchor = ball_pos
            self._clear_hold_cycles = 0
        else:
            self._clear_hold_cycles += 1

        target = safe_clear_target(
            self_xy, ball_pos, self.side, _opponents(self.teamname, game_state)
        )
        target_angle = math.atan2(
            target[1] - self_pose_rad[1], target[0] - self_pose_rad[0]
        )
        angle_diff = normalize_angle(target_angle - self_pose_rad[2])

        carried = _dist(ball_pos, self._clear_anchor)
        if self.dribbling and (
            self._clear_hold_cycles >= CLEAR_HOLD_CYCLE_LIMIT
            or carried >= CLEAR_CARRY_LIMIT
        ):
            if abs(angle_diff) <= math.radians(25.0):
                self.dribbling = False
                self._clear_anchor = None
                self._clear_hold_cycles = 0
                return "kick 70 0"
            self.dribbling = False
            self._clear_anchor = None
            self._clear_hold_cycles = 0
            return "drop"

        cmd = self.kick(target_angle, self_pose_rad, ball_pos, kick_power=85)
        if cmd.startswith("kick"):
            self._clear_anchor = None
            self._clear_hold_cycles = 0
        return cmd if cmd != "failed" else "turn 0"

    def action(
        self,
        ball_pos: Tuple[float, float],
        defender_pose: Tuple[float, float, float],
        game_state=None,
    ) -> str:
        bx, by = float(ball_pos[0]), float(ball_pos[1])
        ball_xy = (bx, by)
        dx, dy, dtheta_deg = (
            float(defender_pose[0]),
            float(defender_pose[1]),
            float(defender_pose[2]),
        )
        self_xy = (dx, dy)
        self_pose_rad = (dx, dy, math.radians(dtheta_deg))
        def_to_ball = _dist(self_xy, ball_xy)

        ball_vel = self._update_ball_velocity(ball_xy, game_state)
        handler = (
            detect_ball_handler(self.teamname, ball_xy, game_state)
            if game_state is not None
            else None
        )

        # Priority 1: clear if ball is at feet and legal
        if (
            def_to_ball <= KICKABLE
            and not is_inside_own_defense_area(self_xy, self.side)
            and not is_inside_own_defense_area(ball_xy, self.side)
        ):
            return self._clear(self_pose_rad, ball_xy, game_state)

        self._clear_anchor = None
        self._clear_hold_cycles = 0
        self.dribbling = False

        # Priority 2: intercept a pass in flight
        ball_speed = math.hypot(ball_vel[0], ball_vel[1])
        if ball_speed >= PASS_SPEED_THRESHOLD:
            intercept = pass_intercept_point(self_xy, ball_xy, ball_vel, self.side)
            if intercept is not None:
                self.mode = "INTERCEPT_PASS"
                return self._move_to(
                    self_pose_rad,
                    intercept[0],
                    intercept[1],
                    game_state,
                    margin=0.2,
                    face=ball_xy,
                    speed=INTERCEPT_PASS_SPEED,
                    full_speed=True,
                )

        # Priority 3: mark the most dangerous unmarked attacker
        mark_target = select_mark_target(
            self.teamname, ball_xy, handler, self.goal, game_state
        )
        if mark_target is not None:
            self.mode = "MARK"
            target = mark_position(mark_target, self.goal, self.side)
            return self._move_to(
                self_pose_rad,
                target[0],
                target[1],
                game_state,
                margin=0.3,
                face=ball_xy,
                speed=MARK_SPEED,
            )

        # Priority 4: center-back cover
        self.mode = "COVER"
        target = cover_position(ball_xy, self.goal, self.side)
        return self._move_to(
            self_pose_rad,
            target[0],
            target[1],
            game_state,
            margin=0.35,
            face=ball_xy,
            speed=COVER_SPEED,
        )

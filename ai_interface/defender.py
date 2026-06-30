"""Scripted threat defender for Stage 3 and future multi-defender stages.

The defender follows the same high-level idea used by strong SSL teams: defend
the active threat line, not just a static point. In Stage 3 there is still only
one field defender, so this class remains a single-robot controller, but the
helpers are intentionally factored so later stages can assign multiple defenders
to ball pressure, receiver marking, and pass disruption.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple

from ai_interface.constants.field_constants import FIELD_X, FIELD_Y, GOAL_L, GOAL_R
from ai_interface.constants.player_constants import BALL_SIZE, KICKABLE_MARGIN, PLAYER_SIZE
from ai_interface.player import Player
from ai_interface.utils import basic_commands
from ai_interface.utils.algo_utils import normalize_angle


KICKABLE = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE

# Approximate defense-area guard for the 90x60 simulator field. Non-goalie
# defenders should stay outside this rectangle; the goalie owns the deep mouth.
DEFENSE_AREA_X_EDGE = 35.0
DEFENSE_AREA_Y_LIMIT = 9.0
DEFENSE_AREA_MARGIN = 0.6

# General movement tuning. PRESS is deliberately slower than emergency
# interception to reduce pushing/crashing risk when the attacker is also on the
# ball.
PROTECT_STANDOFF = 4.0
PROTECT_SPEED = 72.0
PRESS_SPEED = 58.0
PRESS_CONTACT_SPEED = 42.0
INTERCEPT_SPEED = 95.0
RECOVER_SPEED = 45.0

DANGER_X = 14.0
PRESS_RADIUS = 7.0
PRESS_GOAL_SIDE_OFFSET = 1.05
PRESS_LATERAL_OFFSET = 0.95
MIN_ATTACKER_SEPARATION = 1.25

BALL_SPEED_INTERCEPT = 0.28
GOAL_SHOT_Y_MARGIN = 7.5
RECEIVER_MAX_LINE_DIST = 3.0
RECEIVER_MIN_BALL_SPEED = 0.55
INTERCEPT_LOOKAHEAD_STEPS = 16
DEFENDER_REACH_PER_STEP = 0.45

SHARED_CONTACT_LIMIT = 5
RECOVER_CYCLES = 7
# The defender catch-glues the ball then rotates (capped at ~2 deg/cycle) to aim a
# clear. 8 cycles / 0.75 units was far too tight: it could not point upfield before
# the limit hit, so it released early. Give the geometric turn time to swing roughly
# toward the clear target before the forced release (which is now a kick, not a drop).
CLEAR_HOLD_CYCLE_LIMIT = 24
CLEAR_CARRY_LIMIT = 1.4
LOOSE_CLEAR_RADIUS = KICKABLE + 1.25
FIELD_MARGIN = 2.0


@dataclass(frozen=True)
class OpponentInfo:
    team: str
    unum: int
    pose: Tuple[float, float, float]

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.pose[0], self.pose[1])


@dataclass(frozen=True)
class ThreatInfo:
    source: Tuple[float, float]
    target: Tuple[float, float]
    kind: str
    actor: Optional[OpponentInfo] = None


def _as_xy(value: Sequence[float]) -> Tuple[float, float]:
    return (float(value[0]), float(value[1]))


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _unit_from_to(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return (1.0, 0.0)
    return (dx / norm, dy / norm)


def _point_to_segment_distance(
    point: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
) -> Tuple[float, float]:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    vx, vy = bx - ax, by - ay
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1e-9:
        return (math.hypot(px - ax, py - ay), 0.0)
    t = ((px - ax) * vx + (py - ay) * vy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * vx, ay + t * vy
    return (math.hypot(px - cx, py - cy), t)


def _opponents(teamname: str, game_state) -> list[OpponentInfo]:
    opponents: list[OpponentInfo] = []
    for team, robots in getattr(game_state, "robot_poses", {}).items():
        if team == teamname:
            continue
        for robot in robots or []:
            if not isinstance(robot, dict):
                continue
            for unum, pose in robot.items():
                if pose is None or len(pose) < 3:
                    continue
                opponents.append(
                    OpponentInfo(
                        team=str(team),
                        unum=int(unum),
                        pose=(float(pose[0]), float(pose[1]), float(pose[2])),
                    )
                )
    return opponents


def _defended_goal(side: str) -> Tuple[float, float]:
    return _as_xy(GOAL_R if side == "right" else GOAL_L)


def _clear_goal(side: str) -> Tuple[float, float]:
    return _as_xy(GOAL_L if side == "right" else GOAL_R)


def _toward_own_goal_sign(side: str) -> float:
    return 1.0 if side == "right" else -1.0


def _is_goalward(ball_vel: Tuple[float, float], side: str) -> bool:
    vx = float(ball_vel[0])
    return vx * _toward_own_goal_sign(side) > BALL_SPEED_INTERCEPT


def is_inside_own_defense_area(point: Sequence[float], side: str) -> bool:
    x, y = float(point[0]), float(point[1])
    if abs(y) > DEFENSE_AREA_Y_LIMIT:
        return False
    return x >= DEFENSE_AREA_X_EDGE if side == "right" else x <= -DEFENSE_AREA_X_EDGE


def is_legal_defender_target(point: Sequence[float], side: str) -> bool:
    x, y = float(point[0]), float(point[1])
    if x < FIELD_X[0] + FIELD_MARGIN or x > FIELD_X[1] - FIELD_MARGIN:
        return False
    if y < FIELD_Y[0] + FIELD_MARGIN or y > FIELD_Y[1] - FIELD_MARGIN:
        return False
    return not is_inside_own_defense_area((x, y), side)


def clamp_legal_defender_target(point: Sequence[float], side: str) -> Tuple[float, float]:
    x = max(FIELD_X[0] + FIELD_MARGIN, min(FIELD_X[1] - FIELD_MARGIN, float(point[0])))
    y = max(FIELD_Y[0] + FIELD_MARGIN, min(FIELD_Y[1] - FIELD_MARGIN, float(point[1])))
    if is_inside_own_defense_area((x, y), side):
        if side == "right":
            x = min(x, DEFENSE_AREA_X_EDGE - DEFENSE_AREA_MARGIN)
        else:
            x = max(x, -DEFENSE_AREA_X_EDGE + DEFENSE_AREA_MARGIN)
    return (x, y)


def detect_ball_handler(
    teamname: str,
    ball_pos: Tuple[float, float],
    game_state,
    possession_margin: float = 0.35,
) -> Optional[OpponentInfo]:
    """Return the nearest opponent controlling the ball, if any."""

    opponents = _opponents(teamname, game_state)
    if not opponents:
        return None
    nearest = min(opponents, key=lambda opp: _dist(opp.xy, ball_pos))
    if _dist(nearest.xy, ball_pos) <= KICKABLE + possession_margin:
        return nearest
    return None


def predict_receiver(
    teamname: str,
    ball_pos: Tuple[float, float],
    ball_vel: Tuple[float, float],
    game_state,
) -> Optional[OpponentInfo]:
    """Predict an opponent receiver from the current ball trajectory."""

    speed = math.hypot(float(ball_vel[0]), float(ball_vel[1]))
    if speed < RECEIVER_MIN_BALL_SPEED:
        return None
    end = (
        ball_pos[0] + ball_vel[0] * INTERCEPT_LOOKAHEAD_STEPS,
        ball_pos[1] + ball_vel[1] * INTERCEPT_LOOKAHEAD_STEPS,
    )
    candidates = []
    for opp in _opponents(teamname, game_state):
        line_dist, t = _point_to_segment_distance(opp.xy, ball_pos, end)
        if 0.05 < t <= 1.0 and line_dist <= RECEIVER_MAX_LINE_DIST:
            candidates.append((line_dist, _dist(ball_pos, opp.xy), opp))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def threat_line(
    side: str,
    ball_pos: Tuple[float, float],
    ball_vel: Tuple[float, float],
    game_state,
    teamname: str,
) -> ThreatInfo:
    """Choose the threat source and its target line.

    Priority mirrors the Sumatra shape at a simpler scale: a moving ball aimed at
    a receiver is a receiver threat; a controlled ball is a ball-handler threat;
    otherwise the current ball position is the threat.
    """

    goal = _defended_goal(side)
    receiver = predict_receiver(teamname, ball_pos, ball_vel, game_state)
    if receiver is not None:
        return ThreatInfo(source=receiver.xy, target=goal, kind="receiver", actor=receiver)

    handler = detect_ball_handler(teamname, ball_pos, game_state)
    if handler is not None:
        return ThreatInfo(source=ball_pos, target=goal, kind="handler", actor=handler)

    if _is_goalward(ball_vel, side):
        projected = _project_y_at_goal(ball_pos, ball_vel, side)
        if projected is not None:
            goal_x = goal[0]
            target_y = max(-GOAL_SHOT_Y_MARGIN, min(GOAL_SHOT_Y_MARGIN, projected))
            return ThreatInfo(source=ball_pos, target=(goal_x, target_y), kind="shot")

    return ThreatInfo(source=ball_pos, target=goal, kind="ball")


def protection_point(
    threat: ThreatInfo,
    side: str,
    standoff: float = PROTECT_STANDOFF,
) -> Tuple[float, float]:
    """Point on the threat-to-goal line, goal-side of the threat source."""

    ux, uy = _unit_from_to(threat.source, threat.target)
    point = (threat.source[0] + ux * standoff, threat.source[1] + uy * standoff)
    return clamp_legal_defender_target(point, side)


def pressure_point(
    ball_pos: Tuple[float, float],
    attacker: OpponentInfo,
    side: str,
) -> Tuple[float, float]:
    """Legal pressure point near the ball without driving through the attacker."""

    goal = _defended_goal(side)
    ux, uy = _unit_from_to(ball_pos, goal)
    px, py = -uy, ux
    attacker_side = (attacker.xy[0] - ball_pos[0]) * px + (attacker.xy[1] - ball_pos[1]) * py
    lateral_sign = -1.0 if attacker_side >= 0.0 else 1.0
    candidate = (
        ball_pos[0] + ux * PRESS_GOAL_SIDE_OFFSET + px * PRESS_LATERAL_OFFSET * lateral_sign,
        ball_pos[1] + uy * PRESS_GOAL_SIDE_OFFSET + py * PRESS_LATERAL_OFFSET * lateral_sign,
    )

    if _dist(candidate, attacker.xy) < MIN_ATTACKER_SEPARATION:
        candidate = (
            candidate[0] + px * PRESS_LATERAL_OFFSET * lateral_sign,
            candidate[1] + py * PRESS_LATERAL_OFFSET * lateral_sign,
        )
    return clamp_legal_defender_target(candidate, side)


def safe_clear_target(
    self_xy: Tuple[float, float],
    ball_pos: Tuple[float, float],
    side: str,
    opponents: Iterable[OpponentInfo] = (),
) -> Tuple[float, float]:
    """Choose an in-field upfield clearance target away from own goal."""

    clear_goal = _clear_goal(side)
    target_x = clear_goal[0] - math.copysign(3.0, clear_goal[0])
    target_y = max(FIELD_Y[0] + 6.0, min(FIELD_Y[1] - 6.0, self_xy[1] * 0.4))
    direct = (target_x, target_y)

    blockers = []
    for opp in opponents:
        d, t = _point_to_segment_distance(opp.xy, ball_pos, direct)
        if 0.0 < t < 1.0 and d < 2.2:
            blockers.append(opp)
    if blockers:
        blocker_y = blockers[0].xy[1]
        offset_sign = -1.0 if blocker_y > ball_pos[1] else 1.0
        target_y = max(FIELD_Y[0] + 6.0, min(FIELD_Y[1] - 6.0, ball_pos[1] + 12.0 * offset_sign))
    return (target_x, target_y)


def _project_y_at_goal(
    ball_pos: Tuple[float, float],
    ball_vel: Tuple[float, float],
    side: str,
) -> Optional[float]:
    vx, vy = float(ball_vel[0]), float(ball_vel[1])
    goal_x = _defended_goal(side)[0]
    if abs(vx) < 1e-6:
        return None
    t = (goal_x - ball_pos[0]) / vx
    if t <= 0.0:
        return None
    return float(ball_pos[1] + vy * t)


def _goalward_intercept_target(
    defender_xy: Tuple[float, float],
    ball_pos: Tuple[float, float],
    ball_vel: Tuple[float, float],
    side: str,
) -> Optional[Tuple[float, float]]:
    if not _is_goalward(ball_vel, side):
        return None
    projected_y = _project_y_at_goal(ball_pos, ball_vel, side)
    if projected_y is None or abs(projected_y) > GOAL_SHOT_Y_MARGIN:
        return None

    best = None
    best_margin = -math.inf
    for step in range(2, INTERCEPT_LOOKAHEAD_STEPS + 1):
        point = (
            ball_pos[0] + ball_vel[0] * step,
            ball_pos[1] + ball_vel[1] * step,
        )
        if not is_legal_defender_target(point, side):
            continue
        reach = DEFENDER_REACH_PER_STEP * step + 0.5
        margin = reach - _dist(defender_xy, point)
        if margin >= 0.0:
            return point
        if margin > best_margin:
            best = point
            best_margin = margin
    if best is None:
        return None
    return clamp_legal_defender_target(best, side)


class Defender(Player):
    """A single rule-conscious Stage 3 defender."""

    def __init__(self, teamname: str, unum: int, side: str = "right") -> None:
        super().__init__(teamname=teamname, unum=unum)
        self.side = side
        self.goal = _defended_goal(side)
        self.clear_goal = _clear_goal(side)
        self.mode = "PROTECT"
        self._last_ball_pos: Optional[Tuple[float, float]] = None
        self._last_count: Optional[int] = None
        self._ball_vel: Tuple[float, float] = (0.0, 0.0)
        self._shared_contact_cycles = 0
        self._recover_until_count: Optional[int] = None
        self._clear_hold_cycles = 0
        self._clear_anchor: Optional[Tuple[float, float]] = None

    # --- Perception/state -------------------------------------------------

    def _update_ball_velocity(self, ball_pos: Tuple[float, float], game_state) -> Tuple[float, float]:
        count_raw = getattr(game_state, "count", None) if game_state is not None else None
        count = int(count_raw) if count_raw is not None else None
        if self._last_ball_pos is not None:
            dt_count = 1 if count is None or self._last_count is None else max(1, count - self._last_count)
            self._ball_vel = (
                (ball_pos[0] - self._last_ball_pos[0]) / dt_count,
                (ball_pos[1] - self._last_ball_pos[1]) / dt_count,
            )
        self._last_ball_pos = ball_pos
        self._last_count = count
        return self._ball_vel

    def _recover_active(self, game_state) -> bool:
        if self._recover_until_count is None:
            return False
        count_raw = getattr(game_state, "count", None) if game_state is not None else None
        if count_raw is None:
            return False
        if int(count_raw) <= self._recover_until_count:
            return True
        self._recover_until_count = None
        self._shared_contact_cycles = 0
        return False

    def _update_contact_guard(
        self,
        defender_xy: Tuple[float, float],
        ball_pos: Tuple[float, float],
        handler: Optional[OpponentInfo],
        game_state,
    ) -> None:
        if handler is None:
            self._shared_contact_cycles = 0
            return
        both_on_ball = (
            _dist(defender_xy, ball_pos) <= KICKABLE + 0.18
            and _dist(handler.xy, ball_pos) <= KICKABLE + 0.18
        )
        if not both_on_ball:
            self._shared_contact_cycles = 0
            return
        self._shared_contact_cycles += 1
        if self._shared_contact_cycles >= SHARED_CONTACT_LIMIT:
            count_raw = getattr(game_state, "count", None) if game_state is not None else None
            if count_raw is not None:
                self._recover_until_count = int(count_raw) + RECOVER_CYCLES

    # --- Command generation ----------------------------------------------

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

    def _recover(
        self,
        self_pose_rad: Tuple[float, float, float],
        ball_pos: Tuple[float, float],
        game_state,
    ) -> str:
        self.mode = "RECOVER"
        ux, uy = _unit_from_to(ball_pos, self.goal)
        target = clamp_legal_defender_target((ball_pos[0] + ux * 2.7, ball_pos[1] + uy * 2.7), self.side)
        return self._move_to(self_pose_rad, target[0], target[1], game_state, 0.35, ball_pos, RECOVER_SPEED)

    def _clear(self, self_pose_rad: Tuple[float, float, float], ball_pos: Tuple[float, float], game_state) -> str:
        self.mode = "CLEAR"
        self_xy = (self_pose_rad[0], self_pose_rad[1])
        if self._clear_anchor is None:
            self._clear_anchor = ball_pos
            self._clear_hold_cycles = 0
        else:
            self._clear_hold_cycles += 1

        target = safe_clear_target(self_xy, ball_pos, self.side, _opponents(self.teamname, game_state))
        target_angle = math.atan2(target[1] - self_pose_rad[1], target[0] - self_pose_rad[0])
        angle_diff = normalize_angle(target_angle - self_pose_rad[2])
        dist_to_ball = _dist(self_xy, ball_pos)

        if dist_to_ball > KICKABLE:
            ux, uy = _unit_from_to(ball_pos, target)
            stand_off = PLAYER_SIZE + BALL_SIZE + 0.2
            contact_x = ball_pos[0] - ux * stand_off
            contact_y = ball_pos[1] - uy * stand_off
            contact = clamp_legal_defender_target((contact_x, contact_y), self.side)
            return basic_commands.goto(
                self_pose_rad,
                contact[0],
                contact[1],
                game_state,
                margin=0.18,
                theta=math.atan2(ball_pos[1] - self_pose_rad[1], ball_pos[0] - self_pose_rad[0]),
                speed=PRESS_CONTACT_SPEED,
                obstacle_avoidance=True,
                include_ball_obstacle=False,
                include_player_obstacles=True,
            )

        if not basic_commands.ball_in_front_reception_cone(self_pose_rad, ball_pos):
            cmd = self.dribble(self_pose_rad, ball_pos, kickable_tolerance=KICKABLE + 0.25)
            return cmd if cmd != "failed" else f"turn {normalize_angle(math.atan2(ball_pos[1] - self_pose_rad[1], ball_pos[0] - self_pose_rad[0]) - self_pose_rad[2])}"

        carried = _dist(ball_pos, self._clear_anchor)
        if self.dribbling and (
            self._clear_hold_cycles >= CLEAR_HOLD_CYCLE_LIMIT or carried >= CLEAR_CARRY_LIMIT
        ):
            self.dribbling = False
            self._clear_anchor = None
            self._clear_hold_cycles = 0
            # Forced release: boot the ball forward. NEVER `drop` it here — a drop
            # releases the glued ball at our own feet, where the pressing attacker
            # is, which hands them possession. The geometric turn above has already
            # swung the body roughly toward the upfield clear target, so a forward
            # kick clears it away from our goal (stronger when better aligned).
            power = 70 if abs(angle_diff) <= math.radians(25.0) else 85
            return f"kick {power} 0"

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
        handler = detect_ball_handler(self.teamname, ball_xy, game_state) if game_state is not None else None
        self._update_contact_guard(self_xy, ball_xy, handler, game_state)

        if self._recover_active(game_state):
            self.dribbling = False
            self._clear_anchor = None
            self._clear_hold_cycles = 0
            return self._recover(self_pose_rad, ball_xy, game_state)

        if (
            def_to_ball <= LOOSE_CLEAR_RADIUS
            and not is_inside_own_defense_area(self_xy, self.side)
            and not is_inside_own_defense_area(ball_xy, self.side)
        ):
            return self._clear(self_pose_rad, ball_xy, game_state)

        self._clear_anchor = None
        self._clear_hold_cycles = 0
        self.dribbling = False

        intercept = _goalward_intercept_target(self_xy, ball_xy, ball_vel, self.side)
        receiver = predict_receiver(self.teamname, ball_xy, ball_vel, game_state) if game_state is not None else None
        if intercept is not None:
            self.mode = "INTERCEPT"
            return self._move_to(
                self_pose_rad,
                intercept[0],
                intercept[1],
                game_state,
                margin=0.2,
                face=ball_xy,
                speed=INTERCEPT_SPEED,
                full_speed=True,
            )
        if receiver is not None and math.hypot(ball_vel[0], ball_vel[1]) >= RECEIVER_MIN_BALL_SPEED:
            self.mode = "INTERCEPT"
            line_target = clamp_legal_defender_target(receiver.xy, self.side)
            return self._move_to(
                self_pose_rad,
                line_target[0],
                line_target[1],
                game_state,
                margin=0.25,
                face=ball_xy,
                speed=INTERCEPT_SPEED,
                full_speed=True,
            )

        if handler is not None and self._ball_in_danger_zone(ball_xy) and def_to_ball <= PRESS_RADIUS:
            self.mode = "PRESS"
            target = pressure_point(ball_xy, handler, self.side)
            attacker_close = _dist(handler.xy, self_xy) <= 2.4
            return self._move_to(
                self_pose_rad,
                target[0],
                target[1],
                game_state,
                margin=0.28,
                face=ball_xy,
                speed=PRESS_CONTACT_SPEED if attacker_close else PRESS_SPEED,
            )

        threat = threat_line(self.side, ball_xy, ball_vel, game_state, self.teamname)
        target = protection_point(threat, self.side)
        target = self._mimic_threat_lateral_velocity(target, threat, ball_vel)
        self.mode = "PROTECT"
        return self._move_to(
            self_pose_rad,
            target[0],
            target[1],
            game_state,
            margin=0.35,
            face=ball_xy,
            speed=PROTECT_SPEED,
        )

    def _ball_in_danger_zone(self, ball_pos: Tuple[float, float]) -> bool:
        return ball_pos[0] >= DANGER_X if self.side == "right" else ball_pos[0] <= -DANGER_X

    def _mimic_threat_lateral_velocity(
        self,
        target: Tuple[float, float],
        threat: ThreatInfo,
        ball_vel: Tuple[float, float],
    ) -> Tuple[float, float]:
        ux, uy = _unit_from_to(threat.source, threat.target)
        nx, ny = -uy, ux
        lateral_vel = ball_vel[0] * nx + ball_vel[1] * ny
        adjusted = (target[0] + nx * lateral_vel * 1.5, target[1] + ny * lateral_vel * 1.5)
        return clamp_legal_defender_target(adjusted, self.side)

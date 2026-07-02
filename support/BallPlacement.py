import math
from collections import deque

from ai_interface.constants.field_constants import FIELD_X, FIELD_Y
from ai_interface.constants.player_constants import BALL_SIZE, KICKABLE_MARGIN, PLAYER_SIZE
from ai_interface.utils.algo_utils import (
    distance,
    field_units_per_meter,
    normalize_angle,
    robot_id_and_pose,
)
from ai_interface.utils.basic_commands import goto
from networking.data_utils import GameState, TeamInfo


FIELD_LENGTH_M = 9.0
GOALIE_ID = 1
MOVE_SPEED = 80.0
PLACEMENT_TIMEOUT_S = 30.0

# SSL rule distances (meters)
PLACEMENT_SUCCESS_RADIUS_M = 0.15
FREE_KICK_CLEARANCE_M = 0.05
FORCE_START_CLEARANCE_M = 0.5
NON_PLACING_CLEARANCE_M = 0.5
BALL_STATIONARY_SPEED_M_S = 0.05

# "No placement needed" thresholds (meters)
NO_PLACEMENT_RADIUS_M = 1.0
DEFENSE_BALL_CLEARANCE_M = 0.7
DEFENSE_AREA_DEPTH_M = 1.0
DEFENSE_AREA_HALF_WIDTH_M = 1.0

# Kickable reach in field units (constants are already in field units)
KICKABLE_REACH = PLAYER_SIZE + BALL_SIZE + KICKABLE_MARGIN


class AccessoryAlgo:
    """
    Ball placement game state.

    Placing team robots carry the ball to the designated position; the
    non-placing team clears the area to avoid interference.

    External callers must configure before each placement command:
        self.is_placing_team  bool
        self.next_command     'free_kick' | 'force_start'
    and call update_designated_ball_pos(x, y).
    """

    def __init__(self, team_infos: list[TeamInfo]):
        self.team_infos = team_infos
        self.designated_ball_pos: tuple[float | None, float | None] = (None, None)
        self.is_placing_team: bool = True
        self.next_command: str = 'free_kick'

        self._upm: float = field_units_per_meter(FIELD_LENGTH_M)
        self._placement_start_time: float | None = None
        self._placer_id: int | None = None
        self._all_dropped: bool = False
        self._ball_history: deque[tuple[float, tuple[float, float]]] = deque(maxlen=8)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update_designated_ball_pos(self, x: float, y: float) -> None:
        """Update the designated ball position and reset placement state."""
        self.designated_ball_pos = (x, y)
        self._placement_start_time = None
        self._placer_id = None
        self._all_dropped = False

    def decide_action(self, game_state: GameState, teamname: str) -> list[str]:
        """Return one action string per robot on teamname."""
        self._update_ball_history(game_state)

        if self._placement_start_time is None:
            self._placement_start_time = self._get_time(game_state)

        if not self._all_dropped:
            self._all_dropped = True
            return ['drop'] * len(game_state.robot_poses[teamname])

        if self.is_placing_team:
            return self._placing_team_actions(game_state, teamname)
        return self._non_placing_team_actions(game_state, teamname)

    # ------------------------------------------------------------------ #
    #  Placement status checks (callable externally by game controller)   #
    # ------------------------------------------------------------------ #

    def placement_needed(self, game_state: GameState) -> bool:
        """
        Return False when automatic placement is unnecessary:
        ball is already near the target, inside the field, away from defense
        areas, and stationary — so the game can continue without placement.
        """
        ball = game_state.ball_pos
        target = self.designated_ball_pos
        if ball is None or target[0] is None:
            return True
        if distance(ball[:2], target) >= self._u(NO_PLACEMENT_RADIUS_M):
            return True
        if not self._in_field(ball[:2]):
            return True
        if self._dist_to_nearest_defense_area(ball[:2]) < self._u(DEFENSE_BALL_CLEARANCE_M):
            return True
        if not self._is_ball_stationary():
            return True
        return False

    def placement_succeeded(self, game_state: GameState) -> bool:
        """
        Return True when all SSL placement success conditions are met.
        The game controller waits at least 2 s after the command before
        continuing, enforced externally.
        """
        ball = game_state.ball_pos
        target = self.designated_ball_pos
        if ball is None or target[0] is None or self._placement_start_time is None:
            return False

        now = self._get_time(game_state)
        if now - self._placement_start_time > PLACEMENT_TIMEOUT_S:
            return False
        if not self._is_ball_stationary():
            return False
        if distance(ball[:2], target) > self._u(PLACEMENT_SUCCESS_RADIUS_M):
            return False

        # All robots (both teams) must clear the required radius
        clearance = self._robot_clearance_radius()
        for robots in game_state.robot_poses.values():
            for robot in robots:
                _, pose = robot_id_and_pose(robot)
                if distance(pose[:2], ball[:2]) < clearance:
                    return False
        return True

    # ------------------------------------------------------------------ #
    #  Placing team                                                        #
    # ------------------------------------------------------------------ #

    def _placing_team_actions(self, game_state: GameState, teamname: str) -> list[str]:
        ball = game_state.ball_pos
        target = self.designated_ball_pos
        now = self._get_time(game_state)

        if self._placer_id is None:
            self._placer_id = self._select_placer(game_state, teamname)

        timed_out = (now - self._placement_start_time) > PLACEMENT_TIMEOUT_S

        actions = []
        for robot in game_state.robot_poses[teamname]:
            unum, pose = robot_id_and_pose(robot)
            if timed_out or unum != self._placer_id:
                actions.append('dash 0 0')
            else:
                actions.append(self._placer_action(pose, ball, target, game_state))
        return actions

    def _placer_action(
        self,
        pose: tuple,
        ball: tuple | None,
        target: tuple,
        game_state: GameState,
    ) -> str:
        if ball is None or target[0] is None:
            return 'dash 0 0'

        ball_xy = ball[:2]
        to_target = (target[0] - ball_xy[0], target[1] - ball_xy[1])
        dist_ball_to_target = math.hypot(*to_target)

        if dist_ball_to_target < 1e-6:
            return 'dash 0 0'

        unit = (to_target[0] / dist_ball_to_target, to_target[1] / dist_ball_to_target)
        clearance = self._robot_clearance_radius()

        # Ball is at target: back away so success conditions can be verified
        if dist_ball_to_target <= self._u(PLACEMENT_SUCCESS_RADIUS_M):
            d_from_ball = distance(pose[:2], ball_xy)
            if d_from_ball < clearance:
                away = (pose[0] - ball_xy[0], pose[1] - ball_xy[1])
                away_norm = math.hypot(*away)
                if away_norm < 1e-6:
                    away = (-unit[0], -unit[1])
                else:
                    away = (away[0] / away_norm, away[1] / away_norm)
                away_angle = math.atan2(away[1], away[0])
                return f'dash {MOVE_SPEED} {normalize_angle(away_angle - pose[2])}'
            return 'dash 0 0'

        d_from_ball = distance(pose[:2], ball_xy)

        # Within kickable reach: align then kick toward target
        if d_from_ball <= KICKABLE_REACH:
            kick_angle = math.atan2(to_target[1], to_target[0])
            angle_diff = normalize_angle(kick_angle - pose[2])
            if abs(angle_diff) > math.radians(10.0):
                return f'turn {angle_diff}'
            # Proportional power, capped so ball doesn't overshoot
            kick_power = min(100.0, max(20.0, dist_ball_to_target * 3.0))
            return f'kick {kick_power:.1f} 0'

        # Move to the approach point: directly behind the ball relative to target
        approach = (
            ball_xy[0] - unit[0] * KICKABLE_REACH,
            ball_xy[1] - unit[1] * KICKABLE_REACH,
        )
        return goto(
            pose, approach[0], approach[1], game_state,
            margin=self._u(0.05), speed=MOVE_SPEED,
        )

    def _select_placer(self, game_state: GameState, teamname: str) -> int:
        """Pick the non-goalie robot closest to the ball."""
        ball = game_state.ball_pos
        best_id, best_dist = GOALIE_ID + 1, float('inf')
        for robot in game_state.robot_poses.get(teamname, []):
            unum, pose = robot_id_and_pose(robot)
            if unum == GOALIE_ID:
                continue
            d = distance(pose[:2], ball[:2]) if ball is not None else float('inf')
            if d < best_dist:
                best_dist, best_id = d, unum
        return best_id

    # ------------------------------------------------------------------ #
    #  Non-placing team                                                    #
    # ------------------------------------------------------------------ #

    def _non_placing_team_actions(self, game_state: GameState, teamname: str) -> list[str]:
        """Keep all robots at least NON_PLACING_CLEARANCE_M from the ball."""
        ball = game_state.ball_pos
        clearance = self._u(NON_PLACING_CLEARANCE_M)

        actions = []
        for robot in game_state.robot_poses[teamname]:
            unum, pose = robot_id_and_pose(robot)
            if ball is None or distance(pose[:2], ball[:2]) >= clearance:
                actions.append('dash 0 0')
                continue
            away_angle = math.atan2(pose[1] - ball[1], pose[0] - ball[0])
            actions.append(f'dash {MOVE_SPEED} {normalize_angle(away_angle - pose[2])}')
        return actions

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _u(self, meters: float) -> float:
        """Convert meters to field units."""
        return meters * self._upm

    def _robot_clearance_radius(self) -> float:
        """Required robot-to-ball clearance for placement success."""
        if self.next_command == 'force_start':
            return self._u(FORCE_START_CLEARANCE_M)
        return self._u(FREE_KICK_CLEARANCE_M)

    def _get_time(self, game_state: GameState) -> float:
        ts = getattr(game_state, 'timestamp', None)
        if ts is not None:
            return float(ts)
        return float(getattr(game_state, 'count', 0)) * 0.1

    def _update_ball_history(self, game_state: GameState) -> None:
        if game_state.ball_pos is not None:
            self._ball_history.append(
                (self._get_time(game_state), tuple(game_state.ball_pos[:2]))
            )

    def _is_ball_stationary(self) -> bool:
        if len(self._ball_history) < 2:
            return False
        t0, p0 = self._ball_history[-2]
        t1, p1 = self._ball_history[-1]
        dt = max(t1 - t0, 1e-6)
        speed = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / dt
        return speed < self._u(BALL_STATIONARY_SPEED_M_S)

    def _in_field(self, pos: tuple[float, float]) -> bool:
        return FIELD_X[0] <= pos[0] <= FIELD_X[1] and FIELD_Y[0] <= pos[1] <= FIELD_Y[1]

    def _dist_to_nearest_defense_area(self, pos: tuple[float, float]) -> float:
        """Distance from pos to the nearest defense area boundary (0 if inside)."""
        depth = self._u(DEFENSE_AREA_DEPTH_M)
        half_w = self._u(DEFENSE_AREA_HALF_WIDTH_M)
        px, py = pos

        def dist_to_rect(x0: float, x1: float, y0: float, y1: float) -> float:
            dx = max(x0 - px, 0.0, px - x1)
            dy = max(y0 - py, 0.0, py - y1)
            return math.hypot(dx, dy)

        d_right = dist_to_rect(FIELD_X[1] - depth, FIELD_X[1], -half_w, half_w)
        d_left = dist_to_rect(FIELD_X[0], FIELD_X[0] + depth, -half_w, half_w)
        return min(d_right, d_left)

# Assignee: Nikitha Maderamitla
import math

from ai_interface.constants.field_constants import FIELD_X, FIELD_Y
from ai_interface.constants.player_constants import BALL_SIZE, KICKABLE_MARGIN, PLAYER_SIZE
from ai_interface.utils.algo_utils import (
    clamp,
    distance,
    face_ball_angle,
    field_units_per_meter,
    get_goal_params,
    robot_id_and_pose,
)
from ai_interface.utils.basic_commands import goto
from networking.data_utils import GameState, TeamInfo


class AccessoryAlgo:
    """
    Prepare kickoff positioning.

    The attacking team stages one kicker behind the ball. Every other robot
    remains in its own half, and the defending team also stays outside the
    center circle.
    """

    GOALIE_ID = 6
    KICKER_ID = 1

    FIELD_LENGTH_M = 9.0
    CENTER_CIRCLE_RADIUS_M = 0.5
    CENTER_CIRCLE_CLEARANCE_M = 0.15
    KICKER_STANDOFF_M = 0.25
    FORMATION_DEPTH_M = 0.85
    FORMATION_ROW_SPACING_M = 0.9
    FORMATION_LATERAL_SPACING_M = 1.1
    GOALIE_OUT_M = 0.35
    HALF_CLEARANCE_M = 0.15
    FIELD_MARGIN_M = 0.15
    MOVE_MARGIN_M = 0.06
    MOVE_SPEED = 80.0

    def __init__(self, team_infos: list[TeamInfo]):
        self.team_infos = team_infos

        # These are expected to be set by the referee/playmode layer.
        # kickoff_against=True means the other team is taking the kickoff.
        self.kickoff_against = False
        self.attack_direction = 1

        self.units_per_meter = field_units_per_meter(self.FIELD_LENGTH_M)
        self.center_circle_clearance = (
            self.CENTER_CIRCLE_RADIUS_M + self.CENTER_CIRCLE_CLEARANCE_M
        ) * self.units_per_meter
        self.kicker_standoff = max(
            self.KICKER_STANDOFF_M * self.units_per_meter,
            PLAYER_SIZE + BALL_SIZE + KICKABLE_MARGIN + 0.1,
        )
        self.formation_depth = self.FORMATION_DEPTH_M * self.units_per_meter
        self.formation_row_spacing = self.FORMATION_ROW_SPACING_M * self.units_per_meter
        self.formation_lateral_spacing = (
            self.FORMATION_LATERAL_SPACING_M * self.units_per_meter
        )
        self.goalie_out = self.GOALIE_OUT_M * self.units_per_meter
        self.half_clearance = self.HALF_CLEARANCE_M * self.units_per_meter
        self.field_margin = self.FIELD_MARGIN_M * self.units_per_meter
        self.move_margin = self.MOVE_MARGIN_M * self.units_per_meter

    def decide_action(self, game_state: GameState, teamname: str):
        """
        Decide actions for all robots on the team based on game state.

        Args:
            game_state: Current game state with ball and robot positions
            teamname: Name of the team to generate actions for

        Returns:
            Decided actions for all the robots
        """
        team_robots = game_state.robot_poses.get(teamname, [])
        ball = game_state.ball_pos
        if ball is None:
            return ["dash 0 0" for _ in team_robots]

        robot_poses = [robot_id_and_pose(robot) for robot in team_robots]
        goalie_id = self._goalie_id(robot_poses)
        kicker_id = self._kicker_id(robot_poses, goalie_id, ball)
        formation_targets = self._formation_targets(robot_poses, goalie_id, kicker_id)

        actions = []
        for unum, pose in robot_poses:
            if unum == goalie_id:
                target = self._goalie_target()
                speed = 70.0
            elif unum == kicker_id:
                target = self._kicker_target(ball)
                speed = self.MOVE_SPEED
            else:
                target = formation_targets[unum]
                speed = 70.0

            target_theta = (
                self._goal_angle()
                if unum == kicker_id
                else face_ball_angle(pose, ball)
            )
            actions.append(self._move_to_pose(pose, target, target_theta, game_state, speed))
        return actions

    def _goalie_id(
        self,
        robot_poses: list[tuple[int, tuple[float, float, float]]],
    ) -> int | None:
        if any(unum == self.GOALIE_ID for unum, _ in robot_poses):
            return self.GOALIE_ID
        if not robot_poses:
            return None

        goal = self._goalie_target()
        return min(robot_poses, key=lambda item: distance(item[1][:2], goal))[0]

    def _kicker_id(
        self,
        robot_poses: list[tuple[int, tuple[float, float, float]]],
        goalie_id: int | None,
        ball,
    ) -> int | None:
        if self.kickoff_against:
            return None
        if any(unum == self.KICKER_ID and unum != goalie_id for unum, _ in robot_poses):
            return self.KICKER_ID

        candidates = [item for item in robot_poses if item[0] != goalie_id]
        if not candidates:
            return goalie_id
        return min(candidates, key=lambda item: distance(item[1][:2], ball[:2]))[0]

    def _goalie_target(self) -> tuple[float, float]:
        _, _, goal_center = get_goal_params(self._defended_goal_side())
        target = (
            goal_center[0] + self.attack_direction * self.goalie_out,
            goal_center[1],
        )
        return self._clamp_to_own_half(target)

    def _kicker_target(self, ball) -> tuple[float, float]:
        target = (
            ball[0] - self.attack_direction * self.kicker_standoff,
            ball[1],
        )
        return self._clamp_to_own_half(target)

    def _formation_targets(
        self,
        robot_poses: list[tuple[int, tuple[float, float, float]]],
        goalie_id: int | None,
        kicker_id: int | None,
    ) -> dict[int, tuple[float, float]]:
        field_ids = sorted(
            unum for unum, _ in robot_poses if unum not in (goalie_id, kicker_id)
        )
        targets = {}
        for index, unum in enumerate(field_ids):
            row = index // 3
            lateral_index = index % 3
            y_offset = (0.0, self.formation_lateral_spacing, -self.formation_lateral_spacing)[
                lateral_index
            ]
            depth = self.formation_depth + row * self.formation_row_spacing
            depth = max(depth, self.center_circle_clearance)
            target = (-self.attack_direction * depth, y_offset)
            targets[unum] = self._clamp_to_own_half(target)
        return targets

    def _move_to_pose(
        self,
        pose,
        target: tuple[float, float],
        target_theta: float,
        game_state: GameState,
        speed: float,
    ) -> str:
        cmd = goto(
            pose,
            target[0],
            target[1],
            game_state,
            margin=self.move_margin,
            theta=target_theta,
            speed=speed,
        )
        return "dash 0 0" if cmd == "done" else cmd

    def _clamp_to_own_half(self, target: tuple[float, float]) -> tuple[float, float]:
        x = clamp(target[0], FIELD_X[0] + self.field_margin, FIELD_X[1] - self.field_margin)
        y = clamp(target[1], FIELD_Y[0] + self.field_margin, FIELD_Y[1] - self.field_margin)
        if self.attack_direction > 0:
            x = min(x, -self.half_clearance)
        else:
            x = max(x, self.half_clearance)
        return (x, y)

    def _defended_goal_side(self) -> str:
        return "left" if self.attack_direction > 0 else "right"

    def _goal_angle(self) -> float:
        return 0.0 if self.attack_direction > 0 else math.pi

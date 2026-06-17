# Assignee: Alex Meng
import math

from ai_interface.constants.field_constants import FIELD_X, FIELD_Y
from ai_interface.constants.player_constants import BALL_SIZE, KICKABLE_MARGIN, PLAYER_SIZE
from ai_interface.utils.algo_utils import (
    clamp,
    distance,
    face_ball_angle,
    field_units_per_meter,
    get_goal_params,
    normalize_angle,
    robot_id_and_pose,
)
from ai_interface.utils.basic_commands import goto
from networking.data_utils import GameState, TeamInfo


class AccessoryAlgo:
    """
    Kickoff behavior for NORMAL_START after our prepare kickoff.

    This file intentionally assumes that if Kickoff is active, we are the
    attacking team. The dispatcher does not currently pass kickoff ownership
    into this module, so ownership is not inferred from referee state here.
    """

    GOALIE_ID = 1
    KICKER_ID = 5

    FIELD_LENGTH_M = 9.0
    GOALIE_OUT_M = 0.35
    FIELD_MARGIN_M = 0.15
    MOVE_MARGIN_M = 0.06
    KICKER_STANDOFF_M = 0.25
    BALL_MOVED_M = 0.18
    CHASE_RELEASE_M = 0.5

    MOVE_SPEED = 85.0
    SUPPORT_SPEED = 75.0
    KICK_POWER = 100.0
    CLEAR_POWER = 80.0
    ALIGN_TOLERANCE_RAD = math.radians(8.0)

    def __init__(self, team_infos: list[TeamInfo]):
        self.team_infos = team_infos
        self.units_per_meter = field_units_per_meter(self.FIELD_LENGTH_M)

        self.goalie_out = self.GOALIE_OUT_M * self.units_per_meter
        self.field_margin = self.FIELD_MARGIN_M * self.units_per_meter
        self.move_margin = self.MOVE_MARGIN_M * self.units_per_meter
        self.kicker_standoff = max(
            self.KICKER_STANDOFF_M * self.units_per_meter,
            PLAYER_SIZE + BALL_SIZE + KICKABLE_MARGIN + 0.1,
        )
        self.ball_moved_distance = self.BALL_MOVED_M * self.units_per_meter
        self.chase_release_distance = self.CHASE_RELEASE_M * self.units_per_meter
        self.kickable_distance = PLAYER_SIZE + BALL_SIZE + KICKABLE_MARGIN

        self.attack_direction: int | None = None
        self.kicker_id: int | None = None
        self.start_ball_pos: tuple[float, float] | None = None
        self.ball_kicked = False

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
        if not robot_poses:
            return []

        if self.attack_direction is None:
            self.attack_direction = self._infer_attack_direction(robot_poses)
        if self.start_ball_pos is None:
            self.start_ball_pos = ball[:2]

        goalie_id = self._goalie_id(robot_poses)
        if self.kicker_id is None:
            self.kicker_id = self._kicker_id(robot_poses, goalie_id, ball)

        if self._ball_has_moved(ball):
            self.ball_kicked = True

        chaser_id = self._chaser_id(robot_poses, goalie_id, ball)
        support_targets = self._support_targets(robot_poses, goalie_id, chaser_id)

        actions = []
        for unum, pose in robot_poses:
            if unum == goalie_id:
                action = self._goalie_action(pose, ball, game_state)
            elif unum == self.kicker_id and not self.ball_kicked:
                action = self._kicker_action(pose, ball, game_state)
            elif unum == chaser_id and self.ball_kicked:
                action = self._chaser_action(pose, ball, game_state)
            else:
                target = support_targets.get(unum, self._fallback_support_target(unum))
                action = self._move_to_pose(
                    pose,
                    target,
                    face_ball_angle(pose, ball),
                    game_state,
                    self.SUPPORT_SPEED,
                )
            actions.append(action)

        return actions

    def _infer_attack_direction(
        self,
        robot_poses: list[tuple[int, tuple[float, float, float]]],
    ) -> int:
        avg_x = sum(pose[0] for _, pose in robot_poses) / len(robot_poses)
        return 1 if avg_x < 0 else -1

    def _goalie_id(
        self,
        robot_poses: list[tuple[int, tuple[float, float, float]]],
    ) -> int | None:
        if any(unum == self.GOALIE_ID for unum, _ in robot_poses):
            return self.GOALIE_ID
        goal = self._goalie_target()
        return min(robot_poses, key=lambda item: distance(item[1][:2], goal))[0]

    def _kicker_id(
        self,
        robot_poses: list[tuple[int, tuple[float, float, float]]],
        goalie_id: int | None,
        ball,
    ) -> int | None:
        if any(unum == self.KICKER_ID and unum != goalie_id for unum, _ in robot_poses):
            return self.KICKER_ID

        candidates = [item for item in robot_poses if item[0] != goalie_id]
        if not candidates:
            return goalie_id
        return min(candidates, key=lambda item: distance(item[1][:2], ball[:2]))[0]

    def _chaser_id(
        self,
        robot_poses: list[tuple[int, tuple[float, float, float]]],
        goalie_id: int | None,
        ball,
    ) -> int | None:
        if not self.ball_kicked:
            return None

        candidates = [
            item
            for item in robot_poses
            if item[0] not in (goalie_id, self.kicker_id)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: distance(item[1][:2], ball[:2]))[0]

    def _ball_has_moved(self, ball) -> bool:
        if self.start_ball_pos is None:
            return False
        return distance(ball[:2], self.start_ball_pos) >= self.ball_moved_distance

    def _kicker_action(self, pose, ball, game_state: GameState) -> str:
        target = self._shot_target()
        cmd = self._kick_toward_target(pose, ball, target, self.KICK_POWER, game_state)
        if cmd.startswith("kick "):
            self.ball_kicked = True
        return cmd

    def _chaser_action(self, pose, ball, game_state: GameState) -> str:
        if distance(pose[:2], ball[:2]) <= self.kickable_distance:
            return self._kick_toward_target(
                pose,
                ball,
                self._shot_target(),
                self.CLEAR_POWER,
                game_state,
            )

        target = (
            ball[0] - self.attack_direction * self.kicker_standoff,
            ball[1],
        )
        return self._move_to_pose(
            pose,
            self._clamp_to_field(target),
            self._goal_angle(),
            game_state,
            self.MOVE_SPEED,
        )

    def _goalie_action(self, pose, ball, game_state: GameState) -> str:
        return self._move_to_pose(
            pose,
            self._goalie_target(),
            face_ball_angle(pose, ball),
            game_state,
            self.SUPPORT_SPEED,
        )

    def _kick_toward_target(
        self,
        pose,
        ball,
        target: tuple[float, float],
        power: float,
        game_state: GameState,
    ) -> str:
        target_angle = math.atan2(target[1] - ball[1], target[0] - ball[0])
        angle_diff = normalize_angle(target_angle - pose[2])
        if abs(angle_diff) > self.ALIGN_TOLERANCE_RAD:
            return f"turn {angle_diff}"

        kick_pos = (
            ball[0] - math.cos(target_angle) * self.kicker_standoff,
            ball[1] - math.sin(target_angle) * self.kicker_standoff,
        )
        if distance(pose[:2], kick_pos) > self.move_margin:
            return self._move_to_pose(
                pose,
                self._clamp_to_field(kick_pos),
                target_angle,
                game_state,
                self.MOVE_SPEED,
            )

        return f"kick {power:.1f} 0"

    def _support_targets(
        self,
        robot_poses: list[tuple[int, tuple[float, float, float]]],
        goalie_id: int | None,
        chaser_id: int | None,
    ) -> dict[int, tuple[float, float]]:
        support_ids = sorted(
            unum
            for unum, _ in robot_poses
            if unum not in (goalie_id, chaser_id)
        )
        targets = {}
        for index, unum in enumerate(support_ids):
            if self.ball_kicked and unum == self.kicker_id:
                targets[unum] = self._kicker_release_target()
            else:
                targets[unum] = self._fallback_support_target(index)
        return targets

    def _fallback_support_target(self, index_or_unum: int) -> tuple[float, float]:
        lanes = (
            (1.15, 0.75),
            (1.15, -0.75),
            (0.25, 1.35),
            (0.25, -1.35),
            (-0.95, 0.0),
            (-1.55, 1.0),
            (-1.55, -1.0),
        )
        x_m, y_m = lanes[index_or_unum % len(lanes)]
        target = (
            self.attack_direction * x_m * self.units_per_meter,
            y_m * self.units_per_meter,
        )

        return self._clamp_to_field(target)

    def _kicker_release_target(self) -> tuple[float, float]:
        return self._clamp_to_field(
            (
                -self.attack_direction * self.chase_release_distance,
                1.4 * self.units_per_meter,
            )
        )

    def _goalie_target(self) -> tuple[float, float]:
        _, _, goal_center = get_goal_params(self._defended_goal_side())
        target = (
            goal_center[0] + self.attack_direction * self.goalie_out,
            goal_center[1],
        )
        return self._clamp_to_field(target)

    def _shot_target(self) -> tuple[float, float]:
        return (
            FIELD_X[1] - self.field_margin
            if self.attack_direction > 0
            else FIELD_X[0] + self.field_margin,
            0.0,
        )

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

    def _clamp_to_field(self, target: tuple[float, float]) -> tuple[float, float]:
        return (
            clamp(target[0], FIELD_X[0] + self.field_margin, FIELD_X[1] - self.field_margin),
            clamp(target[1], FIELD_Y[0] + self.field_margin, FIELD_Y[1] - self.field_margin),
        )

    def _defended_goal_side(self) -> str:
        return "left" if self.attack_direction > 0 else "right"

    def _goal_angle(self) -> float:
        return 0.0 if self.attack_direction > 0 else math.pi

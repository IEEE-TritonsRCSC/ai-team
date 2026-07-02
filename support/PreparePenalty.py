# Assignee: Wing Huang
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
    Prepare penalty positioning.

    Rulebook constraints:
    - defending keeper on the goal line between the posts
    - one attacking robot may approach the ball but must not touch it
    - all other robots stay at least 1 m behind the ball
    """

    GOALIE_ID = 1
    PENALTY_TAKER_ID = 5

    FIELD_LENGTH_M = 9.0
    ATTACKER_STANDOFF_M = 0.25
    OTHER_ROBOT_BEHIND_M = 1.15
    OTHER_ROBOT_SPACING_M = 0.8
    FIELD_MARGIN_M = 0.15
    MOVE_MARGIN_M = 0.06
    MOVE_SPEED = 80.0

    def __init__(self, team_infos: list[TeamInfo]):
        self.team_infos = team_infos

        # These are expected to be set by the referee/playmode layer.
        # penalty_against=True means this team is defending the penalty.
        self.penalty_against = False
        self.attack_direction = 1

        self.units_per_meter = field_units_per_meter(self.FIELD_LENGTH_M)
        self.attacker_standoff = max(
            self.ATTACKER_STANDOFF_M * self.units_per_meter,
            PLAYER_SIZE + BALL_SIZE + KICKABLE_MARGIN + 0.1,
        )
        self.other_robot_behind = self.OTHER_ROBOT_BEHIND_M * self.units_per_meter
        self.other_robot_spacing = self.OTHER_ROBOT_SPACING_M * self.units_per_meter
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
        ball = game_state.ball_pos
        team_robots = game_state.robot_poses.get(teamname, [])
        if ball is None:
            return ["dash 0 0" for _ in team_robots]

        robot_poses = [robot_id_and_pose(robot) for robot in team_robots]
        special_id = self._special_robot_id(robot_poses, ball)
        other_targets = self._other_robot_targets(robot_poses, special_id, ball)

        actions = []
        for unum, pose in robot_poses:
            if self.penalty_against and unum == special_id:
                actions.append(self._goalie_action(pose, ball, game_state))
            elif not self.penalty_against and unum == special_id:
                actions.append(self._attacker_action(pose, ball, game_state))
            else:
                actions.append(self._other_robot_action(pose, other_targets[unum], ball, game_state))
        return actions

    def _special_robot_id(
        self,
        robot_poses: list[tuple[int, tuple[float, float, float]]],
        ball,
    ) -> int | None:
        preferred_id = self.GOALIE_ID if self.penalty_against else self.PENALTY_TAKER_ID
        if any(unum == preferred_id for unum, _ in robot_poses):
            return preferred_id
        if not robot_poses:
            return None
        return min(robot_poses, key=lambda item: distance(item[1][:2], ball[:2]))[0]

    def _attacker_action(self, pose, ball, game_state: GameState) -> str:
        attack_direction = self._active_penalty_attack_direction()
        target = (
            ball[0] - attack_direction * self.attacker_standoff,
            ball[1],
        )
        return self._move_to_pose(pose, self._clamp_to_field(target), self._goal_angle(), game_state)

    def _goalie_action(self, pose, ball, game_state: GameState) -> str:
        target = self._goal_line_target(ball)
        return self._move_to_pose(pose, target, face_ball_angle(pose, ball), game_state, speed=100.0)

    def _other_robot_action(self, pose, target, ball, game_state: GameState) -> str:
        return self._move_to_pose(pose, target, face_ball_angle(pose, ball), game_state, speed=70.0)

    def _other_robot_targets(
        self,
        robot_poses: list[tuple[int, tuple[float, float, float]]],
        special_id: int | None,
        ball,
    ) -> dict[int, tuple[float, float]]:
        other_ids = sorted(unum for unum, _ in robot_poses if unum != special_id)
        offsets = self._lateral_offsets(len(other_ids))
        attack_direction = self._active_penalty_attack_direction()
        target_x = ball[0] - attack_direction * self.other_robot_behind

        targets = {}
        for unum, y_offset in zip(other_ids, offsets):
            target = (target_x, ball[1] + y_offset)
            targets[unum] = self._clamp_to_field(target)
        return targets

    def _lateral_offsets(self, count: int) -> list[float]:
        offsets = []
        for index in range(count):
            magnitude = ((index // 2) + 1) * self.other_robot_spacing
            sign = 1.0 if index % 2 == 0 else -1.0
            offsets.append(sign * magnitude)
        return offsets

    def _goal_line_target(self, ball) -> tuple[float, float]:
        post_top, post_bottom, goal_center = get_goal_params(self._defended_goal_side())
        min_y = min(post_top[1], post_bottom[1])
        max_y = max(post_top[1], post_bottom[1])
        return (goal_center[0], clamp(ball[1], min_y, max_y))

    def _move_to_pose(
        self,
        pose,
        target: tuple[float, float],
        target_theta: float | None,
        game_state: GameState,
        speed: float | None = None,
    ) -> str:
        cmd = goto(
            pose,
            target[0],
            target[1],
            game_state,
            margin=self.move_margin,
            theta=target_theta,
            speed=self.MOVE_SPEED if speed is None else speed,
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

    def _active_penalty_attack_direction(self) -> int:
        if self.penalty_against:
            return -self.attack_direction
        return self.attack_direction

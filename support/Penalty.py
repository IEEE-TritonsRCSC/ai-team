# Assignee: Yash Tandon
import math

from networking.data_utils import GameState, TeamInfo

from ai_interface.constants.field_constants import (
    FIELD_X,
    GOAL_L,
    GOAL_R,
)
from ai_interface.constants.player_constants import BALL_SIZE, KICKABLE_MARGIN, PLAYER_SIZE
from ai_interface.utils.algo_utils import (
    distance,
    normalize_angle,
)
from ai_interface.goalie import Goalie
from ai_interface.utils.basic_commands import goto


class AccessoryAlgo:

    GOALIE_ID = 1
    PENALTY_TAKER_ID = 5

    FIELD_LENGTH_M = 9.0
    FINAL_SHOT_DISTANCE_M = 2.0
    ADVANCE_TOUCH_M = 0.8
    RELEASE_GAP_M = 0.22
    APPROACH_STANDOFF_M = 0.15
    PENALTY_TIMEOUT_S = 10.0

    KICKABLE_DISTANCE_M = 0.18
    ALIGN_TOLERANCE_RAD = math.radians(7.0)
    MOVE_MARGIN_M = 0.08

    TAP_POWER = 18.0
    SHOT_POWER = 100.0
    MOVE_SPEED = 80.0

    def __init__(self, team_infos: list[TeamInfo]):
        self.team_infos = team_infos

        # Provided by the referee/playmode layer: True means we are defending.
        self.penalty_against = False
        self.attack_direction = 1

        self.phase = "WAIT_FOR_START"
        self.phase_started_at = None
        self.penalty_started_at = None
        self.touch_start_ball_x = None
        self.last_ball_x = None
        self._goalie: Goalie | None = None

        self.units_per_meter = self._units_per_meter()
        self.final_shot_distance = self.FINAL_SHOT_DISTANCE_M * self.units_per_meter
        self.advance_touch = self.ADVANCE_TOUCH_M * self.units_per_meter
        self.release_gap = self.RELEASE_GAP_M * self.units_per_meter
        self.approach_standoff = self.APPROACH_STANDOFF_M * self.units_per_meter
        self.kickable_distance = max(
            self.KICKABLE_DISTANCE_M * self.units_per_meter,
            PLAYER_SIZE + BALL_SIZE + KICKABLE_MARGIN,
        )
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
        if self.penalty_against:
            return self._defend_penalty(game_state, teamname)
        return self._attack_penalty(game_state, teamname)

    def _attack_penalty(self, game_state: GameState, teamname: str) -> list[str]:
        ball = game_state.ball_pos
        now = self._time(game_state)
        if self.penalty_started_at is None:
            self.penalty_started_at = now
            self.phase_started_at = now
            self.touch_start_ball_x = ball[0]
            self.last_ball_x = ball[0]

        actions = []
        for robot in game_state.robot_poses[teamname]:
            unum, pose = self._robot_id_and_pose(robot)
            if unum == self.PENALTY_TAKER_ID:
                actions.append(self._penalty_taker_action(pose, ball, game_state, teamname, now))
            else:
                actions.append("dash 0 0")
        return actions

    def _defend_penalty(self, game_state: GameState, teamname: str) -> list[str]:
        ball = game_state.ball_pos
        actions = []
        for robot in game_state.robot_poses[teamname]:
            unum, pose = self._robot_id_and_pose(robot)
            if unum == self.GOALIE_ID:
                actions.append(self._penalty_goalie_action(pose, ball, game_state, teamname))
            else:
                actions.append("dash 0 0")
        return actions

    def _penalty_taker_action(self, pose, ball, game_state: GameState, teamname: str, now: float) -> str:
        if now - self.penalty_started_at >= self.PENALTY_TIMEOUT_S:
            self.phase = "DONE"

        if self.phase == "DONE":
            return "dash 0 0"

        if self._ball_regressed(ball):
            return "dash 0 0"

        if self._within_final_shot_distance(ball):
            self.phase = "SHOOT"

        if self.phase == "WAIT_FOR_START":
            # Penalty.py is invoked for the penalty action itself, so one aligned
            # cycle is enough before beginning controlled forward touches.
            cmd = self._move_behind_ball(pose, ball, game_state)
            if cmd == "done":
                self._set_phase("TAP_ADVANCE", now, ball)
                return "dash 0 0"
            return cmd

        if self.phase == "TAP_ADVANCE":
            if distance(pose[:2], ball[:2]) > self.kickable_distance:
                return self._move_behind_ball(pose, ball, game_state)

            if abs(ball[0] - self.touch_start_ball_x) >= self.advance_touch:
                self._set_phase("RELEASE", now, ball)
                return "dash 0 0"

            cmd = self._kick_straight_if_aligned(pose, ball, self.TAP_POWER, game_state)
            if cmd.startswith("kick "):
                self._set_phase("RELEASE", now, ball)
            return cmd

        if self.phase == "RELEASE":
            if self._has_visible_gap(pose, ball):
                self._set_phase("REACQUIRE", now, ball)
                return "dash 0 0"

            target = (
                ball[0] - self.attack_direction * (self.kickable_distance + self.release_gap),
                ball[1],
            )
            return self._move_to_pose(pose, target, self._goal_angle(ball), game_state, speed=35.0)

        if self.phase == "REACQUIRE":
            cmd = self._move_behind_ball(pose, ball, game_state)
            if cmd == "done":
                self._set_phase("TAP_ADVANCE", now, ball)
                return "dash 0 0"
            return cmd

        if self.phase == "SHOOT":
            target = self._shot_target(game_state, teamname)
            return self._kick_toward_target(pose, ball, target, self.SHOT_POWER, game_state)

        return "dash 0 0"

    def _move_behind_ball(self, pose, ball, game_state: GameState) -> str:
        offset = self._approach_offset()
        target = (
            ball[0] - self.attack_direction * offset,
            ball[1],
        )
        return self._move_to_pose(pose, target, self._goal_angle(ball), game_state)

    def _kick_straight_if_aligned(self, pose, ball, power: float, game_state: GameState) -> str:
        target_x = ball[0] + self.attack_direction * self.advance_touch
        target = (target_x, ball[1])
        return self._kick_toward_target(pose, ball, target, power, game_state)

    def _kick_toward_target(self, pose, ball, target, power: float, game_state: GameState) -> str:
        target_angle = math.atan2(target[1] - ball[1], target[0] - ball[0])
        angle_diff = normalize_angle(target_angle - pose[2])
        if abs(angle_diff) > self.ALIGN_TOLERANCE_RAD:
            return f"turn {angle_diff}"

        offset = self._approach_offset()
        kick_pos = (
            ball[0] - math.cos(target_angle) * offset,
            ball[1] - math.sin(target_angle) * offset,
        )
        if distance(pose[:2], kick_pos) > self.move_margin:
            return self._move_to_pose(pose, kick_pos, target_angle, game_state)
        return f"kick {power:.1f} 0"

    def _penalty_goalie_action(self, pose, ball, game_state: GameState, teamname: str) -> str:
        if self._goalie is None:
            self._goalie = Goalie(teamname=teamname, unum=self.GOALIE_ID)
        goalie_pose_deg = (pose[0], pose[1], math.degrees(pose[2]))
        return self._goalie.action(
            ball_pos=tuple(ball[:2]),
            goalie_pose=goalie_pose_deg,
            goalie_to_ball_dist=distance(pose[:2], ball[:2]),
            game_state=game_state,
        )

    def _move_to_pose(
        self,
        pose,
        target,
        target_theta: float | None,
        game_state: GameState,
        speed: float | None = None,
        margin: float | None = None,
    ) -> str:
        speed = self.MOVE_SPEED if speed is None else speed
        margin = self.move_margin if margin is None else margin
        return goto(
            pose,
            target[0],
            target[1],
            game_state,
            margin=margin,
            theta=target_theta,
            speed=speed,
        )

    def _shot_target(self, game_state: GameState, our_teamname: str | None = None) -> tuple[float, float]:
        goal = GOAL_R if self.attack_direction > 0 else GOAL_L
        goal_x = goal[0]
        keeper = self._opponent_keeper_pose(game_state, our_teamname)
        goal_half_width = self.units_per_meter * 0.5
        if keeper is None:
            target_y = goal_half_width
        elif keeper[1] >= 0:
            target_y = -goal_half_width
        else:
            target_y = goal_half_width
        return (goal_x, target_y)

    def _opponent_keeper_pose(self, game_state: GameState, our_teamname: str | None = None):
        for teamname, robots in game_state.robot_poses.items():
            if teamname == our_teamname:
                continue
            for robot in robots:
                unum, pose = self._robot_id_and_pose(robot)
                if unum == self.GOALIE_ID:
                    return pose
        return None

    def _approach_offset(self) -> float:
        return max(self.kickable_distance * 0.9, self.kickable_distance - self.approach_standoff)

    def _within_final_shot_distance(self, ball) -> bool:
        goal_x = GOAL_R[0] if self.attack_direction > 0 else GOAL_L[0]
        return abs(goal_x - ball[0]) <= self.final_shot_distance

    def _ball_regressed(self, ball) -> bool:
        if self.last_ball_x is None:
            self.last_ball_x = ball[0]
            return False
        regressed = self.attack_direction * (ball[0] - self.last_ball_x) < -0.05 * self.units_per_meter
        self.last_ball_x = max(self.last_ball_x, ball[0]) if self.attack_direction > 0 else min(self.last_ball_x, ball[0])
        return regressed

    def _has_visible_gap(self, pose, ball) -> bool:
        return distance(pose[:2], ball[:2]) >= self.kickable_distance + self.release_gap

    def _set_phase(self, phase: str, now: float, ball) -> None:
        self.phase = phase
        self.phase_started_at = now
        if phase == "TAP_ADVANCE":
            self.touch_start_ball_x = ball[0]

    def _goal_angle(self, ball) -> float:
        return 0.0 if self.attack_direction > 0 else math.pi

    def _time(self, game_state: GameState) -> float:
        timestamp = getattr(game_state, "timestamp", None)
        if timestamp is not None:
            return float(timestamp)
        return float(getattr(game_state, "count", 0)) * 0.1

    def _robot_id_and_pose(self, robot) -> tuple[int, tuple[float, float, float]]:
        unum = int(next(iter(robot.keys())))
        raw_pose = robot[unum]
        theta = raw_pose[2]
        if abs(theta) > 2 * math.pi:
            theta = math.radians(theta)
        return unum, (float(raw_pose[0]), float(raw_pose[1]), normalize_angle(float(theta)))

    def _units_per_meter(self) -> float:
        field_units = abs(FIELD_X[1] - FIELD_X[0])
        return field_units / self.FIELD_LENGTH_M

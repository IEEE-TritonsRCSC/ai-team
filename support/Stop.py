# Assignee: Travis Wu
import math

from ai_interface.constants.field_constants import *
from ai_interface.constants.player_constants import *
from networking.data_utils import GameState, TeamInfo


STOP_RADIUS = 0.5
MAX_SPEED = 1.5
SAFE_POSITION_SAMPLES = 24

class AccessoryAlgo:
    """
    Helpers for SSL stop-state behavior.
    """
    def __init__(self, team_infos: list[TeamInfo]):
        """
        Initialize stop-state behavior.
        """
        self.team_infos = team_infos
        self.max_dash = max(MAX_SPEED * (1 - PLAYER_DECAY) / DASH_POWER_RATE, 100)
        self.all_dropped = False

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize an angle to [-pi, pi].
        """
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _robots_within_stop_radius(
        self, game_state: GameState, teamname: str
    ) -> list[tuple[int, tuple[float, float, float]]]:
        """
        Return robots whose centers are within STOP_RADIUS of the ball.
        """
        ball_pos = game_state.ball_pos
        if ball_pos is None:
            return []

        violating_robots = []
        ball_x, ball_y = ball_pos[:2]
        for robot in game_state.robot_poses.get(teamname, []):
            unum = int(next(iter(robot.keys())))
            pose = robot[unum]
            distance = math.hypot(pose[0] - ball_x, pose[1] - ball_y)
            if distance < STOP_RADIUS:
                violating_robots.append((unum, pose))
        return violating_robots

    def _sample_safe_positions(self, ball_pos: tuple[float, float]) -> list[tuple[float, float]]:
        """
        Sample a dense ring of candidate safe positions around the ball.
        """
        ball_x, ball_y = ball_pos[:2]
        safe_positions = []
        for i in range(SAFE_POSITION_SAMPLES):
            angle = (2.0 * math.pi * i) / SAFE_POSITION_SAMPLES
            candidate = (
                ball_x + STOP_RADIUS * math.cos(angle),
                ball_y + STOP_RADIUS * math.sin(angle),
            )
            if FIELD_X[0] <= candidate[0] <= FIELD_X[1] and FIELD_Y[0] <= candidate[1] <= FIELD_Y[1]:
                safe_positions.append(candidate)
        return safe_positions

    def _assign_safe_positions(
        self,
        ball_pos: tuple[float, float],
        violating_robots: list[tuple[int, tuple[float, float, float]]],
        safe_positions: list[tuple[float, float]],
    ) -> dict[int, tuple[float, float]]:
        """
        Assign each violating robot the closest safe position that is still unclaimed.
        """
        ball_x, ball_y = ball_pos[:2]
        available_positions = list(safe_positions)
        assignments = {}

        for unum, pose in violating_robots:
            if not available_positions:
                break

            robot_from_ball_x = pose[0] - ball_x
            robot_from_ball_y = pose[1] - ball_y
            acute_positions = [
                pos
                for pos in available_positions
                if (
                    robot_from_ball_x * (pos[0] - ball_x)
                    + robot_from_ball_y * (pos[1] - ball_y)
                ) > 0.0
            ]
            if not acute_positions:
                continue

            closest_position = min(
                acute_positions,
                key=lambda pos: math.hypot(pose[0] - pos[0], pose[1] - pos[1]),
            )
            assignments[unum] = closest_position
            available_positions.remove(closest_position)

        return assignments

    def decide_action(self, game_state: GameState, teamname: str):
        """
        Decide actions for all robots on the team based on game state.
        
        Args:
            game_state: Current game state with ball and robot positions
            teamname: Name of the team to generate actions for
            
        Returns:
            Decided actions for all the robots
        """
        actions = []
        violating_robots = self._robots_within_stop_radius(game_state, teamname)
        safe_positions = (
            self._sample_safe_positions(game_state.ball_pos)
            if game_state.ball_pos is not None
            else []
        )
        position_assignments = (
            self._assign_safe_positions(game_state.ball_pos, violating_robots, safe_positions)
            if game_state.ball_pos is not None
            else {}
        )

        team_robots = game_state.robot_poses[teamname]
        if not self.all_dropped:
            for robot in team_robots:
                actions.append("drop")
            self.all_dropped = True
            return actions
        for robot in team_robots:
            unum = int(next(iter(robot.keys())))
            pose = robot[unum]
            if unum not in position_assignments:
                actions.append("dash 0 0")
                continue

            target_x, target_y = position_assignments[unum]
            dx = target_x - pose[0]
            dy = target_y - pose[1]
            if math.hypot(dx, dy) <= 0.05:
                actions.append("dash 0 0")
                continue

            heading = math.radians(pose[2])
            target_heading = math.atan2(dy, dx)
            relative_angle = self._normalize_angle(target_heading - heading)
            actions.append(f"dash {self.max_dash} {relative_angle}")
        return actions

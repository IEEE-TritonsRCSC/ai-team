"""
Player wrapper around ai_interface.utils.basic_commands.
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Tuple, Optional

from ai_interface.utils import basic_commands
from ai_interface.constants.player_constants import *

class Player:
    """
    Convenience wrapper for basic command helpers using defaults.
    """

    kick_power = 100
    pass_power = 60
    kickable_tolerance = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE

    angle_tolerance = math.radians(5.0)
    goto_margin = 0.1
    goto_speed = 100
    detour_margin = 1.0

    def __init__(self,
                 teamname: str,
                 unum: int,
                 dribbling: bool = False,
                 is_goalie: bool = False) -> None:
        self.teamname = teamname
        self.unum = unum
        self.dribbling = dribbling
        self.is_goalie = is_goalie

    def goto(self, x: float, y: float,
             self_pose: List | Tuple,
             game_state,
             margin: float | None = None,
             theta: float | None = None,
             speed: float | None = None,
             detour_margin: float | None = None,
             is_goalie: bool = False) -> str:
        cmd = basic_commands.goto(
            self_pose,
            x,
            y,
            game_state,
            margin=self.goto_margin if margin is None else margin,
            theta=theta,
            speed=self.goto_speed if speed is None else speed,
            detour_margin=self.detour_margin if detour_margin is None else detour_margin,
            is_goalie=is_goalie, allow_decay=False
        )
        return cmd
    
    def reset(self) -> None:
        self.dribbling = False

    def estimate_velocity(self,
                          position_history: List[np.ndarray],
                          dt: float = 0.1) -> np.ndarray:
        if len(position_history) < 2:
            return np.zeros(2)
        try:
            prev_pos = position_history[-2]
            last_pos = position_history[-1]
            if dt <= 0:
                return np.zeros(2)
            return (last_pos - prev_pos) / dt
        except Exception:
            return np.zeros(2)

    def find_nearest_teammate(self,
                              self_pos: Tuple[float, float],
                              game_state) -> Optional[Tuple[float, float]]:
        """
        Find nearest teammate position (x, y).

        Robust to missing/partial `game_state` and to unexpected robot pose structures.
        Excludes this player (`self.unum`) when possible.
        """
        if game_state is None:
            return None
        robot_poses = getattr(game_state, "robot_poses", None)
        if not isinstance(robot_poses, dict):
            return None

        team_robots = robot_poses.get(self.teamname)
        if not isinstance(team_robots, list):
            return None

        origin = np.array(self_pos, dtype=float)
        best_pos: Optional[Tuple[float, float]] = None
        best_dist = float("inf")

        for robot in team_robots:
            if not isinstance(robot, dict) or not robot:
                continue
            try:
                unum = int(next(iter(robot.keys())))
            except Exception:
                continue
            if unum == self.unum:
                continue
            pose = robot.get(unum)
            if pose is None or len(pose) < 2:
                continue
            try:
                pos = np.array([float(pose[0]), float(pose[1])], dtype=float)
            except Exception:
                continue
            d = float(np.linalg.norm(pos - origin))
            if d < best_dist:
                best_dist = d
                best_pos = (float(pose[0]), float(pose[1]))

        return best_pos

    def kick(self, target_angle: float,
             self_pose: List | Tuple,
             ball_pose: List | Tuple,
             kick_power: int | None = None,
             allow_dribble: bool = True,
             game_state = None) -> str:
        assert allow_dribble or game_state is not None, "game_state must be provided when allow_dribble is False"
        kick_cmd = basic_commands.kick(
            self_pose,
            ball_pose,
            target_angle,
            kick_power=self.kick_power if kick_power is None else kick_power,
            dribbling=self.dribbling,
            allow_dribble=allow_dribble,
            game_state=game_state,
        )
        if "kick" in kick_cmd:
            self.dribbling = False
        elif "catch" in kick_cmd:
            self.dribbling = True
        return kick_cmd

    def shoot_at_goal(self, goal: List | Tuple,
                      self_pose: List | Tuple,
                      ball_pose: List | Tuple,
                      kick_power: int | None = None,) -> str:
        target_angle = math.atan2(goal[1] - self_pose[1], goal[0] - self_pose[0])
        return self.kick(
            target_angle,
            self_pose,
            ball_pose,
            kick_power=self.kick_power if kick_power is None else kick_power,
        )

    def pass_to_teammate(self, teammate_pose: List | Tuple,
                         self_pose: List | Tuple,
                         ball_pose: List | Tuple,
                         kick_power: int | None = None) -> str:
        if np.linalg.norm([ball_pose[0] - self_pose[0], ball_pose[1] - self_pose[1]]) > self.kickable_tolerance:
            return "failed"
        target_angle = math.atan2(teammate_pose[1] - self_pose[1], teammate_pose[0] - self_pose[0])
        return self.kick(
            target_angle,
            self_pose,
            ball_pose,
            kick_power=self.pass_power if kick_power is None else kick_power,
        )

    def dribble(self,
                self_pose: List | Tuple,
                ball_pose: List | Tuple,
                kickable_tolerance: float | None = None,
                angle_tolerance: float | None = None) -> str:
        cmd = basic_commands.dribble(
            self_pose,
            ball_pose,
            kickable_tolerance=self.kickable_tolerance if kickable_tolerance is None else kickable_tolerance,
            angle_tolerance=self.angle_tolerance if angle_tolerance is None else angle_tolerance,
        )
        if "catch" in cmd:
            self.dribbling = True
        return cmd
    
    def hasBall(self, self_pose: List | Tuple, ball_pose: List | Tuple, check_angle=False) -> bool:
        if check_angle:
            to_ball = np.array(ball_pose[:2]) - np.array(self_pose[:2])
            heading = self_pose[2]
            angle_diff=abs(heading - math.atan2(to_ball[1], to_ball[0]))
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        robot_to_ball_vec = np.array(ball_pose[:2]) - np.array(self_pose[:2])
        robot_to_ball_dist = np.linalg.norm(robot_to_ball_vec)
        return abs(robot_to_ball_dist - (PLAYER_SIZE + BALL_SIZE)) < KICKABLE_MARGIN and \
            (not check_angle or abs(angle_diff) < math.radians(5))

    def approach_offset(self, target: Tuple[float, float], ball_xy: Tuple[float, float], approach_back_extra: float = 1.2) -> np.ndarray:
        """Vector from ball to a staging point behind it along the shot line."""
        bx, by = float(ball_xy[0]), float(ball_xy[1])
        dir_vec = np.array([float(target[0]) - bx, float(target[1]) - by], dtype=float)
        norm = float(np.linalg.norm(dir_vec))
        if norm < 1e-6:
            dir_vec = np.array([1.0, 0.0], dtype=float)
            norm = 1.0
        back = BALL_SIZE + PLAYER_SIZE + KICKABLE_MARGIN / 2 + approach_back_extra
        return -(dir_vec / norm) * back
    
    
    def contact_point(self, target: Tuple[float, float], ball_xy: Tuple[float, float]) -> np.ndarray:
        """Point just behind the ball along the shot line for contact."""
        bx, by = float(ball_xy[0]), float(ball_xy[1])
        dir_vec = np.array([float(target[0]) - bx, float(target[1]) - by], dtype=float)
        norm = float(np.linalg.norm(dir_vec))
        if norm < 1e-6:
            dir_vec = np.array([1.0, 0.0], dtype=float)
            norm = 1.0
        back = BALL_SIZE + PLAYER_SIZE + KICKABLE_MARGIN / 2
        return np.array([bx, by], dtype=float) - (dir_vec / norm) * back

    def approach_then_contact(self, ball_xy: Tuple[float, float], target_xy: Tuple[float, float], self_pose: Tuple[float, float, float],
                              game_state, use_deep_approach: bool, approach_back_extra: float, deep_margin: float) -> tuple[Optional[str], bool]:
        """
        Stage behind the ball (deep approach), then move to contact point.
        Returns (cmd, use_deep_approach). cmd is None when already at contact.
        """
        deep_speed = 100.0
        contact_speed = 100.0
        contact_margin = 0.1
        bx, by = float(ball_xy[0]), float(ball_xy[1])
        rx, ry = float(self_pose[0]), float(self_pose[1])
        tx, ty = float(target_xy[0]), float(target_xy[1])

        approach_p = np.array([bx, by], dtype=float) + self.approach_offset((tx, ty), (bx, by), approach_back_extra=approach_back_extra)
        if use_deep_approach and math.hypot(float(approach_p[0]) - rx, float(approach_p[1]) - ry) > deep_margin:
            theta = math.atan2(ty - approach_p[1], tx - approach_p[0])
            cmd = self.goto(float(approach_p[0]), float(approach_p[1]), self_pose, game_state,
                            margin=deep_margin, theta=theta, speed=deep_speed)
            return cmd, use_deep_approach

        use_deep_approach = False
        contact_p = self.contact_point((tx, ty), (bx, by))
        if math.hypot(float(contact_p[0]) - rx, float(contact_p[1]) - ry) > contact_margin:
            theta = math.atan2(ty - contact_p[1], tx - contact_p[0])
            cmd = self.goto(float(contact_p[0]), float(contact_p[1]), self_pose, game_state,
                            margin=contact_margin, theta=theta, speed=contact_speed)
            return cmd, use_deep_approach

        return None, use_deep_approach

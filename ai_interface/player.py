"""
Player wrapper around ai_interface.utils.basic_commands.
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Tuple

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
    avoid_radius = 1.0
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
             avoid_radius: float | None = None,
             detour_margin: float | None = None) -> str:
        cmd = basic_commands.goto(
            self_pose,
            x,
            y,
            game_state,
            margin=self.goto_margin if margin is None else margin,
            theta=theta,
            speed=self.goto_speed if speed is None else speed,
            avoid_radius=self.avoid_radius if avoid_radius is None else avoid_radius,
            detour_margin=self.detour_margin if detour_margin is None else detour_margin,
        )
        return cmd
    
    def reset(self) -> None:
        self.dribbling = False

    def kick(self, target_angle: float,
             self_pose: List | Tuple,
             ball_pose: List | Tuple,
             kick_power: int | None = None) -> str:
        kick_cmd = basic_commands.kick(
            self_pose,
            ball_pose,
            target_angle,
            kick_power=self.kick_power if kick_power is None else kick_power,
            dribbling=self.dribbling,
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
        return abs(robot_to_ball_dist - (PLAYER_SIZE + BALL_SIZE)) < KICKABLE_MARGIN / 2 and \
            (not check_angle or abs(angle_diff) < math.radians(5))

"""
Naive AI implementation for soccer robot control.

This module provides a simple AI strategy that demonstrates basic robot behaviors
like kicking and movement in a RoboCup soccer environment.
"""

import math
from networking.data_utils import TeamInfo, GameState

class SoccerAI:
    """
    Simple AI implementation for controlling soccer robots.
    
    This AI uses basic heuristics: the first robot alternates between kicking
    and dashing, while other robots perform simple movement.
    """
    
    def __init__(self, team_info: TeamInfo):
        """Initialize the AI system."""
        self.team_info = team_info
        self.goalie_ids = {}
        for team in team_info:
            self.goalie_ids[team.name] = team.goalie_id

    def decide_action(self, game_state: GameState, teamname: str):
        """
        Decide actions for all robots on the team based on game state.
        
        Args:
            game_state: Current game state with ball and robot positions
            teamname: Name of the team to generate actions for
            
        Returns:
            Raw output from AI decision making
        """
        actions = []
        goalie_id = self.goalie_ids[teamname]

        i = game_state.count
        ball_pos = game_state.ball_pos
        for robot in game_state.robot_poses[teamname]:
            unum = int(next(iter(robot.keys())))
            robot_pose = robot[unum]
            if unum == 1:
                action = self.get_robot1_action(i, ball_pos, robot_pose)
            elif unum == goalie_id:
                action = self.get_goalie_action(ball_pos, robot_pose)
            else:
                action = "dash 20 0"
            actions.append(action)
        return actions

    def translate_ai_output(self, ai_output) -> list[str]:
        """
        Translate AI output to command format.
        
        Args:
            ai_output: Raw output from AI decision making
            
        Returns:
            Translated commands ready for execution
        """
        return ai_output
    
    def get_robot1_action(self, i: int, ball_pos: tuple, robot_pose: tuple) -> str:
        """
        Get action for robot 1.
        
        Args:
            ball_pos: Current position of the ball
            robot_pose: Current position of robot 1
        Returns:
            Action command for robot 1
        """
        kick = (i % 2 == 0)
        return "kick 100 0" if kick else "dash 100 0"
    
    def get_goalie_action(self, ball_pos: tuple, goalie_pose: tuple) -> str:
        """
        Get action for the goalie robot.
        
        Args:
            ball_pos: Current position of the ball
            goalie_pose: Current position of the goalie robot
        Returns:
            Action command for the goalie robot
        """
        ball_x, ball_y = ball_pos
        goalie_x, goalie_y, goalie_dir = goalie_pose

        delta_x = ball_x - goalie_x
        delta_y = ball_y - goalie_y
        dist = math.hypot(delta_x, delta_y)
        if abs(dist - 0.385) < 1e-4:
            return "kick 100 0"
        elif dist < 2.0:
            return "catch 0"

        goalie_dir = math.radians(goalie_dir)
        power = delta_y * (1e+3 / 32)
        if power > 0:
            direction = goalie_dir + (math.pi / 2)
        else:
            direction = (goalie_dir - (math.pi / 2))
        direction = ((direction + math.pi) % (2 * math.pi)) - math.pi

        return f"dash {abs(power)} {direction}"

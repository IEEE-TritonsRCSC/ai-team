"""
Naive AI implementation for soccer robot control.

This module provides a simple AI strategy that demonstrates basic robot behaviors
like kicking and movement in a RoboCup soccer environment.
"""

import math
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from networking.data_utils import TeamInfo, GameState
from ai_interface.goalie import goalie_action
from ai_interface.constants.player_constants import PLAYER_SIZE, BALL_SIZE, KICKABLE_MARGIN
from ai_interface.constants.field_constants import GOAL_R, GOAL_L
from ai_interface.utils.basic_commands import dribble_to, DribbleState

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
        self.side = {}
        # Persistent dribble phase machine for robot 1 (see dribble_to).
        self.robot1_dribble_state = DribbleState()
        for i, team in enumerate(team_info):
            self.goalie_ids[team.name] = team.goalie_id
            self.side[team.name] = "left" if i == 0 else "right"

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
            if unum == goalie_id:
                action = self.get_goalie_action(ball_pos, robot_pose, self.side[teamname])
            elif unum == 1:
                action = self.get_robot1_action(i, ball_pos, robot_pose, self.side[teamname])
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
    
    def get_dist(self, pos1: tuple, pos2: tuple) -> float:
        """
        Calculate Euclidean distance between two positions.
        
        Args:
            pos1: First position
            pos2: Second position
        Returns:
            Euclidean distance between pos1 and pos2
        Raises:
            ValueError: If positions do not have the same dimensions
        """
        if len(pos1) != len(pos2):
            raise ValueError("Positions must have the same dimensions")
        deltas = [a - b for a, b in zip(pos1, pos2)]
        return math.hypot(*deltas)
    
    def hasBall(self, robot_to_ball_dist: float) -> bool:
        """
        Determine if the robot has possession of the ball.
        Args:
            robot_to_ball_dist: Distance between robot and ball
        Returns:
            True if robot has the ball, False otherwise
        """
        return abs(robot_to_ball_dist - (PLAYER_SIZE + BALL_SIZE)) < KICKABLE_MARGIN / 2
    
    def get_robot1_action(self, i: int, ball_pos: tuple, robot_pose: tuple, side: str) -> str:
        """
        Get action for robot 1: dribble the ball to the opponent's goal.

        Delegates to `dribble_to`, which runs the full rule-compliant transport
        cycle (approach -> grab -> carry <=1 m -> release with separation ->
        repeat) until the ball reaches the goal. The per-robot phase machine is
        persisted on the instance across cycles.

        Args:
            ball_pos: Current position of the ball
            robot_pose: Current pose of robot 1 as (x, y, theta_deg)
            side: 'left' if attacking the right goal, 'right' otherwise
        Returns:
            Action command for robot 1
        """
        goal = GOAL_R if side == "left" else GOAL_L
        # robot_pose theta is in degrees here; dribble_to expects radians.
        self_pose_rad = (robot_pose[0], robot_pose[1], math.radians(robot_pose[2]))
        # dribble_to only calls goto with obstacle avoidance off, so it never
        # reads game_state -> None is safe.
        cmd = dribble_to(
            self_pose_rad,
            (ball_pos[0], ball_pos[1]),
            goal,
            None,
            self.robot1_dribble_state,
        )
        if cmd in ("done", "failed"):    # ball on target / transient miss: idle
            return "turn 0"
        return cmd
    
    def get_goalie_action(self, ball_pos: tuple, goalie_pose: tuple, side: str) -> str:
        """
        Get action for the goalie robot using angle-bisector positioning.
        
        Args:
            ball_pos: Current position of the ball
            goalie_pose: Current position of the goalie robot (x, y, theta_deg)
            side: 'left' if our team starts on the left attacking right goal,
                  'right' if our team starts on the right attacking left goal
        Returns:
            Action command for the goalie robot
        """
        goalie_pos = (goalie_pose[0], goalie_pose[1])
        goalie_to_ball_dist = self.get_dist(ball_pos, goalie_pos)
        has_ball = self.hasBall(goalie_to_ball_dist)
        
        return goalie_action(
            ball_pos=ball_pos,
            goalie_pose=goalie_pose,
            has_ball=has_ball,
            goalie_to_ball_dist=goalie_to_ball_dist,
            side=side
        )

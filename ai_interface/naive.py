"""
Naive AI implementation for soccer robot control.

This module provides a simple AI strategy that demonstrates basic robot behaviors
like kicking and movement in a RoboCup soccer environment.
"""

from networking.data_utils import GameState

class SoccerAI:
    """
    Simple AI implementation for controlling soccer robots.
    
    This AI uses basic heuristics: the first robot alternates between kicking
    and dashing, while other robots perform simple movement.
    """
    
    def __init__(self):
        """Initialize the AI system."""
        pass

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
        kick = game_state.count % 2 == 0
        for robot in game_state.robot_poses[teamname]:
            unum = int(list(robot.keys())[0])
            if unum == 1:
                actions.append("kick 100 0" if kick else "dash 100")
            else:
                actions.append("dash 20")
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

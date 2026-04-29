# Assignee: Evan Chou
from networking.data_utils import GameState, TeamInfo


class AccessoryAlgo:
    """
    Halt state behavior - prevents all robots from moving.
    
    Used to completely immobilize a team when the game is halted.
    Robots are dropped initially to stop momentum, then held in place.
    """
    def __init__(self, team_infos: list[TeamInfo]):
        """
        Initialize halt state behavior.
        
        Args:
            team_infos: List of team information (name and player count)
        """
        self.team_infos = team_infos
        self.all_dropped = False

    def decide_action(self, game_state: GameState, teamname: str):
        """
        Decide actions for all robots on the team based on game state.
        
        Ensures all robots are halted and cannot move.
        
        Args:
            game_state: Current game state with ball and robot positions
            teamname: Name of the team to generate actions for
            
        Returns:
            Decided actions for all the robots (all immobilized)
        """
        actions = []
        team_robots = game_state.robot_poses[teamname]
        
        # First cycle: drop all robots to stop them
        if not self.all_dropped:
            for robot in team_robots:
                actions.append("drop")
            self.all_dropped = True
            return actions
        
        # Subsequent cycles: keep all robots immobilized with zero power
        for robot in team_robots:
            actions.append("dash 0 0")
        
        return actions


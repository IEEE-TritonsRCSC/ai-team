from networking.data_utils import GameState, TeamInfo

class AccessoryAlgo:
    """
    # TODO
    """
    def __init__(self, team_infos: list[TeamInfo]):
        """
        # TODO
        """
        # TODO
        self.designated_ball_pos = (None, None)
        pass

    def update_designated_ball_pos(self, x, y):
        """
        Update the designated ball position for placement.
        Args:
            x: X-coordinate of the designated ball position
            y: Y-coordinate of the designated ball position
        """
        self.designated_ball_pos = (x, y)

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
        team_robots = game_state.robot_poses[teamname]
        for robot in team_robots:
            # TODO
            actions.append("dash 0 0")
        return actions


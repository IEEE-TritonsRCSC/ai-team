"""
Main networking interface for the soccer AI system.

This module provides the Networker class which coordinates communication
between the AI system and various game environments (simulators and real robots).
"""

from .data_utils import GameState, TeamInfo, Serializer
from .socket_utils import Listener, Commander

class Networker:
    """
    Coordinates networking operations between AI system and game environments.
    
    This class manages communication with both simulated and real robot environments,
    handling game state reception and command execution.
    """
    
    def __init__(self, team_infos: list[TeamInfo], environment: str):
        """
        Initialize the networker with team information and environment settings.
        
        Args:
            team_infos: List of team information including names and player counts
            environment: Environment type for the game
        """
        self.environment = environment
        self.serializer = Serializer()
        self.commander = Commander(team_infos, environment)
        self.game_watcher = Listener(team_infos, environment)

    def get_game_state(self) -> GameState:
        """
        Retrieve the current game state from the appropriate source.
        
        Returns:
            Current game state including ball position, robot poses, and timing information
        """
        return self.game_watcher.watch_game()

    def execute_ai_output(self, output: list[str], team_name: str):
        """
        Execute AI-generated commands by sending them to the appropriate targets.
        
        Args:
            output: List of command strings from the AI system
            team_name: Name of the team executing the commands
        """
        if self.environment in ["sim-only", "sim-mixed"]:
            messages = self.serializer.sim_serialize(output)
            self.commander.send_to_sim(team_name, messages)

        if self.environment != "sim-only":
            messages = self.serializer.robot_serialize(output)
            self.commander.send_to_robots(team_name, messages)

    def disconnect_from_sim(self):
        """Cleanly disconnect from simulator connections."""
        self.commander.disconnect_from_sim()
        self.game_watcher.disconnect_from_sim()

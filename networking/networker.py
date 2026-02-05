"""
Main networking interface for the soccer AI system.

This module provides the Networker class which coordinates communication
between the AI system and various game environments (simulators and real robots).
"""

import threading
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
        self.game_watcher = Listener(team_infos, environment, self.commander.desired_init_poses)
        self._client_data_lock = threading.Lock()
        self._latest_client_data = None

    def get_game_state(self) -> GameState:
        """
        Retrieve the current game state from the appropriate source.
        
        Returns:
            Current game state including ball position, robot poses, and timing information
        """
        game_state = self.game_watcher.watch_game()
        if game_state is None:
            return None
        client_data = self._pop_latest_client_data()
        if client_data and "playmode" in client_data:
            return game_state._replace(playmode=client_data["playmode"])
        return game_state

    def update_client_data(self, client_data: dict) -> None:
        """
        Store the latest client data from the simulator (e.g., playmode).
        """
        if not client_data or "playmode" not in client_data:
            return
        with self._client_data_lock:
            self._latest_client_data = client_data

    def _pop_latest_client_data(self):
        with self._client_data_lock:
            client_data = self._latest_client_data
            self._latest_client_data = None
            return client_data

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

    def reset_sim(self):
        """Reset the simulator to initial state."""
        if hasattr(self.commander, 'reset_sim'):
            self.commander.reset_sim()

    def shutdown(self):
        """Cleanly shutdown all networking connections."""
        if self.environment in ["sim-only", "sim-mixed"]:
            self.commander.disconnect_from_sim()
            self.game_watcher.disconnect_from_sim()
        else:
            self.game_watcher.disconnect_from_camera()
        if self.environment != "sim-only":
            self.commander.stop_robots()

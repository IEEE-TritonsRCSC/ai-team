"""
Main networking interface for the soccer AI system.

This module provides the Networker class which coordinates communication
between the AI system and various game environments (simulators and real robots).
"""

import threading
from .data_utils import GameState, TeamInfo, Serializer
from .socket_utils import (
    Listener,
    Commander,
    _uses_embedded_simulator,
    _uses_simulator,
    _uses_robot_multicast,
)

class Networker:
    """
    Coordinates networking operations between AI system and game environments.
    
    This class manages communication with both simulated and real robot environments,
    handling game state reception and command execution.
    """
    
    def __init__(self, team_infos: list[TeamInfo], environment: str,
                 sim_host: str = "127.0.0.1",
                 sim_player_port: int = 6000,
                 sim_trainer_port: int = 6001):
        """
        Initialize the networker with team information and environment settings.
        
        Args:
            team_infos: List of team information including names and player counts
            environment: Environment type for the game
            sim_host: Simulator host (for sim-only/sim-mixed)
            sim_player_port: Simulator player port (default 6000)
            sim_trainer_port: Simulator trainer/monitor port (default 6001)
        """
        self.environment = environment
        self.serializer = Serializer()
        self.commander = Commander(team_infos, environment,
                                   sim_host=sim_host,
                                   sim_player_port=sim_player_port)
        self.game_watcher = Listener(team_infos, environment,
                                     self.commander.desired_init_poses,
                                     sim_host=sim_host,
                                     sim_trainer_port=sim_trainer_port,
                                     embedded_backend=self.commander.embedded_backend)
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
        if _uses_simulator(self.environment):
            messages = self.serializer.sim_serialize(output)
            self.commander.send_to_sim(team_name, messages)

        if _uses_robot_multicast(self.environment):
            messages = self.serializer.robot_serialize(output)
            self.commander.send_to_robots(team_name, messages)

    def reset_sim(self):
        """Reset the simulator: move all players to initial poses, reset ball, restart play."""
        if _uses_embedded_simulator(self.environment):
            self.commander.reset_sim()
            return

        import time

        # The Listener (game_watcher) holds the monitor socket which has authority
        # to move any object via (move <obj> x y theta).
        monitor_sock = getattr(self.game_watcher, 'sock', None)
        monitor_addr = getattr(self.game_watcher, 'addr', None)
        if monitor_sock is None or monitor_addr is None:
            return

        # 1. Move every player back to its initial pose
        for obj_name, pose in self.commander.desired_init_poses:
            cmd = f"(move {obj_name} {pose[0]} {pose[1]} {pose[2]})\0".encode()
            try:
                monitor_sock.sendto(cmd, monitor_addr)
                monitor_sock.recvfrom(16)  # consume (ok move)
            except Exception:
                pass
            time.sleep(0.02)

        # 2. Reset the ball to centre
        try:
            monitor_sock.sendto(b"(move (ball) 0 0)\0", monitor_addr)
            monitor_sock.recvfrom(16)
        except Exception:
            pass

        # 3. Restart play so players can act immediately
        self.game_watcher.restart_game()

    def shutdown(self):
        """Cleanly shutdown all networking connections."""
        if _uses_simulator(self.environment):
            self.commander.disconnect_from_sim()
            self.game_watcher.disconnect_from_sim()
        else:
            self.game_watcher.disconnect_from_camera()
        if _uses_robot_multicast(self.environment):
            self.commander.stop_robots()

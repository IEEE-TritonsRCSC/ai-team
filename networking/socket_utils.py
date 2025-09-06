"""
Socket-based networking utilities for simulator and robot communication.

This module provides classes for listening to game data, managing client connections,
and commanding both simulated and physical robots through various network protocols.
"""

import re
import random
import time
import socket
import threading
from collections import namedtuple
from .data_utils import GameState, Deserializer

BUFFER_SIZE = 1536
LOCALHOST_IP = "127.0.0.1"
SIM_CLIENT_ADDR = (LOCALHOST_IP, 6000)
SIM_TRAINER_ADDR = (LOCALHOST_IP, 6001)
INIT_PATTERN = r"\(init ([lr]) (1[0-1]|[1-9]) before_kick_off\)"

# Multicast settings for real robots
COMMAND_IP = "239.42.42.42"
COMMAND_PORT = 10000

TeamInfo = namedtuple("TeamInfo", ["name", "n_players"])

class Listener:
    """Listens for game state updates from simulators or cameras."""
    def __init__(self, environment: str):
        """
        Initialize listener for the specified environment.
        
        Args:
            environment: Type of environment to listen to
        """
        self.parser = Deserializer()
        self.addr = SIM_TRAINER_ADDR
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.2)    # Non-blocking with timeout
        
        if environment in ["sim-only", "sim-mixed"]:
            self.source = "simulator"
            self.connect_to_sim()
        else:
            self.source = "camera"
            # TODO: setup multicast socket for camera data

    def watch_game(self) -> GameState:
        """
        Watch for and return the next game state update.
        
        Returns:
            Current game state or None if no valid data received
        """
        if self.source == "simulator":
            (data, address) = self.sock.recvfrom(BUFFER_SIZE)
            if address == self.addr:
                game_state = self.parser.sim_deserialize(data)
                return game_state
        else:
            # TODO: receive multicasted data, call deserializer.cam_deserialize
            pass

    def connect_to_sim(self):
        """Establish connection to simulator and initialize monitoring."""
        self.sock.bind((LOCALHOST_IP, 0))
        self.sock.sendto(b"(init (version 19))\0", self.addr)
        (data, address) = self.sock.recvfrom(16)
        if data != b"(init ok)\0":
            raise Exception(f"Unexpected response: {data} from {address}")
        else:
            self.addr = address    # Save address for subsequent communication
            self.sock.sendto(b"(eye on)\0", self.addr)
            self.sock.sendto(b"(change_mode play_on)\0", self.addr)

        # skip initialization messages
        eye_on = False
        play_on = False
        while not (eye_on and play_on):
            (data, address) = self.sock.recvfrom(16)
            if address == self.addr and data == b"(ok eye on)\0":
                eye_on = True
            if address == self.addr and data == b"(ok change_mode)":
                play_on = True

    def disconnect_from_sim(self):
        """Disconnect from simulator and close socket."""
        self.sock.sendto(b"(bye)\0", self.addr)
        time.sleep(0.1)
        self.sock.close()


class Client:
    """Represents a single robot client connection to the simulator."""
    def __init__(self, teamname: str, side: str = "left", first: bool = False):
        """
        Initialize a client connection for a single robot.
        
        Args:
            teamname: Name of the team this robot belongs to
            side: Which side of field ("left" or "right")
            first: Whether this is the first robot
        """
        self.teamname = teamname
        self.init_pose = self.get_init_pose(first, side)

        self.addr = SIM_CLIENT_ADDR
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.connect_to_sim()

    def get_init_pose(self, first: bool, side: str):
        """
        Calculate initial pose for the robot.
        
        Args:
            first: Whether this is the first robot
            side: Which side of field ("left" or "right")
            
        Returns:
            Tuple of (x, y, theta) initial pose
        """
        if first:
            x, y, theta = (-9.5, 0.0, 0.0)
        else:
            x, y = random.uniform(-30, -15), random.uniform(-10, 10)
            theta = random.uniform(-180, 180)
        
        if side == "right" and first:
            y = 5
            theta = 180.0
        
        return (x, y, theta)

    def send_command(self, command: bytes):
        """Send a command to the simulator for this robot."""
        self.sock.sendto(command, self.addr)

    def connect_to_sim(self):
        """Connect to simulator and initialize robot pose."""
        init_args = f"{self.teamname} (version 19)".encode()
        move_args = f"{self.init_pose[0]} {self.init_pose[1]}".encode()
        turn_args = f"{self.init_pose[2]}".encode()

        # Initialize the connection
        time.sleep(0.1)
        self.send_command(b"(init %b)\0" % init_args)
        (data, address) = self.sock.recvfrom(64)
        if m := re.search(INIT_PATTERN, data.decode()):
            self.side = m.group(1)
            self.id = int(m.group(2))
            self.addr = address    # Save the address for later use
        else:
            raise Exception(f"Unexpected response: {data} from {address}")

        # Send initial position
        time.sleep(0.1)
        self.send_command(b"(move %b)\0" % move_args)

        # Send initial rotation
        time.sleep(0.1)
        self.send_command(b"(turn %b)\0" % turn_args)

    def disconnect_from_sim(self):
        """Disconnect from simulator and close socket."""
        self.send_command(b"(bye)\0")
        self.sock.close()


class Commander:
    """Manages command sending to both simulated and physical robots."""
    def __init__(self, team_infos: list[TeamInfo], environment: str):
        """
        Initialize commander for the given teams and environment.
        
        Args:
            team_infos: Information about teams to command
            environment: Type of environment for command routing
        """
        self.team_infos = team_infos
        self.environment = environment

        if environment in ["sim-only", "sim-mixed"]:
            self.create_sim_clients()

        self.socks, self.addrs = {}, {}
        if environment != "sim-only":
            for i, team_info in enumerate(team_infos):
                teamname = team_info.name
                port_num = COMMAND_PORT + (i * 1000)
                self.socks[teamname] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.addrs[teamname] = (COMMAND_IP, port_num)

    def create_sim_clients(self):
        """Create simulator client connections for all teams and robots."""
        self.sim_clients = {}
        for team_info, side in zip(self.team_infos, ["left", "right"]):
            self.sim_clients[team_info.name] = [None] * team_info.n_players
            # Populate clients for each team
            for i in range(team_info.n_players):
                client = Client(team_info.name, side, i == 0)
                self.sim_clients[team_info.name][i] = client

    def send_to_sim(self, teamname: str, commands: list[bytes]):
        """
        Send commands to simulated robots using threading.
        
        Args:
            teamname: Name of team to send commands to
            commands: List of command bytes for each robot
        """
        threads = []
        for (client, command) in zip(self.sim_clients[teamname], commands):
            if command is None:
                continue
            args = (command,)
            thread = threading.Thread(target=client.send_command, args=args)
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()

    def send_to_robots(self, teamname: str, command: bytes):
        """
        Send commands to physical robots via multicast.
        
        Args:
            teamname: Name of team to send commands to
            command: Command bytes to send
        """
        sock, addr = self.socks[teamname], self.addrs[teamname]
        sock.sendto(command, addr)

    def disconnect_from_sim(self):
        """Disconnect all simulator clients."""
        for team_clients in self.sim_clients.values():
            for client in team_clients:
                time.sleep(0.1)
                client.disconnect_from_sim()

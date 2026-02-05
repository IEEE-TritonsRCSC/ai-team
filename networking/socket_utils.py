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
import sslclient
from .data_utils import GameState, TeamInfo, Deserializer

# Network constants for listening to simulator data
BUFFER_SIZE = 1536
LOCALHOST_IP = "127.0.0.1"
SIM_CLIENT_ADDR = (LOCALHOST_IP, 6000)
SIM_TRAINER_ADDR = (LOCALHOST_IP, 6001)
INIT_PATTERN = r"\(init ([lr]) (1[0-1]|[1-9]) before_kick_off\)"
PLAYMODE_REGEX = r"\(hear \d+ referee (\w+)\)"

# Multicast settings for real robots
COMMAND_IP = "239.42.42.42"
COMMAND_PORT = 10000

class Listener:
    """Listens for game state updates from simulators or cameras."""
    def __init__(self, team_infos: list[TeamInfo], environment: str, desired_init_poses: list):
        """
        Initialize listener for the specified environment.
        
        Args:
            team_infos: List of team information including names and player counts
            environment: Type of environment to listen to
            desired_init_poses: List of desired initial poses for simulated robots
        """
        self.parser = Deserializer(team_infos)

        if environment in ["sim-only", "sim-mixed"]:
            self.source = "simulator"
            self.addr = SIM_TRAINER_ADDR
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(0.2)    # Non-blocking with timeout
            self.connect_to_sim(desired_init_poses)
        else:
            self.source = "camera"
            self.vision_client = sslclient.client()
            self.vision_client.connect()

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
                return None
        else:
            data = self.vision_client.receive()
            if data.HasField("detection"):
                game_state = self.parser.cam_deserialize(data.detection)
                return game_state
            else:
                return None

    def connect_to_sim(self, desired_init_poses: list):
        """
        Establish connection to simulator and initialize monitoring.
        Args:
            desired_init_poses: List of desired initial poses for simulated robots
        """
        self.sock.bind((LOCALHOST_IP, 0))
        self.sock.sendto(b"(init (version 19))\0", self.addr)
        (data, address) = self.sock.recvfrom(16)
        if data != b"(init ok)\0":
            raise Exception(f"Unexpected response: {data} from {address}")
        else:
            self.addr = address    # Save address for subsequent communication

        # Drain any pending initialization messages from the socket buffer
        try:
            while True:
                self.sock.recvfrom(16)
        except TimeoutError:
            pass

        # Set desired initial poses
        for (obj_name, pose) in desired_init_poses:
            init_command = f"(move {obj_name} {pose[0]} {pose[1]} {pose[2]})\0".encode()
            self.sock.sendto(init_command, self.addr)
            (data, address) = self.sock.recvfrom(16)
            if (self.addr != address or data != b"(ok move)\0"):
                raise Exception(f"Unexpected response: {data} from {address}")
            time.sleep(0.1)   # Allow time for simulator to process
        
        # Enable "eye on" to start receiving data about the game state and start the game
        self.sock.sendto(b"(eye on)\0", self.addr)
        self.sock.sendto(b"(change_mode play_on)\0", self.addr)

        (data, address) = self.sock.recvfrom(16)
        if (self.addr != address or data != b"(ok eye on)\0"):
            raise Exception(f"Unexpected response: {data} from {address}")

        (data, address) = self.sock.recvfrom(16)
        if (self.addr != address or data != b"(ok change_mode)"):
            raise Exception(f"Unexpected response: {data} from {address}")

    def disconnect_from_sim(self):
        """Disconnect from simulator and close socket."""
        self.sock.sendto(b"(bye)\0", self.addr)
        time.sleep(0.1)
        self.sock.close()

    def disconnect_from_camera(self):
        """Disconnect from camera vision client."""
        self.vision_client.sock.close()


class Client:
    """Represents a single robot client connection to the simulator."""
    def __init__(self, teamname: str, side: str = "left", 
                 first: bool = False, goalie: bool = False):
        """
        Initialize a client connection for a single robot.
        
        Args:
            teamname: Name of the team this robot belongs to
            side: Which side of field ("left" or "right")
            first: Whether this is the first robot
            goalie: Whether this robot is a goalie
        """
        self.teamname = teamname
        self.init_pose = self.get_init_pose(side, first, goalie)

        self.addr = SIM_CLIENT_ADDR
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.connect_to_sim(goalie)

    def get_init_pose(self, side: str, first: bool, goalie: bool):
        """
        Calculate initial pose for the robot.
        
        Args:
            side: Which side of field ("left" or "right")
            first: Whether this is the first robot
            goalie: Whether this robot is a goalie
            
        Returns:
            Tuple of (x, y, theta) initial pose
        """
        x, y = random.uniform(15, 30), random.uniform(-25, 25)
        theta = random.uniform(-180, 180)

        if side == "left":
            x = -x
            if first:
                x, y, theta = (-10, 0, 0.0)
            if goalie:
                x, y, theta = -41.4, 0.0, 0.0
        else:
            if first:
                x, y, theta = (20, 10, 180.0)
            if goalie:
                x, y, theta = (41.4, 0.0, 180.0)

        return (x, y, theta)

    def send_command(self, command: bytes):
        """Send a command to the simulator for this robot."""
        self.sock.sendto(command, self.addr)

    def watch_game(self) -> dict:
        (data, address) = self.sock.recvfrom(BUFFER_SIZE)
        if address == self.addr:
            data = data.decode()
            if m := re.search(PLAYMODE_REGEX, data):
                playmode = m.group(1).strip()
                return {"playmode": playmode}
            else:
                return {}
        else:
            return None

    def connect_to_sim(self, goalie: bool):
        """Connect to simulator and initialize robot pose.
        Args:
            goalie: Whether this robot is a goalie
        """
        init_args = f"{self.teamname} (version 19)".encode()
        init_args += b" (goalie)" if goalie else b""

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
        self.desired_init_poses = []
        self.sample_client = None

        if environment in ["sim-only", "sim-mixed"]:
            self.create_sim_clients()
            time.sleep(0.1)    # Allow time for simulator to set up

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
            goalie_0idx = team_info.goalie_id - 1    # Convert goalie_id to 0-based index
            for i in range(team_info.n_players):
                teamname = team_info.name
                goalie = (i == goalie_0idx)

                client = Client(teamname, side, i == 0, goalie)
                self.sim_clients[teamname][i] = client
                if self.sample_client is None:
                    self.sample_client = client    # Save one sample client for reference

                # Store desired initial poses by tuple (ObjName, (x, y, theta))
                obj_name = f"(player {teamname} {i+1}{' goalie' if goalie else ''})"
                self.desired_init_poses.append((obj_name, client.init_pose))

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
        print(command, addr)

    def reset_sim(self):
        """Reset simulator by moving players to initial positions."""
        if not hasattr(self, 'sim_clients'):
            return
        
        # Reset each client to initial position instead of full disconnect/reconnect
        for team_info, side in zip(self.team_infos, ["left", "right"]):
            team_clients = self.sim_clients[team_info.name]
            for i, client in enumerate(team_clients):
                if client:
                    # Get new initial pose
                    init_pose = client.get_init_pose(i == 0, side)
                    # Move to initial position
                    move_args = f"{init_pose[0]} {init_pose[1]}".encode()
                    turn_args = f"{init_pose[2]}".encode()
                    
                    try:
                        client.send_command(b"(move %b)\0" % move_args)
                        time.sleep(0.05)
                        client.send_command(b"(turn %b)\0" % turn_args)
                    except Exception:
                        # If command fails, continue with other clients
                        pass

    def disconnect_from_sim(self):
        """Disconnect all simulator clients."""
        for team_clients in self.sim_clients.values():
            for client in team_clients:
                time.sleep(0.1)
                client.disconnect_from_sim()

    def stop_robots(self):
        """Send stop commands to all robots."""
        stop_command = b"stop\0"
        for teamname in self.socks.keys():
            # Send stop command twice because OS-level buffering may drop the last packet
            self.send_to_robots(teamname, stop_command)
            self.send_to_robots(teamname, stop_command)
            time.sleep(0.1)

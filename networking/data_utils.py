"""
Data structures and serialization utilities for game state processing.

This module handles parsing and serializing game data between different formats
used by simulators, cameras, and robot communication protocols.
"""

import re
import time
import math
from collections import namedtuple

SIM_TIMESTEP = 0.1    # seconds
# Matches "(see_global <digits> <content>)" and captures server cycle number and
# remaining data until " ((b"
SIM_COUNT_REGEX = r"\(see_global (\d+) (.*?)(?=\s\(\(b)"
# Matches " ((b) <data>)" and captures ball position data between ((b) and 
# " ((p" markers
SIM_BALL_POS_REGEX = r"\s\(\(b\) (.*?)(?=\s\(\(p)"
# Matches " ((p "<team>" <uniform_num>) <pose_data>)" capturing team name, 
# uniform number (1-11), and pose coordinates
SIM_ROBOT_POSE_REGEX = r"\s\(\(p \"(\w*)\" (1[0-1]|[1-9])\) ([^\)]+)\)"

GameState = namedtuple(
    "GameState", ["count", "timestamp", "ball_pos", "robot_poses"]
)

class Deserializer:
    """Deserializes game data from various sources into GameState objects."""
    def sim_deserialize(self, data: bytes) -> GameState:
        """
        Parse simulator data into a GameState object.
        
        Args:
            data: Raw bytes from simulator
            
        Returns:
            Parsed GameState or None if parsing fails
        """
        message = data.decode()
        
        # count (server cycle number)
        if m := re.match(SIM_COUNT_REGEX, message):
            count = int(m.group(1))
            message = message[m.end():]
        else:
            return None

        # timestamp
        timestamp = time.time()

        # ball position
        if result := self.sim_get_ball_pos(message):
            message, ball_pos = result
        else:
            return None

        # robot poses
        robot_poses = self.sim_get_robot_poses(message)
        if robot_poses is None:
            return None

        return GameState(count, timestamp, ball_pos, robot_poses)

    def sim_get_ball_pos(self, message: str) -> tuple:
        """
        Extract ball position from simulator message.
        
        Args:
            message: Simulator message string
            
        Returns:
            Tuple of (remaining_message, ball_position) or None
        """
        if m := re.search(SIM_BALL_POS_REGEX, message):
            description = m.group(1).split()
            ball_pos = tuple(map(float, description[:2]))
            return message[m.end():], ball_pos
        return None

    def sim_get_robot_poses(self, message: str) -> dict[str, list]:
        """
        Extract robot poses from simulator message.
        
        Args:
            message: Simulator message string
            
        Returns:
            Dictionary mapping team names to lists of robot poses
        """
        robot_poses = {}
        while m := re.match(SIM_ROBOT_POSE_REGEX, message):
            teamname = m.group(1)
            unum = int(m.group(2))
            description = m.group(3).split()
            pose = tuple(float(description[i]) for i in [0, 1, 4])

            if teamname not in robot_poses:
                robot_poses[teamname] = []
            robot_poses[teamname].append({unum: pose})
            message = message[m.end():]
        
        return robot_poses

    def cam_deserialize(self, data: bytes) -> GameState:
        """Parse camera data into GameState (placeholder implementation)."""
        return None

    def cam_get_ball_pos(self, message: str) -> tuple:
        """Extract ball position from camera data (placeholder implementation)."""
        return None

    def cam_get_robot_poses(self, message: str) -> dict[str, list]:
        """Extract robot poses from camera data (placeholder implementation)."""
        return None


class Serializer:
    """Serializes commands into formats suitable for different targets."""
    def _convert_command_for_simulator(self, action: str, convert_index: int) -> str:
        """
        Convert robot angle commands (rad/s) to simulator commands (degrees).
        
        Args:
            action: Robot command string
            convert_index: Index of the rad/s value to convert
            
        Returns:
            Simulator command string with degrees
        """
        parts = action.split()
        head = " ".join(parts[:convert_index])
        tail = " ".join(parts[convert_index + 1:])

        rad_per_sec = float(parts[convert_index])
        degrees_per_sec = math.degrees(rad_per_sec)
        degrees =  degrees_per_sec * SIM_TIMESTEP
        normalized_degrees = ((degrees + 180) % 360) - 180

        return f"{head} {normalized_degrees} {tail}".strip()

    def sim_serialize(self, actions) -> list[bytes]:
        """
        Serialize actions for simulator communication.
        
        Args:
            actions: List of action strings
            
        Returns:
            List of serialized command bytes
        """
        messages = [None] * len(actions)
        for i, action in enumerate(actions):
            if action is None:
                continue

            # Convert rad/s to degrees for simulator
            if action.startswith("turn "):
                action = self._convert_command_for_simulator(action, 1)
            elif action.startswith("dash "):
                action = self._convert_command_for_simulator(action, 2)

            messages[i] = b"(" + action.encode() + b")\0"
        return messages

    def robot_serialize(self, actions) -> bytes:
        """
        Serialize actions for robot communication.
        
        Args:
            actions: List of action strings
            
        Returns:
            Serialized command bytes for robot transmission
        """
        message = ""
        for action in actions:
            if action is None:
                message += "None\n"
            else:
                message += action + "\n"
        
        return message.encode()


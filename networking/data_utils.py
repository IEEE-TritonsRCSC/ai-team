import re
import time
from collections import namedtuple

SIM_BALL_POS_REGEX = r"\s\(\(b\) (.*?)(?=\s\(\(p)"
SIM_ROBOT_POSE_REGEX = r"\s\(\(p \"(\w*)\" (1[0-1]|[1-9])\) ([^\)]+)\)"

GameState = namedtuple(
    "GameState", ["count", "timestamp", "ball_pos", "robot_poses"]
)

class Deserializer:
    def sim_deserialize(self, data: bytes) -> GameState:
        message = data.decode()
        
        # count (server cycle number)
        if m := re.match(r"\(see_global (\d+) (.*?)(?=\s\(\(b)", message):
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
        if robot_poses := self.sim_get_robot_poses(message):
            robot_poses = robot_poses
        else:
            return None

        return GameState(count, timestamp, ball_pos, robot_poses)

    def sim_get_ball_pos(self, message: str) -> tuple:
        if m := re.search(SIM_BALL_POS_REGEX, message):
            description = m.group(1).split()
            ball_pos = tuple(float(value) for value in description[0:2])
            return message[m.end():], ball_pos
        return None

    def sim_get_robot_poses(self, message: str) -> dict[list[tuple]]:
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
        return None

    def cam_get_ball_pos(self, message: str) -> tuple:
        return None

    def cam_get_robot_poses(self, message: str) -> dict[list[tuple]]:
        return None


class Serializer:
    def sim_serialize(self, actions) -> list[bytes]:
        messages = [None] * len(actions)
        for i, action in enumerate(actions):
            if action is None:
                continue

            messages[i] = b"(" + action.encode() + b")\0"
        return messages

    def robot_serialize(self, actions) -> bytes:
        message = ""
        for action in actions:
            if action is None:
                message += "None\n"
            else:
                message += action + "\n"
        
        return message.encode()


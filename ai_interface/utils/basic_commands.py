import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Iterable, List, Tuple
import numpy as np
from constants.player_constants import KICKABLE_MARGIN
from constants.field_constants import GOAL_L, GOAL_R
from algo_utils import normalize_angle

def goto(self_pose: np.ndarray | Tuple | List, x: float, y: float, margin: float = KICKABLE_MARGIN,
         theta: float | None = None, speed: float = 100.0) -> str:
    destination = np.array([x, y])
    distance = np.linalg.norm(destination - np.array(self_pose[:2]))
    angle = np.degrees(np.arctan2(destination[1] - self_pose[1], destination[0] - self_pose[0])) - self_pose[4]
    if distance < margin:
        if theta is not None:
            angle_diff = normalize_angle(theta - self_pose[2])
            return f"turn {angle_diff}" if abs(angle_diff) > 1 else "done"
        else:
            return "done"
    return f"dash {min(speed, distance * 10)} {angle}"

def shoot(self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List, 
          target: np.ndarray | Tuple | List, kick_power: float = 80.0, 
          kickable_tolerance: float = KICKABLE_MARGIN, angle_tolerance: float = 5.0) -> str:
    
    if not np.isclose(self_pose[:2], ball_pose[:2], atol=kickable_tolerance):
        return "failed"
    
    ball_dir = normalize_angle(np.degrees(np.arctan2(ball_pose[1] - self_pose[1], ball_pose[0] - self_pose[0])))
    if not np.isclose(ball_dir, self_pose[2], atol=angle_tolerance):
        return "failed"
    
    angle_to_target = np.degrees(np.arctan2(target[1] - self_pose[1], target[0] - self_pose[0]))
    angle_diff = normalize_angle(angle_to_target - self_pose[2])
    return f"kick {kick_power} {angle_diff}"

def shoot_at_goal(self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List, 
                  goal: np.ndarray | Tuple | List, kick_power: float = 80.0) -> str:
    
    return shoot(self_pose, ball_pose, goal, kick_power)


def pass_to_teammate(self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List,
                     teammate_pose: np.ndarray | Tuple | List, kick_power: float = 60.0) -> str:
    
    return shoot(self_pose, ball_pose, teammate_pose[:2], kick_power)

def dribble(self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List,
            kickable_tolerance: float = KICKABLE_MARGIN, angle_tolerance: float = 5.0,
            angle: float = 0.0) -> str:
 
    if not np.isclose(self_pose[:2], ball_pose[:2], atol=kickable_tolerance):
        return "failed"
    
    ball_dir = normalize_angle(np.degrees(np.arctan2(ball_pose[1] - self_pose[1], ball_pose[0] - self_pose[0])))
    if not np.isclose(ball_dir, self_pose[2], atol=angle_tolerance):
        return "failed"
    
    return f"dribble {angle}"
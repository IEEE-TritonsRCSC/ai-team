import sys
import os

from numpy._typing._array_like import NDArray
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Iterable, List, Tuple
import numpy as np
from constants.player_constants import KICKABLE_MARGIN
from constants.field_constants import GOAL_L, GOAL_R

def _normalize_angle(angle: float) -> float:
    """Normalize angle to be within [-180, 180] degrees."""
    return (angle + 180) % 360 - 180

def goto(self_pose: np.ndarray | Tuple | List, x: float, y: float, margin: float = KICKABLE_MARGIN,
         theta: float | None = None, speed: float = 100.0) -> str:
    destination = np.array([x, y])
    distance = np.linalg.norm(destination - np.array(self_pose[:2]))
    angle = np.degrees(np.arctan2(destination[1] - self_pose[1], destination[0] - self_pose[0])) - self_pose[4]
    if distance < margin:
        if theta is not None:
            angle_diff = _normalize_angle(theta - self_pose[2])
            return f"turn {angle_diff}" if abs(angle_diff) > 1 else "done"
        else:
            return "done"
    return f"dash {min(speed, distance * 10)} {angle}"

def shoot(self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List, goal: np.ndarray | Tuple | List) -> str: 
    if not np.isclose(self_pose[:2], ball_pose[:2], atol=KICKABLE_MARGIN):
        return "failed"
    ball_dir = _normalize_angle(np.arctan2(ball_pose[1] - self_pose[1], ball_pose[0] - self_pose[0]))
    if not np.isclose(ball_dir, self_pose[2], atol=5):
        return "failed"
    angle_to_goal = np.degrees(np.arctan2(goal[1] - self_pose[1], goal[0] - self_pose[0]))
    angle_diff = _normalize_angle(angle_to_goal - self_pose[2])
    return f"kick 80 {angle_diff}"

def calculate_shooting_pose(ball_pose: np.ndarray | Tuple | List, goal: np.ndarray | Tuple | List) -> np.ndarray:
    vec = np.array(ball_pose[:2]) - np.array(goal)
    destination = np.array(goal) + vec + (vec / np.linalg.norm(vec)) * KICKABLE_MARGIN / 2
    destination_theta = _normalize_angle(np.degrees(np.arctan2(vec[1], vec[0])))
    return np.array([destination[0], destination[1], destination_theta])
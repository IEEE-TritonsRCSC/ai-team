import sys
import os
import time
from turtle import distance

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
from scipy.optimize import differential_evolution
from constants.player_constants import KICKABLE_MARGIN
from constants.field_constants import GOAL_L, GOAL_R

def normalize_angle(angle: float) -> float:
    """Normalize angle to be within [-180, 180] degrees."""
    return (angle + 180) % 360 - 180

def calculate_shooting_pose(ball_pose: np.ndarray, target: np.ndarray) -> np.ndarray:
    ball_pose, target = np.array(ball_pose), np.array(target)
    vec = np.array(ball_pose[:2]) - np.array(target)
    destination = np.array(target) + vec + (vec / np.linalg.norm(vec)) * KICKABLE_MARGIN / 2
    destination_theta = normalize_angle(np.degrees(np.arctan2(vec[1], vec[0])))
    return np.array([destination[0], destination[1], destination_theta])


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
    """Normalize angle to be within [-pi, pi] radians."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_shooting_pose(ball_pose: np.ndarray, target: np.ndarray) -> np.ndarray:
    ball_pose, target = np.array(ball_pose), np.array(target)
    vec = np.array(ball_pose[:2]) - np.array(target)
    destination = np.array(target) + vec + (vec / np.linalg.norm(vec)) * KICKABLE_MARGIN / 2
    destination_theta = normalize_angle(np.arctan2(vec[1], vec[0]))
    return np.array([destination[0], destination[1], destination_theta])

def estimate_ball_velocity(positions: np.ndarray,
                           alpha: float,
                           k: int = 4,
                           max_speed: float | None = 8.0) -> np.ndarray:
    """
    Robust ball velocity estimate.
    positions: (N, 2) recent ball positions.
    Returns v_{t+1} (next-step velocity) under decay alpha.

    Strategy:
    - use last k+1 points
    - compute per-step velocities
    - take median to reject outliers
    - apply decay: v_{t+1} = alpha * v_t
    """
    pos = np.asarray(positions, dtype=float)
    if pos.shape[0] < 2:
        return np.zeros(2, dtype=float)

    if k is not None:
        pos = pos[-(k + 1):]

    v_steps = pos[1:] - pos[:-1]
    v_t = np.median(v_steps, axis=0)
    v_next = float(alpha) * v_t

    if max_speed is not None:
        s = float(np.linalg.norm(v_next))
        if s > max_speed:
            v_next = v_next * (max_speed / (s + 1e-9))

    return np.asarray(v_next, dtype=float).reshape(2,)

import sys
import os
import time
import math

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
from scipy.optimize import differential_evolution
from constants.player_constants import KICKABLE_MARGIN, PLAYER_SIZE, BALL_SIZE
from constants.field_constants import (
    GOAL_L, GOAL_R,
    GOAL_L_Y_TOP, GOAL_L_Y_BOTTOM,
    GOAL_R_Y_TOP, GOAL_R_Y_BOTTOM,
)

from typing import List, Tuple


def _as_float_array(value: np.ndarray | Tuple | List) -> np.ndarray:
    """Normalize vector-like inputs at function boundaries."""
    return np.asarray(value, dtype=float)


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    """Return distance from point to segment and normalized projection t."""
    point = _as_float_array(point)
    start = _as_float_array(start)
    end = _as_float_array(end)
    segment = end - start
    seg_len_sq = float(np.dot(segment, segment))
    if seg_len_sq == 0.0:
        return float(np.linalg.norm(point - start)), 0.0
    t = float(np.dot(point - start, segment) / seg_len_sq)
    t = float(np.clip(t, 0.0, 1.0))
    closest = start + t * segment
    return float(np.linalg.norm(point - closest)), t


def normalize_angle(angle: float) -> float:
    """Normalize angle to be within [-pi, pi] radians."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_shooting_pose(ball_pose: np.ndarray, target: np.ndarray) -> np.ndarray:
    ball_pose, target = np.array(ball_pose), np.array(target)
    vec = np.array(ball_pose[:2]) - np.array(target)
    destination = np.array(target) + vec + (vec / np.linalg.norm(vec)) * KICKABLE_MARGIN / 2
    destination_theta = normalize_angle(np.arctan2(vec[1], vec[0]))
    return np.array([destination[0], destination[1], destination_theta])

def estimate_ball_velocity(positions, alpha):
    """
    Estimate the ball's velocity at the next timestep given its position history.
    Ball has no control input (a=0), so v_{t+1} = α * v_t.
    
    Parameters:
    - positions: list/array of shape (k, 2) with k >= 2 consecutive positions
    - alpha: decay parameter from discrete dynamics
    
    Returns:
    - v_next: estimated velocity at the next timestep (2D vector)
    - metadata: dictionary with info about the estimation method
    """
    
    positions = np.asarray(positions)
    k = len(positions)
    
    if k < 2:
        raise ValueError(f"Need at least 2 positions, got {k}")
    
    if k == 2:
        # Exact dynamics with k=2: v_k = p_k - p_{k-1}, v_{k+1} = α * v_k
        v_k = positions[1] - positions[0]
        v_next = alpha * v_k
        
    else:
        # k >= 3: Use least squares for better noise rejection
        # Compute velocities: v_i = p_i - p_{i-1} for i = 1,...,k-1
        velocities = positions[1:] - positions[:-1]  # shape (k-1, 2)
        
        # We can estimate v_0 using least squares: sum_i ||v_i - α^{i} * v_0||^2    
        i_vals = np.arange(len(velocities))  # 0, 1, ..., k-2
        weights = (alpha**i_vals).reshape(-1, 1)
        v0_est_x = np.linalg.solve(weights.T @ weights, weights.T @ velocities[:, 0])
        v0_est_y = np.linalg.solve(weights.T @ weights, weights.T @ velocities[:, 1])
        v0_est = np.array([v0_est_x, v0_est_y])
        v_next = (alpha**(k-1)) * v0_est
    
    return v_next.reshape(-1)

def has_ball(self_pos_xy, ball_pos_xy, kickable_dist: float) -> bool:
    """Return True if robot is within `kickable_dist` of the ball (range check).

    A generous "can the robot reach the ball" test used by the JAL/attention
    envs. For the tighter in-contact possession test (and optional facing
    check), use `in_possession`.
    """
    if self_pos_xy is None or ball_pos_xy is None:
        return False

    if len(self_pos_xy) < 2 or len(ball_pos_xy) < 2:
        return False

    dx = float(self_pos_xy[0]) - float(ball_pos_xy[0])
    dy = float(self_pos_xy[1]) - float(ball_pos_xy[1])
    return bool(np.hypot(dx, dy) <= float(kickable_dist))


def _compute_goal_params(side: str) -> Tuple[Tuple, Tuple, Tuple]:
    """Return (post_top, post_bottom, goal_center) for the given defending side."""
    if side == "left":
        return GOAL_L_Y_TOP, GOAL_L_Y_BOTTOM, GOAL_L
    return GOAL_R_Y_TOP, GOAL_R_Y_BOTTOM, GOAL_R


def infer_side_from_position(goalie_pos: Tuple[float, float]) -> str:
    """Return 'left' or 'right' defending side inferred from goalie x-position."""
    return "left" if goalie_pos[0] < 0 else "right"


def in_possession(self_pose, ball_pose, check_angle: bool = False) -> bool:
    """Return True when the ball sits in the contact band, optionally faced.

    self_pose is [x, y, theta] in radians; ball_pose is [x, y(, ...)]. Unlike
    `has_ball` (a generous range check), possession requires the robot-ball
    distance to be within KICKABLE_MARGIN/2 of physical contact
    (PLAYER_SIZE + BALL_SIZE). With `check_angle`, the robot must also face the
    ball within 5 degrees.
    """
    to_ball = _as_float_array(ball_pose[:2]) - _as_float_array(self_pose[:2])
    dist = float(np.linalg.norm(to_ball))
    if abs(dist - (PLAYER_SIZE + BALL_SIZE)) >= KICKABLE_MARGIN / 2:
        return False
    if not check_angle:
        return True
    angle_diff = normalize_angle(math.atan2(to_ball[1], to_ball[0]) - float(self_pose[2]))
    return abs(angle_diff) < math.radians(5.0)

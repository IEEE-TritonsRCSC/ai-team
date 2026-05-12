import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import math
from scipy.optimize import differential_evolution
from typing import Tuple
from constants.player_constants import KICKABLE_MARGIN
from constants.field_constants import GOAL_L, GOAL_R, GOAL_L_Y_TOP, GOAL_L_Y_BOTTOM, GOAL_R_Y_TOP, GOAL_R_Y_BOTTOM, MAX_KEEPER_OUT

def normalize_angle(angle: float) -> float:
    """Normalize angle to be within [-pi, pi] radians."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp a value between bounds."""
    return max(lo, min(hi, v))

def distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance between two 2D positions."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def face_ball_angle(pose: Tuple[float, ...], ball_pos: Tuple[float, ...]) -> float:
    """Heading angle from a robot pose toward the ball."""
    return math.atan2(ball_pos[1] - pose[1], ball_pos[0] - pose[0])


def dist_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to segment a->b."""
    v = b - a
    vv = float(np.dot(v, v))
    if vv < 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, v) / vv)
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * v
    return float(np.linalg.norm(p - proj))

def get_side(position: Tuple[float, ...]) -> str:
    """Get the side of the field from a position. position is (x, y) or (x, y, theta)."""
    if position[0] < 0:
        return "left"
    return "right"

def get_goal_params(side: str) -> Tuple[float, float, float]:
    """Get goal parameters based on the side of the field."""
    if side == "left":
        return GOAL_L_Y_TOP, GOAL_L_Y_BOTTOM, GOAL_L
    else:
        return GOAL_R_Y_TOP, GOAL_R_Y_BOTTOM, GOAL_R


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

def compute_bisector_target(ball_pos: Tuple[float, float], 
                            goalie_pos: Tuple[float, float],
                            side: str = None) -> Tuple[float, float]:
    """
    Compute desired target position for goalkeeper using angle-bisector positioning.
    
    Parameters:
    - ball_pos: (x, y) position of the ball
    - goalie_pos: (x, y) position of the goalie (used to infer side if not provided)
    - side: 'left' if defending left goal, 'right' if defending right goal.
        
    Returns:
    - (x, y) target position for the goalkeeper
    """
    if side is None:
        side = get_side(goalie_pos)
    
    bx, by = ball_pos
    post_top, post_bottom, goal_center = get_goal_params(side)
    goal_x = goal_center[0]

    vTx, vTy = post_top[0] - bx, post_top[1] - by
    vBx, vBy = post_bottom[0] - bx, post_bottom[1] - by

    normT = math.hypot(vTx, vTy)
    normB = math.hypot(vBx, vBy)

    if normT < 1e-6 or normB < 1e-6:
        dir_x, dir_y = goal_center[0] - bx, goal_center[1] - by
        dir_norm = math.hypot(dir_x, dir_y)
        if dir_norm < 1e-6:
            return goal_center
        dir_x /= dir_norm
        dir_y /= dir_norm
    else:
        uTx, uTy = vTx / normT, vTy / normT
        uBx, uBy = vBx / normB, vBy / normB

        vb_x, vb_y = uTx + uBx, uTy + uBy
        vb_norm = math.hypot(vb_x, vb_y)
        if vb_norm < 1e-6:
            dir_x, dir_y = goal_center[0] - bx, goal_center[1] - by
            dir_norm = math.hypot(dir_x, dir_y)
            if dir_norm < 1e-6:
                return goal_center
            dir_x /= dir_norm
            dir_y /= dir_norm
        else:
            dir_x, dir_y = vb_x / vb_norm, vb_y / vb_norm
    
    if abs(dir_x) < 1e-6:
        intersect_y = goal_center[1]
        intersect_x = goal_x
    else:
        t_intersect = (goal_x - bx) / dir_x
        intersect_y = by + t_intersect * dir_y
        intersect_x = goal_x
    
    ball_to_goal_dist = math.hypot(goal_x - bx, goal_center[1] - by)
    #step_out_factor = min(1.0, 0.5 + ((abs(ball_to_goal_dist - 30) / 15)))  # Scale based on distance
    step_out_factor = min(1.0, ball_to_goal_dist / 15)
    step_out_dist = MAX_KEEPER_OUT * step_out_factor
    
    step_out_x = intersect_x - step_out_dist * dir_x
    step_out_y = intersect_y - step_out_dist * dir_y
    
    if goal_x < 0:
        min_x = goal_x
        max_x = goal_x + MAX_KEEPER_OUT
    else:
        min_x = goal_x - MAX_KEEPER_OUT
        max_x = goal_x
    kx = clamp(step_out_x, min_x, max_x)
    
    post_y_top = max(post_top[1], post_bottom[1])
    post_y_bottom = min(post_top[1], post_bottom[1])
    margin = 2.0
    ky = clamp(step_out_y, post_y_bottom - margin, post_y_top + margin)

    return (kx, ky)

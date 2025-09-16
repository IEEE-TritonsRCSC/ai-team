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

def calculate_inception_velocity(self_pose: np.ndarray, target_pos: np.ndarray, target_vel: np.ndarray, 
                                 target_acc: float, max_speed: float, zero_vel_stop=True):
    """
    Calculate the interception position and required velocity for a player to intercept a moving target.
    Args:
        self_pose (np.ndarray): The player's current position and orientation as [x, y, theta].
        target_pos (np.ndarray): The target's current position as [x, y].
        target_vel (np.ndarray): The target's current velocity as [vx, vy].
        target_acc (float): The target's acceleration magnitude (colinear with velocity).
        max_speed (float): The player's maximum speed.
        time_limit (float or 'stop'): Maximum time to intercept. If 'stop', intercept before target stops.
    Returns:
        intercept_pos (np.ndarray): The position where interception occurs as [x, y].
        required_velocity (np.ndarray): The required velocity vector for interception as [vx, vy].
        interception_time (float): The time until interception occurs.
        None, None, None if interception is not possible within constraints.
    Raises:
        ValueError: If time_limit is invalid.
    """
    self_pos = np.array(self_pose[:2])
    target_pos = np.array(target_pos[:2])
    target_vel = np.array(target_vel[:2])
    
    # Calculate acceleration vector (colinear with velocity)
    vel_magnitude = np.linalg.norm(target_vel)
    if vel_magnitude > 1e-6:
        vel_direction = target_vel / vel_magnitude
        target_acc_vec = target_acc * vel_direction
        if zero_vel_stop:
            if target_acc < -1e-6:
                time_limit = vel_magnitude / -target_acc  # Time to stop
            else:
                time_limit = float('inf')  # No time limit if accelerating or constant speed
    else:
        target_acc_vec = np.zeros(2)
        # Just go to target
        direction_to_target = target_pos - self_pos
        distance_to_target = np.linalg.norm(direction_to_target)
        if distance_to_target < 1e-6:
            return target_pos, np.zeros(2), 0.0
        required_velocity = (direction_to_target / distance_to_target) * max_speed
        return target_pos, required_velocity, distance_to_target / max_speed

    # Solve for optimal direction angle that minimizes interception time
    # For each direction theta, calculate time to interception with fixed player speed
    def time_to_interception(theta) -> float:
        """Calculate time to interception for a given direction angle theta.
        Returns inf if no feasible interception exists within max_speed constraint.
        """
        # Player velocity direction (unit vector)
        player_dir = np.array([np.cos(theta), np.sin(theta)])
        
        # Set up the interception equation:
        # Player position: self_pos + player_dir * speed * t
        # Target position: target_pos + target_vel * t + 0.5 * target_acc_vec * t^2
        # At interception: player_pos = target_pos
        A = np.array([player_dir, -target_vel]).T
        if np.linalg.matrix_rank(A) < 2:
            return float('inf')  # No valid interception time as directions are colinear
        intersection_parametrized = np.linalg.solve(A, target_pos - self_pose[:2])
        if intersection_parametrized[1] < 0:
            return float('inf')  # No valid interception time as time cannot be negative
        intersection = self_pose[:2] + player_dir * intersection_parametrized[0]
        distance = np.linalg.norm(intersection - target_pos)
        if distance < 1e-6:
            time_to_impact = 0.0  # Already at interception point
        else:
            roots = np.roots([0.5 * target_acc, np.linalg.norm(target_vel), -distance])
            roots = roots[np.isreal(roots)].real  # Keep only real roots
            if len(roots) == 0 or np.all(roots <= 0):
                return float('inf')  # No valid interception time as no positive real roots
            time_to_impact = max(roots) if min(roots) <= 0 else min(roots)
            assert time_to_impact > 0, "Time to impact should be positive"
        if time_to_impact * max_speed < np.linalg.norm(intersection - self_pos):
            return float('inf')  # Cannot reach interception point in time
        return float(time_to_impact)
        
    
    # Find optimal direction angle using optimization
    try:
        def objective_for_de(x):
            return time_to_interception(x[0])
        
        result = differential_evolution(
            objective_for_de,
            bounds=[(0, np.pi)],
            seed=42,
            maxiter=50,
            popsize=20 
        )
        
        if result.success and result.fun < time_limit:
            optimal_theta = result.x[0]
            optimal_time = result.fun
        else:
            return None, None, None
    except Exception as e:
        print(e)
        return None, None, None
    
    # Calculate interception position
    intercept_pos = 0.5 * target_acc_vec * optimal_time**2 + target_vel * optimal_time + target_pos
    # Calculate final interception using optimal direction
    player_vel_dir = np.array([np.cos(optimal_theta), np.sin(optimal_theta)])
    player_intersection_dist = np.linalg.norm(intercept_pos - self_pos)
    required_speed = min(max_speed, player_intersection_dist / optimal_time)
    required_velocity = required_speed * player_vel_dir * np.sign(np.dot(player_vel_dir, intercept_pos - self_pos))
    return intercept_pos, required_velocity, optimal_time


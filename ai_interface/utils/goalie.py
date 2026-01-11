"""
Goalkeeper positioning and control utilities.

This module provides angle-bisector positioning logic for goalkeeper positioning,
which positions the goalkeeper along the angle bisector between the ball and the goalposts.
"""

import math
from typing import Tuple
from ai_interface.constants.field_constants import (
    GOAL_L_Y_TOP,
    GOAL_L_Y_BOTTOM,
    GOAL_R_Y_TOP,
    GOAL_R_Y_BOTTOM,
    GOAL_L,
    GOAL_R,
    MAX_KEEPER_OUT,
)


def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp a value between bounds."""
    return max(lo, min(hi, v))


def _compute_goal_params(side: str) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Get goal parameters based on which side we're defending.
    
    Args:
        side: 'left' if defending left goal, 'right' if defending right goal
        
    Returns:
        Tuple of (post_top, post_bottom, goal_center)
    """
    if side == "left":
        # Defending left goal
        post_top = GOAL_L_Y_TOP
        post_bottom = GOAL_L_Y_BOTTOM
        goal_center = GOAL_L
    else:
        # Defending right goal
        post_top = GOAL_R_Y_TOP
        post_bottom = GOAL_R_Y_BOTTOM
        goal_center = GOAL_R
    return post_top, post_bottom, goal_center


def infer_side_from_position(goalie_pos: Tuple[float, float]) -> str:
    """
    Infer which side we're defending from the goalie's position.
    
    Args:
        goalie_pos: (x, y) position of the goalie
        
    Returns:
        'left' if defending left goal (x < 0), 'right' if defending right goal (x >= 0)
    """
    if goalie_pos[0] < 0:
        return "left"
    else:
        return "right"


def compute_bisector_target(ball_pos: Tuple[float, float], 
                            goalie_pos: Tuple[float, float],
                            side: str = None) -> Tuple[float, float]:
    """
    Compute desired target position for goalkeeper using angle-bisector positioning.
    
    The goalkeeper positions along the angle bisector between the ball and the two goalposts,
    stepping out from the goal line toward the ball to cut the shooting angle.
    
    Args:
        ball_pos: (x, y) position of the ball
        goalie_pos: (x, y) position of the goalie (used to infer side if not provided)
        side: 'left' if defending left goal, 'right' if defending right goal.
              If None, inferred from goalie_pos.
        
    Returns:
        (x, y) target position for the goalkeeper
    """
    if side is None:
        side = infer_side_from_position(goalie_pos)
    
    bx, by = ball_pos
    post_top, post_bottom, goal_center = _compute_goal_params(side)
    goal_x = goal_center[0]

    # Vectors from ball to each post
    vLx, vLy = post_top[0] - bx, post_top[1] - by
    vRx, vRy = post_bottom[0] - bx, post_bottom[1] - by

    # Norms
    normL = math.hypot(vLx, vLy)
    normR = math.hypot(vRx, vRy)

    # Compute a direction from ball toward the angle bisector (or fallback)
    if normL < 1e-6 or normR < 1e-6:
        # Fallback: aim at goal center
        dir_x, dir_y = goal_center[0] - bx, goal_center[1] - by
        dir_norm = math.hypot(dir_x, dir_y)
        if dir_norm < 1e-6:
            # Degenerate case: just sit at goal center
            return goal_center
        dir_x /= dir_norm
        dir_y /= dir_norm
    else:
        # unit vectors from ball to each post
        uLx, uLy = vLx / normL, vLy / normL
        uRx, uRy = vRx / normR, vRy / normR

        # bisector direction
        vb_x, vb_y = uLx + uRx, uLy + uRy
        vb_norm = math.hypot(vb_x, vb_y)
        if vb_norm < 1e-6:
            # Fallback to goal center direction if bisector degenerates
            dir_x, dir_y = goal_center[0] - bx, goal_center[1] - by
            dir_norm = math.hypot(dir_x, dir_y)
            if dir_norm < 1e-6:
                return goal_center
            dir_x /= dir_norm
            dir_y /= dir_norm
        else:
            dir_x, dir_y = vb_x / vb_norm, vb_y / vb_norm

    # Find where the bisector line intersects the goal line
    # Parametric line: P(t) = ball + t * dir
    # Goal line: x = goal_x
    # Intersection: bx + t*dir_x = goal_x  =>  t = (goal_x - bx) / dir_x
    
    if abs(dir_x) < 1e-6:
        # Bisector is nearly vertical (ball directly in front/behind goal)
        # Position keeper at goal center y-coordinate
        intersect_y = goal_center[1]
        intersect_x = goal_x
    else:
        # Find intersection parameter
        t_intersect = (goal_x - bx) / dir_x
        # Intersection point on goal line
        intersect_y = by + t_intersect * dir_y
        intersect_x = goal_x
    
    # Determine how far out from goal line to position keeper
    # When ball is close, stay closer to goal line. When ball is far, can step out more.
    ball_to_goal_dist = math.hypot(goal_x - bx, goal_center[1] - by)
    # Scale step-out distance: closer ball = less step out, farther ball = more step out
    # But never exceed MAX_KEEPER_OUT
    step_out_factor = min(1.0, ball_to_goal_dist / 30.0)  # Scale based on distance
    step_out_dist = MAX_KEEPER_OUT * step_out_factor
    
    # Step out along the bisector direction TOWARD the ball
    # Since dir points from ball toward goal, we step in the opposite direction (-dir)
    step_out_x = intersect_x - step_out_dist * dir_x
    step_out_y = intersect_y - step_out_dist * dir_y
    
    # Clamp x so keeper stays between goal line and MAX_KEEPER_OUT out
    if goal_x < 0:  # left goal
        min_x = goal_x
        max_x = goal_x + MAX_KEEPER_OUT
    else:           # right goal
        min_x = goal_x - MAX_KEEPER_OUT
        max_x = goal_x
    kx = _clamp(step_out_x, min_x, max_x)
    
    # Use the y-coordinate from the stepped-out position along the bisector
    # Clamp y near goal mouth with a small margin
    post_y_top = max(post_top[1], post_bottom[1])
    post_y_bottom = min(post_top[1], post_bottom[1])
    margin = 2.0
    ky = _clamp(step_out_y, post_y_bottom - margin, post_y_top + margin)

    return (kx, ky)


def goalie_action(ball_pos: Tuple[float, float],
                  goalie_pose: Tuple[float, float, float],
                  has_ball: bool,
                  goalie_to_ball_dist: float,
                  side: str = None,
                  charge_distance: float = 20) -> str:
    """
    Compute goalkeeper action command using angle-bisector positioning.
    
    Args:
        ball_pos: (x, y) position of the ball
        goalie_pose: (x, y, theta_deg) position and heading of the goalie
        has_ball: Whether the goalie currently has the ball
        goalie_to_ball_dist: Distance between goalie and ball
        side: 'left' or 'right' - which goal we're defending. If None, inferred from goalie position.
        charge_distance: Distance threshold for charging at the ball (default: 15.0)
        
    Returns:
        Action command string (e.g., "dash 100 0", "catch 0", "kick 100 0", etc.)
    """
    goalie_pos = (goalie_pose[0], goalie_pose[1])
    goalie_dir = math.radians(goalie_pose[2])
    
    if side is None:
        side = infer_side_from_position(goalie_pos)
    
    # Immediate ball interaction logic
    if has_ball:
        return "kick 100 0"
    elif goalie_to_ball_dist < 1.2:
        return "catch 0"

    gx, gy = goalie_pos

    # If the ball is very close to the center of our goal, the keeper must charge
    _, _, goal_center = _compute_goal_params(side)
    ball_goal_center_dist = math.hypot(ball_pos[0] - goal_center[0], 
                                       ball_pos[1] - goal_center[1])
    if ball_goal_center_dist < charge_distance:
        # Charge straight toward the ball with max power
        charge_heading = math.atan2(ball_pos[1] - gy, ball_pos[0] - gx)
        charge_direction = charge_heading - goalie_dir
        charge_direction = (charge_direction + math.pi) % (2 * math.pi) - math.pi
        return f"dash 100 {charge_direction}"

    # Compute desired target using angle-bisector positioning
    target_pos = compute_bisector_target(ball_pos, goalie_pos, side)
    tx, ty = target_pos

    # Convert target position into a dash/turn command
    dist_to_target = math.hypot(tx - gx, ty - gy)
    desired_heading = math.atan2(ty - gy, tx - gx)
    direction = desired_heading - goalie_dir
    direction = (direction + math.pi) % (2 * math.pi) - math.pi

    # If we're close to the target, just align to the ball or stay still
    if dist_to_target < 0.5:
        facing_ball = math.atan2(ball_pos[1] - gy, ball_pos[0] - gx)
        turn_to_ball = facing_ball - goalie_dir
        turn_to_ball = (turn_to_ball + math.pi) % (2 * math.pi) - math.pi
        if abs(turn_to_ball) > math.pi / 36:
            return f"turn {turn_to_ball * 10}"
        return "dash 0 0"

    power = min(100, dist_to_target * 10.0)
    return f"dash {power} {direction}"


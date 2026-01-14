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
from ai_interface.constants.field_constants import FIELD_Y, FIELD_X, GOAL_L, GOAL_R, MAX_KEEPER_OUT
from ai_interface.utils.basic_commands import goto

PENALTY_X = 18.0
PENALTY_Y = 20.0
KEEPER_MARGIN = 0.8

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


PENALTY_X = 18.0
PENALTY_Y = 20.0
KEEPER_MARGIN = 0.8

def _clamp_keeper_xy(x: float, y: float, side: str):
    # goalie x must stay within MAX_KEEPER_OUT of goal line
    if side == "left":
        x = _clamp(x, GOAL_L[0], GOAL_L[0] + MAX_KEEPER_OUT)
    else:
        x = _clamp(x, GOAL_R[0] - MAX_KEEPER_OUT, GOAL_R[0])

    # always keep y inside field
    y = _clamp(y, FIELD_Y[0] + KEEPER_MARGIN, FIELD_Y[1] - KEEPER_MARGIN)
    return float(x), float(y)

def _ball_in_penalty(ball_pos, side: str) -> bool:
    bx, by = float(ball_pos[0]), float(ball_pos[1])
    if abs(by) > PENALTY_Y:
        return False
    if side == "left":
        return bx <= GOAL_L[0] + PENALTY_X
    return bx >= GOAL_R[0] - PENALTY_X

def goalie_action(ball_pos, goalie_pos, side, use_charge=True, charge_distance=20):
    gx, gy, gdeg = goalie_pos
    goalie_dir = math.radians(gdeg)
    bx, by = float(ball_pos[0]), float(ball_pos[1])

    # hard safety: if we are already too far out, retreat immediately
    safe_x, safe_y = _clamp_keeper_xy(gx, gy, side)
    if abs(gx - safe_x) > 0.5 or abs(gy - safe_y) > 0.5:
        return goto([gx, gy, goalie_dir], safe_x, safe_y, margin=0.6, speed=100.0)

    goalie_to_ball_dist = _dist(goalie_pos[:2], ball_pos)
    if goalie_to_ball_dist < 1.2 and _ball_in_penalty(ball_pos, side):
        return "catch 0"

    # if we have ball -> clear to wing / upfield
    if goalie_to_ball_dist <= (KICKABLE_MARGIN + 0.05):
        target_x = 0.0
        target_y = -20.0 if by > 0 else 20.0
        theta = math.atan2(target_y - gy, target_x - gx)
        rel = normalize_angle(theta - goalie_dir)
        return f"kick 100 {rel}"

    # default: cover bisector point (then clamp into keeper zone)
    tx, ty = compute_bisector_target(ball_pos, goalie_pos, side)
    tx, ty = _clamp_keeper_xy(tx, ty, side)

    # controlled charge only if ball is inside penalty and we can reach WITHOUT leaving keeper zone
    if use_charge and _ball_in_penalty(ball_pos, side):
        ball_goal_center_dist = _dist(ball_pos, GOAL_L if side == "left" else GOAL_R)
        if ball_goal_center_dist < charge_distance:
            cx, cy = _clamp_keeper_xy(bx, by, side)
            return goto([gx, gy, goalie_dir], cx, cy, margin=0.6, speed=100.0)

    return goto([gx, gy, goalie_dir], tx, ty, margin=0.6, speed=90.0)



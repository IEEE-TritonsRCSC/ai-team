import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Tuple, List
import numpy as np
from constants.field_constants import GOAL_L, GOAL_R
from constants.player_constants import KICKABLE_MARGIN
from ai_interface.utils.basic_commands import goto

GOAL_WIDTH = 2.1
GOAL_Y_BOUNDS = [-GOAL_WIDTH / 2, GOAL_WIDTH / 2]


def guard_goal(
    gk_pose: np.ndarray | Tuple | List,
    ball_pos: np.ndarray | Tuple | List,
    goal_side: str = "left"
) -> str:
    """
    Position the goalkeeper on the goal-line to follow the ball.
    
    The goalkeeper stays on the goal-line and moves along it to align
    with the ball's y-position, effectively following the line of the ball.
    
    Args:
        gk_pose: Current goalkeeper pose (x, y, theta)
        ball_pos: Current ball position (x, y)
        goal_side: Which goal to guard - "left" or "right" (default: "left")
        
    Returns:
        Command string for the goalkeeper to execute
    """
    gk_pose = np.array(gk_pose)
    ball_pos = np.array(ball_pos)
    
    if goal_side.lower() == "left":
        goal_x = GOAL_L[0]  # -54
        target_theta = 0.0
    else:
        goal_x = GOAL_R[0]  # 54
        target_theta = 180.0
    
    # Calculate target y-position by projecting ball's y-position onto goal-line
    # Constrain to stay within goal bounds
    target_y = np.clip(ball_pos[1], GOAL_Y_BOUNDS[0], GOAL_Y_BOUNDS[1])
    target_pos = np.array([goal_x, target_y])
    
    margin = KICKABLE_MARGIN * 0.5 # tighter margin
    return goto(gk_pose, target_pos[0], target_pos[1], margin=margin, theta=target_theta, speed=80.0)


def get_goal_side(gk_pose: np.ndarray | Tuple | List) -> str:
    """
    Determine which goal side the goalkeeper is defending based on their position.
    
    Args:
        gk_pose: Current goalkeeper pose (x, y, theta)
        
    Returns:
        "left" if defending left goal, "right" if defending right goal
    """
    # Where do we store this? Very naive solution below
    gk_x = gk_pose[0] 
    return "left" if gk_x < 0 else "right"


def guard_goal_auto(
    gk_pose: np.ndarray | Tuple | List,
    ball_pos: np.ndarray | Tuple | List
) -> str:
    """
    Position the goalkeeper on the goal-line to follow the ball.
    Automatically determines which goal to guard based on goalkeeper position.
    
    Args:
        gk_pose: Current goalkeeper pose (x, y, theta)
        ball_pos: Current ball position (x, y)
        
    Returns:
        Command string for the goalkeeper to execute
    """
    goal_side = get_goal_side(gk_pose)
    return guard_goal(gk_pose, ball_pos, goal_side)


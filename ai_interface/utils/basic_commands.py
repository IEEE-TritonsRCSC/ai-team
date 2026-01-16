import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Iterable, List, Tuple
import math
import numpy as np
from constants.player_constants import KICKABLE_MARGIN, PLAYER_SIZE, BALL_SIZE
from constants.field_constants import GOAL_L, GOAL_R
from .algo_utils import normalize_angle


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    """Return distance from point to segment and normalized projection t."""
    segment = end - start
    seg_len_sq = float(np.dot(segment, segment))
    if seg_len_sq == 0.0:
        return float(np.linalg.norm(point - start)), 0.0
    t = float(np.dot(point - start, segment) / seg_len_sq)
    t = float(np.clip(t, 0.0, 1.0))
    closest = start + t * segment
    return float(np.linalg.norm(point - closest)), t


def _select_detour(origin: np.ndarray, destination: np.ndarray,
                   obstacles: Iterable[tuple[np.ndarray, float]],
                   detour_margin: float) -> np.ndarray | None:
    """
    Pick a single waypoint that skirts around the first blocking obstacle.
    """
    path = destination - origin
    if np.linalg.norm(path) < 1e-6:
        return None
    perp = np.array([-path[1], path[0]])
    perp_norm = np.linalg.norm(perp)
    if perp_norm < 1e-6:
        return None
    perp_unit = perp / perp_norm

    best = None
    best_cost = None
    for obs_pos, radius in obstacles:
        distance, t = _segment_distance(obs_pos, origin, destination)
        if distance >= radius or t <= 0.0 or t >= 1.0:
            continue
        clearance = radius + detour_margin
        candidates = [
            obs_pos + perp_unit * clearance,
            obs_pos - perp_unit * clearance,
        ]
        for cand in candidates:
            cand_dist, _ = _segment_distance(obs_pos, origin, cand)
            if cand_dist < radius * 0.9:
                continue
            cost = np.linalg.norm(cand - origin) + np.linalg.norm(destination - cand)
            if best_cost is None or cost < best_cost:
                best = cand
                best_cost = cost
    return best


def goto(self_pose: np.ndarray | Tuple | List, x: float, y: float, margin: float = 0.1,
         theta: float | None = None, speed: float = 100.0,
         avoid_points: Iterable[Tuple] | None = None,
         avoid_radius: float = 1.0, detour_margin: float = 1.0) -> str:
    """
    Create a `dash` or `turn` command to move toward a destination.

    self_pose is [x, y, theta] in radians. If the agent is within `margin` of (x, y),
    it optionally turns to heading `theta`; otherwise it dashes toward (x, y) with a
    speed capped by `speed` and scaled by remaining distance. If `avoid_points` are
    provided, the path will detour around obstacles using a single waypoint.
    """
    origin = np.array(self_pose[:2], dtype=float)
    destination = np.array([x, y], dtype=float)
    if avoid_points:
        obstacles = []
        for item in avoid_points:
            try:
                length = len(item)
            except TypeError:
                continue
            if length >= 3:
                ox, oy, radius = item[0], item[1], item[2]
            elif length == 2:
                ox, oy = item[0], item[1]
                radius = avoid_radius
            else:
                continue
            obstacles.append((np.array([ox, oy], dtype=float), float(radius)))
        waypoint = _select_detour(origin, destination, obstacles, detour_margin)
        if waypoint is not None:
            destination = waypoint

    distance = np.linalg.norm(destination - origin)
    angle = np.arctan2(destination[1] - origin[1], destination[0] - origin[0]) - self_pose[2]
    if distance < margin:
        if theta is not None:
            angle_diff = normalize_angle(theta - self_pose[2])
            return f"turn {angle_diff}" if abs(angle_diff) > math.radians(5.0) else "done"
        else:
            return "done"
    return f"dash {speed} {angle}"


def shoot(AI, self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List,
          target: np.ndarray | Tuple | List, kick_power: float = 80.0,
          kickable_tolerance: float = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE,
          angle_tolerance: float = math.radians(5.0)) -> str:
    """
    Create a `kick` command toward a target if the ball is kickable and aligned.

    self_pose is [x, y, theta] in radians; ball_pose is [x, y]; target is [x, y].
    Returns `kick power rel_angle` when the ball is within kickable_tolerance of the
    agent and facing within angle_tolerance radians; otherwise returns `"failed"`.
    """
    if np.linalg.norm(ball_pose - self_pose[:2]) > kickable_tolerance:
        return "failed"


    angle_to_target = np.arctan2(target[1] - self_pose[1], target[0] - self_pose[0])
    return kick(AI, self_pose, ball_pose, angle_to_target, kick_power)

def kick(AI, self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List, 
         target_angle: float, kick_power: float = 80.0):
    """
    Creates 'kick' or 'turn' commands to aim and kick the ball towards a specific global angle
    
    self_pose is [x, y, theta] in radians; target_angle is in radians;
    
    Returns 'kick {kick_power} 0'
    """
    
    angle_diff = normalize_angle(target_angle - self_pose[2])
    
    if AI.dribble_state_shooter and AI.shooter_turn:
        if np.abs(angle_diff) > math.radians(5.0):
            return f"turn {angle_diff}"
        if AI.shooter_turn == True:
            AI.dribble_state_shooter = False
            AI.shooter_turn = False
        else:
            AI.dribble_state_reciever = False
        return f"kick {kick_power} {0}"
    elif AI.dribble_state_reciever and not AI.shooter_turn:
        if np.abs(angle_diff) > math.radians(5.0):
            return f"turn {angle_diff}"
        if AI.shooter_turn == True:
            AI.dribble_state_shooter = False
            AI.shooter_turn = False
        else:
            AI.dribble_state_reciever = False
        return f"kick {kick_power} {0}"
    else:
        return dribble(AI, self_pose, ball_pose)
        


def shoot_at_goal(AI, self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List,
                  goal: np.ndarray | Tuple | List, kick_power: float = 80.0) -> str:
    """
    Convenience wrapper around `shoot` that aims at the provided goal position.
    """
    return shoot(AI, self_pose, ball_pose, goal, kick_power)


def pass_to_teammate(AI, self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List,
                     teammate_pose: np.ndarray | Tuple | List, kick_power: float = 60.0) -> str:
    """
    Convenience wrapper around `shoot` that aims at a teammate's position.

    teammate_pose is [x, y, theta]; only the [x, y] components are used.
    """
    return shoot(AI, self_pose, ball_pose, teammate_pose[:2], kick_power)

def dribble(AI, self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List,
            kickable_tolerance: float = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE,
            angle_tolerance: float = math.radians(5.0),
            angle: float = 0.0) -> str:
    """
    Create a `dribble` command in the given relative angle when the ball is controllable.

    self_pose is [x, y, theta] in radians; ball_pose is [x, y]. Returns `dribble angle`
    when the ball is within kickable_tolerance and aligned within angle_tolerance radians;
    otherwise returns `"failed"`.
    """
    if np.linalg.norm(ball_pose-self_pose[:2]) > kickable_tolerance:
        return "failed"

    ball_dir = normalize_angle(np.arctan2(ball_pose[1] - self_pose[1], ball_pose[0] - self_pose[0]))
    angle_diff = normalize_angle(ball_dir - self_pose[2])
    if abs(angle_diff) > angle_tolerance:
        return f"turn {angle_diff}"

    if AI.shooter_turn:
        AI.dribble_state_shooter = True
    else:
        AI.dribble_state_reciever = True
    return f"catch 0"

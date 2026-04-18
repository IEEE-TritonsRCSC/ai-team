import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Iterable, List, Tuple
import math
import numpy as np
from constants.player_constants import *
from constants.field_constants import *
from .algo_utils import normalize_angle


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


def _select_detour(origin: np.ndarray, destination: np.ndarray,
                   obstacles: Iterable[tuple[np.ndarray, float]],
                   detour_margin: float) -> np.ndarray | None:
    """
    Pick a single waypoint that skirts around the first blocking obstacle.
    """
    origin = _as_float_array(origin)
    destination = _as_float_array(destination)
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
        clearance = radius + detour_margin
        if distance >= clearance or t <= 0.0 or t >= 1.0:
            continue
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


def build_avoid_points(game_state, self_pose,
                       ball_radius: float = KICKABLE_MARGIN,
                       player_radius: float = 1.0) -> list[tuple[float, float, float]]:
    """
    Build avoid points for a player, skipping itself.
    """
    self_pose = _as_float_array(self_pose)
    avoid_points = []
    ball_pos = getattr(game_state, "ball_pos", None)
    if ball_pos is not None:
        ball_pos = _as_float_array(ball_pos)
        avoid_points.append((ball_pos[0], ball_pos[1], ball_radius))
    for other_team, team_robots in game_state.robot_poses.items():
        for robot in team_robots:
            other_unum = int(next(iter(robot.keys())))
            pose = _as_float_array(robot[other_unum])
            if np.isclose(pose, self_pose).all():
                continue
            avoid_points.append((pose[0], pose[1], player_radius))
    return avoid_points


def goto(self_pose: np.ndarray | Tuple | List, x: float, y: float, game_state,
         margin: float = 0.1, theta: float | None = None, speed: float = 100.0,
         detour_margin: float = 1.5, is_goalie: bool = False) -> str:
    """
    Create a `dash` or `turn` command to move toward a destination.

    self_pose is [x, y, theta] in radians. If the agent is within `margin` of (x, y),
    it optionally turns to heading `theta`; otherwise it dashes toward (x, y) with a
    speed capped by `speed` and scaled by remaining distance. If `avoid_points` are
    provided (or can be built from game_state), the path will detour
    around obstacles using a single waypoint.
    """
    self_pose = _as_float_array(self_pose)
    origin = _as_float_array(self_pose[:2])
    destination = _as_float_array([x, y])
    avoid_points = build_avoid_points(game_state, self_pose, ball_radius=0.215, player_radius=0.9)

    if avoid_points:
        obstacles = []
        for item in avoid_points:
            try:
                length = len(item)
            except TypeError:
                continue
            assert length == 3, "Each avoid point must be a tuple of (x, y, radius)"
            ox, oy, radius = item[0], item[1], item[2]
            obstacles.append((_as_float_array([ox, oy]), float(radius)))
        waypoint = _select_detour(origin, destination, obstacles, detour_margin)
        if waypoint is not None:
            print('Going to', destination, 'Detouring via', waypoint)
            destination = waypoint

    distance = np.linalg.norm(destination - origin)
    angle = np.arctan2(destination[1] - origin[1], destination[0] - origin[0]) - self_pose[2]
    if distance < margin:
        if theta is not None:
            angle_diff = normalize_angle(theta - self_pose[2])
            return f"turn {angle_diff}" if abs(angle_diff) > math.radians(5.0) else "done"
        else:
            return "done"
    if not is_goalie:
        speed = min(speed, max(distance * (1 / PLAYER_DECAY - 1) / dt, 20))
    return f"dash {speed} {angle}"


def shoot(self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List,
          target: np.ndarray | Tuple | List, kick_power: float = 80.0,
          kickable_tolerance: float = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE,
          angle_tolerance: float = math.radians(5.0), dribbling=False) -> str:
    """
    Create a `kick` command toward a target if the ball is kickable and aligned.

    self_pose is [x, y, theta] in radians; ball_pose is [x, y]; target is [x, y].
    Returns `kick power rel_angle` when the ball is within kickable_tolerance of the
    agent and facing within angle_tolerance radians; otherwise returns `"failed"`.
    """
    ball_xy = np.array(ball_pose[:2], dtype=float)
    self_xy = np.array(self_pose[:2], dtype=float)
    if np.linalg.norm(ball_xy - self_xy) > kickable_tolerance:
        return "failed"

    angle_to_target = np.arctan2(target[1] - self_xy[1], target[0] - self_xy[0])
    return kick(self_pose, ball_pose, angle_to_target, kick_power, dribbling=dribbling)
    
def kick(self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List, 
         target_angle: float, kick_power: float = 80.0, dribbling=False, ) -> str:
    """
    Creates 'kick' or 'turn' commands to aim and kick the ball towards a specific global angle
    
    self_pose is [x, y, theta] in radians; target_angle is in radians;
    
    Returns 'kick {kick_power} 0'
    """
    self_pose = _as_float_array(self_pose)
    ball_pose = _as_float_array(ball_pose)
    angle_diff = normalize_angle(target_angle - self_pose[2])
    if np.abs(angle_diff) > math.radians(5.0):
        if dribbling:
            return f"turn {angle_diff}"
        else:
            return f"kick {kick_power:.1f} {0}"
    else:
        ball_xy = np.array(ball_pose[:2], dtype=float)
        target_pos = ball_xy - np.array([math.cos(target_angle), math.sin(target_angle)]) * (PLAYER_SIZE + BALL_SIZE + KICKABLE_MARGIN * 0.5)
        print('[Kick] Target position for kicking:', target_pos)
        cmd = goto(self_pose, target_pos[0], target_pos[1], game_state=game_state, margin=0.1, speed=80.0, theta=target_angle)
        if "done" in cmd:
            return f"kick {kick_power:.1f} {0}"
        else:
            return cmd


def shoot_at_goal(self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List,
                  goal: np.ndarray | Tuple | List, kick_power: float = 80.0, dribbling=False) -> str:
    """
    Convenience wrapper around `shoot` that aims at the provided goal position.
    """
    self_pose = _as_float_array(self_pose)
    ball_pose = _as_float_array(ball_pose)
    goal = _as_float_array(goal)
    return shoot(self_pose, ball_pose, goal, kick_power, dribbling=dribbling)

def pass_to_teammate(self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List,
                     teammate_pose: np.ndarray | Tuple | List, kick_power: float = 60.0, dribbling=False) -> str:
    """
    Convenience wrapper around `shoot` that aims at a teammate's position.

    teammate_pose is [x, y, theta]; only the [x, y] components are used.
    """
    self_pose = _as_float_array(self_pose)
    ball_pose = _as_float_array(ball_pose)
    teammate_pose = _as_float_array(teammate_pose)
    return shoot(self_pose, ball_pose, teammate_pose[:2], kick_power, dribbling=dribbling)

def dribble(self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List,
            kickable_tolerance: float = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE,
            angle_tolerance: float = math.radians(5.0)) -> str:
    """
    Create a `dribble` command in the given relative angle when the ball is controllable.

    self_pose is [x, y, theta] in radians; ball_pose is [x, y]. Returns `dribble angle`
    when the ball is within kickable_tolerance and aligned within angle_tolerance radians;
    otherwise returns `"failed"`.
    """
    ball_xy = np.array(ball_pose[:2], dtype=float)
    self_xy = np.array(self_pose[:2], dtype=float)
    if np.linalg.norm(ball_xy - self_xy) > kickable_tolerance:
        return "failed"

    ball_dir = normalize_angle(np.arctan2(ball_xy[1] - self_xy[1], ball_xy[0] - self_xy[0]))
    angle_diff = normalize_angle(ball_dir - self_pose[2])
    if abs(angle_diff) > angle_tolerance:
        return f"turn {angle_diff}"

    return f"catch 0"


def dribble_to(
    self_pose: np.ndarray | Tuple | List,
    ball_pose: np.ndarray | Tuple | List,
    target: np.ndarray | Tuple | List,
    dribble_start_pos: Tuple[float, float] | None,
    max_dribble_dist: float = 1.0,
    release_distance: float = 0.10,
    kickable_tolerance: float = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE,
    angle_tolerance: float = math.radians(5.0),
    is_dribbling: bool = False,
) -> tuple[str, Tuple[float, float] | None]:
    """
    Dribble the ball toward a target with:
    - A maximum continuous dribble distance (max_dribble_dist, meters)
    - A required release distance (release_distance, meters) before starting a new dribble

    Args:
        self_pose: [x, y, theta] in radians.
        ball_pose: [x, y].
        target:   [x, y] dribble target position.
        dribble_start_pos: (x, y) where the current dribble segment started,
                           or None if not currently dribbling.

    Returns:
        (cmd, new_dribble_start_pos)
        - cmd: single command string (e.g. "catch 0", "turn ...", "dash ...", "failed")
        - new_dribble_start_pos: updated start position for the current segment,
                                 or None if not in a dribble segment.
        - is_dribbling: boolean indicating if the agent is currently dribbling the ball.
    """
    robot_xy = np.array(self_pose[:2], dtype=float)
    ball_xy = np.array(ball_pose[:2], dtype=float)
    target_xy = np.array(target[:2], dtype=float)

    # If we are too far from the ball, move to a good approach point just behind the ball
    # (relative to the dribble target) so we are ready to start dribbling.
    dist_to_ball = float(np.linalg.norm(ball_xy - robot_xy))
    if dist_to_ball > kickable_tolerance:
        # Direction from ball to dribble target.
        to_target = target_xy - ball_xy
        norm = float(np.linalg.norm(to_target))
        if norm < 1e-6:
            angle_to_ball = math.atan2(ball_xy[1] - robot_xy[1], ball_xy[0] - robot_xy[0])
            dash_power = 40.0
            return f"dash {dash_power} {angle_to_ball}", dribble_start_pos, False

        to_target_unit = to_target / norm
        # Stand off just behind the ball so we can dribble through it toward target.
        stand_off_dist = PLAYER_SIZE + BALL_SIZE + KICKABLE_MARGIN * 0.5
        behind_ball = ball_xy - to_target_unit * stand_off_dist

        angle_to_spot = math.atan2(behind_ball[1] - robot_xy[1], behind_ball[0] - robot_xy[0])
        dash_power = 40.0
        return f"dash {dash_power} {angle_to_spot}", dribble_start_pos, False

    # If we are in an active dribble segment, enforce the max distance rule.
    if dribble_start_pos is not None:
        start_xy = np.array(dribble_start_pos, dtype=float)
        dribble_dist = float(np.linalg.norm(robot_xy - start_xy))

        if dribble_dist >= max_dribble_dist:
            if is_dribbling:
                return "kick 1 0", dribble_start_pos, False # TODO : Lukas can you change the simulator code to disable dribbling instead of me using kick?
            
            away_angle = math.atan2(robot_xy[1] - ball_xy[1], robot_xy[0] - ball_xy[0])
            dash_power = 40.0
            return f"dash {dash_power} {away_angle}", None, False

        # Still within allowed dribble distance: orient toward target and keep control.
        angle_to_target = math.atan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0])
        angle_diff = normalize_angle(angle_to_target - self_pose[2])

        if abs(angle_diff) > angle_tolerance:
            # Turn toward where we want to dribble.
            return turn(self_pose, angle_to_target), dribble_start_pos, is_dribbling

        if is_dribbling:
            dash_power = 40.0
            return f"dash {dash_power} {angle_to_target}", dribble_start_pos, True

        return "catch 0", dribble_start_pos, True

    if dist_to_ball < release_distance:
        away_angle = math.atan2(robot_xy[1] - ball_xy[1], robot_xy[0] - ball_xy[0])
        dash_power = 40.0
        return f"dash {dash_power} {away_angle}", None, False

    # We are at least release_distance away; align to the ball and start a new dribble segment.
    cmd = dribble(self_pose, ball_pose, kickable_tolerance=kickable_tolerance, angle_tolerance=angle_tolerance, is_dribbling=is_dribbling)
    if cmd == "catch 0":
        return cmd, (float(robot_xy[0]), float(robot_xy[1])), True

    # Either we need to turn toward the ball first, or dribble failed – propagate.
    return cmd, None, False

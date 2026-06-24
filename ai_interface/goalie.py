"""
Goalkeeper positioning and control.

Decision making is a small state machine evaluated once per 100ms cycle:

- POSITION: stand on the angle bisector between the ball and the goalposts,
  stepped out a bounded distance from the goal line to cut the shooting angle.
- BLOCK: a shot has been detected (ball velocity ray crosses the goal mouth);
  move at full speed to the earliest reachable point on the ball's travel line.
- CLEAR: the ball is held or close and slow; grab it with the dribbler, turn
  to a safe upfield direction, and kick it clear.
- CHARGE: loose ball very near our goal with no opponent contesting it;
  advance to claim it.

Ball velocity is not provided by GameState, so the Goalie estimates it by
differencing ball positions across cycles (state kept on the instance).
"""

import math
from collections import deque
from typing import Optional, Tuple

import numpy as np

from ai_interface.constants.field_constants import (
    GOAL_L_Y_TOP,
    GOAL_L_Y_BOTTOM,
    GOAL_R_Y_TOP,
    GOAL_R_Y_BOTTOM,
    GOAL_L,
    GOAL_R,
    MAX_KEEPER_OUT,
    BALL_DECAY,
)
from ai_interface.utils import basic_commands
from ai_interface.utils.algo_utils import normalize_angle, estimate_ball_velocity
from ai_interface.player import Player


# --- Tuning constants -------------------------------------------------------

# Hard cap on how far the keeper steps out from the goal line. MAX_KEEPER_OUT
# (10) leaves the goal open to chips; 4 units can be recovered in ~0.4s at the
# keeper's top speed, which matches the flight time of a shot from ~10 units.
MAX_STEP_OUT = min(4.0, MAX_KEEPER_OUT)

# Step-out scaling: hug the goal line when the ball is inside CLOSE_ZONE of
# the goal, step out fully when it is beyond FAR_ZONE.
CLOSE_ZONE = 8.0
FAR_ZONE = 25.0

# Ball speed (units per 0.1s step) above which a goalward ball counts as a shot.
SHOT_SPEED_THRESH = 0.3

# Extra width added to the goal mouth when testing whether a shot is on target.
GOAL_INTERCEPT_MARGIN = 1.5

# Conservative effective keeper speed in units per step (steady state is ~0.4;
# this allows for acceleration from rest).
KEEPER_SPEED = 0.35

# Distance at which the dribbler can grab the ball.
CATCH_DIST = 1.2


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
        post_top = GOAL_L_Y_TOP
        post_bottom = GOAL_L_Y_BOTTOM
        goal_center = GOAL_L
    else:
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

    The goalkeeper positions along the angle bisector between the ball and the two
    goalposts, stepping out from the goal line toward the ball to cut the shooting
    angle. Step-out is zero when the ball is within CLOSE_ZONE of the goal and grows
    to MAX_STEP_OUT when the ball is beyond FAR_ZONE.

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

    normL = math.hypot(vLx, vLy)
    normR = math.hypot(vRx, vRy)

    # Direction from ball along the angle bisector (or fallback toward goal center)
    if normL < 1e-6 or normR < 1e-6:
        dir_x, dir_y = goal_center[0] - bx, goal_center[1] - by
        dir_norm = math.hypot(dir_x, dir_y)
        if dir_norm < 1e-6:
            return tuple(goal_center)
        dir_x /= dir_norm
        dir_y /= dir_norm
    else:
        uLx, uLy = vLx / normL, vLy / normL
        uRx, uRy = vRx / normR, vRy / normR
        vb_x, vb_y = uLx + uRx, uLy + uRy
        vb_norm = math.hypot(vb_x, vb_y)
        if vb_norm < 1e-6:
            dir_x, dir_y = goal_center[0] - bx, goal_center[1] - by
            dir_norm = math.hypot(dir_x, dir_y)
            if dir_norm < 1e-6:
                return tuple(goal_center)
            dir_x /= dir_norm
            dir_y /= dir_norm
        else:
            dir_x, dir_y = vb_x / vb_norm, vb_y / vb_norm

    # Intersect the bisector ray with the goal line x = goal_x
    if abs(dir_x) < 1e-6:
        intersect_y = goal_center[1]
        intersect_x = goal_x
    else:
        t_intersect = (goal_x - bx) / dir_x
        intersect_y = by + t_intersect * dir_y
        intersect_x = goal_x

    # Step out from the goal line toward the ball, scaled by ball distance
    ball_to_goal_dist = math.hypot(goal_x - bx, goal_center[1] - by)
    step_out_factor = _clamp((ball_to_goal_dist - CLOSE_ZONE) / (FAR_ZONE - CLOSE_ZONE),
                             0.0, 1.0)
    step_out_dist = MAX_STEP_OUT * step_out_factor

    # dir points from ball toward goal, so step in the opposite direction (-dir)
    step_out_x = intersect_x - step_out_dist * dir_x
    step_out_y = intersect_y - step_out_dist * dir_y

    # Clamp x between the goal line and MAX_STEP_OUT in front of it
    if goal_x < 0:
        kx = _clamp(step_out_x, goal_x, goal_x + MAX_STEP_OUT)
    else:
        kx = _clamp(step_out_x, goal_x - MAX_STEP_OUT, goal_x)

    # Clamp y near the goal mouth with a small margin
    post_y_top = max(post_top[1], post_bottom[1])
    post_y_bottom = min(post_top[1], post_bottom[1])
    margin = 2.0
    ky = _clamp(step_out_y, post_y_bottom - margin, post_y_top + margin)

    return (kx, ky)


class Goalie(Player):
    """
    Goalkeeper class that inherits from Player.

    Keeps per-cycle state (ball position history, current mode) on the instance,
    so a single Goalie object must be reused across cycles for shot detection
    to work.
    """

    def __init__(self,
                 teamname: str,
                 unum: int,
                 side: str = None,
                 charge_distance: float = 6.0,
                 reaction_lag: int = 0,
                 dribbling: bool = False) -> None:
        """
        Initialize the Goalie.

        Args:
            teamname: Team name
            unum: Uniform number
            side: 'left' or 'right' - which goal we're defending. If None, inferred
                  from position.
            charge_distance: Ball distance from goal center below which the keeper
                             may charge a loose ball (default: 6.0)
            reaction_lag: Perception latency in cycles applied ONLY to the steady-state
                          POSITION bisector tracking. The keeper steers toward where the
                          ball was `reaction_lag` cycles ago, so a lateral carry trails it
                          and opens a lane. Shot detection / BLOCK / CHARGE / catch stay on
                          the true ball, so real shots are still defended instantly.
                          0 = no lag (perfect tracking).
            dribbling: Whether the goalie is currently dribbling
        """
        super().__init__(teamname, unum, dribbling=dribbling, is_goalie=True)
        self.side = side
        self.charge_distance = charge_distance
        self.reaction_lag = int(reaction_lag)

        # Ball state estimation
        self._ball_history = deque(maxlen=8)  # (cycle, x, y)
        self._ball_vel = np.zeros(2)
        self._ball_speed = 0.0
        self._last_cycle: Optional[int] = None
        self._stale_cycles = 0

        # Mode state machine
        self._mode = "POSITION"
        self._block_release_count = 0

    # --- Side helpers -------------------------------------------------------

    def infer_side(self, goalie_pos: Tuple[float, float] = None) -> str:
        """
        Infer which side we're defending.

        Args:
            goalie_pos: (x, y) position of the goalie. If None, uses stored side.

        Returns:
            'left' or 'right'
        """
        if self.side is not None:
            return self.side
        if goalie_pos is None:
            raise ValueError("Cannot infer side: no goalie position provided and side not set")
        return infer_side_from_position(goalie_pos)

    def compute_target_position(self, ball_pos: Tuple[float, float],
                                goalie_pos: Tuple[float, float] = None) -> Tuple[float, float]:
        """
        Compute desired default-positioning target using angle-bisector positioning.

        Args:
            ball_pos: (x, y) position of the ball
            goalie_pos: (x, y) position of the goalie. If None, side must be set.

        Returns:
            (x, y) target position for the goalkeeper
        """
        if goalie_pos is None:
            raise ValueError("goalie_pos is required")
        side = self.infer_side(goalie_pos)
        return compute_bisector_target(ball_pos, goalie_pos, side)

    # --- Ball state estimation ------------------------------------------------

    def _update_ball_estimate(self, ball_pos: Tuple[float, float], game_state) -> None:
        """
        Update the ball velocity estimate from the position history.

        Uses game_state.count to detect stale (repeated) frames and large jumps
        to detect teleports/resets, then fits a decay-weighted velocity over the
        last few positions and smooths it with an EMA.
        """
        count = getattr(game_state, "count", None) if game_state is not None else None
        if count is None:
            count = (self._last_cycle + 1) if self._last_cycle is not None else 0

        if self._last_cycle is not None:
            if count == self._last_cycle:
                # Frame did not advance: keep previous estimate, flag staleness
                self._stale_cycles += 1
                if self._stale_cycles >= 2:
                    self._ball_vel = np.zeros(2)
                    self._ball_speed = 0.0
                return
            if count - self._last_cycle > 3:
                # Missed too many frames for differencing to be meaningful
                self._ball_history.clear()
        self._stale_cycles = 0

        if self._ball_history:
            _, px, py = self._ball_history[-1]
            if math.hypot(ball_pos[0] - px, ball_pos[1] - py) > 5.0:
                # Teleport/reset: restart estimation
                self._ball_history.clear()

        self._ball_history.append((count, float(ball_pos[0]), float(ball_pos[1])))
        self._last_cycle = count

        if len(self._ball_history) < 2:
            self._ball_vel = np.zeros(2)
            self._ball_speed = 0.0
            return

        window = list(self._ball_history)[-5:]
        positions = np.array([[p[1], p[2]] for p in window])
        raw_vel = np.asarray(estimate_ball_velocity(positions, alpha=BALL_DECAY),
                             dtype=float).reshape(-1)
        self._ball_vel = 0.6 * raw_vel + 0.4 * self._ball_vel
        self._ball_speed = float(math.hypot(self._ball_vel[0], self._ball_vel[1]))

    def _is_shot_incoming(self, ball_pos: Tuple[float, float],
                          side: str) -> Tuple[bool, Optional[float]]:
        """
        Detect whether the ball is a shot on our goal.

        A shot is a ball moving faster than SHOT_SPEED_THRESH whose velocity ray
        crosses the goal mouth (inflated by GOAL_INTERCEPT_MARGIN) and that has
        enough energy to actually reach the goal line before friction stops it.

        Returns:
            (is_shot, predicted_y_crossing)
        """
        if self._ball_speed < SHOT_SPEED_THRESH:
            return False, None

        post_top, post_bottom, goal_center = _compute_goal_params(side)
        goal_x = goal_center[0]
        bx, by = ball_pos
        vx, vy = self._ball_vel

        if abs(vx) < 1e-9:
            return False, None
        # Must be moving toward our goal line
        if vx * (goal_x - bx) <= 0:
            return False, None

        t_goal = (goal_x - bx) / vx
        y_cross = by + t_goal * vy

        y_lo = min(post_top[1], post_bottom[1]) - GOAL_INTERCEPT_MARGIN
        y_hi = max(post_top[1], post_bottom[1]) + GOAL_INTERCEPT_MARGIN
        if not (y_lo <= y_cross <= y_hi):
            return False, None

        # Total distance a decaying ball can still travel is speed / (1 - decay)
        dist_to_goal_line = math.hypot(goal_x - bx, y_cross - by)
        if self._ball_speed / (1.0 - BALL_DECAY) < dist_to_goal_line:
            return False, None

        return True, y_cross

    @staticmethod
    def _ball_steps_to_travel(dist: float, v0: float) -> float:
        """
        Number of 0.1s steps a ball with initial speed v0 needs to cover dist.

        Solves v0 * (1 - decay^n) / (1 - decay) = dist; returns inf if the ball
        stops before covering dist.
        """
        if v0 <= 1e-9:
            return math.inf
        ratio = dist * (1.0 - BALL_DECAY) / v0
        if ratio >= 1.0:
            return math.inf
        if ratio <= 0.0:
            return 0.0
        return math.log(1.0 - ratio) / math.log(BALL_DECAY)

    def _compute_block_point(self, ball_pos: Tuple[float, float],
                             goalie_pos: Tuple[float, float],
                             side: str) -> Tuple[float, float]:
        """
        Compute where to move to block a detected shot.

        Searches the segment of the ball's travel ray between the perpendicular
        foot of the keeper (closest point, fastest to reach) and the goal-line
        crossing, and returns the earliest point the keeper can reach no later
        than the ball. Falls back to guarding the predicted goal-line crossing.
        """
        post_top, post_bottom, goal_center = _compute_goal_params(side)
        goal_x = goal_center[0]
        bx, by = ball_pos
        gx, gy = goalie_pos

        speed = self._ball_speed
        if speed < 1e-6:
            return (goal_x, goal_center[1])
        dx, dy = self._ball_vel[0] / speed, self._ball_vel[1] / speed
        if abs(dx) < 1e-9:
            return (gx, _clamp(by, min(post_top[1], post_bottom[1]),
                               max(post_top[1], post_bottom[1])))

        t_goal = (goal_x - bx) / dx
        # Perpendicular foot of the keeper onto the ball ray
        t_foot = (gx - bx) * dx + (gy - by) * dy
        t_foot = _clamp(t_foot, 0.0, max(t_goal, 0.0))

        # Points before the foot are both farther from the keeper and reached
        # sooner by the ball, so only [t_foot, t_goal] needs to be searched.
        n_samples = 24
        for i in range(n_samples + 1):
            t = t_foot + (t_goal - t_foot) * i / n_samples
            px, py = bx + t * dx, by + t * dy
            ball_steps = self._ball_steps_to_travel(t, speed)
            keeper_steps = math.hypot(px - gx, py - gy) / KEEPER_SPEED
            if keeper_steps <= ball_steps:
                return (px, py)

        # Nothing reachable in time: guard the predicted crossing point
        return (goal_x, by + t_goal * dy)

    # --- Mode selection ---------------------------------------------------------

    def _opponent_near_ball(self, ball_pos: Tuple[float, float], game_state,
                            radius: float = 3.0) -> bool:
        """Return True if any opponent robot is within radius of the ball."""
        if game_state is None or not getattr(game_state, "robot_poses", None):
            return False
        bx, by = ball_pos
        for team, robots in game_state.robot_poses.items():
            if team == self.teamname:
                continue
            for robot in robots:
                pose = robot[int(next(iter(robot.keys())))]
                if math.hypot(pose[0] - bx, pose[1] - by) < radius:
                    return True
        return False

    def _select_mode(self, ball_pos: Tuple[float, float],
                     goalie_pos: Tuple[float, float],
                     has_ball: bool,
                     goalie_to_ball_dist: float,
                     side: str,
                     game_state) -> str:
        """Pick the desired mode for this cycle (before hysteresis)."""
        if has_ball or (goalie_to_ball_dist < CATCH_DIST
                        and self._ball_speed < SHOT_SPEED_THRESH):
            return "CLEAR"

        shot, _ = self._is_shot_incoming(ball_pos, side)
        if shot:
            return "BLOCK"

        _, _, goal_center = _compute_goal_params(side)
        ball_goal_dist = math.hypot(ball_pos[0] - goal_center[0],
                                    ball_pos[1] - goal_center[1])
        if (ball_goal_dist < self.charge_distance
                and self._ball_speed < 0.5
                and self._stale_cycles < 2
                and not self._opponent_near_ball(ball_pos, game_state)):
            return "CHARGE"

        return "POSITION"

    # --- Clearance --------------------------------------------------------------

    def _choose_clearance_angle(self, goalie_pos: Tuple[float, float],
                                side: str, game_state) -> float:
        """
        Pick a safe global angle to clear the ball toward.

        Candidates fan out around straight-upfield (never toward our own goal),
        scored by how upfield they are, a bias toward the nearer sideline, and a
        bonus when a teammate sits in the candidate's cone.
        """
        gx, gy = goalie_pos
        base = 0.0 if side == "left" else math.pi

        teammates = []
        if game_state is not None and getattr(game_state, "robot_poses", None):
            for team, robots in game_state.robot_poses.items():
                if team != self.teamname:
                    continue
                for robot in robots:
                    unum = int(next(iter(robot.keys())))
                    if unum == self.unum:
                        continue
                    pose = robot[unum]
                    teammates.append((pose[0], pose[1]))

        best_angle, best_score = base, -math.inf
        for offset in (-0.9, -0.5, -0.25, 0.0, 0.25, 0.5, 0.9):
            cand = normalize_angle(base + offset)
            score = math.cos(offset)  # prefer straight upfield
            if math.sin(cand) * gy > 0:
                score += 0.15  # slight bias toward the nearer sideline
            for tx, ty in teammates:
                tm_angle = math.atan2(ty - gy, tx - gx)
                if abs(normalize_angle(tm_angle - cand)) < 0.3:
                    score += 1.0
                    break
            if score > best_score:
                best_score = score
                best_angle = cand
        return best_angle

    # --- Per-mode command generation ---------------------------------------------

    def _goto(self, goalie_pose_rad, tx: float, ty: float, game_state,
              margin: float, theta: float = None, full_speed: bool = False) -> str:
        """goto wrapper that tolerates game_state=None and maps 'done' to idle."""
        avoid = game_state is not None and getattr(game_state, "robot_poses", None) is not None
        cmd = basic_commands.goto(
            goalie_pose_rad, tx, ty, game_state,
            margin=margin,
            theta=theta,
            speed=100.0,
            is_goalie=full_speed,  # skips deceleration scaling near the target
            obstacle_avoidance=avoid and not full_speed,
        )
        return cmd

    def _lagged_ball_pos(self, ball_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Return the ball position the keeper *perceives* for steady-state tracking,
        i.e. where the ball was ~`reaction_lag` cycles ago. Models perception +
        actuation latency: while the striker carries the ball laterally the keeper
        steers toward the stale position and trails, opening a lane.

        Reads from `_ball_history` (the true ball is appended each cycle by
        `_update_ball_estimate`, which runs before this in `action()`). Falls back to
        the current ball when lag is disabled or history is too short.
        """
        if self.reaction_lag <= 0 or not self._ball_history:
            return ball_pos
        target_cycle = self._last_cycle - self.reaction_lag if self._last_cycle is not None else None
        if target_cycle is None:
            return ball_pos
        # Walk newest->oldest; pick the most recent entry at or before target_cycle.
        for cycle, px, py in reversed(self._ball_history):
            if cycle <= target_cycle:
                return (px, py)
        # Lag exceeds buffered history: use the oldest entry we have.
        _, ox, oy = self._ball_history[0]
        return (ox, oy)

    def _position_command(self, ball_pos, goalie_pose_rad, goalie_pos,
                          side: str, game_state) -> str:
        post_top, post_bottom, goal_center = _compute_goal_params(side)
        goal_x = goal_center[0]
        inward = 1.0 if side == "left" else -1.0
        bx, by = ball_pos

        if (bx - goal_x) * inward < 0.5:
            # Ball at or behind our goal line (corner, noise): don't chase it;
            # hold the near post area just in front of the line.
            post_y_top = max(post_top[1], post_bottom[1])
            post_y_bottom = min(post_top[1], post_bottom[1])
            target = (goal_x + inward * 1.0,
                      _clamp(by, post_y_bottom + 1.0, post_y_top - 1.0))
        else:
            # Steady-state tracking uses the *perceived* (lagged) ball so a lateral
            # carry trails the keeper and opens a lane; shot defense (BLOCK) uses the
            # true ball elsewhere and is unaffected.
            target = compute_bisector_target(self._lagged_ball_pos(ball_pos), goalie_pos, side)

        gx, gy = goalie_pos
        facing = math.atan2(by - gy, bx - gx)
        cmd = self._goto(goalie_pose_rad, target[0], target[1], game_state,
                         margin=0.3, theta=facing)
        return cmd if cmd != "done" else "dash 0 0"

    def _block_command(self, ball_pos, goalie_pose_rad, goalie_pos,
                       side: str, game_state) -> str:
        post_top, post_bottom, goal_center = _compute_goal_params(side)
        goal_x = goal_center[0]

        tx, ty = self._compute_block_point(ball_pos, goalie_pos, side)
        if goal_x < 0:
            tx = _clamp(tx, goal_x, goal_x + MAX_STEP_OUT)
        else:
            tx = _clamp(tx, goal_x - MAX_STEP_OUT, goal_x)
        post_y_top = max(post_top[1], post_bottom[1])
        post_y_bottom = min(post_top[1], post_bottom[1])
        ty = _clamp(ty, post_y_bottom - 0.5, post_y_top + 0.5)

        gx, gy = goalie_pos
        facing = math.atan2(ball_pos[1] - gy, ball_pos[0] - gx)
        # Full speed, no obstacle avoidance: reaching the line beats detouring.
        cmd = self._goto(goalie_pose_rad, tx, ty, game_state,
                         margin=0.2, theta=facing, full_speed=True)
        return cmd if cmd != "done" else "dash 0 0"

    def _clear_command(self, ball_pos, goalie_pose_rad, goalie_pos,
                       goalie_to_ball_dist: float, has_ball: bool,
                       side: str, game_state) -> str:
        if has_ball:
            # Holding the ball: turn to a safe direction, then kick. Player.kick
            # handles the turn-until-aligned-then-kick sequencing and only kicks
            # within ~5 degrees of the target, so we never clear blindly.
            self.dribbling = True
            target_angle = self._choose_clearance_angle(goalie_pos, side, game_state)
            return self.kick(target_angle, goalie_pose_rad, ball_pos)
        if goalie_to_ball_dist < CATCH_DIST:
            self.dribbling = True
            return "catch 0"
        # Ball got away before we grabbed it: close back in on it.
        cmd = self._goto(goalie_pose_rad, ball_pos[0], ball_pos[1], game_state,
                         margin=1.0, full_speed=True)
        return cmd if cmd != "done" else "catch 0"

    def _charge_command(self, ball_pos, goalie_pose_rad,
                        game_state) -> str:
        # No obstacle avoidance: the ball itself is an avoid point and would
        # otherwise cause a detour around the thing we are charging.
        cmd = self._goto(goalie_pose_rad, ball_pos[0], ball_pos[1], game_state,
                         margin=1.0, full_speed=True)
        return cmd if cmd != "done" else "catch 0"

    # --- Main entry point -------------------------------------------------------

    def action(self, ball_pos: Tuple[float, float],
               goalie_pose: Tuple[float, float, float],
               has_ball: bool = None,
               goalie_to_ball_dist: float = None,
               game_state=None) -> str:
        """
        Compute the goalkeeper command for this cycle.

        Args:
            ball_pos: (x, y) position of the ball
            goalie_pose: (x, y, theta_deg) position and heading of the goalie
            has_ball: Whether the goalie currently has the ball. If None, computed.
            goalie_to_ball_dist: Distance between goalie and ball. If None, computed.
            game_state: Game state object (cycle count, robot poses) used for
                        velocity estimation, clearance targeting and avoidance

        Returns:
            Action command string (e.g., "dash 100 0", "catch 0", "kick 100 0", ...)
        """
        goalie_pos = (goalie_pose[0], goalie_pose[1])
        goalie_dir = math.radians(goalie_pose[2])
        goalie_pose_rad = (goalie_pose[0], goalie_pose[1], goalie_dir)

        if goalie_to_ball_dist is None:
            goalie_to_ball_dist = math.hypot(ball_pos[0] - goalie_pos[0],
                                             ball_pos[1] - goalie_pos[1])

        ball_pose = (ball_pos[0], ball_pos[1], 0)  # dummy heading for hasBall check
        own_has_ball = self.hasBall(goalie_pose_rad, ball_pose, check_angle=False)
        has_ball = own_has_ball if has_ball is None else (has_ball or own_has_ball)
        if not has_ball and goalie_to_ball_dist > CATCH_DIST:
            self.dribbling = False

        side = self.infer_side(goalie_pos)

        self._update_ball_estimate(ball_pos, game_state)

        desired = self._select_mode(ball_pos, goalie_pos, has_ball,
                                    goalie_to_ball_dist, side, game_state)
        # Hysteresis: leaving BLOCK for POSITION requires two consecutive
        # non-shot cycles so a single noisy velocity frame doesn't drop the block.
        if self._mode == "BLOCK" and desired == "POSITION":
            self._block_release_count += 1
            if self._block_release_count < 2:
                desired = "BLOCK"
            else:
                self._block_release_count = 0
        else:
            self._block_release_count = 0
        self._mode = desired

        if self._mode == "CLEAR":
            return self._clear_command(ball_pos, goalie_pose_rad, goalie_pos,
                                       goalie_to_ball_dist, has_ball, side, game_state)
        if self._mode == "BLOCK":
            return self._block_command(ball_pos, goalie_pose_rad, goalie_pos,
                                       side, game_state)
        if self._mode == "CHARGE":
            return self._charge_command(ball_pos, goalie_pose_rad, game_state)
        return self._position_command(ball_pos, goalie_pose_rad, goalie_pos,
                                      side, game_state)


# Backward compatibility: keep the function interface.
# Instances are cached per side so ball-velocity state survives across cycles
# (a fresh Goalie every call would never detect a shot).
_GOALIE_CACHE: dict = {}


def goalie_action(ball_pos: Tuple[float, float],
                  goalie_pose: Tuple[float, float, float],
                  has_ball: bool,
                  goalie_to_ball_dist: float,
                  side: str = None,
                  charge_distance: float = 6.0,
                  game_state=None) -> str:
    """
    Compute goalkeeper action command.

    This is a backward-compatibility wrapper around the Goalie class.

    Args:
        ball_pos: (x, y) position of the ball
        goalie_pose: (x, y, theta_deg) position and heading of the goalie
        has_ball: Whether the goalie currently has the ball
        goalie_to_ball_dist: Distance between goalie and ball
        side: 'left' or 'right' - which goal we're defending. If None, inferred
              from goalie position.
        charge_distance: Ball distance from goal center below which the keeper
                         may charge a loose ball (default: 6.0)
        game_state: Game state object used for velocity estimation and avoidance

    Returns:
        Action command string (e.g., "dash 100 0", "catch 0", "kick 100 0", etc.)
    """
    goalie_pos = (goalie_pose[0], goalie_pose[1])
    inferred_side = infer_side_from_position(goalie_pos) if side is None else side

    goalie = _GOALIE_CACHE.get(inferred_side)
    if goalie is None:
        goalie = Goalie(
            teamname="temp",
            unum=1,
            side=inferred_side,
            charge_distance=charge_distance,
        )
        _GOALIE_CACHE[inferred_side] = goalie
    else:
        goalie.charge_distance = charge_distance

    return goalie.action(
        ball_pos=ball_pos,
        goalie_pose=goalie_pose,
        has_ball=has_ball,
        goalie_to_ball_dist=goalie_to_ball_dist,
        game_state=game_state,
    )

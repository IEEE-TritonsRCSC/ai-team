import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dataclasses import dataclass
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
         detour_margin: float = 1.5, is_goalie: bool = False,
         obstacle_avoidance: bool = True) -> str:
    """
    Create a `dash` or `turn` command to move toward a destination.

    self_pose is [x, y, theta] in radians. If the agent is within `margin` of (x, y),
    it optionally turns to heading `theta`; otherwise it dashes toward (x, y) with a
    speed capped by `speed` and scaled by remaining distance. When `obstacle_avoidance`
    is True, avoid points built from game_state steer the path with a single detour waypoint.
    """
    self_pose = _as_float_array(self_pose)
    origin = _as_float_array(self_pose[:2])
    destination = _as_float_array([x, y])
    avoid_points = []
    if obstacle_avoidance:
        avoid_points = build_avoid_points(
            game_state, self_pose, ball_radius=0.215, player_radius=0.9
        )

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


def approach_ball(self_pose: np.ndarray | Tuple | List, game_state,
                  margin: float = 1.0, theta: float | None = None, speed: float = 100.0,
                  is_goalie: bool = False) -> str:
    """
    Dash or turn toward the ball from ``game_state.ball_pos`` without obstacle avoidance.

    Default margin=1.0 matches kickable-distance semantics expected by the
    JAL env (kicker cone is sized around this range).
    """
    ball_pos = getattr(game_state, "ball_pos", None)
    if ball_pos is None:
        raise ValueError("approach_ball requires game_state.ball_pos")
    ball_pos = _as_float_array(ball_pos)
    return goto(
        self_pose,
        float(ball_pos[0]),
        float(ball_pos[1]),
        game_state,
        margin=margin,
        theta=theta,
        speed=speed,
        is_goalie=is_goalie,
        obstacle_avoidance=False,
    )


def shoot(self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List,
          target: np.ndarray | Tuple | List, kick_power: float = 100.0,
          kickable_tolerance: float = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE,
          angle_tolerance: float = math.radians(5.0), dribbling=False) -> str:
    """
    Create a `kick` command toward a target if the ball is kickable and aligned.

    self_pose is [x, y, theta] in radians; ball_pose is [x, y]; target is [x, y].
    Returns `kick power rel_angle` when the ball is within kickable_tolerance of the
    agent and facing within angle_tolerance radians; otherwise returns `"failed"`.
    """
    self_pose = _as_float_array(self_pose)
    ball_pose = _as_float_array(ball_pose)
    target = _as_float_array(target)
    if np.linalg.norm(ball_pose - self_pose[:2]) > kickable_tolerance:
        return "failed"

    angle_to_target = np.arctan2(target[1] - self_pose[1], target[0] - self_pose[0])
    return kick(self_pose, ball_pose, angle_to_target, kick_power, dribbling=dribbling)


def ball_in_front_reception_cone(
    self_pose: np.ndarray | Tuple | List,
    ball_pose: np.ndarray | Tuple | List,
    *,
    kickable_tolerance: float = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE,
    reception_angle_deg: float = FRONT_RECEPTION_CENTER_ANGLE_DEG,
) -> bool:
    """Return true when the ball center is kickable in the physical front mouth."""

    self_pose = _as_float_array(self_pose)
    ball_pose = _as_float_array(ball_pose)
    delta = ball_pose[:2] - self_pose[:2]
    if float(np.linalg.norm(delta)) > kickable_tolerance:
        return False
    ball_dir = float(np.arctan2(delta[1], delta[0]))
    rel_angle = normalize_angle(ball_dir - float(self_pose[2]))
    half_angle = math.radians(float(reception_angle_deg) * 0.5)
    return abs(rel_angle) <= half_angle

    
def kick(self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List, 
         target_angle: float, kick_power: float = 100.0, dribbling=False, ) -> str:
    """
    Creates 'kick' or 'turn' commands to aim and kick the ball towards a specific global angle
    
    self_pose is [x, y, theta] in radians; target_angle is in radians;
    
    Returns 'kick {kick_power} 0'
    """
    self_pose = _as_float_array(self_pose)
    ball_pose = _as_float_array(ball_pose)
    if not ball_in_front_reception_cone(self_pose, ball_pose):
        return "failed"
    angle_diff = normalize_angle(target_angle - self_pose[2])
    if np.abs(angle_diff) > math.radians(5.0):
        if dribbling:
            # Commands are angular velocity (rad/s); the serializer multiplies by
            # dt. Divide by dt so the requested per-cycle rotation equals the
            # remaining error — a geometric, cap-robust turn-to-align (the 20 deg/s
            # limiter only slows it, it cannot break convergence).
            return f"turn {angle_diff / dt}"
        else:
            return dribble(self_pose, ball_pose) # "failed", "turn {angle_diff}" or "catch 0"
    else:
        return f"kick {kick_power} {0}"
        

def shoot_at_goal(self_pose: np.ndarray | Tuple | List, ball_pose: np.ndarray | Tuple | List,
                  goal: np.ndarray | Tuple | List, kick_power: float = 100.0, dribbling=False) -> str:
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
    self_pose = _as_float_array(self_pose)
    ball_pose = _as_float_array(ball_pose)
    if np.linalg.norm(ball_pose-self_pose[:2]) > kickable_tolerance:
        return "failed"

    ball_dir = normalize_angle(np.arctan2(ball_pose[1] - self_pose[1], ball_pose[0] - self_pose[0]))
    angle_diff = normalize_angle(ball_dir - self_pose[2])
    if abs(angle_diff) > angle_tolerance:
        return f"turn {angle_diff}"

    return f"catch 0"


# ---------------------------------------------------------------------------
# dribble_to: rule-compliant ball transport to a target coordinate.
#
# RoboCup SSL "Excessive Dribbling": a robot may not dribble the ball further
# than 1 m from where dribbling started (the ball location at first contact).
# It may, however, cover large distances by periodically losing possession.
#
# This skill mirrors that rule with a small phase machine. Each call returns a
# single command for the current timestep; per-robot state is held in a caller-
# owned DribbleState so the helper stays as stateless as its siblings here.
#
# One transport cycle:
#   APPROACH  -> close on the ball until it is kickable.
#   GRAB      -> turn to face the BALL, then issue "catch 0" inside the server's
#                body-relative catch rectangle.
#   VERIFY    -> use a small probe dash and require coupled robot/ball motion;
#                retry a failed catch a bounded number of times.
#   CARRY     -> translate toward the target with directional dash and no turns.
#   ALIGN_RELEASE -> perform a bounded target-facing turn immediately before drop.
#   RELEASE   -> finish the segment so the policy can re-approach and re-select.
# Repeats until the ball is within `arrival_margin` of the target.
# ---------------------------------------------------------------------------

DRIBBLE_PHASE_APPROACH = "approach"
DRIBBLE_PHASE_GRAB = "grab"
DRIBBLE_PHASE_SETTLE = "settle"
DRIBBLE_PHASE_VERIFY = "verify"
DRIBBLE_PHASE_CARRY = "carry"
DRIBBLE_PHASE_ALIGN_RELEASE = "align_release"
DRIBBLE_PHASE_RELEASE = "release"
DRIBBLE_PHASE_DONE = "done"


@dataclass
class DribbleState:
    """Per-robot state for `dribble_to`, persisted by the caller across steps.

    phase is one of the DRIBBLE_PHASE_* constants. segment_start is the ball
    position [x, y] captured at the moment of the catch that opened the current
    possession segment; the 1 m limit is measured from it.

    target is the (Dx, Dy) latched on session open and reused for the whole
    transport — the policy re-emits a noisy target every step, so committing
    once keeps the carry heading stable. best_ball_to_target / no_progress_steps
    drive the stall watchdog: best is the closest the ball has ever been to the
    target this session; no_progress_steps counts consecutive steps without a
    new best, and the machine aborts once it exceeds the caller's stall limit.

    committed is set by the caller the moment the policy actually picks
    dribble_to at the ball; it tells the caller's macro to keep driving the
    phase machine through GRAB, SETTLE, VERIFY, CARRY, ALIGN_RELEASE, and RELEASE
    across later steps even if the policy samples other primitives. It is False
    by default and after reset(), so the default GRAB phase before any real pick
    does not hijack ordinary approach steps.
    """

    phase: str = DRIBBLE_PHASE_GRAB
    segment_start: Tuple[float, float] | None = None
    target: Tuple[float, float] | None = None
    best_ball_to_target: float | None = None
    no_progress_steps: int = 0
    committed: bool = False
    catch_attempts: int = 0
    catch_game_count: int | None = None
    verify_robot_start: Tuple[float, float] | None = None
    verify_ball_start: Tuple[float, float] | None = None
    verify_game_count: int | None = None
    verify_steps: int = 0
    align_game_count: int | None = None
    align_steps: int = 0
    grab_steps: int = 0
    release_game_count: int | None = None
    retry_after_release: bool = False
    release_at_limit: bool = False
    last_verify_robot_moved: float | None = None
    last_verify_ball_moved: float | None = None
    last_verify_offset_change: float | None = None
    last_target_heading_error: float | None = None

    def reset(self) -> None:
        self.phase = DRIBBLE_PHASE_GRAB
        self.segment_start = None
        self.target = None
        self.best_ball_to_target = None
        self.no_progress_steps = 0
        self.committed = False
        self.catch_attempts = 0
        self.catch_game_count = None
        self.verify_robot_start = None
        self.verify_ball_start = None
        self.verify_game_count = None
        self.verify_steps = 0
        self.align_game_count = None
        self.align_steps = 0
        self.grab_steps = 0
        self.release_game_count = None
        self.retry_after_release = False
        self.release_at_limit = False
        self.last_verify_robot_moved = None
        self.last_verify_ball_moved = None
        self.last_verify_offset_change = None
        self.last_target_heading_error = None


def dribble_to(self_pose: np.ndarray | Tuple | List,
               ball_pose: np.ndarray | Tuple | List,
               target: np.ndarray | Tuple | List,
               game_state,
               state: DribbleState,
               arrival_margin: float = 0.3,
               segment_limit: float = 0.85,
               separation_margin: float = 0.2,
               speed: float = 100.0,
               angle_tolerance: float = math.radians(5.0),
               kickable_tolerance: float = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE,
               carry_loss_margin: float = 0.6,
               stall_limit: int = 25,
               progress_margin: float = 0.05,
               max_catch_attempts: int = 3,
               verify_probe_steps: int = 2,
               verify_probe_speed: float = 5.0,
               verify_min_displacement: float = 0.005,
               verify_offset_tolerance: float = 0.12,
               max_align_steps: int = 90,
               max_grab_steps: int = 30) -> str:
    """
    Transport the ball to `target` within the excessive-dribbling rule.

    This primitive does NOT walk to the ball: it is a no-op ("done") unless the
    robot is already within `kickable_tolerance` of the ball. Getting to the ball
    is the `approach_ball` primitive's job — keeping that out of here makes a
    `dribble_to` selection mean exactly "carry the ball" (clean credit; it can no
    longer masquerade as an approach). One selection = one carry segment: after
    the ball is released at the segment limit, the machine waits for release to
    settle and returns DONE. Failed verification may retry within the same macro;
    after terminal release the policy must re-select approach_ball → dribble_to.

    self_pose is [x, y, theta] in radians; ball_pose and target are [x, y].
    `state` is mutated in place to advance the phase machine. Returns a single
    command string ("dash …", "turn …", "catch 0", "drop", or "done").

    The first call of a transport latches `target` into `state.target`; every
    later call ignores the passed-in `target` and uses the latched one, so the
    policy's per-step target jitter (it re-emits (Dx, Dy) each step from a noisy
    head) does not wobble the carry heading. The latch clears when the caller
    resets `state` (on arrival, kick, abandon, or stall).

    `segment_limit` (default 0.85 m) is the per-possession carry distance at
    which the ball is released, kept under the 1 m rule with margin for the
    per-step travel of the released ball. `separation_margin` mirrors the env's
    re-dribble gap: after releasing, the robot must reach a robot-ball distance
    of `kickable_tolerance + separation_margin` before re-grabbing.

    Catch success is verified from motion, not proximity. The machine first
    waits for a new simulator count so the catch teleport has settled, then a
    short positive-power probe away from the ball must move the robot and ball
    together while preserving their relative offset. Moving away prevents an
    uncaught robot from colliding with the free ball and producing a false
    positive; low power limits probe-induced displacement. Failed verification
    always drops and waits for the drop to settle before retrying or aborting.

    `stall_limit` / `progress_margin` drive the progress watchdog: the ball must
    reach a new closest-ever distance to the target (improving by at least
    `progress_margin`) at least once every `stall_limit` steps, else the machine
    aborts to DONE so a wedged dribble frees the policy instead of burning the
    episode. It is distance-agnostic — far targets legitimately take more steps;
    only genuine lack of progress trips it.
    """
    self_pose = _as_float_array(self_pose)
    ball_pose = _as_float_array(ball_pose)

    # Latch the target on session open; reuse it for the whole transport.
    if state.target is None:
        state.target = (float(target[0]), float(target[1]))
    target = _as_float_array(state.target)

    robot_xy = self_pose[:2]
    heading = float(self_pose[2])
    raw_game_count = getattr(game_state, "count", None)
    game_count = int(raw_game_count) if raw_game_count is not None else None

    robot_ball_dist = float(np.linalg.norm(ball_pose - robot_xy))
    angle_to_target = float(np.arctan2(target[1] - robot_xy[1], target[0] - robot_xy[0]))
    angle_to_ball = float(np.arctan2(ball_pose[1] - robot_xy[1], ball_pose[0] - robot_xy[0]))
    ball_to_target = float(np.linalg.norm(ball_pose - target))
    state.last_target_heading_error = normalize_angle(angle_to_target - heading)

    # Global arrival check: stop once the ball itself is on target.
    if (
        ball_to_target <= arrival_margin
        and state.phase not in (DRIBBLE_PHASE_ALIGN_RELEASE, DRIBBLE_PHASE_RELEASE)
    ):
        if state.phase == DRIBBLE_PHASE_CARRY:
            state.phase = DRIBBLE_PHASE_ALIGN_RELEASE
            state.align_game_count = None
            state.align_steps = 0
            state.release_at_limit = False
        elif state.phase in (DRIBBLE_PHASE_SETTLE, DRIBBLE_PHASE_VERIFY):
            state.phase = DRIBBLE_PHASE_RELEASE
            state.release_game_count = game_count
            state.retry_after_release = False
            state.release_at_limit = False
            return "drop"
        else:
            state.phase = DRIBBLE_PHASE_DONE
            return "done"

    # Progress watchdog applies only to verified transport. Acquisition can spend
    # several simulator frames aligning/settling/probing without moving the ball;
    # charging those frames against carry progress prematurely killed retries.
    if state.phase == DRIBBLE_PHASE_CARRY:
        if state.best_ball_to_target is None or ball_to_target < state.best_ball_to_target - progress_margin:
            state.best_ball_to_target = ball_to_target
            state.no_progress_steps = 0
        else:
            state.no_progress_steps += 1
            if state.no_progress_steps >= stall_limit:
                if state.phase in (
                    DRIBBLE_PHASE_SETTLE, DRIBBLE_PHASE_VERIFY, DRIBBLE_PHASE_CARRY
                ):
                    state.phase = DRIBBLE_PHASE_RELEASE
                    state.release_game_count = game_count
                    state.retry_after_release = False
                    state.release_at_limit = False
                    return "drop"
                state.phase = DRIBBLE_PHASE_DONE
                return "done"

    # Near-ball gate. A FRESH selection (not yet carrying) must already be within
    # possession range — dribble_to never walks; reaching the ball is
    # approach_ball's job. But once a carry is OPEN (CARRY/RELEASE) the held ball
    # rides a small offset off the body, and re-applying the tight possession
    # threshold every step killed the carry one step after the catch — the ball
    # was transported nothing (TRAINING.md §37: ~55/55 carries showed realized
    # gap-delta ≈ 0, all dying 1–2 steps after open). While carrying, only a
    # genuine LOSS — the ball breaking well clear of the body by more than
    # `carry_loss_margin` — aborts the segment.
    if state.phase in (DRIBBLE_PHASE_CARRY, DRIBBLE_PHASE_ALIGN_RELEASE):
        if robot_ball_dist > kickable_tolerance + carry_loss_margin:
            state.phase = DRIBBLE_PHASE_DONE
            return "done"
    elif (
        state.phase not in (
            DRIBBLE_PHASE_SETTLE, DRIBBLE_PHASE_VERIFY, DRIBBLE_PHASE_RELEASE
        )
        and robot_ball_dist > kickable_tolerance
    ):
        state.phase = DRIBBLE_PHASE_DONE
        return "done"

    # GRAB: catch direction is body-relative in rcssserver. Face the ball before
    # issuing `catch 0`; facing the target can put a wide-spawn ball outside the
    # catch rectangle even though it is within the proximity-only possession
    # threshold. Do not assume the command succeeded — VERIFY owns that decision.
    if state.phase == DRIBBLE_PHASE_GRAB:
        state.grab_steps += 1
        if state.grab_steps > max_grab_steps:
            state.phase = DRIBBLE_PHASE_DONE
            return "done"
        grab_angle_diff = normalize_angle(angle_to_ball - heading)
        if abs(grab_angle_diff) > angle_tolerance:
            return f"turn {grab_angle_diff / dt}"
        state.segment_start = None
        state.verify_robot_start = None
        state.verify_ball_start = None
        state.verify_game_count = None
        state.verify_steps = 0
        state.align_game_count = None
        state.align_steps = 0
        state.last_verify_robot_moved = None
        state.last_verify_ball_moved = None
        state.last_verify_offset_change = None
        state.catch_attempts += 1
        state.catch_game_count = game_count
        state.phase = DRIBBLE_PHASE_SETTLE
        return "catch 0"

    # SETTLE: the state returned immediately after `catch 0` may predate command
    # processing. Wait for a new simulator cycle before taking the verification
    # baseline, otherwise the catch teleport contaminates the displacement test.
    if state.phase == DRIBBLE_PHASE_SETTLE:
        if (
            game_count is not None
            and state.catch_game_count is not None
            and game_count == state.catch_game_count
        ):
            return "turn 0"
        state.verify_robot_start = (float(robot_xy[0]), float(robot_xy[1]))
        state.verify_ball_start = (float(ball_pose[0]), float(ball_pose[1]))
        state.verify_game_count = game_count
        state.verify_steps = 1
        state.phase = DRIBBLE_PHASE_VERIFY
        probe_direction = normalize_angle(angle_to_ball - heading + math.pi)
        return f"dash {abs(verify_probe_speed)} {probe_direction}"

    # VERIFY: proximity cannot distinguish a caught ball from a free ball beside
    # the robot. Probe away from the ball using positive directional dash power.
    # This avoids colliding with a free ball while working around this server's
    # MIN_DASH_POWER=0 clamp, which makes negative-power backward dash a no-op.
    # With catch glue, robot and ball displacement match and their offset stays
    # stable. With a catch fault, the robot moves away from a stationary ball.
    if state.phase == DRIBBLE_PHASE_VERIFY:
        if (
            game_count is not None
            and state.verify_game_count is not None
            and game_count == state.verify_game_count
        ):
            return "turn 0"

        robot_delta = robot_xy - _as_float_array(state.verify_robot_start)
        ball_delta = ball_pose - _as_float_array(state.verify_ball_start)
        robot_moved = float(np.linalg.norm(robot_delta))
        ball_moved = float(np.linalg.norm(ball_delta))
        offset_change = float(np.linalg.norm(ball_delta - robot_delta))
        state.last_verify_robot_moved = robot_moved
        state.last_verify_ball_moved = ball_moved
        state.last_verify_offset_change = offset_change
        catch_verified = (
            robot_moved >= verify_min_displacement
            and ball_moved >= verify_min_displacement
            and offset_change <= verify_offset_tolerance
        )
        if catch_verified:
            state.phase = DRIBBLE_PHASE_CARRY
            state.segment_start = (float(ball_pose[0]), float(ball_pose[1]))
            state.verify_robot_start = None
            state.verify_ball_start = None
            state.verify_game_count = None
            state.verify_steps = 0
        elif state.verify_steps < verify_probe_steps:
            state.verify_robot_start = (float(robot_xy[0]), float(robot_xy[1]))
            state.verify_ball_start = (float(ball_pose[0]), float(ball_pose[1]))
            state.verify_game_count = game_count
            state.verify_steps += 1
            probe_direction = normalize_angle(angle_to_ball - heading + math.pi)
            return f"dash {abs(verify_probe_speed)} {probe_direction}"
        else:
            state.phase = DRIBBLE_PHASE_RELEASE
            state.release_game_count = game_count
            state.retry_after_release = state.catch_attempts < max_catch_attempts
            state.release_at_limit = False
            return "drop"

    # CARRY: move toward the target with a directional dash while preserving body
    # orientation. Turning the body while caught makes rcssserver orbit the glued
    # ball around the player, including visible 180-degree sweeps.
    if state.phase == DRIBBLE_PHASE_CARRY:
        if state.segment_start is None:
            state.segment_start = (float(ball_pose[0]), float(ball_pose[1]))
        carried = float(np.linalg.norm(ball_pose - _as_float_array(state.segment_start)))
        if carried >= segment_limit:  # approaching the foul line: align, then release
            state.phase = DRIBBLE_PHASE_ALIGN_RELEASE
            state.align_game_count = None
            state.align_steps = 0
            state.release_at_limit = True
        else:
            angle_diff = normalize_angle(angle_to_target - heading)
            return f"dash {speed} {angle_diff}"

    # ALIGN_RELEASE: catch glue pins the ball in front of the body, but successful
    # catch also forces the body to face the ball. Directional CARRY therefore
    # transports correctly while leaving a poor sideways shooting pose. Perform
    # only this bounded end-of-segment alignment, then drop with the ball in front
    # toward the remaining target. Simulator-count gating prevents duplicate
    # observations from issuing repeated turns and recreating glued-ball circles.
    if state.phase == DRIBBLE_PHASE_ALIGN_RELEASE:
        if (
            game_count is not None
            and state.align_game_count is not None
            and game_count == state.align_game_count
        ):
            return "turn 0"
        angle_diff = normalize_angle(angle_to_target - heading)
        if abs(angle_diff) > angle_tolerance and state.align_steps < max_align_steps:
            state.align_steps += 1
            state.align_game_count = game_count
            return f"turn {angle_diff / dt}"
        state.phase = DRIBBLE_PHASE_RELEASE
        state.release_game_count = game_count
        state.retry_after_release = False
        state.align_game_count = None
        state.align_steps = 0
        return "drop"

    # RELEASE: keep macro ownership until at least one new simulator state proves
    # the `drop` command had a chance to clear server-side catch ownership. A
    # failed verification may then retry from GRAB; normal release terminates.
    if state.phase == DRIBBLE_PHASE_RELEASE:
        if (
            game_count is not None
            and state.release_game_count is not None
            and game_count == state.release_game_count
        ):
            return "turn 0"
        if state.retry_after_release and robot_ball_dist <= kickable_tolerance:
            state.phase = DRIBBLE_PHASE_GRAB
            state.release_game_count = None
            state.retry_after_release = False
            state.release_at_limit = False
            state.verify_robot_start = None
            state.verify_ball_start = None
            state.verify_game_count = None
            state.verify_steps = 0
            state.grab_steps = 0
            retry_angle_diff = normalize_angle(angle_to_ball - heading)
            if abs(retry_angle_diff) > angle_tolerance:
                return f"turn {retry_angle_diff / dt}"
            return "turn 0"
        state.phase = DRIBBLE_PHASE_DONE
        state.segment_start = None
        state.release_game_count = None
        state.retry_after_release = False
        state.release_at_limit = False
        return "done"

    # DONE or unknown phase: idle.
    return "done"

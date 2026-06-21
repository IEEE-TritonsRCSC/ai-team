"""Scripted goal-side container defender for 1-attacker-vs-defender+GK training.

Stage 3 pairs a single trained attacker against this hardcoded defender plus the
scripted goalkeeper. The defender is deliberately a *containment* defender, not an
aggressive ball-winner:

- CONTAIN (default): hold a point on the ball->goal line, a fixed stand-off
  goal-side of the ball, so the defender occupies the direct shooting lane and
  mirrors the ball's lateral moves. This forces the attacker to dribble to an
  angle the defender doesn't cover before shooting — exactly the skill Stage 3
  trains.
- CHALLENGE: only when the ball is genuinely loose (no attacker controlling it)
  AND very close, step onto it and clear it upfield (away from the defended
  goal). This punishes careless loss of possession without making the defender a
  ball magnet.

A moderate move speed and small challenge radius keep the defender beatable by a
good dribble. Shot blocking on the goal line is left to the goalkeeper; this
defender's job is to deny the easy central lane.
"""

from __future__ import annotations

import math
from typing import Tuple

from ai_interface.constants.field_constants import GOAL_R, GOAL_L
from ai_interface.constants.player_constants import (
    KICKABLE_MARGIN, PLAYER_SIZE, BALL_SIZE,
)
from ai_interface.player import Player
from ai_interface.utils import basic_commands
from ai_interface.utils.algo_utils import normalize_angle


# --- Tuning constants -------------------------------------------------------

# How far goal-side of the ball to hold the containing position.
CONTAIN_STANDOFF = 4.0
# Moderate (beatable) move speed — slower than the keeper's full-speed blocks so a
# good dribble can create separation.
CONTAIN_SPEED = 70.0
# Never sit deeper than this x; the deep mouth is the keeper's responsibility and
# overlapping the keeper would just double-cover the centre.
CONTAIN_X_MAX = 39.0
# Vertical band the defender patrols (the keeper covers the rest of the mouth).
CONTAIN_Y_LIMIT = 9.0
# Only leave containment to challenge a loose ball within this distance.
CHALLENGE_RADIUS = 2.5
# Ball is "loose" once the nearest attacker is at least this far from it.
LOOSE_BALL_MARGIN = 0.4

KICKABLE = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE


class Defender(Player):
    """A goal-side container defender driven once per simulator cycle."""

    def __init__(self, teamname: str, unum: int, side: str = "right") -> None:
        super().__init__(teamname=teamname, unum=unum)
        self.side = side
        # Goal we defend, and the goal we clear toward (the opposite one).
        self.goal = GOAL_R if side == "right" else GOAL_L
        self.clear_goal = GOAL_L if side == "right" else GOAL_R

    # --- Perception ---------------------------------------------------------

    def _nearest_attacker_ball_dist(self, ball_pos: Tuple[float, float], game_state) -> float:
        """Distance from the ball to the nearest non-teammate (attacker)."""
        bx, by = float(ball_pos[0]), float(ball_pos[1])
        best = math.inf
        for team, robots in getattr(game_state, "robot_poses", {}).items():
            if team == self.teamname:
                continue
            for robot in robots:
                if not isinstance(robot, dict):
                    continue
                for _unum, pose in robot.items():
                    if pose is None or len(pose) < 2:
                        continue
                    best = min(best, math.hypot(float(pose[0]) - bx, float(pose[1]) - by))
        return best

    # --- Command generation -------------------------------------------------

    def _move_to(self, self_pose_rad, tx: float, ty: float, game_state,
                 margin: float, face: Tuple[float, float] | None,
                 speed: float, full_speed: bool = False) -> str:
        """goto wrapper that maps the 'done' sentinel to an idle no-op."""
        theta = None
        if face is not None:
            theta = math.atan2(face[1] - self_pose_rad[1], face[0] - self_pose_rad[0])
        cmd = basic_commands.goto(
            self_pose_rad, tx, ty, game_state,
            margin=margin,
            theta=theta,
            speed=(100.0 if full_speed else speed),
            is_goalie=full_speed,           # skip deceleration scaling on a challenge
            obstacle_avoidance=not full_speed,
        )
        return cmd if cmd != "done" else "turn 0"

    def _clear(self, self_pose_rad, ball_pos: Tuple[float, float]) -> str:
        """Catch the ball then kick it clear toward the opposite goal.

        Uses Player.kick() which sequences: face ball → catch (dribbling=False),
        then turn to clear_angle → kick (dribbling=True), updating self.dribbling
        automatically each call.
        """
        dx, dy, _ = self_pose_rad
        clear_angle = math.atan2(self.clear_goal[1] - dy, self.clear_goal[0] - dx)
        return self.kick(clear_angle, self_pose_rad, ball_pos)

    def action(self, ball_pos: Tuple[float, float],
               defender_pose: Tuple[float, float, float],
               game_state=None) -> str:
        """Compute the defender command for this cycle.

        Args:
            ball_pos: (x, y) position of the ball.
            defender_pose: (x, y, theta_deg) of this defender.
            game_state: full game state (used for attacker detection / avoidance).

        Returns:
            A single command string ("dash …", "turn …", "kick …").
        """
        bx, by = float(ball_pos[0]), float(ball_pos[1])
        dx, dy, dtheta_deg = (
            float(defender_pose[0]), float(defender_pose[1]), float(defender_pose[2]),
        )
        self_pose_rad = (dx, dy, math.radians(dtheta_deg))
        def_to_ball = math.hypot(bx - dx, by - dy)

        # CHALLENGE: only contest a genuinely loose ball that is very close.
        ball_loose = (
            game_state is not None
            and self._nearest_attacker_ball_dist((bx, by), game_state) > KICKABLE + LOOSE_BALL_MARGIN
        )
        if ball_loose and def_to_ball < CHALLENGE_RADIUS:
            if def_to_ball <= KICKABLE:
                return self._clear(self_pose_rad, (bx, by))
            return self._move_to(
                self_pose_rad, bx, by, game_state,
                margin=0.2, face=(bx, by), speed=CONTAIN_SPEED, full_speed=True,
            )

        # Not challenging: clear any stale dribble state so the next _clear()
        # starts fresh with the face-ball→catch phase.
        self.dribbling = False

        # CONTAIN: hold a point on the ball->goal line, goal-side of the ball.
        gx, gy = float(self.goal[0]), float(self.goal[1])
        vx, vy = gx - bx, gy - by
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = vx / norm, vy / norm
        tx = bx + ux * CONTAIN_STANDOFF
        ty = by + uy * CONTAIN_STANDOFF
        # Stay strictly goal-side of the ball but out of the keeper's deep mouth.
        tx = min(max(tx, bx + 1.5), CONTAIN_X_MAX)
        ty = max(-CONTAIN_Y_LIMIT, min(CONTAIN_Y_LIMIT, ty))
        return self._move_to(
            self_pose_rad, tx, ty, game_state,
            margin=0.3, face=(bx, by), speed=CONTAIN_SPEED,
        )

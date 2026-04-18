"""Single-agent hierarchical state machine controller backed by hardcoded skills."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple
import math

import numpy as np

from ai_interface.constants.field_constants import FIELD_X, FIELD_Y
from ai_interface.constants.player_constants import BALL_SIZE, KICKABLE_MARGIN, PLAYER_SIZE
from ai_interface.player import Player
from ai_interface.utils.basic_commands import goto, shoot_at_goal


class FSMIntent(IntEnum):
    """High-level intent selected by PPO."""

    AUTO = 0
    FORCE_CHASE = 1
    FORCE_DRIBBLE = 2
    FORCE_SHOOT = 3
    FORCE_CLEAR = 4
    FORCE_REPOSITION = 5


class FSMState(IntEnum):
    """Low-level deterministic skill state."""

    CHASE_BALL = 0
    SECURE_POSSESSION = 1
    DRIBBLE_TO_GOAL = 2
    SHOOT_ON_GOAL = 3
    CLEAR_BALL = 4
    REPOSITION = 5


@dataclass
class FSMContext:
    """Compact world state for the deterministic controller."""

    ball_pos: Tuple[float, float]
    ball_vel: Tuple[float, float]
    self_pose_rad: Tuple[float, float, float]
    self_pose_deg: Tuple[float, float, float]
    ball_dist: float
    has_ball: bool
    goal_dist: float


class SingleAgentHSMController:
    """Deterministic FSM skill router used under PPO high-level intents."""

    def __init__(self, team_name: str, unum: int = 1):
        self.team_name = team_name
        self.unum = int(unum)
        self.player = Player(team_name, self.unum)
        self.kickable_dist = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE

        self.goal_x = float(FIELD_X[1])
        self.state = FSMState.CHASE_BALL
        self.last_intent = FSMIntent.AUTO

    def reset(self) -> None:
        self.state = FSMState.CHASE_BALL
        self.last_intent = FSMIntent.AUTO
        self.player.reset()

    def step(self, intent: int, context: FSMContext, game_state) -> Tuple[str, FSMState]:
        intent_enum = FSMIntent(int(intent)) if int(intent) in set(i.value for i in FSMIntent) else FSMIntent.AUTO
        self.last_intent = intent_enum
        self.state = self._next_state(intent_enum, context)
        command = self._state_to_command(self.state, context, game_state)
        return command, self.state

    def _next_state(self, intent: FSMIntent, ctx: FSMContext) -> FSMState:
        if intent == FSMIntent.FORCE_CHASE:
            return FSMState.CHASE_BALL
        if intent == FSMIntent.FORCE_DRIBBLE:
            return FSMState.DRIBBLE_TO_GOAL if ctx.has_ball else FSMState.CHASE_BALL
        if intent == FSMIntent.FORCE_SHOOT:
            return FSMState.SHOOT_ON_GOAL if ctx.has_ball else FSMState.SECURE_POSSESSION
        if intent == FSMIntent.FORCE_CLEAR:
            return FSMState.CLEAR_BALL if ctx.has_ball else FSMState.SECURE_POSSESSION
        if intent == FSMIntent.FORCE_REPOSITION:
            return FSMState.REPOSITION

        # AUTO transition logic for stable, reusable tactical behavior.
        if not ctx.has_ball:
            if ctx.ball_dist > 1.8:
                return FSMState.CHASE_BALL
            return FSMState.SECURE_POSSESSION

        if ctx.goal_dist < 16.0:
            return FSMState.SHOOT_ON_GOAL
        if ctx.self_pose_rad[0] < 0.0:
            return FSMState.DRIBBLE_TO_GOAL
        return FSMState.DRIBBLE_TO_GOAL

    def _state_to_command(self, state: FSMState, ctx: FSMContext, game_state) -> str:
        if state == FSMState.CHASE_BALL:
            return self._cmd_chase(ctx, game_state)
        if state == FSMState.SECURE_POSSESSION:
            return self._cmd_secure(ctx, game_state)
        if state == FSMState.DRIBBLE_TO_GOAL:
            return self._cmd_dribble(ctx)
        if state == FSMState.SHOOT_ON_GOAL:
            return self._cmd_shoot(ctx)
        if state == FSMState.CLEAR_BALL:
            return self._cmd_clear(ctx)
        return self._cmd_reposition(ctx, game_state)

    def _cmd_chase(self, ctx: FSMContext, game_state) -> str:
        bx, by = ctx.ball_pos
        vx, vy = ctx.ball_vel
        lead = 2.5 if ctx.ball_dist > 2.5 else 1.0
        tx = float(np.clip(bx + lead * vx, FIELD_X[0] + 1.0, FIELD_X[1] - 1.0))
        ty = float(np.clip(by + lead * vy, FIELD_Y[0] + 1.0, FIELD_Y[1] - 1.0))
        if game_state is None:
            heading = math.atan2(ty - ctx.self_pose_rad[1], tx - ctx.self_pose_rad[0])
            angle_diff = heading - ctx.self_pose_rad[2]
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            if abs(angle_diff) > math.radians(8.0):
                return f"turn {angle_diff:.4f}"
            return "dash 85.0 0.0"
        cmd = goto(ctx.self_pose_rad, tx, ty, game_state, margin=0.15, theta=None, speed=90.0)
        return None if cmd in ("done", "failed") else cmd

    def _cmd_secure(self, ctx: FSMContext, game_state) -> str:
        bx, by = ctx.ball_pos
        rx, ry, _ = ctx.self_pose_rad
        to_goal = np.array([self.goal_x - bx, -by], dtype=float)
        norm = float(np.linalg.norm(to_goal)) + 1e-8
        behind = np.array([bx, by], dtype=float) - 0.7 * to_goal / norm
        tx = float(np.clip(behind[0], FIELD_X[0] + 1.0, FIELD_X[1] - 1.0))
        ty = float(np.clip(behind[1], FIELD_Y[0] + 1.0, FIELD_Y[1] - 1.0))

        face = math.atan2(by - ry, bx - rx)
        if game_state is None:
            angle_diff = face - ctx.self_pose_rad[2]
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            if abs(angle_diff) > math.radians(6.0):
                return f"turn {angle_diff:.4f}"
            return "dash 65.0 0.0"
        cmd = goto(ctx.self_pose_rad, tx, ty, game_state, margin=0.1, theta=face, speed=75.0)
        return None if cmd in ("done", "failed") else cmd

    def _cmd_dribble(self, ctx: FSMContext) -> str:
        bx, by = ctx.ball_pos
        rx, ry, _ = ctx.self_pose_rad
        target = (self.goal_x, 0.0)
        cmd = self.player.shoot_at_goal(target, ctx.self_pose_rad, (bx, by), kick_power=35)
        if cmd == "failed":
            angle = math.atan2(by - ry, bx - rx)
            angle_diff = angle - ctx.self_pose_rad[2]
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            return f"turn {angle_diff:.4f}"
        return cmd

    def _cmd_shoot(self, ctx: FSMContext) -> str:
        bx, by = ctx.ball_pos
        goal_corner = (self.goal_x, -3.3 if by > 0 else 3.3)
        self_pose_arr = np.asarray(ctx.self_pose_rad, dtype=float)
        ball_pose_arr = np.asarray((bx, by), dtype=float)
        goal_corner_arr = np.asarray(goal_corner, dtype=float)
        cmd = shoot_at_goal(self_pose_arr, ball_pose_arr, goal_corner_arr, kick_power=100.0)
        if cmd == "failed":
            return self._cmd_dribble(ctx)
        return cmd

    def _cmd_clear(self, ctx: FSMContext) -> str:
        bx, by = ctx.ball_pos
        clear_target = (self.goal_x, 0.0)
        self_pose_arr = np.asarray(ctx.self_pose_rad, dtype=float)
        ball_pose_arr = np.asarray((bx, by), dtype=float)
        clear_target_arr = np.asarray(clear_target, dtype=float)
        cmd = shoot_at_goal(self_pose_arr, ball_pose_arr, clear_target_arr, kick_power=100.0)
        if cmd == "failed":
            cmd = self.player.shoot_at_goal(clear_target, ctx.self_pose_rad, (bx, by), kick_power=90)
        return None if cmd in ("done", "failed") else cmd

    def _cmd_reposition(self, ctx: FSMContext, game_state) -> str:
        bx, by = ctx.ball_pos
        tx = float(np.clip(bx - 2.5, FIELD_X[0] + 1.0, FIELD_X[1] - 1.0))
        ty = float(np.clip(0.5 * by, FIELD_Y[0] + 1.0, FIELD_Y[1] - 1.0))
        if game_state is None:
            heading = math.atan2(ty - ctx.self_pose_rad[1], tx - ctx.self_pose_rad[0])
            angle_diff = heading - ctx.self_pose_rad[2]
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            if abs(angle_diff) > math.radians(8.0):
                return f"turn {angle_diff:.4f}"
            return "dash 60.0 0.0"
        cmd = goto(ctx.self_pose_rad, tx, ty, game_state, margin=0.2, theta=0.0, speed=70.0)
        return None if cmd in ("done", "failed") else cmd

    def build_context(self, game_state, prev_game_state) -> Optional[FSMContext]:
        if game_state is None:
            return None

        ball = game_state.ball_pos or (0.0, 0.0)
        ball_vel = self._estimate_ball_velocity(game_state, prev_game_state)

        pose_deg = self._extract_self_pose(game_state)
        if pose_deg is None:
            return None

        rx, ry, theta_deg = pose_deg
        theta_rad = math.radians(theta_deg)
        bx, by = float(ball[0]), float(ball[1])
        ball_dist = math.hypot(bx - rx, by - ry)

        pose_rad = (rx, ry, theta_rad)
        has_ball = self.player.hasBall(pose_rad, (bx, by), check_angle=False)
        goal_dist = math.hypot(self.goal_x - rx, -ry)

        return FSMContext(
            ball_pos=(bx, by),
            ball_vel=ball_vel,
            self_pose_rad=pose_rad,
            self_pose_deg=(rx, ry, theta_deg),
            ball_dist=ball_dist,
            has_ball=bool(has_ball),
            goal_dist=goal_dist,
        )

    def _extract_self_pose(self, game_state) -> Optional[Tuple[float, float, float]]:
        team_poses = getattr(game_state, "robot_poses", {}).get(self.team_name, [])
        if not team_poses:
            return None

        for pose_dict in team_poses:
            if self.unum in pose_dict:
                x, y, theta = pose_dict[self.unum]
                return float(x), float(y), float(theta)

        first = team_poses[0]
        if isinstance(first, dict):
            unum = int(next(iter(first.keys())))
            x, y, theta = first[unum]
            return float(x), float(y), float(theta)
        return None

    @staticmethod
    def _estimate_ball_velocity(game_state, prev_game_state) -> Tuple[float, float]:
        ball_vel = getattr(game_state, "ball_vel", None)
        if ball_vel is not None:
            return float(ball_vel[0]), float(ball_vel[1])

        if prev_game_state is None:
            return 0.0, 0.0

        p = getattr(prev_game_state, "ball_pos", None)
        c = getattr(game_state, "ball_pos", None)
        if p is None or c is None:
            return 0.0, 0.0

        return float(c[0] - p[0]), float(c[1] - p[1])

    @property
    def intent_count(self) -> int:
        return len(FSMIntent)

    @property
    def state_count(self) -> int:
        return len(FSMState)

    def state_dict(self) -> Dict[str, int]:
        return {
            "state": int(self.state),
            "intent": int(self.last_intent),
        }

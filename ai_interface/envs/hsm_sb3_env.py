"""Single-agent hierarchical FSM environment for SB3 PPO training."""

from __future__ import annotations

from typing import Optional, Tuple
import logging
import math
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ai_interface.constants.field_constants import FIELD_X, FIELD_Y
from ai_interface.constants.player_constants import BALL_SIZE, KICKABLE_MARGIN, PLAYER_SIZE
from ai_interface.hsm.single_agent_fsm import FSMIntent, FSMState, SingleAgentHSMController
from networking.networker import Networker


class HSMSingleAgentEnv(gym.Env):
    """PPO trains high-level intents while deterministic FSM issues simulator commands."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        networker: Networker,
        team_name: str,
        obs_dim: int = 28,
        max_steps: int = 200,
        unum: int = 1,
        debug: bool = False,
    ):
        super().__init__()
        self.networker = networker
        self.team_name = team_name
        self.obs_dim = int(obs_dim)
        self.max_steps = int(max_steps)
        self.unum = int(unum)

        self.logger = logging.getLogger(f"HSMSingleAgentEnv[{team_name}]")
        if debug:
            self.logger.setLevel(logging.DEBUG)

        self.controller = SingleAgentHSMController(team_name=team_name, unum=unum)
        self.action_space = spaces.Discrete(self.controller.intent_count)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.kickable_dist = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE
        self.current_step = 0
        self.prev_game_state = None

        self.prev_ball_dist = None
        self.prev_ball_to_goal_dist = None
        self.prev_robot_pos = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        try:
            self.networker.reset_sim()
        except Exception:
            pass

        time.sleep(0.25)
        for _ in range(3):
            try:
                self.networker.get_game_state()
            except Exception:
                break
            time.sleep(0.05)

        game_state = self.networker.get_game_state()
        self.current_step = 0
        self.controller.reset()
        self.prev_game_state = game_state
        self.prev_ball_dist = None
        self.prev_ball_to_goal_dist = None
        self.prev_robot_pos = None

        obs = self._build_observation(game_state)
        return obs, {}

    def step(self, action: int):
        game_state_before = self.networker.get_game_state()
        game_state_before = self._recover_play_on(game_state_before)
        context = self.controller.build_context(game_state_before, self.prev_game_state)

        command = None
        state = FSMState.CHASE_BALL
        if context is not None:
            command, state = self.controller.step(action, context, game_state_before)

        try:
            if command:
                self.networker.execute_ai_output([command], self.team_name)
        except Exception:
            pass

        time.sleep(0.1)
        game_state = self.networker.get_game_state()
        game_state = self._recover_play_on(game_state)

        reward, goal_scored, robot_oob, ball_oob = self._compute_reward(game_state, int(action), state)

        self.current_step += 1
        terminated = bool(goal_scored or robot_oob or ball_oob)
        truncated = bool(self.current_step >= self.max_steps)

        self.prev_game_state = game_state
        obs = self._build_observation(game_state)
        info = {
            "fsm_state": FSMState(state).name,
            "intent": FSMIntent(int(action)).name if int(action) in set(i.value for i in FSMIntent) else "AUTO",
            "command": command,
            "goal_scored": goal_scored,
            "robot_out_of_bounds": robot_oob,
            "ball_out_of_bounds": ball_oob,
            "step": self.current_step,
        }

        return obs, float(reward), terminated, truncated, info

    def _recover_play_on(self, game_state):
        """Restart simulator match if playmode drifts away from active play."""
        playmode = getattr(game_state, "playmode", None) if game_state is not None else None
        if not playmode or str(playmode).startswith("play_on"):
            return game_state

        watcher = getattr(self.networker, "game_watcher", None)
        restart_fn = getattr(watcher, "restart_game", None)
        if callable(restart_fn):
            try:
                restart_fn()
                time.sleep(0.05)
                refreshed = self.networker.get_game_state()
                if refreshed is not None:
                    return refreshed
            except Exception:
                pass
        return game_state

    def _build_observation(self, game_state) -> np.ndarray:
        if game_state is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        context = self.controller.build_context(game_state, self.prev_game_state)
        if context is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        bx, by = context.ball_pos
        bvx, bvy = context.ball_vel
        rx, ry, rtheta = context.self_pose_rad

        goal_dx = FIELD_X[1] - rx
        goal_dy = -ry
        goal_dist = math.hypot(goal_dx, goal_dy)
        goal_angle = math.atan2(goal_dy, goal_dx)
        goal_angle_diff = math.atan2(math.sin(goal_angle - rtheta), math.cos(goal_angle - rtheta))

        ball_dx = bx - rx
        ball_dy = by - ry
        ball_angle = math.atan2(ball_dy, ball_dx)
        ball_angle_diff = math.atan2(math.sin(ball_angle - rtheta), math.cos(ball_angle - rtheta))

        robot_vx, robot_vy = 0.0, 0.0
        if self.prev_robot_pos is not None:
            robot_vx = rx - self.prev_robot_pos[0]
            robot_vy = ry - self.prev_robot_pos[1]

        base = np.array(
            [
                ball_dx,
                ball_dy,
                context.ball_dist,
                bvx,
                bvy,
                float(context.has_ball),
                goal_dx,
                goal_dy,
                goal_dist,
                math.cos(rtheta),
                math.sin(rtheta),
                ball_angle_diff,
                goal_angle_diff,
                rx,
                ry,
                robot_vx,
                robot_vy,
            ],
            dtype=np.float32,
        )

        fsm_onehot = np.zeros(self.controller.state_count, dtype=np.float32)
        fsm_onehot[int(self.controller.state)] = 1.0

        intent_onehot = np.zeros(self.controller.intent_count, dtype=np.float32)
        intent_onehot[int(self.controller.last_intent)] = 1.0

        obs = np.concatenate([base, fsm_onehot, intent_onehot], axis=0)
        if obs.shape[0] < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]))
        else:
            obs = obs[: self.obs_dim]

        self.prev_robot_pos = (rx, ry)
        return obs.astype(np.float32)

    def _compute_reward(
        self,
        game_state,
        action: int,
        state: FSMState,
    ) -> Tuple[float, bool, bool, bool]:
        if game_state is None:
            return -0.1, False, False, False

        context = self.controller.build_context(game_state, self.prev_game_state)
        if context is None:
            return -0.1, False, False, False

        bx, by = context.ball_pos
        rx, ry, _ = context.self_pose_rad
        ball_dist = context.ball_dist
        ball_to_goal_dist = math.hypot(FIELD_X[1] - bx, -by)

        reward = 0.0

        if self.prev_ball_dist is not None:
            approach = np.clip(self.prev_ball_dist - ball_dist, -0.5, 0.5)
            reward += float(approach) * 1.8

        if self.prev_ball_to_goal_dist is not None:
            progress = np.clip(self.prev_ball_to_goal_dist - ball_to_goal_dist, -0.4, 0.4)
            reward += float(progress) * 3.0

        if context.has_ball:
            reward += 0.8
        if ball_dist < self.kickable_dist * 1.5:
            reward += 0.2

        if state == FSMState.SHOOT_ON_GOAL:
            reward += 0.25
        if state == FSMState.DRIBBLE_TO_GOAL:
            reward += 0.15

        if int(action) == FSMIntent.FORCE_SHOOT and not context.has_ball:
            reward -= 0.2

        goal_scored = bx >= FIELD_X[1] and abs(by) < 3.66
        if goal_scored:
            reward += 70.0

        robot_oob = abs(rx) > FIELD_X[1] or abs(ry) > FIELD_Y[1]
        ball_oob = abs(by) > FIELD_Y[1] or bx < FIELD_X[0]
        if robot_oob:
            reward -= 3.0
        if ball_oob:
            reward -= 1.5

        reward += 0.03

        self.prev_ball_dist = ball_dist
        self.prev_ball_to_goal_dist = ball_to_goal_dist

        return float(reward), bool(goal_scored), bool(robot_oob), bool(ball_oob)

    def close(self):
        try:
            self.networker.shutdown()
        except Exception:
            pass

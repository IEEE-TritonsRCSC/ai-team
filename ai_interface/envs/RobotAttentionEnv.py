"""Robot attention environment."""

from __future__ import annotations

# Issues to note:
# 1. Observations currently include only teammate poses and the ball in each
#    robot's ego frame. If you need opponent awareness, extend the observation
#    layout explicitly rather than reusing the old global encoding.
# 2. Past 5 planned actions per robot are intentionally deferred for now. A
#    future update should add a fixed-size action-history encoding to the
#    observation.

from typing import Optional, Tuple, Dict, Any, List
import logging
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import obs_as_tensor

from ai_interface.utils.algo_utils import has_ball
from ai_interface.constants.field_constants import *
from ai_interface.constants.player_constants import *
from ai_interface.envs.reward import RewardInputs, evaluate_reward, extract_opponent_positions
from networking.networker import Networker
from networking.data_utils import GameState


class RobotAttentionEnv(gym.Env):
    def __init__(
        self, 
        networker: Networker,  
        team_name: str,
        robot_ids: Optional[List[int]] = None,
        opponent_model: Optional[BaseAlgorithm] = None,
        opponent_name: Optional[str] = None,
        max_steps: int = 200,
        debug: bool = False,
        time_window: int = 1,
        state_retry_count: int = 10,
        state_retry_sleep_s: float = 0.02,
        none_state_warn_every: int = 50,
        position_noise_std: float = 0.05,
        invalid_action_penalty: float = 0.2,
        some_arg=None
        ):
        
        super().__init__()
        
        self.networker = networker
        self.team_name = team_name
        self.opponent_model = opponent_model
        self.opponent_name = opponent_name
        if robot_ids is None:
            robot_ids = [k for k in range(1, 7)]
        self.robot_ids = list(robot_ids)
        self.num_robots = len(self.robot_ids)
        self.max_steps = max_steps
        self.debug = debug
        self.time_window = max(1, int(time_window))
        self.state_retry_count = max(0, int(state_retry_count))
        self.state_retry_sleep_s = max(0.0, float(state_retry_sleep_s))
        self.none_state_warn_every = max(1, int(none_state_warn_every))
        self.position_noise_std = float(position_noise_std)
        self.invalid_action_penalty = max(0.0, float(invalid_action_penalty))
        
        
        # Create a logger for this environment
        self.logger = logging.getLogger(f"RobotAttentionEnv[{team_name}]")
        if self.debug:
            self.logger.setLevel(logging.DEBUG)

        # Observation design per ego robot:
        # [ball_rel_x, ball_rel_y] +
        # [other_rel_x, other_rel_y, cos(other_rel_theta), sin(other_rel_theta)]
        # for each other robot.
        # Observations are returned as stacked frame histories with shape
        # (num_robots, obs_dim_per_robot, time_window).
        # Legacy obs_dim args stay in the signature for compatibility, but the
        # actual observation shape is derived from num_robots.
        self.obs_dim_per_robot = 2 + 4 * max(0, self.num_robots - 1)
        self.non_robot_obs_dim = 0
        self.obs_frame_dim = self.num_robots * self.obs_dim_per_robot
        self.obs_dim = self.obs_frame_dim * self.time_window
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_robots, self.obs_dim_per_robot, self.time_window),
            dtype=np.float32
            )
        self.obs_frame_buffer_by_team: Dict[str, List[np.ndarray]] = {}
        
        # Action design per robot (10D):
        # [dash_logit, turn_logit, kick_logit, start_dribble_logit,
        #  stop_dribble_logit, dash_power, dash_cos, dash_sin, turn_cos, turn_sin]
        self.action_dim_per_robot = 10
        action_low = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            dtype=np.float32,
        )
        action_high = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.tile(action_low, self.num_robots),
            high=np.tile(action_high, self.num_robots),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.episode_num = 0
        
        # Field dimensions (from your existing code)
        self.field_half_width = 45.0
        self.field_half_height = 30.0

        self.kickable_dist = KICKABLE_MARGIN + BALL_SIZE + PLAYER_SIZE

        self.prev_reward_ball_dist_by_id: Dict[int, float] = {}
        self.prev_reward_ball_to_goal_dist: Optional[float] = None

        # Statistics
        self.total_rewards = 0.0
        self.episode_actions = []  # Track action distribution
        self._none_state_counter = 0
        self.obs_frame_buffer_by_team = self._init_obs_frame_buffers()
        
        self.logger.info(
            f"Initialized RobotAttentionEnv for team '{team_name}' with robots {robot_ids}, "
            f"num_robots={self.num_robots}, obs_dim={self.obs_dim}, action_dim={self.action_space.shape[0]}"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
        ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial observation tensor with shape
                (num_robots, obs_dim_per_robot, time_window)
            info: Additional information dictionary
        """
        
        
        super().reset(seed=seed)
        self.networker.reset_sim()
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_num += 1
        self.total_rewards = 0.0
        self.episode_actions = []
        self._none_state_counter = 0
        
        self.prev_reward_ball_dist_by_id = {}
        self.prev_reward_ball_to_goal_dist = None
        self.obs_frame_buffer_by_team = self._init_obs_frame_buffers()

        # Get initial game state from simulator
        game_state = self._get_game_state(
            retries=max(self.state_retry_count, 20),
            sleep_s=self.state_retry_sleep_s,
        )
        
        # Build initial observation
        obs = self._game_state_to_obs(game_state, team_name=self.team_name)

        
        info = {
            "episode_num": self.episode_num,
            "step": self.current_step
        }
        
        if self.debug:
            self.logger.debug(f"Episode {self.episode_num} started")
        
        return obs, info

    def step(
        self, 
        action: np.ndarray
        ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Joint action vector with shape (10 * num_robots,).
        
        Returns:
            observation: Next observation tensor with shape
                (num_robots, obs_dim_per_robot, time_window)
            reward: Reward for this step
            terminated: Whether episode ended (goal/out-of-bounds)
            truncated: Whether episode hit max steps
            info: Additional information
        """
        self.current_step += 1

        # Use current state to validate ball-dependent commands.
        current_game_state = self._get_game_state(
            retries=self.state_retry_count,
            sleep_s=self.state_retry_sleep_s,
        )

        # Decode action vector into simulator commands
        commands, action_info = self._action_to_commands(
            action,
            current_game_state,
            team_name=self.team_name,
        )

        # Track action distribution for debugging
        self.episode_actions.append(action_info["action_type"])

        opponent_action_info = None
        opponent_commands: List[str] = []
        if self.opponent_name:
            opponent_game_state = self._get_game_state(
                retries=self.state_retry_count,
                sleep_s=self.state_retry_sleep_s,
            )
            opponent_obs = self._game_state_to_obs(
                opponent_game_state, team_name=self.opponent_name
            )
            opponent_action = action
            if self.opponent_model is not None:
                opponent_action = self._predict_opponent_action(opponent_obs)
            opponent_commands, opponent_action_info = self._action_to_commands(
                opponent_action,
                opponent_game_state,
                team_name=self.opponent_name,
            )

        # Send commands to simulator
        self._send_commands(
            commands,
            team_name=self.team_name,
        )
        if opponent_commands:
            self._send_commands(
                opponent_commands,
                team_name=self.opponent_name,
            )
        
        # Read next state after sending commands, then build next observation.
        next_game_state = self._get_game_state(
            retries=self.state_retry_count,
            sleep_s=self.state_retry_sleep_s,
        )
        obs = self._game_state_to_obs(next_game_state, team_name=self.team_name)

        reward = self._calculate_reward(next_game_state)
        invalid_action_count = int(action_info.get("invalid_action_count", 0))
        if invalid_action_count > 0 and self.invalid_action_penalty > 0.0:
            reward -= self.invalid_action_penalty * invalid_action_count
        self.total_rewards += reward
        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {
            "action_info": action_info,
            "opponent_action_info": opponent_action_info,
            "reward": reward,
            "total_reward": self.total_rewards,
            "invalid_action_count": invalid_action_count,
        }
        return obs, reward, terminated, truncated, info
    
    
    def _get_game_state(self, retries: int = 0, sleep_s: float = 0.0) -> Optional[GameState]:
        """
        Get current game state from networker.
        
        Returns:
            Game state dictionary or None if unavailable
        """
        attempts = max(0, int(retries)) + 1
        delay = max(0.0, float(sleep_s))

        for attempt in range(attempts):
            try:
                game_state = self.networker.get_game_state()
                if game_state is not None:
                    return game_state
            except Exception as e:
                if attempt == attempts - 1:
                    self.logger.error(f"Error getting game state: {e}")

            if attempt < attempts - 1 and delay > 0.0:
                time.sleep(delay)

        return None

    def _preprocess_game_state(self, game_state: GameState) -> GameState:
        """
        Add observation noise to positional fields before the state is consumed.

        Noise is applied to x/y positions only. Robot headings are intentionally
        left unchanged.
        """
        if game_state is None or self.position_noise_std <= 0.0:
            return game_state

        try:
            ball_pos = game_state.ball_pos
            noisy_ball_pos = ball_pos
            if ball_pos is not None and len(ball_pos) >= 2:
                ball_noise = self.np_random.normal(0.0, self.position_noise_std, size=2)
                noisy_ball_pos = (
                    float(ball_pos[0]) + float(ball_noise[0]),
                    float(ball_pos[1]) + float(ball_noise[1]),
                    *ball_pos[2:],
                )

            robot_poses = game_state.robot_poses
            noisy_robot_poses = {}
            for team_name, team_pose_entries in robot_poses.items():
                noisy_entries = []
                for entry in team_pose_entries:
                    if not isinstance(entry, dict):
                        noisy_entries.append(entry)
                        continue

                    noisy_entry = {}
                    for robot_id, pose in entry.items():
                        if pose is None or len(pose) < 2:
                            noisy_entry[robot_id] = pose
                            continue

                        pose_noise = self.np_random.normal(0.0, self.position_noise_std, size=2)
                        noisy_entry[robot_id] = (
                            float(pose[0]) + float(pose_noise[0]),
                            float(pose[1]) + float(pose_noise[1]),
                            *pose[2:],
                        )
                    noisy_entries.append(noisy_entry)
                noisy_robot_poses[team_name] = noisy_entries

            return game_state._replace(
                ball_pos=noisy_ball_pos,
                robot_poses=noisy_robot_poses,
            )
        except Exception as e:
            self.logger.error(f"Error preprocessing game state: {e}")
            return game_state

    @staticmethod
    def _world_to_ego(dx: float, dy: float, ego_theta: float) -> Tuple[float, float]:
        """Rotate a world-frame offset into the ego robot's frame."""

        cos_theta = float(np.cos(ego_theta))
        sin_theta = float(np.sin(ego_theta))
        rel_x = cos_theta * dx + sin_theta * dy
        rel_y = -sin_theta * dx + cos_theta * dy
        return rel_x, rel_y

    @staticmethod
    def _clockwise_deg_to_math_rad(angle_deg: float) -> float:
        """
        Convert simulator/game headings into standard math radians.

        Game convention:
        - x+ is right
        - y+ is up
        - 0 degrees points right
        - positive angles are clockwise

        Internal convention:
        - 0 radians points right
        - positive angles are counterclockwise
        """

        return float(-np.deg2rad(angle_deg))

    def _init_obs_frame_buffers(self) -> Dict[str, List[np.ndarray]]:
        """Create empty temporal buffers for each tracked team name."""

        buffer_by_team: Dict[str, List[np.ndarray]] = {self.team_name: []}
        if self.opponent_name:
            buffer_by_team[self.opponent_name] = []
        return buffer_by_team

    def _stack_obs_frames(self, obs_frame: np.ndarray, team_name: str) -> np.ndarray:
        """Append one per-frame observation and return a time-stacked tensor."""

        obs_frame = np.asarray(obs_frame, dtype=np.float32)
        team_buffer = self.obs_frame_buffer_by_team.setdefault(team_name, [])
        team_buffer.append(obs_frame)
        if len(team_buffer) > self.time_window:
            team_buffer.pop(0)

        earliest_frame = team_buffer[0]
        if len(team_buffer) < self.time_window:
            pad_count = self.time_window - len(team_buffer)
            frames = [earliest_frame] * pad_count + list(team_buffer)
        else:
            frames = list(team_buffer)

        stacked = np.stack(frames, axis=-1)
        return stacked.astype(np.float32, copy=False)

    def _predict_opponent_action(self, obs: np.ndarray) -> np.ndarray:
        """Query ``opponent_model.policy`` without tracking gradients."""

        if self.opponent_model is None:
            raise RuntimeError("opponent_model is not configured")

        policy = getattr(self.opponent_model, "policy", None)
        if policy is None:
            raise RuntimeError("opponent_model does not expose a policy")

        was_training = bool(policy.training)
        try:
            policy.set_training_mode(False)
            with th.no_grad():
                obs_tensor = obs_as_tensor(obs, self.opponent_model.device)
                if obs_tensor.dim() == len(self.observation_space.shape):
                    obs_tensor = obs_tensor.unsqueeze(0)
                actions, _, _ = policy(obs_tensor)
        finally:
            policy.set_training_mode(was_training)

        actions = actions.cpu().numpy()
        if actions.shape[0] == 1:
            actions = actions[0]

        if isinstance(self.action_space, spaces.Box):
            if policy.squash_output:
                actions = policy.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        return np.asarray(actions, dtype=np.float32)

    def _game_state_to_obs(self, game_state, team_name: str) -> np.ndarray:
        """
        Convert game state to an ego-centric observation tensor.

        Per-frame layout:
        - Per ego robot in self.robot_ids order:
            - [ball_rel_x, ball_rel_y]
            - [other_rel_x, other_rel_y, cos(other_rel_theta), sin(other_rel_theta)]
              for each other robot
        
        Args:
            game_state: GameState object from networker
            team_name: Team name whose controlled robots define the ego observations
        
        Returns:
            Observation tensor (num_robots, obs_dim_per_robot, time_window)
        """
        if game_state is None:
            self._none_state_counter += 1
            if (
                self._none_state_counter == 1
                or self._none_state_counter % self.none_state_warn_every == 0
            ):
                self.logger.warning(
                    "Game state is None, returning zero observation "
                    f"(count={self._none_state_counter})"
                )
            zero_frame = np.zeros(
                (self.num_robots, self.obs_dim_per_robot), dtype=np.float32
            )
            return self._stack_obs_frames(zero_frame, team_name=team_name)

        self._none_state_counter = 0
        
        try:
            game_state = self._preprocess_game_state(game_state)

            ball_pos = game_state.ball_pos
            ball_x = float(ball_pos[0]) if ball_pos is not None and len(ball_pos) >= 2 else 0.0
            ball_y = float(ball_pos[1]) if ball_pos is not None and len(ball_pos) >= 2 else 0.0

            team_pose_entries = game_state.robot_poses.get(team_name, [])
            pose_by_robot_id = {}
            for entry in team_pose_entries:
                if isinstance(entry, dict):
                    pose_by_robot_id.update(entry)

            obs_values: list[list[float]] = []
            for ego_robot_id in self.robot_ids:
                ego_pose = pose_by_robot_id.get(ego_robot_id)

                if ego_pose is None or len(ego_pose) < 3:
                    obs_values.append([0.0] * self.obs_dim_per_robot)
                    continue

                ego_x = float(ego_pose[0])
                ego_y = float(ego_pose[1])
                ego_theta = self._clockwise_deg_to_math_rad(float(ego_pose[2]))

                robot_obs: list[float] = []
                ball_rel_x, ball_rel_y = self._world_to_ego(
                    dx=ball_x - ego_x,
                    dy=ball_y - ego_y,
                    ego_theta=ego_theta,
                )
                robot_obs.extend([ball_rel_x, ball_rel_y])

                for other_robot_id in self.robot_ids:
                    if other_robot_id == ego_robot_id:
                        continue

                    other_pose = pose_by_robot_id.get(other_robot_id)
                    if other_pose is None or len(other_pose) < 3:
                        robot_obs.extend([0.0, 0.0, 0.0, 0.0])
                        continue

                    other_x = float(other_pose[0])
                    other_y = float(other_pose[1])
                    other_theta = self._clockwise_deg_to_math_rad(float(other_pose[2]))
                    other_rel_x, other_rel_y = self._world_to_ego(
                        dx=other_x - ego_x,
                        dy=other_y - ego_y,
                        ego_theta=ego_theta,
                    )
                    other_rel_theta = other_theta - ego_theta
                    robot_obs.extend(
                        [
                            other_rel_x,
                            other_rel_y,
                            float(np.cos(other_rel_theta)),
                            float(np.sin(other_rel_theta)),
                        ]
                    )
                obs_values.append(robot_obs)

            obs_frame = np.array(obs_values, dtype=np.float32)

            if obs_frame.shape != (self.num_robots, self.obs_dim_per_robot):
                if obs_frame.size == 0:
                    obs_frame = np.zeros(
                        (self.num_robots, self.obs_dim_per_robot), dtype=np.float32
                    )
                else:
                    obs_frame = obs_frame.reshape(self.num_robots, self.obs_dim_per_robot)

            return self._stack_obs_frames(obs_frame, team_name=team_name)
            
        except Exception as e:
            self.logger.error(f"Error building observation: {e}")
            zero_frame = np.zeros(
                (self.num_robots, self.obs_dim_per_robot), dtype=np.float32
            )
            return self._stack_obs_frames(zero_frame, team_name=team_name)

    def _action_to_commands(
        self,
        action: np.ndarray,
        game_state: Optional[GameState],
        team_name: str,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Convert action vector to simulator command strings.
        
        Action vector format (10*number of robots) Dimensions:
        [0]: dash_logit
        [1]: turn_logit
        [2]: kick_logit
        [3]: start_dribble_logit
        [4]: stop_dribble_logit
        [5]: dash_power  (range: [0, 1])
        [6]: dash_cos  (range: [-1, 1])
        [7]: dash_sin  (range: [-1, 1])
        [8]: turn_cos  (range: [-1, 1])
        [9]: turn_sin  (range: [-1, 1])
        
        Args:
            action: Action vector from policy
            game_state: Latest game state
            team_name: Team name whose robots are being controlled
        
        Returns:
            commands: List of simulator command strings in robot_ids order
            action_info: Dictionary with action details (for logging/debugging)
        """
        action_arr = np.asarray(action, dtype=np.float32).flatten()
        expected_dim = self.num_robots * self.action_dim_per_robot
        if action_arr.shape[0] != expected_dim:
            raise ValueError(f"Expected action dim {expected_dim}, got {action_arr.shape[0]}")

        pose_by_robot_id: Dict[int, Any] = {}
        if game_state is not None:
            team_pose_entries = game_state.robot_poses.get(team_name, [])
            for entry in team_pose_entries:
                if isinstance(entry, dict):
                    pose_by_robot_id.update(entry)

        commands: List[str] = []
        per_robot_info: List[Dict[str, Any]] = []
        invalid_action_count = 0
        ball_pos = game_state.ball_pos if game_state is not None else None

        for i, robot_id in enumerate(self.robot_ids):
            base = i * self.action_dim_per_robot
            dash_logit = float(action_arr[base + 0])
            turn_logit = float(action_arr[base + 1])
            kick_logit = float(action_arr[base + 2])
            start_dribble_logit = float(action_arr[base + 3])
            stop_dribble_logit = float(action_arr[base + 4])
            dash_power = float(action_arr[base + 5])
            dash_cos = float(action_arr[base + 6])
            dash_sin = float(action_arr[base + 7])
            turn_cos = float(action_arr[base + 8])
            turn_sin = float(action_arr[base + 9])

            logits = np.array([dash_logit, turn_logit, kick_logit, start_dribble_logit, stop_dribble_logit], dtype=np.float32)

            # Numerical stability: subtract max before exp
            logits_shifted = logits - np.max(logits)
            exp_logits = np.exp(logits_shifted)
            probs = exp_logits / np.sum(exp_logits)

            action_idx = int(np.argmax(probs))
            action_types = ["dash", "turn", "kick", "start_dribble", "stop_dribble"]
            action_type = action_types[action_idx]

            dash_angle = float(np.arctan2(dash_sin, dash_cos))
            turn_angle = float(np.arctan2(turn_sin, turn_cos))
            dash_power_cmd = float(100.0 * dash_power)

            pose = pose_by_robot_id.get(robot_id)
            has_ball_now = False
            if pose is not None and ball_pos is not None and len(ball_pos) >= 2:
                has_ball_now = has_ball(
                    self_pos_xy=pose,
                    ball_pos_xy=ball_pos,
                    kickable_dist=self.kickable_dist,
                )

            invalid_action_requested = (
                (action_type == "kick" and not has_ball_now)
                or (action_type == "start_dribble" and not has_ball_now)
                or (action_type == "stop_dribble" and not has_ball_now)
            )
            if invalid_action_requested:
                invalid_action_count += 1

            if action_type == "kick":
                if not has_ball_now:
                    command = "turn 0"
                else:
                    command = "kick 100 0"
            elif action_type == "start_dribble":
                if not has_ball_now:
                    command = "turn 0"
                else:
                    command = "catch 0"  # Start dribble
            elif action_type == "stop_dribble":
                if not has_ball_now:
                    command = "turn 0"
                else:
                    command = "drop"  # Stop dribble
            elif action_type == "turn":
                command = f"turn {turn_angle:.4f}"
            else:
                command = f"dash {dash_power_cmd:.2f} {dash_angle:.4f}"

            commands.append(command)
            per_robot_info.append(
                {
                    "robot_id": robot_id,
                    "action_type": action_type,
                    "action_idx": action_idx,
                    "probs": probs.tolist(),
                    "logits": logits.tolist(),
                    "dash_power": dash_power,
                    "dash_power_cmd": dash_power_cmd,
                    "dash_cos": dash_cos,
                    "dash_sin": dash_sin,
                    "dash_angle": dash_angle,
                    "turn_cos": turn_cos,
                    "turn_sin": turn_sin,
                    "turn_angle": turn_angle,
                    "has_ball_now": has_ball_now,
                    "invalid_action_requested": invalid_action_requested,
                    "command": command,
                }
            )

            if self.debug and self.current_step % 10 == 0:
                self.logger.debug(
                    f"Step {self.current_step} robot {robot_id}: {action_type} -> {command}"
                )

        action_info: Dict[str, Any] = {
            "action_type": "multi" if self.num_robots > 1 else per_robot_info[0]["action_type"],
            "per_robot": per_robot_info,
            "invalid_action_count": invalid_action_count,
        }

        # print(f"[DEBUG] Team {team_name} actions: {commands}")
        return commands, action_info

    def build_reward_inputs(
        self,
        current_game_state: GameState,
        robot_id: Optional[int] = None,
        state: Optional[str] = None,
        prev_ball_dist: Optional[float] = None,
        prev_ball_to_goal_dist: Optional[float] = None,
    ) -> RewardInputs:
        """Build a shared `RewardInputs` object from the current game state."""

        if current_game_state is None or current_game_state.ball_pos is None:
            raise ValueError("current_game_state with ball_pos is required")

        if robot_id is None:
            if not self.robot_ids:
                raise ValueError("robot_id is required when no robots are configured")
            robot_id = self.robot_ids[0]

        pose_by_robot_id: Dict[int, Any] = {}
        team_pose_entries = current_game_state.robot_poses.get(self.team_name, [])
        for entry in team_pose_entries:
            if isinstance(entry, dict):
                pose_by_robot_id.update(entry)

        pose = pose_by_robot_id.get(robot_id)
        if pose is None:
            raise ValueError(f"Robot pose for robot_id={robot_id} is unavailable")

        ball_x, ball_y = float(current_game_state.ball_pos[0]), float(current_game_state.ball_pos[1])
        robot_x, robot_y = float(pose[0]), float(pose[1])
        robot_theta = self._clockwise_deg_to_math_rad(float(pose[2]))
        ball_dist = float(np.hypot(ball_x - robot_x, ball_y - robot_y))
        has_ball = bool(ball_dist <= self.kickable_dist)
        reward_state = state if state is not None else self._default_reward_state(has_ball)

        return RewardInputs(
            ball_pos=(ball_x, ball_y),
            self_pose_rad=(robot_x, robot_y, robot_theta),
            ball_dist=ball_dist,
            has_ball=has_ball,
            kickable_dist=self.kickable_dist,
            state=reward_state,
            opponent_positions=extract_opponent_positions(
                robot_poses=getattr(current_game_state, "robot_poses", {}),
                team_name=self.team_name,
            ),
            prev_ball_dist=prev_ball_dist,
            prev_ball_to_goal_dist=prev_ball_to_goal_dist,
        )

    @staticmethod
    def _default_reward_state(has_ball: bool) -> str:
        """Infer a default reward state when no explicit state label is provided."""

        if has_ball:
            return "secure_possession"
        return "chase_ball"

    def _calculate_reward(self, current_game_state: Optional[GameState]) -> float:
        """Calculate the shared team reward from the latest game state."""

        if current_game_state is None:
            return 0.0

        rewards: List[float] = []
        next_ball_to_goal_dist: Optional[float] = None

        for robot_id in self.robot_ids:
            try:
                reward_inputs = self.build_reward_inputs(
                    current_game_state=current_game_state,
                    robot_id=robot_id,
                    prev_ball_dist=self.prev_reward_ball_dist_by_id.get(robot_id),
                    prev_ball_to_goal_dist=self.prev_reward_ball_to_goal_dist,
                )
            except ValueError:
                continue

            reward_result = evaluate_reward(reward_inputs)
            rewards.append(float(reward_result.reward))
            self.prev_reward_ball_dist_by_id[robot_id] = reward_result.intermediates.ball_dist
            next_ball_to_goal_dist = reward_result.intermediates.ball_to_goal_dist

        if next_ball_to_goal_dist is not None:
            self.prev_reward_ball_to_goal_dist = next_ball_to_goal_dist

        if not rewards:
            return 0.0

        return float(np.mean(rewards) + np.max(rewards) * 2) / 3.0

    def _send_commands(
        self,
        commands: List[str],
        team_name: str,
    ):
        """
        Send commands to simulator via networker.
        
        Args:
            commands: List of per-robot command strings
            team_name: Team name that should receive the commands
        """
        try:
            if not commands:
                return

            serialized_commands = self._preprocess_commands_for_send(commands)

            if self.debug:
                self.logger.debug(f"Sending {len(serialized_commands)} commands: {serialized_commands}")

            # Send exactly one batched command list per step to keep robot actions synchronized.
            self.networker.execute_ai_output(serialized_commands, team_name)
            
        except Exception as e:
            self.logger.error(f"Error sending commands: {e}")

    def _preprocess_commands_for_send(self, commands: List[str]) -> List[str]:
        """
        Validate/normalize command strings before sending to simulator.

        Accepted final formats include: "dash p a", "turn a", "kick p a", "catch", "drop".
        """
        valid_prefixes = ("dash ", "turn ", "kick ", "catch", "drop")
        out: List[str] = []
        for cmd in commands:
            clean_cmd = cmd.strip()
            if clean_cmd.startswith(valid_prefixes):
                out.append(clean_cmd)
            else:
                out.append("turn 0")

        return out

"""JAL environment."""

from __future__ import annotations

# Issues to note:
# 1. Ball velocity estimation does not account for ball being kicked by a robot, which causes discontinuities. We could add a heuristic to detect when the ball is likely being kicked (e.g. sudden large velocity change near a robot) and reset the velocity estimate in those cases.
# 2. Robot velocity estimation is a simple finite difference which can be noisy. We could maintain a short history of robot poses and use a more robust method like least squares to estimate velocity, similar to the ball velocity estimation.
# 3. Past 5 planned actions per robot are intentionally deferred for now. A future update should add a fixed-size action-history encoding to the observation.

from typing import Optional, Tuple, Dict, Any, List
import logging
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ai_interface.utils.algo_utils import estimate_ball_velocity, has_ball
from ai_interface.utils.basic_commands import goto
from ai_interface.constants.field_constants import *
from ai_interface.constants.player_constants import *
from ai_interface.envs.reward import RewardInputs, evaluate_reward, extract_opponent_positions
from networking.networker import Networker
from networking.data_utils import GameState


class JALTeamEnv(gym.Env):
    def __init__(
        self, 
        networker: Networker,  
        team_name: str,
        robot_ids: Optional[List[int]] = None,
        obs_dim_per_robot: int = 8,
        non_robot_obs_dim: int = 4,
        max_steps: int = 200,
        debug: bool = False,
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
        if robot_ids is None:
            robot_ids = [1]
        self.robot_ids = list(robot_ids)
        self.num_robots = len(self.robot_ids)
        self.max_steps = max_steps
        self.debug = debug
        self.state_retry_count = max(0, int(state_retry_count))
        self.state_retry_sleep_s = max(0.0, float(state_retry_sleep_s))
        self.none_state_warn_every = max(1, int(none_state_warn_every))
        self.position_noise_std = float(position_noise_std)
        self.invalid_action_penalty = max(0.0, float(invalid_action_penalty))
        
        
        # Create a logger for this environment
        self.logger = logging.getLogger(f"JALTeamEnv[{team_name}]")
        if self.debug:
            self.logger.setLevel(logging.DEBUG)

        # Observation design:
        # global: [ball_x, ball_y, ball_vx, ball_vy] (4)
        # per robot: [robot_x, robot_y, robot_theta, robot_vx, robot_vy, is_dribbling, start_dribble_x, start_dribble_y] (8)
        self.obs_dim_per_robot = obs_dim_per_robot
        self.non_robot_obs_dim = non_robot_obs_dim
        self.obs_dim = obs_dim_per_robot * self.num_robots + non_robot_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.obs_dim,), 
            dtype=np.float32
            )
        self.is_dribbling = {robot_id: False for robot_id in self.robot_ids}  # Track dribble state per robot
        self.start_dribble_pos = {robot_id: [-1.0, -1.0] for robot_id in self.robot_ids}  # Placeholder for dribble start position, can be updated in step() when dribble starts
        
        # Action design per robot (8D): [goto_logit, turn_logit, kick_logit, start_dribble_logit, stop_dribble_logit, goto_x, goto_y, turn_theta]
        self.action_dim_per_robot = 8
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_robots * self.action_dim_per_robot,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.episode_num = 0
        
        # Ball position history for estimate_ball_velocity()
        self.ball_pos_history: List[np.ndarray] = []
        self.ball_history_max = 5

        # Field dimensions (from your existing code)
        self.field_half_width = 45.0
        self.field_half_height = 30.0

        self.kickable_dist = KICKABLE_MARGIN + BALL_SIZE / 2 + PLAYER_SIZE / 2

        # Per-robot pose history for velocity estimation
        self.prev_robot_pose_by_id: Dict[int, np.ndarray] = {}
        self.prev_reward_ball_dist_by_id: Dict[int, float] = {}
        self.prev_reward_ball_to_goal_dist: Optional[float] = None

        # Statistics
        self.total_rewards = 0.0
        self.episode_actions = []  # Track action distribution
        self._none_state_counter = 0
        
        self.logger.info(f"Initialized JALTeamEnv for team '{team_name}' with robots {robot_ids}, num_robots={self.num_robots}, obs_dim={self.obs_dim}, action_dim={self.action_space.shape[0]}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
        ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial observation vector (obs_dim,)
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
        
        # Clear ball history so velocity starts fresh each episode
        self.ball_pos_history = []

        # Clear robot pose memory
        self.prev_robot_pose_by_id = {}
        self.prev_reward_ball_dist_by_id = {}
        self.prev_reward_ball_to_goal_dist = None

        # Get initial game state from simulator
        game_state = self._get_game_state(
            retries=max(self.state_retry_count, 20),
            sleep_s=self.state_retry_sleep_s,
        )
        
        # Build initial observation
        obs = self._game_state_to_obs(game_state)

        
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
            action: Joint action vector with shape (6 * num_robots,).
        
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended (goal/out-of-bounds)
            truncated: Whether episode hit max steps
            info: Additional information
        """
        self.current_step += 1

        # Use current state to decode pose-aware commands (especially goto()).
        current_game_state = self._get_game_state(
            retries=self.state_retry_count,
            sleep_s=self.state_retry_sleep_s,
        )

        # Decode action vector into simulator commands
        commands, action_info = self._action_to_commands(action, current_game_state)

        # Track action distribution for debugging
        self.episode_actions.append(action_info["action_type"])

        # Send commands to simulator
        self._send_commands(commands)
        
        # Read next state after sending commands, then build next observation.
        next_game_state = self._get_game_state(
            retries=self.state_retry_count,
            sleep_s=self.state_retry_sleep_s,
        )
        obs = self._game_state_to_obs(next_game_state)

        reward = self._calculate_reward(next_game_state)
        invalid_action_count = int(action_info.get("invalid_action_count", 0))
        if invalid_action_count > 0 and self.invalid_action_penalty > 0.0:
            reward -= self.invalid_action_penalty * invalid_action_count
        self.total_rewards += reward
        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {
            "action_info": action_info,
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
        
    def _game_state_to_obs(self, game_state) -> np.ndarray:
        """
        Convert game state to observation vector.

        Flat list format:
        - Global (4):
            - [ball_x, ball_y, ball_vx, ball_vy]
        - Per robot in self.robot_ids order (8 each):
            - [robot_x, robot_y, robot_theta, robot_vx, robot_vy, is_dribbling, start_dribble_x, start_dribble_y]
        
        Ball velocity is estimated using estimate_ball_velocity() from algo_utils,
        which uses a position history and the known ball decay constant for better
        noise rejection than a raw single-step finite difference.
        
        Args:
            game_state: GameState object from networker
        
        Returns:
            Observation vector (obs_dim,)
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
            return np.zeros(self.obs_dim, dtype=np.float32)

        self._none_state_counter = 0
        
        try:
            game_state = self._preprocess_game_state(game_state)

            # --- Ball position (tuple: x, y) ---
            ball_pos = game_state.ball_pos
            ball_x, ball_y = ball_pos[0], ball_pos[1]
            
            # Maintain a rolling history of ball positions for velocity estimation
            # self.ball_pos_history --> python list
            self.ball_pos_history.append(np.array([ball_x, ball_y], dtype=float))
            if len(self.ball_pos_history) > self.ball_history_max:
                self.ball_pos_history.pop(0)
            
            # Use estimate_ball_velocity() when we have ≥2 frames; zero otherwise
            if len(self.ball_pos_history) >= 2:
                positions = np.stack(self.ball_pos_history)  # Numpy Array --> shape (k, 2)
                ball_vel = estimate_ball_velocity(positions, alpha=BALL_DECAY)
                ball_vx, ball_vy = float(ball_vel[0]), float(ball_vel[1])
            else:
                ball_vx = ball_vy = 0.0
            
            # Build lookup map from robot id -> pose from list-of-dicts format.
            # Expected shape: robot_poses[team_name] = [{1: (x, y, theta)}, {2: (...)}, ...]
            team_pose_entries = game_state.robot_poses.get(self.team_name, [])
            pose_by_robot_id = {}
            for entry in team_pose_entries:
                if isinstance(entry, dict):
                    pose_by_robot_id.update(entry)

            # Build flattened observation list directly.
            obs_values: list[float] = [
                float(ball_x),
                float(ball_y),
                float(ball_vx),
                float(ball_vy),
            ]

            for robot_id in self.robot_ids:
                pose = pose_by_robot_id.get(robot_id)

                if pose is None:
                    obs_values.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0])  # Default values for missing robot
                    continue

                robot_x = float(pose[0])
                robot_y = float(pose[1])
                robot_theta = float(np.deg2rad(pose[2]))

                robot_ball_distance = float(np.hypot(robot_x - ball_x, robot_y - ball_y))
                self.is_dribbling[robot_id] = robot_ball_distance <= 1.115 + 0.05

                if self.is_dribbling[robot_id] and self.start_dribble_pos[robot_id] == [-1.0, -1.0]:
                    self.start_dribble_pos[robot_id] = [robot_x, robot_y]
                elif not self.is_dribbling[robot_id]:
                    self.start_dribble_pos[robot_id] = [-1.0, -1.0]

                current_pose_xy = np.array([robot_x, robot_y], dtype=np.float32)
                prev_pose_xy = self.prev_robot_pose_by_id.get(robot_id)
                if prev_pose_xy is None:
                    robot_vx = 0.0
                    robot_vy = 0.0
                else:
                    robot_vx = float(current_pose_xy[0] - prev_pose_xy[0])
                    robot_vy = float(current_pose_xy[1] - prev_pose_xy[1])

                is_dribbling = self.is_dribbling.get(robot_id, False)
                start_dribble_pos = self.start_dribble_pos.get(robot_id, [-1.0, -1.0])

                self.prev_robot_pose_by_id[robot_id] = current_pose_xy

                obs_values.extend([robot_x, robot_y, robot_theta, robot_vx, robot_vy, float(is_dribbling), float(start_dribble_pos[0]), float(start_dribble_pos[1])])

            obs = np.array(obs_values, dtype=np.float32)

            # Keep output shape consistent with observation_space.
            if obs.shape[0] != self.obs_dim:
                if obs.shape[0] > self.obs_dim:
                    obs = obs[:self.obs_dim]
                else:
                    obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]))

            return obs
            
        except Exception as e:
            self.logger.error(f"Error building observation: {e}")
            return np.zeros(self.obs_dim, dtype=np.float32)

    def _action_to_commands(self, action: np.ndarray, game_state: Optional[GameState]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Convert TD3 action vector to simulator command strings.
        
        Action vector format (8*number of robots) Dimensions:
        [0]: goto_logit
        [1]: turn_logit
        [2]: kick_logit
        [3]: start_dribble_logit
        [4]: stop_dribble_logit
        [5]: goto_x_raw  (range: [-1, 1])
        [6]: goto_y_raw  (range: [-1, 1])
        [7]: turn_theta_raw  (range: [-1, 1])
        
        Args:
            action: Action vector from TD3 policy
        
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
            team_pose_entries = game_state.robot_poses.get(self.team_name, [])
            for entry in team_pose_entries:
                if isinstance(entry, dict):
                    pose_by_robot_id.update(entry)

        commands: List[str] = []
        per_robot_info: List[Dict[str, Any]] = []
        invalid_action_count = 0
        ball_pos = game_state.ball_pos if game_state is not None else None

        for i, robot_id in enumerate(self.robot_ids):
            base = i * self.action_dim_per_robot
            goto_logit = float(action_arr[base + 0])
            turn_logit = float(action_arr[base + 1])
            kick_logit = float(action_arr[base + 2])
            start_dribble_logit = float(action_arr[base + 3])
            stop_dribble_logit = float(action_arr[base + 4])
            goto_x_raw = float(action_arr[base + 5])
            goto_y_raw = float(action_arr[base + 6])
            turn_theta_raw = float(action_arr[base + 7])

            logits = np.array([goto_logit, turn_logit, kick_logit, start_dribble_logit, stop_dribble_logit], dtype=np.float32)

            # Numerical stability: subtract max before exp
            logits_shifted = logits - np.max(logits)
            exp_logits = np.exp(logits_shifted)
            probs = exp_logits / np.sum(exp_logits)

            action_idx = int(np.argmax(probs))
            action_types = ["goto", "turn", "kick", "start_dribble", "stop_dribble"]
            action_type = action_types[action_idx]

            goto_x = float(goto_x_raw * self.field_half_width)
            goto_y = float(goto_y_raw * self.field_half_height)
            turn_theta = float(turn_theta_raw * np.pi)

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
                command = f"turn {turn_theta:.2f}"
            else:
                if pose is None or game_state is None:
                    command = "turn 0"
                else:
                    self_pose = np.array([
                        float(pose[0]),
                        float(pose[1]),
                        float(np.deg2rad(pose[2])),
                    ], dtype=np.float32)
                    command = goto(
                        self_pose=self_pose,
                        x=goto_x,
                        y=goto_y,
                        game_state=game_state,
                    )
                    if command == "done":
                        command = "turn 0"

            commands.append(command)
            per_robot_info.append(
                {
                    "robot_id": robot_id,
                    "action_type": action_type,
                    "action_idx": action_idx,
                    "probs": probs.tolist(),
                    "logits": logits.tolist(),
                    "goto_x": goto_x,
                    "goto_y": goto_y,
                    "turn_theta": turn_theta,
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
        robot_theta = float(np.deg2rad(pose[2]))
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

        return float(np.mean(rewards))

    def _send_commands(self, commands: List[str]):
        """
        Send commands to simulator via networker.
        
        Args:
            commands: List of per-robot command strings
        """
        try:
            if not commands:
                return

            serialized_commands = self._preprocess_commands_for_send(commands)

            if self.debug:
                self.logger.debug(f"Sending {len(serialized_commands)} commands: {serialized_commands}")

            # Send exactly one batched command list per step to keep robot actions synchronized.
            self.networker.execute_ai_output(serialized_commands, self.team_name)
            
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

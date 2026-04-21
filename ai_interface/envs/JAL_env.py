"""JAL environment."""

"""
Issues to note:
1. Ball velocity estimation does not account for ball being kicked by a robot, which causes discontinuities. We could add a heuristic to detect when the ball is likely being kicked (e.g. sudden large velocity change near a robot) and reset the velocity estimate in those cases.
2. Robot velocity estimation is a simple finite difference which can be noisy. We could maintain a short history of robot poses and use a more robust method like least squares to estimate velocity, similar to the ball velocity estimation.
3. Past 5 planned actions per robot are intentionally deferred for now. A future update should add a fixed-size action-history encoding to the observation.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List
import logging

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ai_interface.utils.algo_utils import estimate_ball_velocity
from ai_interface.utils.basic_commands import goto
from ai_interface.constants.field_constants import BALL_DECAY
from networking.networker import Networker


class JALTeamEnv(gym.Env):
    def __init__(
        self, 
        networker: Networker,  
        team_name: str,
        robot_ids: Optional[List[int]] = None,
        obs_dim_per_robot: int = 5,
        non_robot_obs_dim: int = 4,
        max_steps: int = 200,
        debug: bool = False,
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
        
        
        # Create a logger for this environment
        self.logger = logging.getLogger(f"JALTeamEnv[{team_name}]")
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
            
        # Observation design:
        # global: [ball_x, ball_y, ball_vx, ball_vy] (4)
        # per robot: [robot_x, robot_y, robot_theta, robot_vx, robot_vy] (5)
        self.obs_dim_per_robot = obs_dim_per_robot
        self.non_robot_obs_dim = non_robot_obs_dim
        self.obs_dim = obs_dim_per_robot * self.num_robots + non_robot_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.obs_dim,), 
            dtype=np.float32
            )
        
        # Action design per robot (6D): [goto_logit, kick_logit, dribble_logit, goto_x, goto_y, goto_theta]
        self.action_dim_per_robot = 6
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


        # Per-robot pose history for velocity estimation
        self.prev_robot_pose_by_id: Dict[int, np.ndarray] = {}

        # Statistics
        self.total_rewards = 0.0
        self.episode_actions = []  # Track action distribution
        
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
        
        # Clear ball history so velocity starts fresh each episode
        self.ball_pos_history = []

        # Clear robot pose memory
        self.prev_robot_pose_by_id = {}

        # Get initial game state from simulator
        game_state = self._get_game_state()
        
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
        current_game_state = self._get_game_state()

        # Decode action vector into simulator commands
        commands, action_info = self._action_to_commands(action, current_game_state)

        # Track action distribution for debugging
        self.episode_actions.append(action_info["action_type"])

        # Send commands to simulator
        self._send_commands(commands)
        
        # Read next state after sending commands, then build next observation.
        next_game_state = self._get_game_state()
        obs = self._game_state_to_obs(next_game_state)

        reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {
            "action_info": action_info,
        }
        return obs, reward, terminated, truncated, info
    
    
    def _get_game_state(self) -> Optional[Dict]:
        """
        Get current game state from networker.
        
        Returns:
            Game state dictionary or None if unavailable
        """
        try:
            game_state = self.networker.get_game_state()
            return game_state
        except Exception as e:
            self.logger.error(f"Error getting game state: {e}")
            return None
        
    def _game_state_to_obs(self, game_state) -> np.ndarray:
        """
        Convert game state to observation vector.

        Flat list format:
        - Global (4):
            - [ball_x, ball_y, ball_vx, ball_vy]
        - Per robot in self.robot_ids order (5 each):
            - [robot_x, robot_y, robot_theta, robot_vx, robot_vy]
        
        Ball velocity is estimated using estimate_ball_velocity() from algo_utils,
        which uses a position history and the known ball decay constant for better
        noise rejection than a raw single-step finite difference.
        
        Args:
            game_state: GameState object from networker
        
        Returns:
            Observation vector (obs_dim,)
        """
        if game_state is None:
            self.logger.warning("Game state is None, returning zero observation")
            return np.zeros(self.obs_dim, dtype=np.float32)
        
        try:
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
                    obs_values.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    continue

                robot_x = float(pose[0])
                robot_y = float(pose[1])
                robot_theta = float(np.deg2rad(pose[2]))

                current_pose_xy = np.array([robot_x, robot_y], dtype=np.float32)
                prev_pose_xy = self.prev_robot_pose_by_id.get(robot_id)
                if prev_pose_xy is None:
                    robot_vx = 0.0
                    robot_vy = 0.0
                else:
                    robot_vx = float(current_pose_xy[0] - prev_pose_xy[0])
                    robot_vy = float(current_pose_xy[1] - prev_pose_xy[1])

                self.prev_robot_pose_by_id[robot_id] = current_pose_xy

                obs_values.extend([robot_x, robot_y, robot_theta, robot_vx, robot_vy])

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

    def _action_to_commands(self, action: np.ndarray, game_state: Optional[Dict]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Convert TD3 action vector to simulator command strings.
        
        Action vector format (6*number of robots) Dimensions:
        [0]: goto_logit
        [1]: kick_logit
        [2]: dribble_logit
        [3]: goto_x_raw  (range: [-1, 1])
        [4]: goto_y_raw  (range: [-1, 1])
        [5]: goto_theta_raw  (range: [-1, 1])
        
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

        for i, robot_id in enumerate(self.robot_ids):
            base = i * self.action_dim_per_robot
            goto_logit = float(action_arr[base + 0])
            kick_logit = float(action_arr[base + 1])
            dribble_logit = float(action_arr[base + 2])
            goto_x_raw = float(action_arr[base + 3])
            goto_y_raw = float(action_arr[base + 4])
            goto_theta_raw = float(action_arr[base + 5])

            logits = np.array([goto_logit, kick_logit, dribble_logit], dtype=np.float32)

            # Numerical stability: subtract max before exp
            logits_shifted = logits - np.max(logits)
            exp_logits = np.exp(logits_shifted)
            probs = exp_logits / np.sum(exp_logits)

            action_idx = int(np.argmax(probs))
            action_types = ["goto", "kick", "dribble"]
            action_type = action_types[action_idx]

            goto_x = float(goto_x_raw * self.field_half_width)
            goto_y = float(goto_y_raw * self.field_half_height)
            goto_theta = float(goto_theta_raw * np.pi)

            if action_type == "kick":
                command = "kick 100 0"
            elif action_type == "dribble":
                command = "dribble"
            else:
                pose = pose_by_robot_id.get(robot_id)
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
                        theta=goto_theta,
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
                    "goto_theta": goto_theta,
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
        }

        return commands, action_info

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

        Accepted final formats include: "dash p a", "turn a", "kick p a", "dribble".
        """
        valid_prefixes = ("dash ", "turn ", "kick ", "dribble")
        out: List[str] = []
        for cmd in commands:
            clean_cmd = cmd.strip()
            if clean_cmd.startswith(valid_prefixes):
                out.append(clean_cmd)
            else:
                out.append("turn 0")

        return out

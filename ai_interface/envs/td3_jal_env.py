"""
TD3 JAL Environment for RoboCup Soccer (1v0 Scenario)

This environment implements Joint-Action Learning for TD3 (Twin Delayed DDPG).
It treats a single robot as an agent that outputs a 6D continuous action vector:
[kick_logit, dash_logit, turn_logit, dash_power_raw, dash_angle_raw, turn_angle_raw]

The environment handles:
- Logit → probability conversion (softmax)
- Action sampling (kick/dash/turn)
- Parameter scaling (raw values → proper ranges)
- Command generation for the simulator

Compatible with Stable Baselines3's TD3 algorithm.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import logging
import math

from networking.networker import Networker
from ai_interface.utils.algo_utils import estimate_ball_velocity
from ai_interface.constants.field_constants import BALL_DECAY
from ai_interface.constants.player_constants import KICKABLE_MARGIN, PLAYER_SIZE, BALL_SIZE

class TD3JALEnv(gym.Env):
    """
    Gymnasium environment for TD3 with Joint-Action Learning.
    
    Action Space:
        Box(6,) with range [-1, 1]:
        - [0:3]: Logits for kick/dash/turn (unnormalized scores)
        - [3:6]: Raw parameters for dash_power, dash_angle, turn_angle
    
    Observation Space:
        Box(obs_dim,) with normalized state information
    
    For 1v0 scenario: Single robot trying to score a goal with no opponents.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        networker: Networker,
        team_name: str,
        robot_id: int = 1,
        num_robots: int = 1,
        obs_dim: int = 10,
        max_steps: int = 200,
        debug: bool = False
    ):
        """
        Initialize TD3 JAL environment.
        
        Args:
            networker: Networker instance for simulator communication
            team_name: Name of team to control
            robot_id: ID of the robot to control (default: 1)
            num_robots: Number of robots (default: 1 for 1v0)
            obs_dim: Observation dimension (default: 10 for 1v0 scenario)
            max_steps: Maximum steps per episode
            debug: Enable debug logging
        """
        super().__init__()
        
        self.networker = networker
        self.team_name = team_name
        self.robot_id = robot_id
        self.num_robots = num_robots
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.debug = debug
        
        # Setup logging
        self.logger = logging.getLogger(f"TD3JALEnv[{team_name}]")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        
        # Define action space: 6D continuous vector per robot
        # For 1v0: 1 robot × 6D = 6D total
        action_dim = num_robots * 6  # [kick_logit, dash_logit, turn_logit, dash_p, dash_a, turn_a]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # Define observation space: normalized state vector
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.episode_num = 0
        
        # State tracking for reward shaping (will be used by teammate's calculate_reward)
        self.prev_ball_pos = None
        self.prev_robot_pos = None
        
        # Ball position history for estimate_ball_velocity()
        self.ball_pos_history: List[np.ndarray] = []
        self.ball_history_max = 5
        
        # Kickable distance threshold (from player/ball size constants)
        self.kickable_dist = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE
        
        # Statistics
        self.total_rewards = 0.0
        self.episode_actions = []  # Track action distribution
        
        # Field dimensions (from your existing code)
        self.field_width = 90.0
        self.field_height = 60.0
        
        self.CROWDING_THRESHOLD = 2.0
        self.OPPONENT_GOAL = (self.field_width / 2, 0.0)   # GOAL_R
        self.OWN_GOAL      = (-self.field_width / 2, 0.0)  # GOAL_L

        self.logger.info(f"TD3JALEnv initialized: robot_id={robot_id}, {num_robots} robots, {obs_dim}D obs, {action_dim}D action")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
        
        # Clear previous state tracking
        self.prev_ball_pos = None
        self.prev_robot_pos = None
        
        # Clear ball history so velocity starts fresh each episode
        self.ball_pos_history = []
        
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
            action: Action vector (6,) containing [logits(3), params(3)]
        
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended (goal/out-of-bounds)
            truncated: Whether episode hit max steps
            info: Additional information
        """
        self.current_step += 1
        
        # Decode action vector into simulator commands
        commands, action_info = self._action_to_commands(action)
        
        # Track action distribution for debugging
        self.episode_actions.append(action_info["action_type"])
        
        # Send commands to simulator
        # Currently, just sending command to robot id 0
        self._send_commands(commands)
        
        # Get next game state
        # RESOLVED: Do we need to wait before calling GameState after executing the action? Not needed.
        game_state = self._get_game_state()
        
        # Build observation
        obs = self._game_state_to_obs(game_state)
        
        # Check termination conditions
        terminated, truncated, termination_reason = self._check_termination(game_state)

        # Calculate reward
        reward = self._calculate_reward(game_state, action_info, termination_reason=termination_reason)
        self.total_rewards += reward
        
        # Build info dictionary
        info = {
            "step": self.current_step,
            "episode_num": self.episode_num,
            "action_type": action_info["action_type"],
            "action_probs": action_info["probs"],
            "total_reward": self.total_rewards,
            **action_info  # Include all action details
        }
        
        if terminated or truncated:
            info["termination_reason"] = termination_reason
            info["episode_length"] = self.current_step
            info["episode_reward"] = self.total_rewards
            
            # Action distribution statistics
            action_counts = {
                "kick": self.episode_actions.count("kick"),
                "dash": self.episode_actions.count("dash"),
                "turn": self.episode_actions.count("turn")
            }
            info["action_distribution"] = action_counts
            
            if self.debug:
                self.logger.debug(f"Episode {self.episode_num} ended: {termination_reason}")
                self.logger.debug(f"  Length: {self.current_step} steps")
                self.logger.debug(f"  Reward: {self.total_rewards:.2f}")
                self.logger.debug(f"  Actions: {action_counts}")
        
        return obs, reward, terminated, truncated, info
    
    def _action_to_commands(
        self,
        action: np.ndarray
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Convert TD3 action vector to simulator commands.
        
        Action vector format (6D):
        [0]: kick_logit
        [1]: dash_logit
        [2]: turn_logit
        [3]: dash_power_raw  (range: [-1, 1])
        [4]: dash_angle_raw  (range: [-1, 1])
        [5]: turn_angle_raw  (range: [-1, 1])
        
        Args:
            action: Action vector from TD3 policy
        
        Returns:
            commands: List of command strings for simulator
            action_info: Dictionary with action details (for logging/debugging)
        """
        # Extract components
        # FUTURE: Use loop here for if we move to multiple robots
        kick_logit = action[0]
        dash_logit = action[1]
        turn_logit = action[2]
        dash_power_raw = action[3]
        dash_angle_raw = action[4]
        turn_angle_raw = action[5]
        
        # Convert logits to probabilities using softmax
        logits = np.array([kick_logit, dash_logit, turn_logit])
        # Numerical stability: subtract max before exp
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)
        
        # Choose action with highest probability
        action_idx = int(np.argmax(probs))   
        action_types = ["kick", "dash", "turn"]
        action_type = action_types[action_idx]
        
        # Scale raw parameters to proper ranges
        # Dash power: [-1, 1] → [0, 100] as we can't use negative dash
        dash_power = np.clip((dash_power_raw + 1.0) * 50.0, 0.0, 100.0)
        
        # Dash angle: [-1, 1] → [-π, π] (about 180 degrees each direction)
        dash_angle = np.clip(dash_angle_raw * (np.pi), -np.pi, np.pi)
        # Can we use normalise angle here?
        
        # Turn angle: [-1, 1] → [-π, π]
        # Angle already in range of [-1, 1] so we don't need to use normalise_angle() helper function
        turn_angle = np.clip(turn_angle_raw * (np.pi), -np.pi, np.pi)

        # Generate command based on sampled action
        if action_type == "kick":
            # Hardcoded kick: power=100, angle=0 (straight ahead)
            command = "kick 100 0"
        elif action_type == "dash":
            command = f"dash {dash_power:.2f} {dash_angle:.4f}"
        else:  # turn
            command = f"turn {turn_angle:.4f}"
        
        commands = [command]
        
        # Build info dictionary for logging/debugging
        action_info = {
            "action_type": action_type,
            "action_idx": action_idx,
            "probs": probs.tolist(),
            "logits": logits.tolist(),
            "dash_power": float(dash_power),
            "dash_angle": float(dash_angle),
            "turn_angle": float(turn_angle),
            "command": command
        }
        
        if self.debug and self.current_step % 10 == 0:
            self.logger.debug(f"Step {self.current_step}: {action_type} (p={probs[action_idx]:.3f})")
            if action_type == "dash":
                self.logger.debug(f"  Dash: power={dash_power:.1f}, angle={dash_angle:.2f}")
            elif action_type == "turn":
                self.logger.debug(f"  Turn: angle={turn_angle:.2f}")
        
        return commands, action_info
    
    def _game_state_to_obs(self, game_state) -> np.ndarray:
        """
        Convert game state to observation vector.
        
        For 1v0 scenario, observation is ego-centric (all positions relative to robot):
        - Ball relative position + velocity: [ball_dx, ball_dy, ball_vx, ball_vy] (4D)
        - Goal relative position:            [goal_dx, goal_dy]                    (2D)
        - Robot orientation:                 [sin_θ, cos_θ]                        (2D)
        - Flags:                             [kickable, step_fraction]              (2D)
        Total: 10D
        
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
            
            # --- Robot pose: robot_poses[team_name][0][robot_id] → (x, y, theta_degrees) ---
            # RESOLVED: confirm if team_poses is a list of dictionaries or a list of just one dictionary with all the robot ids?
            # It's a list of multiple dictionaries

            # GameState(count=178, timestamp=1770400207.583633, ball_pos=(43.4443, -5.45664),
            # robot_poses={'TritonBots': [{1: (39.2051, -10.907, 10.0)}],
            # 'TeamB': [{1: (43.9682, -4.0878, -160.0)}]}, playmode=None)

            team_poses = game_state.robot_poses.get(self.team_name, [])
            if team_poses and self.robot_id in team_poses[0]:
                robot_pose = team_poses[0][self.robot_id]  # (x, y, theta_degrees)
                robot_x, robot_y = robot_pose[0], robot_pose[1]
                robot_theta = np.deg2rad(robot_pose[2])    # degrees → radians
            else:
                robot_x = robot_y = robot_theta = 0.0
            
            # --- Ego-centric relative vectors ---
            ball_dx = (ball_x - robot_x) / self.field_width
            ball_dy = (ball_y - robot_y) / self.field_height
            
            goal_x = self.field_width / 2   # Opponent goal
            goal_y = 0.0
            goal_dx = (goal_x - robot_x) / self.field_width
            goal_dy = (goal_y - robot_y) / self.field_height
            
            # --- Orientation: sin/cos avoids angular discontinuity ---
            sin_theta = np.sin(robot_theta)
            cos_theta = np.cos(robot_theta)
            
            # --- Kickable flag using proper size constants ---
            ball_dist = np.sqrt((ball_x - robot_x)**2 + (ball_y - robot_y)**2)
            kickable = 1.0 if ball_dist < self.kickable_dist else 0.0
            
            # --- Step fraction ---
            step_fraction = self.current_step / self.max_steps
            
            # --- Build 10D ego-centric observation ---
            obs = np.array([
                ball_dx, ball_dy, ball_vx / 10.0, ball_vy / 10.0,  # Ball relative (4D) # RESOLVED: What's the max speed of the ball? Any speed works. Can just scale up.
                goal_dx, goal_dy,                                    # Goal relative (2D)
                sin_theta, cos_theta,                                # Orientation (2D)
                kickable, step_fraction                              # Flags (2D)
            ], dtype=np.float32)
            
            return obs
            
        except Exception as e:
            self.logger.error(f"Error building observation: {e}")
            return np.zeros(self.obs_dim, dtype=np.float32)
    
    def _calculate_reward(
        self,
        game_state,
        action_info: Dict[str, Any],
        termination_reason: str = "ongoing"
    ) -> float:
        if game_state is None:
            return -0.1

        try:
            if termination_reason == "goal_scored":
                return 1.0

            next_ball = game_state.ball_pos

            team_poses = game_state.robot_poses.get(self.team_name, [])
            if not team_poses or self.robot_id not in team_poses[0]:
                return -0.1
            rp = team_poses[0][self.robot_id]
            next_robot = (rp[0], rp[1])
            next_robots = [next_robot]

            # On first step prev_* is None — skip shaped rewards, just update tracking
            if self.prev_ball_pos is None or self.prev_robot_pos is None:
                self.prev_ball_pos  = next_ball
                self.prev_robot_pos = next_robot
                return 0.0

            curr_ball   = self.prev_ball_pos
            curr_robots = [self.prev_robot_pos]

            reward = 0.0

            distances_to_ball = [math.hypot(r[0] - curr_ball[0], r[1] - curr_ball[1]) for r in curr_robots]
            min_dist_to_ball  = min(distances_to_ball)
            closest_robot_id  = distances_to_ball.index(min_dist_to_ball)

            team_has_possession = min_dist_to_ball <= self.kickable_dist

            if not team_has_possession:
                # Hunter mode
                reward += (-min_dist_to_ball * 0.1) - 0.1

            else:
                # Attacker mode
                curr_dist_to_opp_goal = math.hypot(curr_ball[0] - self.OPPONENT_GOAL[0], curr_ball[1] - self.OPPONENT_GOAL[1])
                next_dist_to_opp_goal = math.hypot(next_ball[0] - self.OPPONENT_GOAL[0], next_ball[1] - self.OPPONENT_GOAL[1])
                v_prog = curr_dist_to_opp_goal - next_dist_to_opp_goal

                curr_dist_to_own_goal = math.hypot(curr_ball[0] - self.OWN_GOAL[0], curr_ball[1] - self.OWN_GOAL[1])
                next_dist_to_own_goal = math.hypot(next_ball[0] - self.OWN_GOAL[0], next_ball[1] - self.OWN_GOAL[1])
                v_reg = curr_dist_to_own_goal - next_dist_to_own_goal

                reward += (1.2 * v_prog) - v_reg

                next_distances  = [math.hypot(r[0] - next_ball[0], r[1] - next_ball[1]) for r in next_robots]
                next_closest_id = next_distances.index(min(next_distances))

                if closest_robot_id != next_closest_id:
                    reward += 0.1

            # Crowding penalty (always 0 in 1v0, kept for parity)
            crowding_penalty = 0.0
            for i in range(len(curr_robots)):
                for j in range(i + 1, len(curr_robots)):
                    if math.hypot(curr_robots[i][0] - curr_robots[j][0], curr_robots[i][1] - curr_robots[j][1]) < self.CROWDING_THRESHOLD:
                        crowding_penalty += 0.005
            reward -= crowding_penalty

            # OOB penalties from termination check
            if termination_reason == "robot_out_of_bounds":
                reward -= 0.5
            if termination_reason == "ball_out_of_bounds":
                reward -= 0.2

            # Update tracking for next step
            self.prev_ball_pos  = next_ball
            self.prev_robot_pos = next_robot

            return float(reward)

        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return -0.1
        
    def _check_termination(
        self,
        game_state
    ) -> Tuple[bool, bool, str]:
        """
        Check if episode should terminate.
        
        Args:
            game_state: GameState object from networker
        
        Returns:
            terminated: Episode ended due to terminal state (goal/out-of-bounds)
            truncated: Episode ended due to max steps
            reason: String describing termination reason
        """
        terminated = False
        truncated = False
        reason = "ongoing"
        
        # Check max steps
        if self.current_step >= self.max_steps:
            truncated = True
            reason = "max_steps"
            return terminated, truncated, reason
        
        # Check for invalid game state
        if game_state is None:
            terminated = True
            reason = "invalid_state"
            return terminated, truncated, reason
        
        try:
            ball_x, ball_y = game_state.ball_pos

            half_length = self.field_width / 2   # 45.0
            half_width  = self.field_height / 2  # 30.0
            goal_half_width = 7.32 / 2           # Standard RoboCup goal width

            # Goal scored: ball has crossed the goal line and is within the posts
            if ball_x >= half_length and abs(ball_y) < goal_half_width:
                terminated = True
                reason = "goal_scored"
                return terminated, truncated, reason

            # Robot out of bounds
            robot_poses = game_state.robot_poses.get(self.team_name, [])
            if robot_poses and self.robot_id in robot_poses[0]:
                robot_x, robot_y, _ = robot_poses[0][self.robot_id]
                if abs(robot_x) > half_length or abs(robot_y) > half_width:
                    terminated = True
                    reason = "robot_out_of_bounds"
                    return terminated, truncated, reason

            # Ball out of bounds (sides or back, but NOT through the goal)
            ball_out_sides = abs(ball_y) > half_width
            ball_out_back  = ball_x < -half_length
            if ball_out_sides or ball_out_back:
                terminated = True
                reason = "ball_out_of_bounds"
                return terminated, truncated, reason

        except Exception as e:
            self.logger.error(f"Error checking termination: {e}")
            terminated = True
            reason = "error"
        
        return terminated, truncated, reason

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
    
    def _send_commands(self, commands: List[str]):
        """
        Send commands to simulator via networker.
        
        Args:
            commands: List of command strings
        """
        try:
            # RESOLVED: Do we need to send the robot_id? How does it know which robots to send the commands to? Not needed.
            self.networker.execute_ai_output(commands, self.team_name)
        except Exception as e:
            self.logger.error(f"Error sending commands: {e}")
    
    def close(self):
        """Clean up resources."""
        if self.debug:
            self.logger.debug("Environment closed")

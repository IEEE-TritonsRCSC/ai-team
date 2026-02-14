import gymnasium as gym
from gymnasium import spaces
import numpy as np
from networking.networker import Networker


class SoccerEnv(gym.Env):
    """
    Hierarchical Soccer Environment
    High-level: discrete actions (GoToBall, Shoot, Reposition)
    Low-level: continuous actions (v_x, v_y, omega, kick_power)
    """

    def __init__(self, networker: Networker, team_name: str, obs_dim: int = 10):
        super().__init__()
        self.networker = networker
        self.team_name = team_name

        # Observation space: e.g., [ball_dx, ball_dy, goal_dx, goal_dy, cos(theta), sin(theta), vx, vy, ...]
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Action space (hierarchical)
        self.action_space = spaces.Dict({
            "high_level": spaces.Discrete(3),  # GoToBall, Shoot, Reposition
            "low_level": spaces.Box(
                low=np.array([-2.0, -2.0, -2.0, 0.0]),  # Increased range for more power
                high=np.array([2.0, 2.0, 2.0, 1.0]),
                dtype=np.float32
            )
        })

        # Step counters
        self.max_steps = 200
        self.current_step = 0
        
        # Reward tracking
        self.prev_ball_dist = None
        self.prev_goal_dist = None
        self.prev_ball_pos = None
        self.prev_robot_pos = None
        
        # Field dimensions (RoboCup soccer field)
        self.field_length = 105.0  # meters
        self.field_width = 68.0    # meters
        self.goal_x = self.field_length / 2  # Goal at x = 52.5
        
    def _game_state_to_obs(self, game_state) -> np.ndarray:
        """Convert game state to observation vector."""
        if game_state is None:
            return np.zeros(self.obs_dim, dtype=np.float32)
        
        try:
            # Extract ball position
            ball_pos = game_state.ball_pos or (0.0, 0.0)
            ball_x, ball_y = ball_pos
            
            # Extract our team's first player position (assuming we control player 1)
            our_team_poses = game_state.robot_poses.get(self.team_name, [])
            if our_team_poses and len(our_team_poses) > 0:
                # Get first player's pose (assuming it has uniform number 1)
                first_player = our_team_poses[0]
                if isinstance(first_player, dict) and 1 in first_player:
                    robot_x, robot_y, robot_theta = first_player[1]
                else:
                    robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
            else:
                robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
            
            # Compute relative positions
            ball_dx = ball_x - robot_x
            ball_dy = ball_y - robot_y
            goal_dx = self.goal_x - robot_x  # Distance to goal
            goal_dy = 0.0 - robot_y  # Goal is centered at y=0
            
            # Robot orientation
            theta_rad = np.radians(robot_theta)
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)
            
            # Distance to ball and goal
            ball_dist = np.sqrt(ball_dx**2 + ball_dy**2)
            goal_dist = np.sqrt(goal_dx**2 + goal_dy**2)
            
            # Construct observation vector
            obs = np.array([
                ball_dx, ball_dy,           # Ball position relative to robot
                goal_dx, goal_dy,           # Goal position relative to robot
                cos_theta, sin_theta,       # Robot orientation
                ball_dist,                  # Distance to ball
                goal_dist,                  # Distance to goal
                robot_x, robot_y           # Robot absolute position
            ], dtype=np.float32)
            
            # Pad or truncate to desired obs_dim
            if len(obs) < self.obs_dim:
                obs = np.pad(obs, (0, self.obs_dim - len(obs)), mode='constant')
            else:
                obs = obs[:self.obs_dim]
            
            return obs
            
        except Exception as e:
            # If extraction fails, return zeros
            return np.zeros(self.obs_dim, dtype=np.float32)

    def reset(self):
        """Reset simulator and step counter"""
        # Reset the simulator to initial state (less aggressive approach)
        reset_ok = False
        try:
            reset_ok = bool(self.networker.reset_sim())
        except Exception as e:
            print(f"[SoccerEnv] reset_sim failed: {e}")

        # In simulator modes, a failed reset should be visible in logs.
        if self.networker.environment in ["sim-only", "sim-mixed"] and not reset_ok:
            print("[SoccerEnv] reset_sim did not complete; continuing current match state.")
        
        # Wait a moment for reset to take effect
        import time
        time.sleep(0.1)
        
        # Get initial observation
        game_state = self.networker.get_game_state()
        obs = self._game_state_to_obs(game_state)
        self.current_step = 0
        
        # Reset reward tracking
        self.prev_ball_dist = None
        self.prev_goal_dist = None
        self.prev_ball_pos = None
        self.prev_robot_pos = None
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """
        action: dict with "high_level" (int) and "low_level" (array)
        Returns: obs, reward, done, info
        """

        # Map hierarchical action to simulator commands
        high = action["high_level"]
        low = action["low_level"]

        # Convert low-level actions to simulator commands
        # low = [v_x, v_y, omega, kick_power]
        commands = self._action_to_commands(high, low)
        
        # Execute commands
        try:
            self.networker.execute_ai_output(commands, self.team_name)
        except Exception as e:
            pass

        # Get next game state
        game_state = self.networker.get_game_state()
        obs = self._game_state_to_obs(game_state)
        
        # Compute reward (TODO: implement meaningful reward function)
        reward = self._compute_reward(game_state, high)
        
        self.current_step += 1

        # Terminate if max steps reached
        done = self.current_step >= self.max_steps

        return np.array(obs, dtype=np.float32), reward, done, {}
    
    def _action_to_commands(self, high_action: int, low_action: np.ndarray) -> list:
        """Convert hierarchical action to simulator command strings.
        
        Args:
            high_action: High-level action (0=GoToBall, 1=Shoot, 2=Reposition)
            low_action: Low-level continuous actions [v_x, v_y, omega, kick_power]
            
        Returns:
            List of command strings for each player
        """
        v_x, v_y, omega, kick_power = low_action
        
        # Create commands based on high-level action
        if high_action == 0:  # GoToBall
            # Use higher power dash commands - RoboCup allows power up to 100
            power = np.clip(np.sqrt(v_x**2 + v_y**2) * 30, 0, 100)  # Scale up power significantly
            direction = np.arctan2(v_y, v_x)
            return [f"dash {power:.1f} {direction:.3f}"]
        
        elif high_action == 1:  # Shoot
            # Use skick with power and direction (RoboCup simulator format)
            kick_power = np.clip(kick_power * 100, 0, 100)  # Scale kick power to 0-100 range
            direction = np.arctan2(v_y, v_x)
            return [f"skick {kick_power:.1f} {direction:.3f}"]
        
        elif high_action == 2:  # Reposition
            # Move and turn with higher power
            power = np.clip(np.sqrt(v_x**2 + v_y**2) * 30, 0, 100)  # Higher power for repositioning
            direction = np.arctan2(v_y, v_x)
            if abs(omega) > 0.1:
                # Turn command - omega is in rad/s, scale it appropriately
                turn_power = np.clip(omega * 50, -180, 180)  # Scale to degrees per step
                return [f"turn {turn_power:.1f}"]
            else:
                return [f"dash {power:.1f} {direction:.3f}"]
        
        return [None]
    
    def _compute_reward(self, game_state, high_action: int) -> float:
        """Compute reward from game state and action taken.
        
        Reward components:
        - Move toward ball: positive reward for reducing distance to ball
        - Move toward goal with ball: positive reward when close to ball and moving toward goal
        - Kick accuracy: reward for kicks when close to ball
        - Movement encouragement: small positive reward for taking actions
        - Position-based rewards: reward for being in good field positions
        """
        if game_state is None:
            return -0.1  # Small penalty for invalid state
        
        total_reward = 0.0
        
        try:
            # Extract current positions
            ball_pos = game_state.ball_pos or (0.0, 0.0)
            ball_x, ball_y = ball_pos
            
            # Get our robot's position
            our_team_poses = game_state.robot_poses.get(self.team_name, [])
            if our_team_poses and len(our_team_poses) > 0:
                first_player = our_team_poses[0]
                if isinstance(first_player, dict) and 1 in first_player:
                    robot_x, robot_y, robot_theta = first_player[1]
                else:
                    return -0.1
            else:
                return -0.1
            
            # Calculate distances
            ball_dist = np.sqrt((ball_x - robot_x)**2 + (ball_y - robot_y)**2)
            goal_dist = np.sqrt((self.goal_x - robot_x)**2 + (0.0 - robot_y)**2)
            ball_to_goal_dist = np.sqrt((self.goal_x - ball_x)**2 + (0.0 - ball_y)**2)
            
            # 1. Ball approach reward
            if self.prev_ball_dist is not None:
                ball_dist_change = self.prev_ball_dist - ball_dist
                total_reward += ball_dist_change * 0.1  # Reward for getting closer to ball
            
            # 2. Close to ball reward
            if ball_dist < 5.0:  # Within 5 meters of ball
                total_reward += 0.2
                if ball_dist < 2.0:  # Very close to ball
                    total_reward += 0.3
            
            # 3. Goal approach reward (when close to ball)
            if ball_dist < 3.0 and self.prev_goal_dist is not None:
                goal_dist_change = self.prev_goal_dist - goal_dist  
                total_reward += goal_dist_change * 0.05  # Smaller reward for goal approach
            
            # 4. Action-specific rewards
            if high_action == 0:  # GoToBall
                if ball_dist > 3.0:  # Far from ball, good to approach
                    total_reward += 0.05
            elif high_action == 1:  # Shoot
                if ball_dist < 2.0:  # Close to ball, good to shoot
                    total_reward += 0.15
                    # Extra reward if shooting toward goal
                    if ball_to_goal_dist > goal_dist:  # Ball is between robot and goal
                        total_reward += 0.1
                else:
                    total_reward -= 0.05  # Penalty for shooting when far from ball
            elif high_action == 2:  # Reposition
                # Reward strategic positioning
                total_reward += 0.02
            
            # 5. Movement encouragement
            if self.prev_robot_pos is not None:
                robot_moved = np.sqrt((robot_x - self.prev_robot_pos[0])**2 + 
                                    (robot_y - self.prev_robot_pos[1])**2)
                total_reward += min(robot_moved * 0.01, 0.05)  # Small reward for movement, capped
            
            # 6. Field position rewards
            # Reward being on the attacking side (positive x)
            if robot_x > 0:
                total_reward += 0.01
            
            # 7. Ball possession reward (very close to ball)
            if ball_dist < 1.0:
                total_reward += 0.1
            
            # 8. Penalty for being too far from play
            if ball_dist > 20.0:
                total_reward -= 0.05
            
            # Update previous values for next step
            self.prev_ball_dist = ball_dist
            self.prev_goal_dist = goal_dist
            self.prev_ball_pos = ball_pos
            self.prev_robot_pos = (robot_x, robot_y)
            
            return total_reward
            
        except Exception as e:
            return -0.1  # Small penalty for errors

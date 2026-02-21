"""
Minimal Discrete Action Soccer Environment.

Simplest possible RL setup:
- 5 raw simulator commands as discrete actions
- No hardcoded logic, no calculations
- Agent learns to combine these primitive commands

Actions:
0. kick 100 0   - Kick forward at full power
1. dash 100 0   - Dash forward at full power  
2. turn 100     - Turn right
3. turn -100    - Turn left
4. dash 50 90   - Dash sideways

Even simpler than our previous "discrete" version which had 
smart action selection logic. This is pure command learning.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from networking.networker import Networker
from ai_interface.constants.player_constants import KICKABLE_MARGIN, PLAYER_SIZE, BALL_SIZE


class SimpleDiscreteEnv(gym.Env):
    """
    Ultra-minimal soccer environment with 5 raw commands.
    
    Perfect for Q-Learning or simple PPO.
    """
    
    def __init__(self, networker: Networker, team_name: str, obs_dim: int = 8):
        super().__init__()
        self.networker = networker
        self.team_name = team_name
        
        self.kickable_dist = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE
        
        # Simple observation: just the basics
        # [ball_dx, ball_dy, ball_dist, goal_dx, goal_dy, goal_dist, cos_theta, sin_theta]
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: 5 discrete commands
        self.action_space = spaces.Discrete(5)
        
        self.max_steps = 200
        self.current_step = 0
        
        # Tracking for rewards
        self.prev_ball_dist = None
        self.prev_ball_to_goal_dist = None
        self.prev_robot_pos = None
        
        # Termination
        self._goal_scored = False
        self._out_of_bounds = False
        
        # Field
        self.field_length = 90.0
        self.field_width = 60.0
        self.goal_x = self.field_length / 2
        
        # Action names for logging
        self.action_names = [
            "KICK_FORWARD",
            "DASH_FORWARD", 
            "TURN_RIGHT",
            "TURN_LEFT",
            "DASH_SIDE"
        ]
    
    def _game_state_to_obs(self, game_state) -> np.ndarray:
        """Convert game state to simple observation."""
        if game_state is None:
            return np.zeros(self.obs_dim, dtype=np.float32)
        
        try:
            # Ball
            ball_pos = game_state.ball_pos or (0.0, 0.0)
            ball_x, ball_y = ball_pos
            
            # Robot
            our_team_poses = game_state.robot_poses.get(self.team_name, [])
            if our_team_poses and len(our_team_poses) > 0:
                first_player = our_team_poses[0]
                if isinstance(first_player, dict) and 1 in first_player:
                    robot_x, robot_y, robot_theta = first_player[1]
                else:
                    robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
            else:
                robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
            
            # Relative positions
            ball_dx = ball_x - robot_x
            ball_dy = ball_y - robot_y
            ball_dist = np.sqrt(ball_dx**2 + ball_dy**2)
            
            goal_dx = self.goal_x - robot_x
            goal_dy = 0.0 - robot_y
            goal_dist = np.sqrt(goal_dx**2 + goal_dy**2)
            
            # Orientation
            theta_rad = np.radians(robot_theta)
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)
            
            # Simple 8-dimensional observation
            obs = np.array([
                ball_dx, ball_dy, ball_dist,
                goal_dx, goal_dy, goal_dist,
                cos_theta, sin_theta
            ], dtype=np.float32)
            
            if len(obs) < self.obs_dim:
                obs = np.pad(obs, (0, self.obs_dim - len(obs)), mode='constant')
            else:
                obs = obs[:self.obs_dim]
            
            return obs
            
        except Exception:
            return np.zeros(self.obs_dim, dtype=np.float32)
    
    def reset(self):
        """Reset environment."""
        try:
            self.networker.reset_sim()
        except Exception:
            pass
        
        import time
        time.sleep(0.3)
        
        for _ in range(3):
            try:
                self.networker.get_game_state()
            except Exception:
                break
            time.sleep(0.05)
        
        game_state = self.networker.get_game_state()
        obs = self._game_state_to_obs(game_state)
        
        self.current_step = 0
        self.prev_ball_dist = None
        self.prev_ball_to_goal_dist = None
        self.prev_robot_pos = None
        self._goal_scored = False
        self._out_of_bounds = False
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action: int):
        """
        Execute one of 5 primitive commands.
        
        Actions:
        0: kick 100 0    - Kick straight ahead
        1: dash 100 0    - Dash forward
        2: turn 100      - Turn right
        3: turn -100     - Turn left  
        4: dash 50 90    - Dash sideways (right)
        """
        # Convert action to command
        command = self._action_to_command(action)
        
        # Execute
        try:
            self.networker.execute_ai_output([command], self.team_name)
        except Exception:
            pass
        
        import time
        time.sleep(0.1)
        
        # Get new state
        game_state = self.networker.get_game_state()
        if game_state is None:
            time.sleep(0.05)
            game_state = self.networker.get_game_state()
        
        obs = self._game_state_to_obs(game_state)
        
        # Compute reward
        self._goal_scored = False
        self._out_of_bounds = False
        reward = self._compute_reward(game_state, action)
        
        self.current_step += 1
        done = (
            self.current_step >= self.max_steps or
            self._goal_scored or
            self._out_of_bounds
        )
        
        info = {
            "action_name": self.action_names[action],
            "goal_scored": self._goal_scored
        }
        
        return obs, reward, done, info
    
    def _action_to_command(self, action: int) -> str:
        """Convert action index to simulator command."""
        if action == 0:
            return "kick 100 0"  # Kick forward, full power
        elif action == 1:
            return "dash 100 0"  # Dash forward, full power
        elif action == 2:
            return "turn 100"    # Turn right
        elif action == 3:
            return "turn -100"   # Turn left
        elif action == 4:
            return "dash 50 90"  # Dash sideways (right)
        else:
            return "dash 0 0"    # Do nothing (shouldn't happen)
    
    def _compute_reward(self, game_state, action: int) -> float:
        """Simple reward function."""
        if game_state is None:
            return -0.1
        
        total_reward = 0.0
        
        try:
            # Extract positions
            ball_pos = game_state.ball_pos or (0.0, 0.0)
            ball_x, ball_y = ball_pos
            
            our_team_poses = game_state.robot_poses.get(self.team_name, [])
            if our_team_poses and len(our_team_poses) > 0:
                first_player = our_team_poses[0]
                if isinstance(first_player, dict) and 1 in first_player:
                    robot_x, robot_y, robot_theta = first_player[1]
                else:
                    return -0.1
            else:
                return -0.1
            
            ball_dist = np.sqrt((ball_x - robot_x)**2 + (ball_y - robot_y)**2)
            ball_to_goal_dist = np.sqrt((self.goal_x - ball_x)**2 + (0.0 - ball_y)**2)
            
            # 1. Reward for getting closer to ball
            if self.prev_ball_dist is not None:
                approach = self.prev_ball_dist - ball_dist
                total_reward += approach * 2.0  # Strong signal
            
            # 2. Reward for ball moving toward goal
            if self.prev_ball_to_goal_dist is not None:
                ball_progress = self.prev_ball_to_goal_dist - ball_to_goal_dist
                if ball_progress > 0.3:  # Ball moved significantly
                    total_reward += ball_progress * 10.0
            
            # 3. Proximity bonus
            if ball_dist < self.kickable_dist:
                total_reward += 1.0
            
            # 4. GOAL!
            if ball_x >= self.goal_x and abs(ball_y) < 7.32 / 2:
                total_reward += 50.0
                self._goal_scored = True
            
            # 5. Out of bounds penalty
            half_length = self.field_length / 2
            half_width = self.field_width / 2
            
            if abs(robot_x) > half_length or abs(robot_y) > half_width:
                total_reward -= 2.0
                self._out_of_bounds = True
            
            # 6. Small alive bonus
            total_reward += 0.05
            
            # Update tracking
            self.prev_ball_dist = ball_dist
            self.prev_ball_to_goal_dist = ball_to_goal_dist
            self.prev_robot_pos = (robot_x, robot_y)
            
            return total_reward
            
        except Exception:
            return -0.1
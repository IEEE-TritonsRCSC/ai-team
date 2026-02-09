"""
Simplified Discrete Action Space for Soccer RL

Instead of learning continuous (vx, vy, omega, kick_power), the agent only
chooses between high-level discrete actions. Each action uses hardcoded logic
from the SmartAttacker class to execute properly.

This reduces the learning problem from:
  - 4 discrete high-level × 4-dimensional continuous low-level
To just:
  - 5 discrete actions total

Much easier to learn!
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from networking.networker import Networker
from ai_interface.constants.player_constants import KICKABLE_MARGIN, PLAYER_SIZE, BALL_SIZE
from ai_interface.constants.field_constants import GOAL_R
import math


class SimplifiedSoccerEnv(gym.Env):
    """
    Simplified Soccer Environment with discrete actions only.
    
    Actions (5 discrete choices):
    0. APPROACH_BALL - Chase the ball intelligently
    1. SHOOT_GOAL - Shoot at the best corner of the goal
    2. DRIBBLE_FORWARD - Dribble ball toward goal
    3. CLEAR_BALL - Kick ball away from danger
    4. REPOSITION - Move to a better position
    """

    def __init__(self, networker: Networker, team_name: str, obs_dim: int = 18):
        super().__init__()
        self.networker = networker
        self.team_name = team_name

        # Kickable distance
        self.kickable_dist = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE

        # Observation space (same as before)
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Action space: 5 discrete actions (MUCH simpler!)
        self.action_space = spaces.Discrete(5)

        # Step counters
        self.max_steps = 200
        self.current_step = 0
        
        # Reward tracking
        self.prev_ball_dist = None
        self.prev_goal_dist = None
        self.prev_ball_pos = None
        self.prev_robot_pos = None
        self.prev_ball_to_goal_dist = None
        
        # Termination flags
        self._out_of_bounds = False
        self._goal_scored = False
        self._ball_out_of_bounds = False
        
        # Field dimensions
        self.field_length = 90.0
        self.field_width = 60.0
        self.goal_x = self.field_length / 2
        
        # Ball velocity history for prediction
        self.ball_history = []
        self.ball_history_max = 5
    
    def _game_state_to_obs(self, game_state) -> np.ndarray:
        """Convert game state to observation (same as curriculum env)."""
        if game_state is None:
            return np.zeros(self.obs_dim, dtype=np.float32)
        
        try:
            ball_pos = game_state.ball_pos or (0.0, 0.0)
            ball_x, ball_y = ball_pos
            ball_vel = game_state.ball_vel or (0.0, 0.0)
            ball_vx, ball_vy = ball_vel
            
            our_team_poses = game_state.robot_poses.get(self.team_name, [])
            if our_team_poses and len(our_team_poses) > 0:
                first_player = our_team_poses[0]
                if isinstance(first_player, dict) and 1 in first_player:
                    robot_x, robot_y, robot_theta = first_player[1]
                else:
                    robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
            else:
                robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
            
            if self.prev_robot_pos is not None:
                robot_vx = robot_x - self.prev_robot_pos[0]
                robot_vy = robot_y - self.prev_robot_pos[1]
            else:
                robot_vx, robot_vy = 0.0, 0.0
            
            ball_dx = ball_x - robot_x
            ball_dy = ball_y - robot_y
            goal_dx = self.goal_x - robot_x
            goal_dy = 0.0 - robot_y
            
            theta_rad = np.radians(robot_theta)
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)
            
            ball_dist = np.sqrt(ball_dx**2 + ball_dy**2)
            goal_dist = np.sqrt(goal_dx**2 + goal_dy**2)
            
            is_kickable = 1.0 if ball_dist < self.kickable_dist else 0.0
            
            to_goal_angle = np.arctan2(goal_dy, goal_dx)
            goal_angle_diff = to_goal_angle - theta_rad
            goal_angle_diff = np.arctan2(np.sin(goal_angle_diff), np.cos(goal_angle_diff))
            
            obs = np.array([
                ball_dx, ball_dy,
                ball_vx, ball_vy,
                goal_dx, goal_dy,
                cos_theta, sin_theta,
                robot_vx, robot_vy,
                ball_dist,
                goal_dist,
                is_kickable,
                goal_angle_diff,
                robot_x, robot_y
            ], dtype=np.float32)
            
            if len(obs) < self.obs_dim:
                obs = np.pad(obs, (0, self.obs_dim - len(obs)), mode='constant')
            else:
                obs = obs[:self.obs_dim]
            
            return obs
            
        except Exception as e:
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
        self.prev_goal_dist = None
        self.prev_ball_pos = None
        self.prev_robot_pos = None
        self.prev_ball_to_goal_dist = None
        self._out_of_bounds = False
        self._goal_scored = False
        self._ball_out_of_bounds = False
        self.ball_history = []
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action: int):
        """
        Execute discrete action.
        
        Args:
            action: Integer 0-4 representing discrete action choice
        """
        # Convert discrete action to simulator commands
        commands = self._action_to_commands(action)
        
        try:
            self.networker.execute_ai_output(commands, self.team_name)
        except Exception:
            pass
        
        import time
        time.sleep(0.1)
        
        game_state = self.networker.get_game_state()
        if game_state is None:
            time.sleep(0.05)
            game_state = self.networker.get_game_state()
        
        obs = self._game_state_to_obs(game_state)
        
        # Reset termination flags before computing reward
        self._out_of_bounds = False
        self._goal_scored = False
        self._ball_out_of_bounds = False
        
        reward = self._compute_reward(game_state, action)
        
        self.current_step += 1
        
        # Check all termination conditions
        done = (
            self.current_step >= self.max_steps or 
            self._out_of_bounds or 
            self._goal_scored or
            self._ball_out_of_bounds
        )
        
        # Provide info about why episode ended
        info = {
            "action": action,
            "goal_scored": self._goal_scored,
            "robot_out_of_bounds": self._out_of_bounds,
            "ball_out_of_bounds": self._ball_out_of_bounds,
            "max_steps_reached": self.current_step >= self.max_steps
        }
        
        return obs, reward, done, info
    
    def _action_to_commands(self, action: int) -> list:
        """
        Convert discrete action to simulator commands using hardcoded logic.
        
        Actions:
        0. APPROACH_BALL - Chase ball (predictive if far, direct if close)
        1. SHOOT_GOAL - Shoot at goal (finds best corner, turns to face)
        2. DRIBBLE_FORWARD - Gentle kick forward
        3. CLEAR_BALL - Strong kick away from danger
        4. REPOSITION - Move to strategic position
        """
        # Get current game state to extract positions
        game_state = self.networker.get_game_state()
        if game_state is None:
            return [None]
        
        try:
            ball_pos = game_state.ball_pos or (0.0, 0.0)
            ball_x, ball_y = ball_pos
            
            our_team_poses = game_state.robot_poses.get(self.team_name, [])
            if not our_team_poses or len(our_team_poses) == 0:
                return [None]
            
            first_player = our_team_poses[0]
            if isinstance(first_player, dict) and 1 in first_player:
                robot_x, robot_y, robot_theta = first_player[1]
            else:
                return [None]
            
            ball_dist = np.sqrt((ball_x - robot_x)**2 + (ball_y - robot_y)**2)
            
            # Update ball history for velocity estimation
            self.ball_history.append((ball_x, ball_y))
            if len(self.ball_history) > self.ball_history_max:
                self.ball_history.pop(0)
            
            # Estimate ball velocity
            ball_vx, ball_vy = 0.0, 0.0
            if len(self.ball_history) >= 2:
                ball_vx = self.ball_history[-1][0] - self.ball_history[-2][0]
                ball_vy = self.ball_history[-1][1] - self.ball_history[-2][1]
            
            # Action 0: APPROACH_BALL
            if action == 0:
                # If far from ball, chase predicted position
                if ball_dist > 3.0:
                    # Predict ball position 3 steps ahead
                    pred_x = ball_x + ball_vx * 3.0
                    pred_y = ball_y + ball_vy * 3.0
                    # Clamp to field
                    pred_x = np.clip(pred_x, -self.field_length/2 + 1, self.field_length/2 - 1)
                    pred_y = np.clip(pred_y, -self.field_width/2 + 1, self.field_width/2 - 1)
                    target_x, target_y = pred_x, pred_y
                else:
                    # Close to ball, approach from behind (relative to goal)
                    to_goal_x = self.goal_x - ball_x
                    to_goal_y = 0.0 - ball_y
                    norm = np.sqrt(to_goal_x**2 + to_goal_y**2) + 1e-6
                    # Position 0.5m behind ball
                    target_x = ball_x - (to_goal_x / norm) * 0.5
                    target_y = ball_y - (to_goal_y / norm) * 0.5
                
                # Calculate dash direction and power
                dx = target_x - robot_x
                dy = target_y - robot_y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < 0.1:
                    return [None]  # Already at target
                
                direction = np.arctan2(dy, dx)
                power = np.clip(distance * 20, 10, 100)
                
                # Turn toward target if not facing it
                theta_rad = np.radians(robot_theta)
                angle_diff = direction - theta_rad
                angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                
                if abs(angle_diff) > np.radians(10):
                    turn_power = np.degrees(angle_diff)
                    return [f"turn {turn_power:.1f}"]
                else:
                    return [f"dash {power:.1f} {direction:.3f}"]
            
            # Action 1: SHOOT_GOAL
            elif action == 1:
                if ball_dist < self.kickable_dist:
                    # In kickable range - shoot at goal
                    # Choose better corner based on simple heuristic
                    goal_y_top = 3.5
                    goal_y_bot = -3.5
                    
                    # Shoot at corner farther from ball's y position
                    if ball_y > 0:
                        target_y = goal_y_bot  # Shoot to opposite corner
                    else:
                        target_y = goal_y_top
                    
                    target_x = self.goal_x
                    
                    # Calculate kick direction
                    to_target_x = target_x - ball_x
                    to_target_y = target_y - ball_y
                    kick_dir = np.arctan2(to_target_y, to_target_x)
                    
                    # Turn to face target if needed
                    theta_rad = np.radians(robot_theta)
                    angle_diff = kick_dir - theta_rad
                    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                    
                    if abs(angle_diff) > np.radians(5):
                        turn_power = np.degrees(angle_diff)
                        return [f"turn {turn_power:.1f}"]
                    else:
                        # Facing target, kick hard!
                        return [f"kick 100 {kick_dir:.3f}"]
                else:
                    # Not in range, approach ball first
                    return self._action_to_commands(0)  # Recursive call to APPROACH
            
            # Action 2: DRIBBLE_FORWARD
            elif action == 2:
                if ball_dist < self.kickable_dist:
                    # Gentle kick toward goal
                    to_goal_x = self.goal_x - ball_x
                    to_goal_y = 0.0 - ball_y
                    
                    # Dribble slightly to the side to avoid defender
                    # (simple heuristic: alternate based on ball y position)
                    side_offset = 1.0 if ball_y < 0 else -1.0
                    
                    target_x = ball_x + to_goal_x * 0.3
                    target_y = ball_y + to_goal_y * 0.3 + side_offset * 0.5
                    
                    to_target_x = target_x - ball_x
                    to_target_y = target_y - ball_y
                    kick_dir = np.arctan2(to_target_y, to_target_x)
                    
                    # Turn to face target
                    theta_rad = np.radians(robot_theta)
                    angle_diff = kick_dir - theta_rad
                    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                    
                    if abs(angle_diff) > np.radians(5):
                        turn_power = np.degrees(angle_diff)
                        return [f"turn {turn_power:.1f}"]
                    else:
                        # Gentle kick (dribble)
                        return [f"kick 30 {kick_dir:.3f}"]
                else:
                    # Not in range, approach ball
                    return self._action_to_commands(0)
            
            # Action 3: CLEAR_BALL
            elif action == 3:
                if ball_dist < self.kickable_dist:
                    # Strong kick forward
                    to_goal_x = self.goal_x - ball_x
                    to_goal_y = 0.0 - ball_y
                    kick_dir = np.arctan2(to_goal_y, to_goal_x)
                    
                    theta_rad = np.radians(robot_theta)
                    angle_diff = kick_dir - theta_rad
                    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                    
                    if abs(angle_diff) > np.radians(5):
                        turn_power = np.degrees(angle_diff)
                        return [f"turn {turn_power:.1f}"]
                    else:
                        return [f"kick 100 {kick_dir:.3f}"]
                else:
                    return self._action_to_commands(0)
            
            # Action 4: REPOSITION
            elif action == 4:
                # Move to a strategic position (between ball and goal)
                mid_x = (robot_x + ball_x) / 2
                mid_y = (robot_y + ball_y) / 2
                
                # Bias toward goal side
                target_x = mid_x + (self.goal_x - ball_x) * 0.2
                target_y = mid_y
                
                dx = target_x - robot_x
                dy = target_y - robot_y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < 0.5:
                    return [None]
                
                direction = np.arctan2(dy, dx)
                power = np.clip(distance * 15, 10, 80)
                
                theta_rad = np.radians(robot_theta)
                angle_diff = direction - theta_rad
                angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                
                if abs(angle_diff) > np.radians(10):
                    turn_power = np.degrees(angle_diff)
                    return [f"turn {turn_power:.1f}"]
                else:
                    return [f"dash {power:.1f} {direction:.3f}"]
            
            return [None]
            
        except Exception as e:
            return [None]
    
    def _compute_reward(self, game_state, action: int) -> float:
        """Compute reward (same as improved env)."""
        if game_state is None:
            return -0.1
        
        total_reward = 0.0
        
        try:
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
            goal_dist = np.sqrt((self.goal_x - robot_x)**2 + (0.0 - robot_y)**2)
            ball_to_goal_dist = np.sqrt((self.goal_x - ball_x)**2 + (0.0 - ball_y)**2)
            
            # Ball approach reward
            if self.prev_ball_dist is not None:
                ball_dist_change = self.prev_ball_dist - ball_dist
                ball_dist_change = np.clip(ball_dist_change, -0.5, 0.5)
                total_reward += ball_dist_change * 1.0
            
            # Facing goal when near ball
            theta_rad = np.radians(robot_theta)
            heading_x, heading_y = np.cos(theta_rad), np.sin(theta_rad)
            if ball_dist < self.kickable_dist * 2:
                to_goal_x = self.goal_x - robot_x
                to_goal_y = 0.0 - robot_y
                to_goal_norm = np.sqrt(to_goal_x**2 + to_goal_y**2) + 1e-8
                facing_goal_cos = (heading_x * to_goal_x + heading_y * to_goal_y) / to_goal_norm
                total_reward += facing_goal_cos * 0.3
            
            # Proximity bonuses
            if ball_dist < self.kickable_dist * 4:
                total_reward += 0.3
                if ball_dist < self.kickable_dist:
                    total_reward += 0.5
            
            # Ball-to-goal approach
            if self.prev_ball_to_goal_dist is not None:
                ball_goal_change = self.prev_ball_to_goal_dist - ball_to_goal_dist
                total_reward += ball_goal_change * 0.5
            
            # GOAL scored! (Episode ends)
            if ball_x >= self.goal_x and abs(ball_y) < 7.32 / 2:
                total_reward += 20.0
                self._goal_scored = True
            
            # Massive kick rewards
            if self.prev_ball_pos is not None:
                ball_moved_toward_goal = self.prev_ball_to_goal_dist - ball_to_goal_dist
                if ball_moved_toward_goal > 0.5:
                    total_reward += 3.0
                if ball_moved_toward_goal > 1.5:
                    total_reward += 2.0
            
            # Movement encouragement
            if self.prev_robot_pos is not None:
                robot_moved = np.sqrt((robot_x - self.prev_robot_pos[0])**2 + 
                                     (robot_y - self.prev_robot_pos[1])**2)
                total_reward += robot_moved * 0.3
                if robot_moved < 0.05:
                    total_reward -= 0.05
            
            # Action-specific bonuses
            if action == 0:  # APPROACH_BALL
                total_reward += 0.05
            elif action == 1:  # SHOOT_GOAL
                if ball_dist < self.kickable_dist:
                    total_reward += 0.5  # Bonus for shooting when possible
            elif action == 2:  # DRIBBLE
                if ball_dist < self.kickable_dist:
                    total_reward += 0.2
            
            total_reward += 0.05  # Alive bonus
            
            # Check robot out of bounds (episode ends with penalty)
            half_length = self.field_length / 2  # 45
            half_width = self.field_width / 2    # 30
            
            if abs(robot_x) > half_length or abs(robot_y) > half_width:
                total_reward -= 1.0  # Penalty for going out
                self._out_of_bounds = True
            
            # Check ball out of bounds (episode ends with penalty)
            # Ball out on sides or back, but NOT through the goal
            ball_out_sides = abs(ball_y) > half_width
            ball_out_back = ball_x < -half_length
            
            if ball_out_sides or ball_out_back:
                total_reward -= 0.5  # Smaller penalty (you might have kicked it out)
                self._ball_out_of_bounds = True
            
            # Update tracking
            self.prev_ball_dist = ball_dist
            self.prev_goal_dist = goal_dist
            self.prev_ball_pos = ball_pos
            self.prev_robot_pos = (robot_x, robot_y)
            self.prev_ball_to_goal_dist = ball_to_goal_dist
            
            return total_reward
            
        except Exception as e:
            return -0.1
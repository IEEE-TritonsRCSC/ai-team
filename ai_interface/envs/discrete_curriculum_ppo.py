"""
Curriculum Learning Environment with DISCRETE Actions.

This combines:
- Discrete actions (5 simple choices) from SimplifiedSoccerEnv
- Curriculum phases (progressive difficulty) from CurriculumSoccerEnv

Perfect for continuing training from your Phase 1-2 model!
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from networking.networker import Networker
from ai_interface.constants.player_constants import KICKABLE_MARGIN, PLAYER_SIZE, BALL_SIZE
from ai_interface.constants.field_constants import GOAL_R
import math


class CurriculumPhase:
    """Defines a single phase of curriculum learning."""
    def __init__(self, name, episodes, reward_fn, success_criteria=None, 
                 ball_start_range=None):
        self.name = name
        self.episodes = episodes
        self.reward_fn = reward_fn
        self.success_criteria = success_criteria
        self.ball_start_range = ball_start_range
        self.episodes_completed = 0
        self.success_count = 0
        
    def is_complete(self):
        """Check if this phase is done."""
        if self.episodes_completed >= self.episodes:
            return True
        if self.success_criteria and self.success_count >= self.success_criteria:
            return True
        return False
    
    def record_episode(self, reward, success=False):
        """Record an episode result."""
        self.episodes_completed += 1
        if success:
            self.success_count += 1


class DiscreteCurriculumEnv(gym.Env):
    """
    Curriculum environment with discrete actions.
    
    Actions (5 discrete):
    0. APPROACH_BALL
    1. SHOOT_GOAL
    2. DRIBBLE_FORWARD
    3. CLEAR_BALL
    4. REPOSITION
    
    Phases:
    1. Ball Contact (0-200 ep): Just touch the ball
    2. Ball Control (200-500 ep): Stay near ball
    3. Kicking Distance (500-800 ep): Kick ball far ← YOUR MODEL NEEDS THIS
    4. Goal Alignment (800-1200 ep): Kick toward goal
    5. Goal Scoring (1200+): Full task
    """
    
    def __init__(self, networker: Networker, team_name: str, obs_dim: int = 18,
                 start_phase: int = 0):
        """
        Args:
            start_phase: Which phase to start from (0-4)
                        Set to 2 to start from Phase 3 (kicking) with your trained model
        """
        super().__init__()
        self.networker = networker
        self.team_name = team_name
        
        self.kickable_dist = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE
        
        # Observation and action spaces (same as SimplifiedSoccerEnv)
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)  # DISCRETE, not hierarchical!
        
        self.max_steps = 200
        self.current_step = 0
        
        # Tracking
        self.prev_ball_dist = None
        self.prev_goal_dist = None
        self.prev_ball_pos = None
        self.prev_robot_pos = None
        self.prev_ball_to_goal_dist = None
        
        # Termination flags
        self._out_of_bounds = False
        self._goal_scored = False
        self._ball_out_of_bounds = False
        
        # Field
        self.field_length = 90.0
        self.field_width = 60.0
        self.goal_x = self.field_length / 2
        
        self.ball_history = []
        self.ball_history_max = 5
        
        # Curriculum
        self.total_episodes = 0
        self.current_phase_idx = start_phase  # Can start from Phase 3!
        self.curriculum = self._create_curriculum()
        
        print(f"\n{'='*60}")
        print(f"DISCRETE CURRICULUM INITIALIZED")
        if start_phase > 0:
            print(f"STARTING FROM: {self.curriculum[start_phase].name}")
        print(f"{'='*60}")
        self._print_curriculum_plan()
    
    def _create_curriculum(self):
        """Create curriculum phases with discrete action rewards."""
        return [
            # PHASE 1: Ball Contact
            CurriculumPhase(
                name="Phase 1: Ball Contact",
                episodes=200,
                reward_fn=self._phase1_reward,
                success_criteria=150,
                ball_start_range=(2.0, 5.0)
            ),
            
            # PHASE 2: Ball Control
            CurriculumPhase(
                name="Phase 2: Ball Control",
                episodes=300,
                reward_fn=self._phase2_reward,
                success_criteria=200,
                ball_start_range=(3.0, 8.0)
            ),
            
            # PHASE 3: Kicking Distance ← YOUR MODEL NEEDS THIS!
            CurriculumPhase(
                name="Phase 3: Kicking Distance",
                episodes=300,
                reward_fn=self._phase3_reward,
                success_criteria=200,
                ball_start_range=(2.0, 10.0)
            ),
            
            # PHASE 4: Goal Alignment
            CurriculumPhase(
                name="Phase 4: Goal Alignment",
                episodes=400,
                reward_fn=self._phase4_reward,
                success_criteria=250,
                ball_start_range=(5.0, 15.0)
            ),
            
            # PHASE 5: Goal Scoring
            CurriculumPhase(
                name="Phase 5: Goal Scoring",
                episodes=float('inf'),
                reward_fn=self._phase5_reward,
                ball_start_range=(5.0, 20.0)
            ),
        ]
    
    def _print_curriculum_plan(self):
        """Print curriculum phases."""
        for i, phase in enumerate(self.curriculum):
            episodes = phase.episodes if phase.episodes != float('inf') else "∞"
            marker = "→" if i == self.current_phase_idx else " "
            print(f"{marker} {i+1}. {phase.name}: {episodes} episodes")
        print(f"{'='*60}\n")
    
    @property
    def current_phase(self):
        return self.curriculum[self.current_phase_idx]
    
    def _advance_phase_if_needed(self):
        """Check if we should move to next phase."""
        if self.current_phase.is_complete() and self.current_phase_idx < len(self.curriculum) - 1:
            old_phase = self.current_phase.name
            self.current_phase_idx += 1
            new_phase = self.current_phase.name
            
            print(f"\n{'='*60}")
            print(f"PHASE COMPLETE! {old_phase}")
            print(f"ADVANCING TO: {new_phase}")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # PHASE REWARD FUNCTIONS (Discrete action versions)
    # ========================================================================
    
    def _phase1_reward(self, robot_x, robot_y, robot_theta, ball_x, ball_y,
                       ball_dist, goal_dist, ball_to_goal_dist, action):
        """Phase 1: Just touch the ball."""
        reward = 0.0
        success = False
        
        # Massive reward for touching
        if ball_dist < self.kickable_dist:
            reward += 10.0
            success = True
        
        # Approach reward
        if self.prev_ball_dist is not None:
            approach = self.prev_ball_dist - ball_dist
            reward += approach * 3.0
        
        # Proximity
        if ball_dist < self.kickable_dist * 2:
            reward += 1.0
        
        # Movement
        if self.prev_robot_pos is not None:
            moved = np.sqrt((robot_x - self.prev_robot_pos[0])**2 + 
                           (robot_y - self.prev_robot_pos[1])**2)
            reward += moved * 0.2
        
        reward += 0.05
        reward += self._check_termination(robot_x, robot_y, ball_x, ball_y)
        
        return reward, success
    
    def _phase2_reward(self, robot_x, robot_y, robot_theta, ball_x, ball_y,
                       ball_dist, goal_dist, ball_to_goal_dist, action):
        """Phase 2: Stay near ball."""
        reward = 0.0
        
        # Continuous reward for being near
        if ball_dist < self.kickable_dist:
            reward += 5.0
        elif ball_dist < self.kickable_dist * 2:
            reward += 2.0
        
        # Approach
        if self.prev_ball_dist is not None:
            approach = self.prev_ball_dist - ball_dist
            reward += approach * 2.0
        
        # Penalty for being far
        if ball_dist > self.kickable_dist * 5:
            reward -= 0.3
        
        # Movement
        if self.prev_robot_pos is not None:
            moved = np.sqrt((robot_x - self.prev_robot_pos[0])**2 + 
                           (robot_y - self.prev_robot_pos[1])**2)
            reward += moved * 0.2
        
        reward += 0.05
        reward += self._check_termination(robot_x, robot_y, ball_x, ball_y)
        
        success = ball_dist < self.kickable_dist * 2
        return reward, success
    
    def _phase3_reward(self, robot_x, robot_y, robot_theta, ball_x, ball_y,
                       ball_dist, goal_dist, ball_to_goal_dist, action):
        """
        Phase 3: KICK BALL FAR (any direction)
        
        THIS IS WHAT YOUR MODEL NEEDS TO LEARN!
        Forces robot to kick hard instead of gentle dribbling.
        """
        reward = 0.0
        success = False
        
        # HUGE reward for ball moving far (from a kick)
        if self.prev_ball_pos is not None:
            ball_moved = np.sqrt((ball_x - self.prev_ball_pos[0])**2 + 
                                (ball_y - self.prev_ball_pos[1])**2)
            
            if ball_moved > 2.0:  # Very powerful kick
                reward += 15.0
                success = True
            elif ball_moved > 1.0:  # Good kick
                reward += 10.0
                success = True
            elif ball_moved > 0.5:  # Decent kick
                reward += 5.0
            elif ball_moved > 0.2:  # Gentle touch
                reward += 1.0
            # NO reward for tiny movements (discourages slow dribbling)
        
        # Reward for being in position to kick
        if ball_dist < self.kickable_dist:
            reward += 1.0  # Smaller than in Phase 2 (don't just stay there)
        
        # Small approach reward
        if self.prev_ball_dist is not None:
            approach = self.prev_ball_dist - ball_dist
            reward += approach * 0.5  # Reduced from Phase 2
        
        # Movement
        if self.prev_robot_pos is not None:
            moved = np.sqrt((robot_x - self.prev_robot_pos[0])**2 + 
                           (robot_y - self.prev_robot_pos[1])**2)
            reward += moved * 0.2
        
        reward += 0.05
        reward += self._check_termination(robot_x, robot_y, ball_x, ball_y)
        
        return reward, success
    
    def _phase4_reward(self, robot_x, robot_y, robot_theta, ball_x, ball_y,
                       ball_dist, goal_dist, ball_to_goal_dist, action):
        """Phase 4: Kick toward goal (directional)."""
        reward = 0.0
        success = False
        
        # Facing goal when near ball
        theta_rad = np.radians(robot_theta)
        heading_x, heading_y = np.cos(theta_rad), np.sin(theta_rad)
        to_goal_x = self.goal_x - robot_x
        to_goal_y = 0.0 - robot_y
        to_goal_norm = np.sqrt(to_goal_x**2 + to_goal_y**2) + 1e-8
        facing_goal_cos = (heading_x * to_goal_x + heading_y * to_goal_y) / to_goal_norm
        
        if ball_dist < self.kickable_dist * 2:
            reward += facing_goal_cos * 3.0  # Strong reward for alignment
        
        # MASSIVE reward for kicking ball toward goal
        if self.prev_ball_to_goal_dist is not None:
            ball_goal_approach = self.prev_ball_to_goal_dist - ball_to_goal_dist
            
            if ball_goal_approach > 1.0:  # Big progress
                reward += 20.0
                success = True
            elif ball_goal_approach > 0.5:  # Good progress
                reward += 10.0
                success = True
            elif ball_goal_approach > 0.0:  # Any progress
                reward += ball_goal_approach * 8.0
        
        # Contact
        if ball_dist < self.kickable_dist:
            reward += 1.0
        
        # Approach
        if self.prev_ball_dist is not None:
            approach = self.prev_ball_dist - ball_dist
            reward += approach * 0.3
        
        # Movement
        if self.prev_robot_pos is not None:
            moved = np.sqrt((robot_x - self.prev_robot_pos[0])**2 + 
                           (robot_y - self.prev_robot_pos[1])**2)
            reward += moved * 0.2
        
        reward += 0.05
        reward += self._check_termination(robot_x, robot_y, ball_x, ball_y)
        
        return reward, success
    
    def _phase5_reward(self, robot_x, robot_y, robot_theta, ball_x, ball_y,
                       ball_dist, goal_dist, ball_to_goal_dist, action):
        """Phase 5: Full task (from simplified_soccer_env.py)."""
        reward = 0.0
        
        # Ball approach
        if self.prev_ball_dist is not None:
            ball_dist_change = self.prev_ball_dist - ball_dist
            ball_dist_change = np.clip(ball_dist_change, -0.5, 0.5)
            reward += ball_dist_change * 1.0
        
        # Facing goal when near ball
        theta_rad = np.radians(robot_theta)
        heading_x, heading_y = np.cos(theta_rad), np.sin(theta_rad)
        if ball_dist < self.kickable_dist * 2:
            to_goal_x = self.goal_x - robot_x
            to_goal_y = 0.0 - robot_y
            to_goal_norm = np.sqrt(to_goal_x**2 + to_goal_y**2) + 1e-8
            facing_goal_cos = (heading_x * to_goal_x + heading_y * to_goal_y) / to_goal_norm
            reward += facing_goal_cos * 0.3
        
        # Proximity
        if ball_dist < self.kickable_dist * 4:
            reward += 0.3
            if ball_dist < self.kickable_dist:
                reward += 0.5
        
        # Ball-to-goal approach
        if self.prev_ball_to_goal_dist is not None:
            ball_goal_change = self.prev_ball_to_goal_dist - ball_to_goal_dist
            reward += ball_goal_change * 0.5
        
        # GOAL scored!
        success = False
        if ball_x >= self.goal_x and abs(ball_y) < 7.32 / 2:
            reward += 20.0
            success = True
            self._goal_scored = True
        
        # Massive kick rewards
        if self.prev_ball_pos is not None:
            ball_moved_toward_goal = self.prev_ball_to_goal_dist - ball_to_goal_dist
            if ball_moved_toward_goal > 0.5:
                reward += 3.0
            if ball_moved_toward_goal > 1.5:
                reward += 2.0
        
        # Movement
        if self.prev_robot_pos is not None:
            robot_moved = np.sqrt((robot_x - self.prev_robot_pos[0])**2 + 
                                 (robot_y - self.prev_robot_pos[1])**2)
            reward += robot_moved * 0.3
            if robot_moved < 0.05:
                reward -= 0.05
        
        # Action-specific
        if action == 0:  # APPROACH
            reward += 0.05
        elif action == 1:  # SHOOT
            if ball_dist < self.kickable_dist:
                reward += 0.5
        elif action == 2:  # DRIBBLE
            if ball_dist < self.kickable_dist:
                reward += 0.2
        
        reward += 0.05
        reward += self._check_termination(robot_x, robot_y, ball_x, ball_y)
        
        return reward, success
    
    def _check_termination(self, robot_x, robot_y, ball_x, ball_y):
        """Check termination conditions, return penalty."""
        penalty = 0.0
        half_length = self.field_length / 2
        half_width = self.field_width / 2
        
        if abs(robot_x) > half_length or abs(robot_y) > half_width:
            penalty -= 1.0
            self._out_of_bounds = True
        
        if abs(ball_y) > half_width or ball_x < -half_length:
            penalty -= 0.5
            self._ball_out_of_bounds = True
        
        return penalty
    
    # ========================================================================
    # STANDARD GYM METHODS (Same as SimplifiedSoccerEnv)
    # ========================================================================
    
    def _game_state_to_obs(self, game_state):
        """Convert game state to observation (same as SimplifiedSoccerEnv)."""
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
        """Execute discrete action (same as SimplifiedSoccerEnv but with curriculum rewards)."""
        # [Action execution code same as SimplifiedSoccerEnv - abbreviated here]
        # Import the full _action_to_commands from SimplifiedSoccerEnv
        
        from ai_interface.envs.simplified_soccer_env import SimplifiedSoccerEnv
        
        # Temporarily create a SimplifiedSoccerEnv instance to use its action conversion
        # (This is a quick way to reuse code - you could also copy the method directly)
        temp_env = SimplifiedSoccerEnv.__new__(SimplifiedSoccerEnv)
        temp_env.networker = self.networker
        temp_env.team_name = self.team_name
        temp_env.kickable_dist = self.kickable_dist
        temp_env.field_length = self.field_length
        temp_env.field_width = self.field_width
        temp_env.goal_x = self.goal_x
        temp_env.prev_ball_dist = self.prev_ball_dist
        temp_env.prev_robot_pos = self.prev_robot_pos
        temp_env.prev_ball_pos = self.prev_ball_pos
        temp_env.ball_history = self.ball_history
        temp_env.ball_history_max = self.ball_history_max
        
        commands = temp_env._action_to_commands(action)
        
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
        
        # Reset termination flags
        self._out_of_bounds = False
        self._goal_scored = False
        self._ball_out_of_bounds = False
        
        # Compute reward using CURRICULUM phase
        reward, success = self._compute_curriculum_reward(game_state, action)
        
        self.current_step += 1
        
        done = (
            self.current_step >= self.max_steps or
            self._out_of_bounds or
            self._goal_scored or
            self._ball_out_of_bounds
        )
        
        if done:
            self.current_phase.record_episode(reward, success)
            self.total_episodes += 1
            self._advance_phase_if_needed()
        
        info = {
            "phase": self.current_phase.name,
            "phase_episode": self.current_phase.episodes_completed,
            "total_episodes": self.total_episodes,
            "goal_scored": self._goal_scored
        }
        
        return obs, reward, done, info
    
    def _compute_curriculum_reward(self, game_state, action):
        """Compute reward using current phase's reward function."""
        if game_state is None:
            return -0.1, False
        
        try:
            ball_pos = game_state.ball_pos or (0.0, 0.0)
            ball_x, ball_y = ball_pos
            
            our_team_poses = game_state.robot_poses.get(self.team_name, [])
            if our_team_poses and len(our_team_poses) > 0:
                first_player = our_team_poses[0]
                if isinstance(first_player, dict) and 1 in first_player:
                    robot_x, robot_y, robot_theta = first_player[1]
                else:
                    return -0.1, False
            else:
                return -0.1, False
            
            ball_dist = np.sqrt((ball_x - robot_x)**2 + (ball_y - robot_y)**2)
            goal_dist = np.sqrt((self.goal_x - robot_x)**2 + (0.0 - robot_y)**2)
            ball_to_goal_dist = np.sqrt((self.goal_x - ball_x)**2 + (0.0 - ball_y)**2)
            
            # Call current phase's reward function
            reward, success = self.current_phase.reward_fn(
                robot_x, robot_y, robot_theta,
                ball_x, ball_y,
                ball_dist, goal_dist, ball_to_goal_dist,
                action
            )
            
            # Update tracking
            self.prev_ball_dist = ball_dist
            self.prev_goal_dist = goal_dist
            self.prev_ball_pos = ball_pos
            self.prev_robot_pos = (robot_x, robot_y)
            self.prev_ball_to_goal_dist = ball_to_goal_dist
            
            return reward, success
            
        except Exception as e:
            return -0.1, False
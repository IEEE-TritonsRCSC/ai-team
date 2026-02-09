import gymnasium as gym
from gymnasium import spaces
import numpy as np
from networking.networker import Networker
from ai_interface.constants.player_constants import KICKABLE_MARGIN, PLAYER_SIZE, BALL_SIZE
from ai_interface.constants.field_constants import GOAL_R


class CurriculumPhase:
    """Defines a single phase of curriculum learning."""
    def __init__(self, name, episodes, reward_fn, success_criteria=None, ball_start_range=None):
        self.name = name
        self.episodes = episodes  # How many episodes to spend in this phase
        self.reward_fn = reward_fn  # Custom reward function for this phase
        self.success_criteria = success_criteria  # Optional: auto-advance if met
        self.ball_start_range = ball_start_range  # Where to spawn ball (for harder phases)
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


class CurriculumSoccerEnv(gym.Env):
    """
    Curriculum-based Soccer Environment for progressive learning.
    
    Phases:
    1. Ball Contact (0-200 ep): Just touch the ball
    2. Ball Control (200-500 ep): Stay near ball for multiple steps
    3. Directional Kicking (500-1000 ep): Kick ball forward
    4. Goal Alignment (1000-1500 ep): Face goal before kicking
    5. Goal Scoring (1500+): Full task
    """

    def __init__(self, networker: Networker, team_name: str, obs_dim: int = 18, 
                 curriculum_file: str = None):
        super().__init__()
        self.networker = networker
        self.team_name = team_name

        # Kickable distance
        self.kickable_dist = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE

        # Observation space
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Action space (hierarchical)
        self.action_space = spaces.Dict({
            "high_level": spaces.Discrete(4),
            "low_level": spaces.Box(
                low=np.array([-2.0, -2.0, -2.0, 0.0]),
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
        self.prev_ball_to_goal_dist = None
        
        self._recent_positions: list = []
        self._out_of_bounds = False
        
        # Field dimensions
        self.field_length = 90.0
        self.field_width = 60.0
        self.goal_x = self.field_length / 2
        
        # Curriculum setup
        self.total_episodes = 0
        self.current_phase_idx = 0
        self.curriculum = self._create_curriculum()
        
        print(f"\n{'='*60}")
        print(f"CURRICULUM LEARNING INITIALIZED")
        print(f"{'='*60}")
        self._print_curriculum_plan()
    
    def _create_curriculum(self):
        """Create the curriculum phases."""
        return [
            # PHASE 1: Ball Contact - just learn to touch the ball
            CurriculumPhase(
                name="Phase 1: Ball Contact",
                episodes=200,
                reward_fn=self._phase1_reward,
                success_criteria=150,  # Auto-advance if 150 successful touches
                ball_start_range=(2.0, 5.0)  # Ball spawns 2-5m away
            ),
            
            # PHASE 2: Ball Control - stay near ball
            CurriculumPhase(
                name="Phase 2: Ball Control", 
                episodes=300,
                reward_fn=self._phase2_reward,
                success_criteria=200,  # 200 episodes with good control
                ball_start_range=(3.0, 8.0)  # Ball spawns 3-8m away
            ),
            
            # PHASE 3: Directional Kicking - kick ball forward
            CurriculumPhase(
                name="Phase 3: Directional Kicking",
                episodes=500,
                reward_fn=self._phase3_reward,
                success_criteria=300,
                ball_start_range=(2.0, 10.0)  # Variable distance
            ),
            
            # PHASE 4: Goal Alignment - face goal before kicking
            CurriculumPhase(
                name="Phase 4: Goal Alignment",
                episodes=500,
                reward_fn=self._phase4_reward,
                success_criteria=300,
                ball_start_range=(5.0, 15.0)
            ),
            
            # PHASE 5: Full Task - score goals!
            CurriculumPhase(
                name="Phase 5: Goal Scoring",
                episodes=float('inf'),  # Never ends
                reward_fn=self._phase5_reward,
                ball_start_range=(5.0, 20.0)
            ),
        ]
    
    def _print_curriculum_plan(self):
        """Print the curriculum plan."""
        for i, phase in enumerate(self.curriculum):
            episodes = phase.episodes if phase.episodes != float('inf') else "∞"
            print(f"{i+1}. {phase.name}: {episodes} episodes")
        print(f"{'='*60}\n")
    
    @property
    def current_phase(self):
        """Get the current curriculum phase."""
        return self.curriculum[self.current_phase_idx]
    
    def _advance_phase_if_needed(self):
        """Check if we should move to next phase."""
        if self.current_phase.is_complete() and self.current_phase_idx < len(self.curriculum) - 1:
            old_phase = self.current_phase.name
            self.current_phase_idx += 1
            new_phase = self.current_phase.name
            
            print(f"\n{'='*60}")
            print(f"CURRICULUM PHASE COMPLETE!")
            print(f"Completed: {old_phase}")
            print(f"Advancing to: {new_phase}")
            print(f"{'='*60}\n")
    
    def _spawn_ball_at_distance(self):
        """Spawn ball at curriculum-appropriate distance."""
        phase = self.current_phase
        if phase.ball_start_range is None:
            return  # Use default spawn
        
        min_dist, max_dist = phase.ball_start_range
        
        # Random distance and angle
        distance = np.random.uniform(min_dist, max_dist)
        angle = np.random.uniform(-np.pi, np.pi)
        
        # Assume robot starts near (0, 0), place ball relative
        ball_x = distance * np.cos(angle)
        ball_y = distance * np.sin(angle)
        
        # Clamp to field boundaries
        ball_x = np.clip(ball_x, -self.field_length/2 + 5, self.field_length/2 - 5)
        ball_y = np.clip(ball_y, -self.field_width/2 + 5, self.field_width/2 - 5)
        
        # Use networker to move ball (if API supports it)
        try:
            # This depends on your networker API - adjust as needed
            self.networker.move_ball(ball_x, ball_y)
        except:
            pass  # If not supported, ball spawns at default location

    # ========================================================================
    # PHASE REWARD FUNCTIONS
    # ========================================================================
    
    def _phase1_reward(self, game_state, high_action, robot_x, robot_y, robot_theta, 
                       ball_x, ball_y, ball_dist, goal_dist, ball_to_goal_dist):
        """
        Phase 1: BALL CONTACT
        Goal: Just touch the ball. That's it.
        Success metric: ball_dist < kickable_dist
        """
        reward = 0.0
        success = False
        
        # MASSIVE reward for touching ball
        if ball_dist < self.kickable_dist:
            reward += 10.0
            success = True
        
        # Strong reward for approaching
        if self.prev_ball_dist is not None:
            approach = self.prev_ball_dist - ball_dist
            reward += approach * 3.0  # 3x stronger than final phase
        
        # Proximity bonus
        if ball_dist < self.kickable_dist * 2:
            reward += 1.0
        if ball_dist < self.kickable_dist * 4:
            reward += 0.5
        
        # Small movement bonus
        if self.prev_robot_pos is not None:
            moved = np.sqrt((robot_x - self.prev_robot_pos[0])**2 + 
                           (robot_y - self.prev_robot_pos[1])**2)
            reward += moved * 0.2
        
        # Small alive bonus
        reward += 0.05
        
        return reward, success
    
    def _phase2_reward(self, game_state, high_action, robot_x, robot_y, robot_theta,
                       ball_x, ball_y, ball_dist, goal_dist, ball_to_goal_dist):
        """
        Phase 2: BALL CONTROL
        Goal: Stay near the ball for extended periods
        Success metric: Average ball_dist < kickable_dist * 2 over episode
        """
        reward = 0.0
        
        # Continuous reward for being near ball
        if ball_dist < self.kickable_dist:
            reward += 5.0  # Very high reward for staying in contact
        elif ball_dist < self.kickable_dist * 2:
            reward += 2.0
        elif ball_dist < self.kickable_dist * 4:
            reward += 0.5
        
        # Approach reward
        if self.prev_ball_dist is not None:
            approach = self.prev_ball_dist - ball_dist
            reward += approach * 2.0
        
        # Penalty for being far
        if ball_dist > self.kickable_dist * 5:
            reward -= 0.3
        
        # Movement bonus
        if self.prev_robot_pos is not None:
            moved = np.sqrt((robot_x - self.prev_robot_pos[0])**2 + 
                           (robot_y - self.prev_robot_pos[1])**2)
            reward += moved * 0.2
        
        reward += 0.05
        
        # Success: stayed close most of the episode
        success = ball_dist < self.kickable_dist * 2
        return reward, success
    
    def _phase3_reward(self, game_state, high_action, robot_x, robot_y, robot_theta,
                       ball_x, ball_y, ball_dist, goal_dist, ball_to_goal_dist):
        """
        Phase 3: DIRECTIONAL KICKING
        Goal: Kick the ball in ANY consistent direction
        Success metric: Ball moves 1m+ from a kick
        """
        reward = 0.0
        success = False
        
        # Huge reward for kicking ball far
        if self.prev_ball_pos is not None:
            ball_moved = np.sqrt((ball_x - self.prev_ball_pos[0])**2 + 
                                (ball_y - self.prev_ball_pos[1])**2)
            
            if ball_moved > 1.0:  # Ball moved 1m+ (from a kick!)
                reward += 8.0
                success = True
            elif ball_moved > 0.5:
                reward += 3.0
            elif ball_moved > 0.2:
                reward += 1.0
        
        # Reward for being in position to kick
        if ball_dist < self.kickable_dist:
            reward += 2.0
        
        # Approach reward (smaller now)
        if self.prev_ball_dist is not None:
            approach = self.prev_ball_dist - ball_dist
            reward += approach * 1.0
        
        # Movement
        if self.prev_robot_pos is not None:
            moved = np.sqrt((robot_x - self.prev_robot_pos[0])**2 + 
                           (robot_y - self.prev_robot_pos[1])**2)
            reward += moved * 0.2
        
        reward += 0.05
        return reward, success
    
    def _phase4_reward(self, game_state, high_action, robot_x, robot_y, robot_theta,
                       ball_x, ball_y, ball_dist, goal_dist, ball_to_goal_dist):
        """
        Phase 4: GOAL ALIGNMENT
        Goal: Face the goal BEFORE kicking
        Success metric: Kick ball while facing goal (angle < 30°)
        """
        reward = 0.0
        success = False
        
        # Calculate facing direction
        theta_rad = np.radians(robot_theta)
        heading_x, heading_y = np.cos(theta_rad), np.sin(theta_rad)
        to_goal_x = self.goal_x - robot_x
        to_goal_y = 0.0 - robot_y
        to_goal_norm = np.sqrt(to_goal_x**2 + to_goal_y**2) + 1e-8
        facing_goal_cos = (heading_x * to_goal_x + heading_y * to_goal_y) / to_goal_norm
        
        # MAJOR reward for facing goal when near ball
        if ball_dist < self.kickable_dist * 2:
            reward += facing_goal_cos * 3.0  # Up to +3 for perfect alignment
        
        # Reward for kicking ball toward goal
        if self.prev_ball_to_goal_dist is not None:
            ball_goal_approach = self.prev_ball_to_goal_dist - ball_to_goal_dist
            if ball_goal_approach > 0.5:
                reward += 10.0  # Huge reward for good kicks!
                success = True
            elif ball_goal_approach > 0.0:
                reward += ball_goal_approach * 5.0
        
        # Ball contact
        if ball_dist < self.kickable_dist:
            reward += 1.0
        
        # Approach
        if self.prev_ball_dist is not None:
            approach = self.prev_ball_dist - ball_dist
            reward += approach * 0.5
        
        # Movement
        if self.prev_robot_pos is not None:
            moved = np.sqrt((robot_x - self.prev_robot_pos[0])**2 + 
                           (robot_y - self.prev_robot_pos[1])**2)
            reward += moved * 0.2
        
        reward += 0.05
        return reward, success
    
    def _phase5_reward(self, game_state, high_action, robot_x, robot_y, robot_theta,
                       ball_x, ball_y, ball_dist, goal_dist, ball_to_goal_dist):
        """
        Phase 5: FULL TASK (this is your improved reward function from before)
        Goal: Score goals!
        """
        reward = 0.0
        
        # Use the full reward function from ppo_env_improved.py
        # (I'll just copy the key parts here)
        
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
        
        # Proximity bonuses
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
        
        reward += 0.05
        return reward, success
    
    # ========================================================================
    # STANDARD GYM METHODS
    # ========================================================================
    
    def _game_state_to_obs(self, game_state) -> np.ndarray:
        """Convert game state to observation (same as before)."""
        if game_state is None:
            return np.zeros(self.obs_dim, dtype=np.float32)
        
        try:
            # [Same observation code as ppo_env_improved.py]
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
        """Reset environment and potentially spawn ball at curriculum distance."""
        try:
            self.networker.reset_sim()
        except Exception:
            pass
        
        import time
        time.sleep(0.3)
        
        # Spawn ball at curriculum-appropriate distance
        self._spawn_ball_at_distance()
        
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
        self._recent_positions = []
        self._out_of_bounds = False
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Execute action and return curriculum-specific reward."""
        high = action["high_level"]
        low = action["low_level"]
        
        commands = self._action_to_commands(high, low)
        
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
        
        # Compute reward using current phase's reward function
        reward, success = self._compute_reward_curriculum(game_state, high)
        
        self.current_step += 1
        done = (self.current_step >= self.max_steps) or self._out_of_bounds
        
        # Record episode result if done
        if done:
            self.current_phase.record_episode(reward, success)
            self.total_episodes += 1
            self._advance_phase_if_needed()
        
        info = {
            "phase": self.current_phase.name,
            "phase_episode": self.current_phase.episodes_completed,
            "total_episodes": self.total_episodes
        }
        
        return obs, reward, done, info
    
    def _compute_reward_curriculum(self, game_state, high_action):
        """Compute reward using current curriculum phase."""
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
                game_state, high_action, robot_x, robot_y, robot_theta,
                ball_x, ball_y, ball_dist, goal_dist, ball_to_goal_dist
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
    
    def _action_to_commands(self, high_action: int, low_action: np.ndarray) -> list:
        """Convert actions to simulator commands (same as before)."""
        v_x, v_y, omega, kick_power = low_action
        
        power = np.clip(np.sqrt(v_x**2 + v_y**2) * 15, 0, 60)
        direction = np.arctan2(v_y, v_x)
        
        if high_action == 0:  # GoToBall
            if self.prev_ball_dist is not None and self.prev_ball_dist < self.kickable_dist:
                goal_x, goal_y = GOAL_R[0], GOAL_R[1]
                if self.prev_robot_pos is not None:
                    kick_dir = np.arctan2(goal_y - self.prev_robot_pos[1],
                                          goal_x - self.prev_robot_pos[0])
                else:
                    kick_dir = 0.0
                return [f"kick 80 {kick_dir:.3f}"]
            return [f"dash {power:.1f} {direction:.3f}"]
        
        elif high_action == 1:  # Shoot
            kick_pwr = np.clip(kick_power * 100, 10, 100)
            return [f"kick {kick_pwr:.1f} {direction:.3f}"]
        
        elif high_action == 2:  # Reposition
            if abs(omega) > 0.1:
                turn_power = np.clip(omega * 50, -180, 180)
                return [f"turn {turn_power:.1f}"]
            else:
                return [f"dash {power:.1f} {direction:.3f}"]
        
        elif high_action == 3:  # CatchHold
            if self.prev_ball_dist is not None and self.prev_ball_dist < self.kickable_dist:
                return [f"catch 0"]
            else:
                return [f"dash {power:.1f} {direction:.3f}"]
        
        return [None]
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from networking.networker import Networker
from ai_interface.constants.player_constants import KICKABLE_MARGIN, PLAYER_SIZE, BALL_SIZE
from ai_interface.constants.field_constants import GOAL_R


class SoccerEnv(gym.Env):
    """
    Hierarchical Soccer Environment
    High-level: discrete actions (GoToBall, Shoot, Reposition, CatchHold)
    Low-level: continuous actions (v_x, v_y, omega, kick_power)
    """

    def __init__(self, networker: Networker, team_name: str, obs_dim: int = 18):
        super().__init__()
        self.networker = networker
        self.team_name = team_name

        # Kickable distance (from server defaults)
        self.kickable_dist = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE  # ~1.215

        # Observation space: e.g., [ball_dx, ball_dy, goal_dx, goal_dy, cos(theta), sin(theta), vx, vy, ...]
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Action space (hierarchical)
        self.action_space = spaces.Dict({
            "high_level": spaces.Discrete(4),  # GoToBall, Shoot, Reposition, CatchHold
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
        
        # Stagnation detection – penalise if robot barely moves over a window
        self._stagnation_window = 5            # number of steps to look back
        self._stagnation_threshold = 0.5       # metres – less than this = stagnant
        self._recent_positions: list = []       # rolling buffer of (x, y)
        self._out_of_bounds = False             # set by reward, checked by step()
        
        # Field dimensions – must match rcssserver (FIELD_X=[-45,45], FIELD_Y=[-30,30])
        self.field_length = 90.0   # metres (not FIFA 105)
        self.field_width  = 60.0   # metres (not FIFA 68)
        self.goal_x = self.field_length / 2  # Goal at x = 45
        
    def _game_state_to_obs(self, game_state) -> np.ndarray:
        """Convert game state to observation vector."""
        if game_state is None:
            return np.zeros(self.obs_dim, dtype=np.float32)
        
        try:
            # Extract ball position and velocity
            ball_pos = game_state.ball_pos or (0.0, 0.0)
            ball_x, ball_y = ball_pos
            ball_vel = game_state.ball_vel or (0.0, 0.0)
            ball_vx, ball_vy = ball_vel
            
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
            
            # Compute robot velocity (if we have previous position)
            if self.prev_robot_pos is not None:
                robot_vx = robot_x - self.prev_robot_pos[0]
                robot_vy = robot_y - self.prev_robot_pos[1]
            else:
                robot_vx, robot_vy = 0.0, 0.0
            
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
            
            # Kickable flag (binary: can kick right now?)
            is_kickable = 1.0 if ball_dist < self.kickable_dist else 0.0
            
            # Goal angle relative to robot heading (how much to turn to face goal)
            to_goal_angle = np.arctan2(goal_dy, goal_dx)
            goal_angle_diff = to_goal_angle - theta_rad
            # Normalize to [-pi, pi]
            goal_angle_diff = np.arctan2(np.sin(goal_angle_diff), np.cos(goal_angle_diff))
            
            # Construct observation vector (expanded from 10 to 18 dimensions)
            obs = np.array([
                ball_dx, ball_dy,           # Ball position relative to robot
                ball_vx, ball_vy,           # Ball velocity (NEW)
                goal_dx, goal_dy,           # Goal position relative to robot
                cos_theta, sin_theta,       # Robot orientation
                robot_vx, robot_vy,         # Robot velocity (NEW)
                ball_dist,                  # Distance to ball
                goal_dist,                  # Distance to goal
                is_kickable,                # Can kick now? (NEW)
                goal_angle_diff,            # Angle to turn to face goal (NEW)
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
        """Reset simulator: move players & ball to initial positions, then observe."""
        # Move all players and ball back to starting positions via the monitor
        try:
            self.networker.reset_sim()
        except Exception:
            pass
        
        # Give the simulator enough time to process all the moves
        import time
        time.sleep(0.3)
        
        # Drain any stale game-state packets so we get a fresh observation
        for _ in range(3):
            try:
                self.networker.get_game_state()
            except Exception:
                break
            time.sleep(0.05)
        
        # Get fresh initial observation
        game_state = self.networker.get_game_state()
        obs = self._game_state_to_obs(game_state)
        self.current_step = 0
        
        # Reset reward tracking
        self.prev_ball_dist = None
        self.prev_goal_dist = None
        self.prev_ball_pos = None
        self.prev_robot_pos = None
        self.prev_ball_to_goal_dist = None
        self._recent_positions = []
        self._out_of_bounds = False
        
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

        # Give the simulator time to process the command and advance one tick
        import time
        time.sleep(0.1)

        # Get next game state (may be None on socket timeout – retry once)
        game_state = self.networker.get_game_state()
        if game_state is None:
            time.sleep(0.05)
            game_state = self.networker.get_game_state()
        obs = self._game_state_to_obs(game_state)
        
        # Compute reward
        self._out_of_bounds = False  # reset before reward computation
        reward = self._compute_reward(game_state, high)
        
        self.current_step += 1

        # Terminate if max steps reached OR robot left the field
        done = self.current_step >= self.max_steps or self._out_of_bounds

        return np.array(obs, dtype=np.float32), reward, done, {}
    
    def _action_to_commands(self, high_action: int, low_action: np.ndarray) -> list:
        """Convert hierarchical action to simulator command strings.
        
        Args:
            high_action: High-level action (0=GoToBall, 1=Shoot, 2=Reposition, 3=CatchHold)
            low_action: Low-level continuous actions [v_x, v_y, omega, kick_power]
            
        Returns:
            List of command strings for each player
        """
        v_x, v_y, omega, kick_power = low_action

        # Pre-compute common values
        # Keep dash power moderate so the robot doesn't rocket off the field
        power = np.clip(np.sqrt(v_x**2 + v_y**2) * 15, 0, 60)
        direction = np.arctan2(v_y, v_x)
        
        if high_action == 0:  # GoToBall
            # If we're within kickable range, auto-kick toward the goal
            # so the robot doesn't just bump into the ball with a dash.
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

        elif high_action == 3:  # CatchHold (trap the ball)
            # Issue catch to hold the ball, then dash gently to stay near it
            if self.prev_ball_dist is not None and self.prev_ball_dist < self.kickable_dist:
                return [f"catch 0"]
            else:
                # Not close enough to catch – dash toward ball instead
                return [f"dash {power:.1f} {direction:.3f}"]
        
        return [None]
    
    def _compute_reward(self, game_state, high_action: int) -> float:
        """Compute reward from game state and action taken.

        Reward components (ordered by importance):
        1. Ball approach – large reward for reducing distance to ball
        2. Facing ball   – reward when the robot's heading points toward the ball
        3. Stagnation    – penalty when the robot barely moves over the last N steps
        4. Proximity bonuses, goal approach, action-specific, field position
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

            # ------------------------------------------------------------------
            # 1. Ball approach reward  (primary learning signal)
            #    CRITICAL: Clip this to prevent huge negative spikes that destroy
            #    the policy when the robot accidentally moves away from the ball
            # ------------------------------------------------------------------
            if self.prev_ball_dist is not None:
                ball_dist_change = self.prev_ball_dist - ball_dist  # positive = closer
                # Clip to [-0.5, 0.5] so one bad move doesn't create -2.0 reward
                ball_dist_change = np.clip(ball_dist_change, -0.5, 0.5)
                total_reward += ball_dist_change * 1.0  # strongest single signal

            # ------------------------------------------------------------------
            # 2. Facing-ball reward
            #    Compute cosine similarity between robot heading and ball direction
            # ------------------------------------------------------------------
            theta_rad = np.radians(robot_theta)
            heading_x, heading_y = np.cos(theta_rad), np.sin(theta_rad)
            to_ball_x, to_ball_y = ball_x - robot_x, ball_y - robot_y
            to_ball_norm = np.sqrt(to_ball_x**2 + to_ball_y**2) + 1e-8
            # cosine ∈ [-1, 1]; reward in [-0.1, 0.1]
            facing_cos = (heading_x * to_ball_x + heading_y * to_ball_y) / to_ball_norm
            total_reward += facing_cos * 0.1
            
            # 2b. Facing-GOAL reward when near ball (encourages lining up shots)
            #     This is MORE important than facing ball when you're close!
            if ball_dist < self.kickable_dist * 2:  # Within ~2.5 meters
                to_goal_x = self.goal_x - robot_x
                to_goal_y = 0.0 - robot_y
                to_goal_norm = np.sqrt(to_goal_x**2 + to_goal_y**2) + 1e-8
                facing_goal_cos = (heading_x * to_goal_x + heading_y * to_goal_y) / to_goal_norm
                total_reward += facing_goal_cos * 0.3  # Stronger than facing ball!

            # ------------------------------------------------------------------
            # 3. Stagnation detection - REMOVED
            #    This was causing the death spiral by double-penalizing stillness.
            #    Movement rewards in section 7 already handle this.
            # ------------------------------------------------------------------
            self._recent_positions.append((robot_x, robot_y))
            if len(self._recent_positions) > self._stagnation_window:
                self._recent_positions.pop(0)
            
            # (Stagnation penalty removed - let movement rewards handle it)

            # ------------------------------------------------------------------
            # 4. Proximity bonuses  (always positive – attract to ball)
            # ------------------------------------------------------------------
            if ball_dist < self.kickable_dist * 4:  # ~5m
                total_reward += 0.3
                if ball_dist < self.kickable_dist:   # actually kickable
                    total_reward += 0.5

            # 5. Ball-to-goal approach reward (fires every step)
            if self.prev_ball_to_goal_dist is not None:
                ball_goal_change = self.prev_ball_to_goal_dist - ball_to_goal_dist
                total_reward += ball_goal_change * 0.5

            # 5b. GOAL scored!  Ball crossed the goal line.
            if ball_x >= self.goal_x and abs(ball_y) < 7.32 / 2:
                total_reward += 10.0
            
            # 5c. MASSIVE reward for kicking ball toward goal (the main objective!)
            #     This fires when the ball moves significantly closer to goal
            if self.prev_ball_pos is not None:
                # Check if ball moved toward goal (not just robot moving toward goal)
                ball_moved_toward_goal = self.prev_ball_to_goal_dist - ball_to_goal_dist
                
                # If ball moved 0.5m+ toward goal, it was likely a good kick!
                if ball_moved_toward_goal > 0.5:
                    total_reward += 3.0  # HUGE reward for successful kicks toward goal
                    
                # Extra bonus if it was a powerful kick (>1m toward goal)
                if ball_moved_toward_goal > 1.5:
                    total_reward += 2.0  # Even bigger for really good kicks!

            # 6. Action-specific rewards (only positives – remove punishments
            #    for "wrong" actions; the policy should explore freely)
            if high_action == 0:  # GoToBall
                total_reward += 0.05  # always mildly encourage approaching ball
            elif high_action == 1:  # Shoot
                if ball_dist < self.kickable_dist:
                    total_reward += 0.3
            elif high_action == 2:  # Reposition
                total_reward += 0.02
            elif high_action == 3:  # CatchHold
                if ball_dist < self.kickable_dist:
                    total_reward += 0.3

            # 7. Movement encouragement (per-step) – THIS IS THE KEY ANTI-COLLAPSE SIGNAL
            #    Make it impossible for "do nothing" to be optimal
            if self.prev_robot_pos is not None:
                robot_moved = np.sqrt((robot_x - self.prev_robot_pos[0])**2 +
                                      (robot_y - self.prev_robot_pos[1])**2)
                # Reward ANY movement - the more you move, the better
                # Remove the cap so large movements are strongly rewarded
                total_reward += robot_moved * 0.3
                
                # Much gentler penalty for being nearly frozen
                # (was -0.2, which created a death spiral)
                if robot_moved < 0.05:
                    total_reward -= 0.05

            # 8. Alive bonus – reward for not ending the episode
            total_reward += 0.05

            # 9. Ball possession reward
            if ball_dist < self.kickable_dist:
                total_reward += 0.2

            # 10. (removed far-from-play penalty – was training inaction)

            # 11. Boundary awareness
            half_length = self.field_length / 2  # 45
            half_width = self.field_width / 2    # 30

            # a) Graduated boundary repulsion
            margin = 5.0
            x_over = max(0.0, abs(robot_x) - (half_length - margin))
            y_over = max(0.0, abs(robot_y) - (half_width  - margin))
            boundary_pen = (x_over + y_over) / margin
            total_reward -= boundary_pen * 0.1   # softer: up to -0.2 at edge

            # b) Hard out-of-bounds – episode ends
            if abs(robot_x) > half_length or abs(robot_y) > half_width:
                total_reward -= 0.5
                self._out_of_bounds = True

            # Update previous values for next step
            self.prev_ball_dist = ball_dist
            self.prev_goal_dist = goal_dist
            self.prev_ball_pos = ball_pos
            self.prev_robot_pos = (robot_x, robot_y)
            self.prev_ball_to_goal_dist = ball_to_goal_dist

            return total_reward

        except Exception as e:
            return -0.1  # Small penalty for errors
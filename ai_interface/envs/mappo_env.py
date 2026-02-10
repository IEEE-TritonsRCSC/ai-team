"""
Multi-Agent Soccer Environment for MAPPO.

Handles multiple agents per team learning to coordinate.
Key features:
- Each agent gets its own observation (local view)
- Agents share team reward (cooperative)
- Supports passing, positioning, and coordinated play
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from networking.networker import Networker
from ai_interface.constants.player_constants import KICKABLE_MARGIN, PLAYER_SIZE, BALL_SIZE
from ai_interface.constants.field_constants import GOAL_R
import math


class MultiAgentSoccerEnv(gym.Env):
    """
    Multi-agent soccer environment.
    
    Actions (6 discrete per agent):
    0. APPROACH_BALL - Move toward ball
    1. SHOOT_GOAL - Shoot at goal
    2. PASS_TEAMMATE - Pass to nearest teammate
    3. DRIBBLE_FORWARD - Dribble toward goal
    4. CLEAR_BALL - Strong kick away
    5. REPOSITION - Move to strategic position
    
    Observations:
    Each agent observes:
    - Ball position/velocity relative to self
    - Goal position relative to self
    - Own position/velocity/orientation
    - Teammates' positions (2 nearest)
    - Opponents' positions (2 nearest)
    Total: ~25 dimensions per agent
    """
    
    def __init__(self, networker: Networker, team_name: str, 
                 num_agents: int = 3, obs_dim: int = 25):
        super().__init__()
        self.networker = networker
        self.team_name = team_name
        self.num_agents = num_agents
        
        # Kickable distance
        self.kickable_dist = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE
        
        # Observation space per agent
        self.obs_dim = obs_dim
        self.observation_space = spaces.Tuple([
            spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for _ in range(num_agents)
        ])
        
        # Action space: each agent picks from 6 discrete actions
        self.action_space = spaces.MultiDiscrete([6] * num_agents)
        
        # Step counters
        self.max_steps = 400  # Longer for multi-agent (more complexity)
        self.current_step = 0
        
        # Reward tracking
        self.prev_ball_pos = None
        self.prev_ball_to_goal_dist = None
        self.prev_team_ball_possession = None  # Which team has ball
        
        # Termination flags
        self._out_of_bounds = False
        self._goal_scored = False
        self._ball_out_of_bounds = False
        
        # Field dimensions
        self.field_length = 90.0
        self.field_width = 60.0
        self.goal_x = self.field_length / 2
        
        # Ball history for velocity estimation
        self.ball_history = []
        self.ball_history_max = 5
    
    def _game_state_to_observations(self, game_state) -> list:
        """
        Convert game state to observations for each agent.
        
        Returns:
            List of observations, one per agent
        """
        if game_state is None:
            return [np.zeros(self.obs_dim, dtype=np.float32) for _ in range(self.num_agents)]
        
        try:
            # Extract ball info
            ball_pos = game_state.ball_pos or (0.0, 0.0)
            ball_x, ball_y = ball_pos
            ball_vel = game_state.ball_vel or (0.0, 0.0)
            ball_vx, ball_vy = ball_vel
            
            # Extract our team's robots
            our_team_poses = game_state.robot_poses.get(self.team_name, [])
            
            # Extract opponent team's robots (first team that's not ours)
            opponent_poses = []
            for team, poses in game_state.robot_poses.items():
                if team != self.team_name:
                    opponent_poses = poses
                    break
            
            # Build observations for each of our agents
            observations = []
            
            for agent_idx in range(self.num_agents):
                obs = self._build_agent_observation(
                    agent_idx, 
                    our_team_poses,
                    opponent_poses,
                    ball_x, ball_y, ball_vx, ball_vy
                )
                observations.append(obs)
            
            return observations
            
        except Exception as e:
            return [np.zeros(self.obs_dim, dtype=np.float32) for _ in range(self.num_agents)]
    
    def _build_agent_observation(self, agent_idx, our_team_poses, opponent_poses,
                                  ball_x, ball_y, ball_vx, ball_vy):
        """Build observation for a single agent."""
        # Get this agent's pose
        if agent_idx < len(our_team_poses):
            agent_pose_dict = our_team_poses[agent_idx]
            # Pose format: {uniform_number: (x, y, theta)}
            # Assume uniform numbers are 1, 2, 3, ...
            uniform_num = agent_idx + 1
            if uniform_num in agent_pose_dict:
                robot_x, robot_y, robot_theta = agent_pose_dict[uniform_num]
            else:
                robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
        else:
            robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
        
        # Ball relative to this agent
        ball_dx = ball_x - robot_x
        ball_dy = ball_y - robot_y
        ball_dist = np.sqrt(ball_dx**2 + ball_dy**2)
        
        # Goal relative to this agent
        goal_dx = self.goal_x - robot_x
        goal_dy = 0.0 - robot_y
        goal_dist = np.sqrt(goal_dx**2 + goal_dy**2)
        
        # Agent's orientation
        theta_rad = np.radians(robot_theta)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        
        # Kickable flag
        is_kickable = 1.0 if ball_dist < self.kickable_dist else 0.0
        
        # Goal angle
        to_goal_angle = np.arctan2(goal_dy, goal_dx)
        goal_angle_diff = to_goal_angle - theta_rad
        goal_angle_diff = np.arctan2(np.sin(goal_angle_diff), np.cos(goal_angle_diff))
        
        # Teammates (2 nearest, excluding self)
        teammate_features = self._get_nearest_teammates(
            robot_x, robot_y, agent_idx, our_team_poses, k=2
        )
        
        # Opponents (2 nearest)
        opponent_features = self._get_nearest_opponents(
            robot_x, robot_y, opponent_poses, k=2
        )
        
        # Construct observation
        obs = np.array([
            # Ball info (6)
            ball_dx, ball_dy,
            ball_vx, ball_vy,
            ball_dist,
            is_kickable,
            
            # Goal info (3)
            goal_dx, goal_dy,
            goal_angle_diff,
            
            # Self info (5)
            cos_theta, sin_theta,
            robot_x, robot_y,
            0.0,  # Placeholder for velocity (would need tracking)
            
            # Teammates (8 = 2 teammates × 4 features)
            *teammate_features,
            
            # Opponents (8 = 2 opponents × 4 features)
            *opponent_features
        ], dtype=np.float32)
        
        # Pad or truncate to obs_dim
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)), mode='constant')
        else:
            obs = obs[:self.obs_dim]
        
        return obs
    
    def _get_nearest_teammates(self, robot_x, robot_y, agent_idx, team_poses, k=2):
        """Get features for k nearest teammates."""
        teammates = []
        
        for idx, pose_dict in enumerate(team_poses):
            if idx == agent_idx:  # Skip self
                continue
            
            uniform_num = idx + 1
            if uniform_num in pose_dict:
                tx, ty, _ = pose_dict[uniform_num]
                dist = np.sqrt((tx - robot_x)**2 + (ty - robot_y)**2)
                teammates.append((dist, tx - robot_x, ty - robot_y))
        
        # Sort by distance, take k nearest
        teammates.sort(key=lambda x: x[0])
        teammates = teammates[:k]
        
        # Extract features: [dx, dy, dist, angle] for each
        features = []
        for dist, dx, dy in teammates:
            angle = np.arctan2(dy, dx)
            features.extend([dx, dy, dist, angle])
        
        # Pad if fewer than k teammates
        while len(features) < k * 4:
            features.extend([0.0, 0.0, 100.0, 0.0])  # Far away placeholder
        
        return features[:k*4]
    
    def _get_nearest_opponents(self, robot_x, robot_y, opponent_poses, k=2):
        """Get features for k nearest opponents."""
        opponents = []
        
        for pose_dict in opponent_poses:
            for uniform_num, (ox, oy, _) in pose_dict.items():
                dist = np.sqrt((ox - robot_x)**2 + (oy - robot_y)**2)
                opponents.append((dist, ox - robot_x, oy - robot_y))
        
        # Sort by distance, take k nearest
        opponents.sort(key=lambda x: x[0])
        opponents = opponents[:k]
        
        # Extract features
        features = []
        for dist, dx, dy in opponents:
            angle = np.arctan2(dy, dx)
            features.extend([dx, dy, dist, angle])
        
        # Pad if fewer than k opponents
        while len(features) < k * 4:
            features.extend([0.0, 0.0, 100.0, 0.0])
        
        return features[:k*4]
    
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
        observations = self._game_state_to_observations(game_state)
        
        self.current_step = 0
        self.prev_ball_pos = None
        self.prev_ball_to_goal_dist = None
        self.prev_team_ball_possession = None
        self._out_of_bounds = False
        self._goal_scored = False
        self._ball_out_of_bounds = False
        self.ball_history = []
        
        return observations
    
    def step(self, actions):
        """
        Execute actions for all agents.
        
        Args:
            actions: List of integers, one action per agent
        
        Returns:
            observations: List of obs arrays, one per agent
            reward: Single float (shared team reward)
            done: Boolean
            info: Dict with episode info
        """
        # Convert actions to commands for each agent
        commands_per_agent = []
        for agent_idx, action in enumerate(actions):
            commands = self._action_to_commands(agent_idx, action)
            commands_per_agent.append(commands)
        
        # Execute all commands
        # Note: This assumes networker can handle multiple agents
        # You may need to modify based on your networker API
        try:
            # Flatten commands for all agents
            all_commands = [cmd for cmds in commands_per_agent for cmd in cmds if cmd]
            if all_commands:
                self.networker.execute_ai_output(all_commands, self.team_name)
        except Exception:
            pass
        
        import time
        time.sleep(0.1)
        
        game_state = self.networker.get_game_state()
        if game_state is None:
            time.sleep(0.05)
            game_state = self.networker.get_game_state()
        
        observations = self._game_state_to_observations(game_state)
        
        # Reset termination flags
        self._out_of_bounds = False
        self._goal_scored = False
        self._ball_out_of_bounds = False
        
        # Compute shared team reward
        reward = self._compute_team_reward(game_state, actions)
        
        self.current_step += 1
        
        # Check termination
        done = (
            self.current_step >= self.max_steps or
            self._out_of_bounds or
            self._goal_scored or
            self._ball_out_of_bounds
        )
        
        info = {
            "goal_scored": self._goal_scored,
            "robot_out_of_bounds": self._out_of_bounds,
            "ball_out_of_bounds": self._ball_out_of_bounds,
            "actions": actions
        }
        
        return observations, reward, done, info
    
    def _action_to_commands(self, agent_idx, action):
        """
        Convert discrete action to simulator commands for specific agent.
        
        Actions:
        0. APPROACH_BALL
        1. SHOOT_GOAL
        2. PASS_TEAMMATE
        3. DRIBBLE_FORWARD
        4. CLEAR_BALL
        5. REPOSITION
        """
        # Get current game state
        game_state = self.networker.get_game_state()
        if game_state is None:
            return [None]
        
        # This is a simplified version
        # In practice, you'd need to:
        # 1. Get agent's current pose
        # 2. Calculate appropriate commands based on action
        # 3. Return commands in format your simulator expects
        
        # For now, return placeholder
        # You'll need to implement the full logic similar to SimplifiedSoccerEnv
        # but tracking which agent (agent_idx) is acting
        
        return [None]  # TODO: Implement full action logic
    
    def _compute_team_reward(self, game_state, actions):
        """
        Compute shared team reward.
        
        Rewards coordination:
        - Goal scored: +50 (huge!)
        - Ball moved toward goal: +3-5
        - Successful pass: +2
        - Ball possession maintained: +0.5
        - Good positioning: +0.3
        """
        if game_state is None:
            return -0.1
        
        total_reward = 0.0
        
        try:
            ball_pos = game_state.ball_pos or (0.0, 0.0)
            ball_x, ball_y = ball_pos
            
            ball_to_goal_dist = np.sqrt((self.goal_x - ball_x)**2 + (0.0 - ball_y)**2)
            
            # Goal scored! (Episode ends)
            if ball_x >= self.goal_x and abs(ball_y) < 7.32 / 2:
                total_reward += 50.0
                self._goal_scored = True
            
            # Ball progress toward goal
            if self.prev_ball_to_goal_dist is not None:
                ball_progress = self.prev_ball_to_goal_dist - ball_to_goal_dist
                if ball_progress > 0.5:
                    total_reward += 5.0
                elif ball_progress > 0.0:
                    total_reward += ball_progress * 3.0
            
            # Team ball possession (any agent close to ball)
            our_team_poses = game_state.robot_poses.get(self.team_name, [])
            min_dist_to_ball = float('inf')
            
            for pose_dict in our_team_poses:
                for uniform_num, (rx, ry, _) in pose_dict.items():
                    dist = np.sqrt((ball_x - rx)**2 + (ball_y - ry)**2)
                    min_dist_to_ball = min(min_dist_to_ball, dist)
            
            if min_dist_to_ball < self.kickable_dist * 2:
                total_reward += 0.5  # Reward possession
            
            # Reward PASS actions when they work
            # (This is simplified - you'd check if ball actually moved to teammate)
            if 2 in actions:  # PASS action taken
                total_reward += 0.3  # Small reward for attempting coordination
            
            # Alive bonus
            total_reward += 0.1
            
            # Check termination conditions
            half_length = self.field_length / 2
            half_width = self.field_width / 2
            
            # Check if ANY agent is out of bounds
            for pose_dict in our_team_poses:
                for uniform_num, (rx, ry, _) in pose_dict.items():
                    if abs(rx) > half_length or abs(ry) > half_width:
                        total_reward -= 2.0
                        self._out_of_bounds = True
                        break
            
            # Ball out of bounds
            if abs(ball_y) > half_width or ball_x < -half_length:
                total_reward -= 1.0
                self._ball_out_of_bounds = True
            
            # Update tracking
            self.prev_ball_pos = ball_pos
            self.prev_ball_to_goal_dist = ball_to_goal_dist
            
            return total_reward
            
        except Exception as e:
            return -0.1


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of MultiAgentSoccerEnv with MAPPO.
    """
    from networking.networker import Networker, TeamInfo
    import json
    
    # Load team config
    with open("team_config.json", 'r') as f:
        team_config = json.load(f)["teams"]
    
    team_infos = [TeamInfo(*team_config[0]), TeamInfo(*team_config[1])]
    networker = Networker(team_infos, "sim-only")
    
    # Create multi-agent environment for 3v3
    env = MultiAgentSoccerEnv(
        networker=networker,
        team_name="YourTeam",
        num_agents=3,
        obs_dim=25
    )
    
    # Reset returns observations for all 3 agents
    observations = env.reset()
    print(f"Observations shape: {[obs.shape for obs in observations]}")
    
    # Step with actions for all 3 agents
    actions = [0, 1, 5]  # Agent 0: APPROACH, Agent 1: SHOOT, Agent 2: REPOSITION
    next_observations, reward, done, info = env.step(actions)
    
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
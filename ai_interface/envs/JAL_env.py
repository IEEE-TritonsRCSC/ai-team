"""JAL environment."""

from __future__ import annotations

# Issues to note:
# 1. Ball velocity estimation does not account for ball being kicked by a robot, which causes discontinuities. We could add a heuristic to detect when the ball is likely being kicked (e.g. sudden large velocity change near a robot) and reset the velocity estimate in those cases.
# 2. Robot velocity estimation is a simple finite difference which can be noisy. We could maintain a short history of robot poses and use a more robust method like least squares to estimate velocity, similar to the ball velocity estimation.
# 3. Past 5 planned actions per robot are intentionally deferred for now. A future update should add a fixed-size action-history encoding to the observation.

from typing import Optional, Tuple, Dict, Any, List, Sequence
import logging
import math
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ai_interface.utils.algo_utils import estimate_ball_velocity, has_ball
from ai_interface.utils.basic_commands import goto, approach_ball
from ai_interface.constants.field_constants import *
from ai_interface.constants.player_constants import *
from ai_interface.envs.reward import (
    RewardConfig,
    RewardInputs,
    aim_quality_from_prediction,
    evaluate_reward,
    extract_opponent_positions,
    goalie_gap_quality,
)
from networking.networker import Networker
from networking.data_utils import GameState


class JALTeamEnv(gym.Env):
    ACTION_TYPES = ["goto", "turn", "kick", "start_dribble", "stop_dribble"]

    def __init__(
        self, 
        networker: Networker,
        team_name: str,
        robot_ids: Optional[List[int]] = None,
        obs_dim_per_robot: int = 8,
        non_robot_obs_dim: int = 4,
        a_max: int = 5,
        c_max: int = 7,
        global_dim: int = 6,
        per_agent_dim: int = 10,
        d_ctx: int = 7,
        max_steps: int = 200,
        debug: bool = False,
        state_retry_count: int = 10,
        state_retry_sleep_s: float = 0.02,
        none_state_warn_every: int = 50,
        position_noise_std: float = 0.05,
        invalid_action_penalty: float = 0.2,
        state_dependent_action_selection: bool = False,
        disabled_actions: Optional[List[str]] = None,
        spawn_robot_at_ball: bool = False,
        spawn_offset_behind_ball: float = 1.0,
        random_ball_x: bool = False,
        random_ball_x_range: Tuple[float, float] = (5.0, 30.0),
        random_ball_y: bool = False,
        random_ball_y_range: Tuple[float, float] = (-3.0, 3.0),
        random_spawn_theta: bool = False,
        random_spawn_theta_range_deg: Tuple[float, float] = (-45.0, 45.0),
        spawn_theta_relative_to_goal: bool = False,
        spawn_theta_min_abs_deg: float = 0.0,
        kick_requires_aim: bool = False,
        kick_min_aim_quality: float = 0.0,
        kick_tie_break_when_aimed: bool = False,
        kick_tie_break_epsilon: float = 0.0,
        approach_defer_when_has_ball: bool = False,
        approach_defer_epsilon: float = 0.0,
        reward_config_overrides: Optional[Dict[str, float]] = None,
        some_arg=None,
        opponent_team_name: Optional[str] = None,
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
        self.state_dependent_action_selection = bool(state_dependent_action_selection)

        # Stage-specific knobs
        self.disabled_actions: List[str] = list(disabled_actions) if disabled_actions else []
        self.spawn_robot_at_ball: bool = bool(spawn_robot_at_ball)
        self.spawn_offset_behind_ball: float = float(spawn_offset_behind_ball)
        self.random_ball_x: bool = bool(random_ball_x)
        self.random_ball_x_range: Tuple[float, float] = (
            float(random_ball_x_range[0]),
            float(random_ball_x_range[1]),
        )
        self.random_ball_y: bool = bool(random_ball_y)
        self.random_ball_y_range: Tuple[float, float] = (
            float(random_ball_y_range[0]),
            float(random_ball_y_range[1]),
        )
        self.random_spawn_theta: bool = bool(random_spawn_theta)
        self.random_spawn_theta_range_deg: Tuple[float, float] = (
            float(random_spawn_theta_range_deg[0]),
            float(random_spawn_theta_range_deg[1]),
        )
        # When True, the sampled spawn heading is interpreted as an OFFSET from
        # the ball→goal direction (so 0° = aimed at goal center, independent of
        # ball_y), rather than an absolute field heading. spawn_theta_min_abs_deg
        # forces a minimum |offset| so the robot always spawns misaligned enough
        # that an immediate kick provably misses — making turn-to-align the only
        # path to a goal, which removes the "free goals" that let the warm kicker
        # win without ever turning. See TRAINING.md §19.
        self.spawn_theta_relative_to_goal: bool = bool(spawn_theta_relative_to_goal)
        self.spawn_theta_min_abs_deg: float = float(spawn_theta_min_abs_deg)
        self.kick_requires_aim: bool = bool(kick_requires_aim)
        self.kick_min_aim_quality: float = max(0.0, float(kick_min_aim_quality))
        self.kick_tie_break_when_aimed: bool = bool(kick_tie_break_when_aimed)
        self.kick_tie_break_epsilon: float = max(0.0, float(kick_tie_break_epsilon))
        self.approach_defer_when_has_ball: bool = bool(approach_defer_when_has_ball)
        self.approach_defer_epsilon: float = max(0.0, float(approach_defer_epsilon))

        # Build a RewardConfig with optional overrides from caller. Field names
        # must match the RewardConfig dataclass attributes in reward.py.
        if reward_config_overrides:
            # Filter to known fields to avoid TypeError on unknown keys.
            known_fields = set(RewardConfig.__dataclass_fields__.keys())
            unknown = [k for k in reward_config_overrides if k not in known_fields]
            if unknown:
                raise ValueError(f"Unknown reward_config_overrides keys: {unknown}")
            self.reward_config = RewardConfig(**{
                k: v for k, v in reward_config_overrides.items() if k in known_fields
            })
        else:
            self.reward_config = RewardConfig()
        
        
        # Create a logger for this environment
        self.logger = logging.getLogger(f"JALTeamEnv[{team_name}]")
        if self.debug:
            self.logger.setLevel(logging.DEBUG)

        # Observation design (expandable backbone — see PPO_EXPANDABLE_PLAN.md):
        # FIXED-MAX, COUNT-AGNOSTIC obs of constant size across the whole
        # curriculum, so adding teammates/opponents/features later never forces
        # a network rebuild. Layout:
        #   global block (global_dim, 4 live): [ball_x, ball_y, ball_vx, ball_vy] + reserved
        #   a_max agent slots (per_agent_dim, 8 live):
        #       [x, y, theta, vx, vy, is_dribbling, start_dribble_x, start_dribble_y] + reserved
        #   c_max context slots (d_ctx, 5 live, ALL ZERO for now — stub):
        #       [x, y, theta, vx, vy] + reserved   (our goalie / opp goalie / opponents)
        # All live values are analytically normalized (see _NORM_* below). Absent
        # entities are zero-filled and flagged via the active masks in info.
        self.obs_dim_per_robot = obs_dim_per_robot   # LIVE per-agent dims written (8)
        self.non_robot_obs_dim = non_robot_obs_dim   # LIVE global dims written (4)
        self.a_max = int(a_max)
        self.c_max = int(c_max)
        self.global_dim = int(global_dim)            # global slot width (>= 4 live)
        self.per_agent_dim = int(per_agent_dim)      # agent slot width (>= 8 live)
        self.d_ctx = int(d_ctx)                      # context slot width (>= 5 live)
        if self.num_robots > self.a_max:
            raise ValueError(f"num_robots={self.num_robots} exceeds a_max={self.a_max}")
        self.obs_dim = self.global_dim + self.per_agent_dim * self.a_max + self.d_ctx * self.c_max
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
            )
        # Opponent team name for filling context slots (slot 1 = opp goalie).
        self.opponent_team_name: Optional[str] = opponent_team_name

        # Active masks: which agent slots are controlled robots, which context
        # slots are present. Static within Stage 1 (1 robot, no context); later
        # stages / foul-randomization will vary them per episode.
        self.agent_active_mask = np.zeros(self.a_max, dtype=np.float32)
        self.agent_active_mask[: self.num_robots] = 1.0
        self.context_active_mask = np.zeros(self.c_max, dtype=np.float32)
        if self.opponent_team_name is not None:
            self.context_active_mask[1] = 1.0  # slot 1 = opp goalie
        self.is_dribbling = {robot_id: False for robot_id in self.robot_ids}  # Track dribble state per robot
        self.start_dribble_pos = {robot_id: [-1.0, -1.0] for robot_id in self.robot_ids}  # Placeholder for dribble start position, can be updated in step() when dribble starts

        # Stage 2 dribble session state (command-level, distinct from the
        # proximity-based is_dribbling/start_dribble_pos used by the obs).
        # A session opens on a VALID start_dribble (catch) and is anchored at
        # the robot's position; it ends on stop_dribble (drop), a fired kick,
        # or exhaustion (anchor distance >= reward_config.dribble_max_radius,
        # which forces a drop). After any session ends the robot must create
        # visible separation from the ball (robot-ball dist > kickable_dist +
        # dribble_redribble_gap_margin) before start_dribble is valid again.
        self.dribble_session_active: Dict[int, bool] = {rid: False for rid in self.robot_ids}
        self.dribble_anchor: Dict[int, Optional[Tuple[float, float]]] = {rid: None for rid in self.robot_ids}
        self.dribble_exhausted: Dict[int, bool] = {rid: False for rid in self.robot_ids}
        self.awaiting_redribble_gap: Dict[int, bool] = {rid: False for rid in self.robot_ids}
        # Steps since the last stop_dribble fired per robot. None = no stop_dribble
        # has occurred yet this episode; 0 = fired this step; > combo_window = expired.
        self.steps_since_stop_dribble: Dict[int, Optional[int]] = {rid: None for rid in self.robot_ids}
        
        # Action design per robot (9D): [goto_logit, approach_ball_logit, turn_logit, kick_logit, start_dribble_logit, stop_dribble_logit, goto_x, goto_y, turn_theta]
        # Matches the layout used to train `stage1_6_final_58pct.zip` so that
        # checkpoint can be warm-started directly. In the 58% run only turn
        # (slot 2) and kick (slot 3) were enabled, so those slots carry
        # trained weights; goto / approach_ball / start_dribble / stop_dribble
        # were masked and have effectively untrained weights. For stage1_9 we
        # enable approach_ball alongside the already-trained turn/kick.
        self.action_dim_per_robot = 9
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_robots * self.action_dim_per_robot,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.episode_num = 0
        self._cached_game_state = None  # reused as current_game_state at next step start
        
        # Ball position history for estimate_ball_velocity()
        self.ball_pos_history: List[np.ndarray] = []
        self.ball_history_max = 5

        # Field dimensions (from your existing code)
        self.field_half_width = 45.0
        self.field_half_height = 30.0

        # Fixed analytic observation normalization (LOCKED — changing these is a
        # distribution shift = retrain; see PPO_EXPANDABLE_PLAN.md). Constant
        # divisors (not running stats), so reserved/masked zero dims stay zero.
        self._NORM_POS_X = self.field_half_width    # 45.0
        self._NORM_POS_Y = self.field_half_height   # 30.0
        self._NORM_THETA = float(np.pi)             # heading in radians → ~[-1, 1]
        self._NORM_VEL = 10.0                        # per-cycle position delta; covers ball_speed_max

        self.kickable_dist = KICKABLE_MARGIN + BALL_SIZE + PLAYER_SIZE
        # Kicker cone gate removed (2026-05-20): the cone restriction caused
        # both TD3 (turn=100% collapse) and PPO (flat 7.6% goal rate) to fail
        # to learn. The 58% checkpoint trained without it, and rcssserver
        # accepts kick commands regardless of orientation — re-add a
        # physically realistic gate only once the simulator policy is solid.

        # Per-robot pose history for velocity estimation
        self.prev_robot_pose_by_id: Dict[int, np.ndarray] = {}
        self.prev_opp_pose_by_id: Dict[int, np.ndarray] = {}
        self.prev_reward_ball_dist_by_id: Dict[int, float] = {}
        self.prev_reward_ball_to_goal_dist: Optional[float] = None
        self.prev_reward_ball_pos: Optional[Tuple[float, float]] = None
        self.prev_reward_facing_goal_cos_by_id: Dict[int, float] = {}
        self._stopped_ball_counter: int = 0

        # Frozen-state detection: counts consecutive steps where BOTH the
        # ball position AND our robots' poses are bit-identical to the prior
        # step. A stalled/stale sim connection (or a stuck-ball edge case)
        # repeats the same frame indefinitely; without this, reward terms
        # computed from static pose (e.g. kick_aim_bonus) get re-granted
        # every step with no actual progress, letting an episode rack up a
        # huge total_reward purely from a frozen frame. Unlike
        # _stopped_ball_counter (which only ends episodes when
        # approach_ball/goto are both disabled), this runs unconditionally.
        self._frozen_state_counter: int = 0

        # Per-robot (rx, ry, theta, bx, by) snapshot at the last step a kick
        # was actually rewarded. If a subsequent kick fires from the exact
        # same snapshot, the underlying frame didn't change — skip awarding
        # kick_aim_bonus/bad_aim_penalty again so a frozen frame can't farm
        # the bonus once per step. See _frozen_state_counter above for the
        # episode-level backstop.
        self._last_rewarded_kick_state: Dict[int, Tuple[float, float, float, float, float]] = {}

        # Statistics
        self.total_rewards = 0.0
        self.episode_actions = []  # Track action distribution
        self._none_state_counter = 0
        
        self.logger.info(
            f"Initialized JALTeamEnv for team '{team_name}' with robots {robot_ids}, "
            f"num_robots={self.num_robots}, obs_dim={self.obs_dim}, "
            f"action_dim={self.action_space.shape[0]}, "
            f"disabled_actions={self.disabled_actions}, "
            f"spawn_robot_at_ball={self.spawn_robot_at_ball}, "
            f"random_ball_x={self.random_ball_x}, "
            f"random_ball_y={self.random_ball_y}, "
            f"random_spawn_theta={self.random_spawn_theta}"
        )

    def _select_action_index(
        self,
        logits: np.ndarray,
        has_ball_now: bool,
    ) -> Tuple[int, np.ndarray]:
        """Select the highest-probability action while masking invalid ball-state actions."""
        if has_ball_now:
            allowed_indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        else:
            allowed_indices = np.array([0, 1], dtype=np.int64)

        allowed_logits = logits[allowed_indices]
        allowed_logits_shifted = allowed_logits - np.max(allowed_logits)
        allowed_exp_logits = np.exp(allowed_logits_shifted)
        allowed_probs = allowed_exp_logits / np.sum(allowed_exp_logits)

        probs = np.zeros_like(logits, dtype=np.float32)
        probs[allowed_indices] = allowed_probs
        action_idx = int(allowed_indices[int(np.argmax(allowed_probs))])
        return action_idx, probs

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

        # Curriculum ball position: start ball near the goal in early episodes
        # so random kicks have a real chance of scoring (gives the model the
        # sparse goal_reward signal it would otherwise never see). Gradually
        # move ball toward centre as the model learns.
        ball_pos = self._get_curriculum_ball_pos()

        # Optionally pre-position the first robot adjacent to the ball, facing
        # +x (opponent goal). Used in narrowed Stage 1 to isolate the kick
        # primitive — the model already starts in kicking range, so it can
        # only learn "kick" rather than "walk to ball, then kick".
        player_poses_override = None
        if self.networker is not None and (self.spawn_robot_at_ball or self.random_spawn_theta):
            commander = getattr(self.networker, "commander", None)
            default_poses = getattr(commander, "desired_init_poses", None) if commander else None
            if default_poses:
                first_obj_name, default_pose = default_poses[0]
                if self.random_spawn_theta:
                    theta_min, theta_max = self.random_spawn_theta_range_deg
                    theta_offset_deg = float(np.random.uniform(theta_min, theta_max))
                    # Force a minimum |offset| so the spawn is never near-aligned:
                    # an immediate kick then provably misses and turning is the
                    # only path to a goal (TRAINING.md §19).
                    min_abs = self.spawn_theta_min_abs_deg
                    if min_abs > 0.0 and abs(theta_offset_deg) < min_abs:
                        if theta_offset_deg == 0.0:
                            sign = 1.0 if np.random.random() < 0.5 else -1.0
                        else:
                            sign = math.copysign(1.0, theta_offset_deg)
                        theta_offset_deg = sign * min_abs
                else:
                    theta_offset_deg = 0.0
                if self.spawn_theta_relative_to_goal:
                    # Offset is measured from the ball→goal-center direction, so
                    # 0° = aimed at the goal regardless of ball_y, giving a
                    # consistent "turn this much to align" target every episode.
                    bx0, by0 = float(ball_pos[0]), float(ball_pos[1])
                    goal_angle_deg = math.degrees(
                        math.atan2(float(GOAL_R[1]) - by0, float(GOAL_R[0]) - bx0)
                    )
                    spawn_theta_deg = goal_angle_deg + theta_offset_deg
                else:
                    spawn_theta_deg = theta_offset_deg
                if self.spawn_robot_at_ball:
                    # Place robot behind ball on the ball-goal axis so that
                    # ball direction ≈ goal direction from robot's spawn pos.
                    # Turning toward goal then also keeps the ball roughly in
                    # front of the chassis, which matches the natural shooting
                    # posture without requiring an explicit cone gate.
                    bx, by = float(ball_pos[0]), float(ball_pos[1])
                    goal_dx = float(GOAL_R[0]) - bx
                    goal_dy = float(GOAL_R[1]) - by
                    goal_dist = float(np.hypot(goal_dx, goal_dy))
                    if goal_dist > 1e-6:
                        goal_dir_x = goal_dx / goal_dist
                        goal_dir_y = goal_dy / goal_dist
                    else:
                        goal_dir_x, goal_dir_y = 1.0, 0.0
                    offset = float(self.spawn_offset_behind_ball)
                    robot_x = bx - offset * goal_dir_x
                    robot_y = by - offset * goal_dir_y
                else:
                    # Keep the default spawn x,y but optionally randomise theta.
                    # rcssserver's dash command translates without rotating, so
                    # a fixed spawn theta means approach_ball arrives at the
                    # ball with the robot still facing its starting orientation
                    # — which trivially lets "kick straight" score whenever the
                    # default theta happens to point at goal. Randomising theta
                    # forces the policy to use the turn skill after approach.
                    robot_x = float(default_pose[0])
                    robot_y = float(default_pose[1])
                robot_pose = (robot_x, robot_y, spawn_theta_deg)
                # Keep all other players at their default poses; only override the first.
                player_poses_override = [(first_obj_name, robot_pose)] + list(default_poses[1:])

        self.networker.reset_sim(ball_pos=ball_pos, player_poses_override=player_poses_override)
        if ball_pos != (0.0, 0.0):
            self.logger.info("Episode %d curriculum ball start: (%.1f, %.1f)",
                             self.episode_num + 1, ball_pos[0], ball_pos[1])

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
        self.prev_opp_pose_by_id = {}
        self.prev_reward_ball_dist_by_id = {}
        self.prev_reward_ball_to_goal_dist = None
        self.prev_reward_ball_pos = None
        self.prev_reward_facing_goal_cos_by_id = {}
        self._stopped_ball_counter = 0
        self._frozen_state_counter = 0
        self._last_rewarded_kick_state = {}

        # Clear stage 2 dribble session state
        self.dribble_session_active = {rid: False for rid in self.robot_ids}
        self.dribble_anchor = {rid: None for rid in self.robot_ids}
        self.dribble_exhausted = {rid: False for rid in self.robot_ids}
        self.awaiting_redribble_gap = {rid: False for rid in self.robot_ids}
        self.steps_since_stop_dribble = {rid: None for rid in self.robot_ids}

        # Get initial game state from simulator
        game_state = self._get_game_state(
            retries=max(self.state_retry_count, 20),
            sleep_s=self.state_retry_sleep_s,
        )
        self._cached_game_state = game_state

        playmode = getattr(game_state, "playmode", None) if game_state is not None else None
        self.logger.info("Episode %d playmode=%s", self.episode_num, playmode)
        self._log_episode_start_state(game_state)
        if playmode == "time_over":
            self.logger.warning(
                "Episode %d started in time_over — physics is frozen; "
                "rewards will be meaningless until the rcssserver session is restarted",
                self.episode_num,
            )

        # Build initial observation
        obs = self._game_state_to_obs(game_state)

        
        info = {
            "episode_num": self.episode_num,
            "step": self.current_step,
            "agent_active_mask": self.agent_active_mask.copy(),
            "context_active_mask": self.context_active_mask.copy(),
        }

        if self.debug:
            self.logger.debug(f"Episode {self.episode_num} started")

        return obs, info

    def _log_episode_start_state(self, game_state) -> None:
        """Log a one-line snapshot of the initial game state for each episode."""
        if game_state is None:
            self.logger.info("Episode %d initial state: <none>", self.episode_num)
            return

        ball_pos = getattr(game_state, "ball_pos", None)
        if ball_pos is not None:
            ball_str = f"ball=({float(ball_pos[0]):.2f}, {float(ball_pos[1]):.2f})"
        else:
            ball_str = "ball=<none>"

        robot_strs = []
        for team_name, team_poses in getattr(game_state, "robot_poses", {}).items():
            for entry in team_poses:
                for unum, pose in entry.items():
                    rx, ry, rtheta = float(pose[0]), float(pose[1]), float(pose[2])
                    robot_strs.append(f"{team_name}#{int(unum)}=({rx:.2f}, {ry:.2f}, {rtheta:.1f}°)")
        robots_str = "  ".join(robot_strs) if robot_strs else "robots=<none>"

        self.logger.info(
            "Episode %d initial state: %s  %s",
            self.episode_num, ball_str, robots_str,
        )

    def step(
        self,
        action
        ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Either
              (a) a flat np.ndarray of shape (9 * num_robots,) — legacy TD3 format
                  (6 action logits + 3 continuous params per robot), or
              (b) a dict with keys "primitive_idx" (np.ndarray[num_robots] int)
                  and "params" (np.ndarray[num_robots*3] float in [-1,1]) —
                  new PPO hybrid format.

        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended (goal/out-of-bounds)
            truncated: Whether episode hit max steps
            info: Additional information
        """
        self.current_step += 1

        # Reuse the state fetched at the end of the previous step (or reset).
        # This eliminates the redundant fetch that consumed a simulator cycle
        # before sending commands, halving cycle usage from 2 per step to 1.
        current_game_state = self._cached_game_state
        if current_game_state is None:
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
        self._cached_game_state = next_game_state
        obs = self._game_state_to_obs(next_game_state)

        reward = self._calculate_reward(next_game_state)
        invalid_action_count = int(action_info.get("invalid_action_count", 0))
        if invalid_action_count > 0 and self.invalid_action_penalty > 0.0:
            reward -= self.invalid_action_penalty * invalid_action_count

        # Kick-aim bonus: one-shot reward at the moment a valid kick action
        # fires, scaled by aim_quality — a *hard cutoff* on whether the kick's
        # straight-line trajectory would actually cross the goal mouth.
        # Replaces the earlier cos^N (cos^2, cos^4) formulation, which awarded
        # large fractions of the bonus to kicks well outside the goal: with
        # ball_x≈10, the goal subtends only ~arctan(5/35)≈8° from the kicker,
        # but cos^4 still yields 0.88 at 10° off, 0.66 at 20° off — so the
        # policy could convergently "turn approximately toward goal then kick"
        # and collect ~80% of the bonus on every miss. aim_quality drops
        # *linearly to zero across the goal mouth* and is exactly 0 for any
        # kick angled outside the mouth, so off-target kicks earn nothing.
        #
        # In addition to the bonus on aimed kicks, a *hard penalty*
        # (`bad_aim_kick_penalty`) is applied on kicks whose straight-line
        # projection misses the goal mouth (|predicted_y| > goal_half_height).
        # Without this, an unaimed kick's reward is zero rather than negative,
        # so the policy can fire on step 1 of every episode with no EV cost
        # and the actor never explores "delay-then-aim" trajectories.
        #
        # predicted_y_at_goal_line = ry + (FIELD_X[1] - rx) * tan(theta);
        # aim_quality = max(0, 1 - |predicted_y| / goal_half_height).
        kick_aim_weight = float(getattr(self.reward_config, "kick_aim_bonus_weight", 0.0))
        goal_half_height = float(getattr(self.reward_config, "goal_half_height", 5.0))
        bad_aim_penalty = float(
            getattr(self.reward_config, "bad_aim_kick_penalty", 0.0)
        )
        # When > 0, the miss penalty ramps linearly with how far past the
        # goalpost the kick projects (0 at the post → bad_aim_penalty at this
        # many units beyond, capped), instead of a flat cliff. Gives an aim
        # gradient on every miss. See TRAINING.md §18.
        bad_aim_grad_scale = float(
            getattr(self.reward_config, "bad_aim_grad_scale", 0.0)
        )

        def _miss_penalty(predicted_y: Optional[float]) -> float:
            """Penalty magnitude for a missed kick. Graded when grad_scale>0."""
            if bad_aim_grad_scale <= 0.0 or predicted_y is None:
                return bad_aim_penalty
            miss = abs(predicted_y) - goal_half_height
            return bad_aim_penalty * min(1.0, max(0.0, miss) / bad_aim_grad_scale)
        # Stage 2: goalie-gap placement gate. When enabled, the one-shot kick
        # bonus pays lateral separation from the keeper (corner placement)
        # instead of center-mouth crossing; mouth validity stays a hard gate
        # via the existing bad-aim penalty. See reward.py goalie_gap_quality.
        use_goalie_gate = bool(getattr(self.reward_config, "use_goalie_aim_gate", False))
        goalie_gap_min_quality = float(
            getattr(self.reward_config, "goalie_gap_min_quality", 0.3)
        )
        kick_into_keeper_penalty = float(
            getattr(self.reward_config, "kick_into_keeper_penalty", 0.0)
        )
        goalie_gap_blend_center = float(
            getattr(self.reward_config, "goalie_gap_blend_center", 0.0)
        )
        if (kick_aim_weight > 0.0 or bad_aim_penalty > 0.0) and current_game_state is not None:
            per_robot = action_info.get("per_robot", [])
            pose_by_id: Dict[int, Any] = {}
            for entry in getattr(current_game_state, "robot_poses", {}).get(self.team_name, []):
                if isinstance(entry, dict):
                    pose_by_id.update(entry)
            goalie_pose = (
                self._opponent_goalie_pose(current_game_state) if use_goalie_gate else None
            )
            gk_y = float(goalie_pose[1]) if goalie_pose is not None else None
            kick_ball_pos = getattr(current_game_state, "ball_pos", None)
            for info_i in per_robot:
                if info_i.get("action_type") != "kick":
                    continue
                # Only score the aim signal on a real kick command (i.e. the
                # robot was in kickable range and the simulator accepted the
                # kick). Awarding it on intent alone would let the policy
                # collect free reward for "picked kick while facing goal"
                # without actually being able to kick.
                if not info_i.get("kick_fired", False):
                    continue
                pose = pose_by_id.get(info_i.get("robot_id"))
                if pose is None:
                    continue
                rx, ry, theta_deg = float(pose[0]), float(pose[1]), float(pose[2])
                robot_id_i = info_i.get("robot_id")
                if kick_ball_pos is not None and len(kick_ball_pos) >= 2:
                    kick_state_snapshot = (
                        round(rx, 3), round(ry, 3), round(theta_deg, 1),
                        round(float(kick_ball_pos[0]), 3), round(float(kick_ball_pos[1]), 3),
                    )
                    # Identical (pose, ball) snapshot as the last kick we
                    # rewarded for this robot means the frame hasn't actually
                    # advanced (frozen/stale state) — the kick didn't do
                    # anything new, so don't pay the aim bonus again.
                    if self._last_rewarded_kick_state.get(robot_id_i) == kick_state_snapshot:
                        continue
                    self._last_rewarded_kick_state[robot_id_i] = kick_state_snapshot
                aim_quality, predicted_y_at_goal_line = self._kick_aim_quality_from_pose(pose)
                # Facing away from / parallel to the goal line: kick can
                # never cross x=FIELD_X[1]. Guaranteed miss — apply the
                # bad-aim penalty and move on.
                if predicted_y_at_goal_line is None:
                    if bad_aim_penalty > 0.0:
                        reward -= bad_aim_penalty
                    self.logger.info(
                        "Kick fired: robot=(%.2f, %.2f, %.1f°)  "
                        "predicted_y=N/A (cos≈0)  aim_quality=0.00  bad_aim=True",
                        rx, ry, theta_deg,
                    )
                    continue
                bad_aim = aim_quality <= 0.0
                gap_quality: Optional[float] = None
                if bad_aim:
                    if bad_aim_penalty > 0.0:
                        reward -= _miss_penalty(predicted_y_at_goal_line)
                elif use_goalie_gate and gk_y is not None:
                    # Pay placement: 0 at the keeper's y, 1 at max in-mouth
                    # separation. The blend term eases transfer from the
                    # stage-1 center-aim checkpoint early in the stage.
                    gap_quality = goalie_gap_quality(
                        predicted_y_at_goal_line, gk_y, goal_half_height
                    )
                    blended_quality = (
                        (1.0 - goalie_gap_blend_center) * gap_quality
                        + goalie_gap_blend_center * aim_quality
                    )
                    reward += blended_quality * kick_aim_weight
                    if (
                        gap_quality < goalie_gap_min_quality
                        and kick_into_keeper_penalty > 0.0
                    ):
                        # On target but into the keeper's cover.
                        reward -= kick_into_keeper_penalty
                else:
                    reward += aim_quality * kick_aim_weight
                # Diagnostic log: confirms predicted_y vs the actual episode
                # outcome. If most ball_in_penalty_off_target episodes show
                # high aim_quality at fire, the divergence is on the physics
                # side (ball deflection / decay), not the policy's aim.
                self.logger.info(
                    "Kick fired: robot=(%.2f, %.2f, %.1f°)  "
                    "predicted_y=%.2f  aim_quality=%.2f  bad_aim=%s  "
                    "gk_y=%s  gap_quality=%s",
                    rx, ry, theta_deg, predicted_y_at_goal_line,
                    aim_quality, bad_aim,
                    f"{gk_y:.2f}" if gk_y is not None else "N/A",
                    f"{gap_quality:.2f}" if gap_quality is not None else "N/A",
                )

        # Stage 2: one-shot bonus for deliberately releasing the ball late in
        # a dribble session (instead of spamming start_dribble at the limit),
        # setting up the stop -> re-approach -> turn -> kick chain.
        stop_release_bonus = float(
            getattr(self.reward_config, "stop_dribble_release_bonus", 0.0)
        )
        if stop_release_bonus > 0.0:
            for info_i in action_info.get("per_robot", []):
                if info_i.get("stop_dribble_fired") and info_i.get("stop_dribble_at_limit"):
                    reward += stop_release_bonus

        # Stage 2: penalty for kicking while the ball is already in the
        # goalie's possession (tug-of-war suppression). Fires when a kick
        # command was actually sent AND the ball is within goalie_possession_dist
        # of the keeper. Complemented by the early termination in _check_terminal.
        kick_near_goalie_penalty = float(
            getattr(self.reward_config, "kick_near_goalie_penalty", 0.0)
        )
        if kick_near_goalie_penalty > 0.0 and current_game_state is not None:
            gk_pose = self._opponent_goalie_pose(current_game_state)
            if gk_pose is not None:
                gk_x, gk_y = gk_pose[0], gk_pose[1]
                goalie_possession_dist = float(
                    getattr(self.reward_config, "goalie_possession_dist", 2.0)
                )
                ball_pos_now = getattr(current_game_state, "ball_pos", None)
                if ball_pos_now is not None:
                    ball_gk_dist = math.hypot(
                        float(ball_pos_now[0]) - gk_x,
                        float(ball_pos_now[1]) - gk_y,
                    )
                    if ball_gk_dist < goalie_possession_dist:
                        for info_i in action_info.get("per_robot", []):
                            if info_i.get("kick_fired"):
                                reward -= kick_near_goalie_penalty
                                self.total_rewards -= kick_near_goalie_penalty

        # Stage 2: dribble→kick combo bonus. Rewards the stop_dribble → kick
        # chain as a unit. Update the per-robot stop_dribble age counter first,
        # then check whether any kick this step qualifies for the bonus.
        combo_bonus = float(getattr(self.reward_config, "post_dribble_kick_bonus", 0.0))
        combo_window = int(getattr(self.reward_config, "post_dribble_kick_combo_window", 5))
        per_robot_info_list = action_info.get("per_robot", [])
        for info_i in per_robot_info_list:
            rid = info_i.get("robot_id")
            if rid is None:
                continue
            if info_i.get("stop_dribble_fired"):
                self.steps_since_stop_dribble[rid] = 0
            elif self.steps_since_stop_dribble.get(rid) is not None:
                self.steps_since_stop_dribble[rid] += 1
        if combo_bonus > 0.0:
            for info_i in per_robot_info_list:
                rid = info_i.get("robot_id")
                if not info_i.get("kick_fired"):
                    continue
                age = self.steps_since_stop_dribble.get(rid)
                if age is not None and age <= combo_window:
                    reward += combo_bonus
                    self.total_rewards += combo_bonus
                    self.logger.info(
                        "Dribble→kick combo bonus +%.1f (steps_since_stop=%d)",
                        combo_bonus, age,
                    )

        self.total_rewards += reward
        terminated, term_reason = self._check_terminal(next_game_state, prev_game_state=current_game_state)
        # Off-target terminal penalty disabled: a -3 penalty here flipped the
        # net-EV of "kick" to slightly negative once the policy was at all
        # uncertain about aim, which collapsed the policy into "turn 100%"
        # action distribution (TD3 is deterministic — once Q(turn)=0 stably
        # exceeds noisy Q(kick), kicking stops entirely). The off-target
        # exploit it was meant to address is better fixed at the shaping
        # level (aim-gate goal_progress) than at the terminal level.
        if terminated and term_reason == "ball_in_penalty_off_target":
            pass
        if terminated and term_reason == "goalie_catch":
            goal_reward = float(getattr(self.reward_config, "goal_reward", 70.0))
            reward -= goal_reward
            self.total_rewards -= goal_reward
        # Small penalty for "wasted kick" outcomes — ball went dead via a
        # non-goal playmode (kick_in/corner/goal_kick), or came to rest where
        # the robot can't recover it. Small enough not to crush kick
        # exploration (cf. the -50 off-target penalty that caused the spin
        # collapse), but nonzero so the model has signal that the kick was
        # unproductive. Self-disables when approach actions are available:
        # with approach_ball enabled the robot could legitimately recover
        # from a dead-ball / stopped-ball state, so the penalty would punish
        # the wrong thing.
        approach_disabled = (
            "approach_ball" in self.disabled_actions
            and "goto" in self.disabled_actions
        )
        if (
            terminated
            and approach_disabled
            and (term_reason.startswith("ball_dead_") or term_reason == "ball_stopped_unreachable")
        ):
            reward -= 3.0
            self.total_rewards -= 3.0
        truncated = (not terminated) and self.current_step >= self.max_steps
        # max_steps terminal penalty: punishes "stand still for 300 steps"
        # so that turn-only / no-kick behavior isn't a stable 0-reward
        # attractor. Without this, an under-trained policy can collapse into
        # turn 100% of the time because stable 0 beats noisy +small from
        # uncertain kicks. -10 is small relative to a successful goal (+70)
        # but large enough to push the deterministic-policy preference back
        # toward acting.
        if truncated:
            reward -= 10.0
            self.total_rewards -= 10.0

        if terminated or truncated:
            total = len(self.episode_actions)
            if total > 0:
                from collections import Counter
                counts = Counter(self.episode_actions)
                dist = "  ".join(
                    f"{k}={100*v//total}%" for k, v in sorted(counts.items())
                )
                self.logger.info(
                    "Episode %d action distribution (%d steps): %s",
                    self.episode_num, total, dist,
                )
            end_reason = term_reason if terminated else "max_steps"
            self.logger.info(
                "Episode %d ended — reason=%s  steps=%d  total_reward=%.2f",
                self.episode_num, end_reason, self.current_step, self.total_rewards,
            )
            if terminated:
                self._cached_game_state = None

        info = {
            "action_info": action_info,
            "reward": reward,
            "total_reward": self.total_rewards,
            "invalid_action_count": invalid_action_count,
            "termination_reason": term_reason if terminated else ("max_steps" if truncated else ""),
            "agent_active_mask": self.agent_active_mask.copy(),
            "context_active_mask": self.context_active_mask.copy(),
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

            # Fixed-max, count-agnostic, analytically-normalized observation.
            # Build a zero vector and fill the live dims of each block; reserved
            # dims, inactive agent slots, and context slots stay 0.
            obs = np.zeros(self.obs_dim, dtype=np.float32)

            # --- global / ball block (4 live, normalized) ---
            obs[0] = float(ball_x) / self._NORM_POS_X
            obs[1] = float(ball_y) / self._NORM_POS_Y
            obs[2] = float(ball_vx) / self._NORM_VEL
            obs[3] = float(ball_vy) / self._NORM_VEL

            # --- agent slots (a_max × per_agent_dim; 8 live, normalized) ---
            agent_base = self.global_dim
            for slot in range(self.a_max):
                if slot >= self.num_robots:
                    continue  # inactive slot → zeros (masked off)
                robot_id = self.robot_ids[slot]
                pose = pose_by_robot_id.get(robot_id)
                if pose is None:
                    continue  # active but unobserved this step → zeros

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

                off = agent_base + slot * self.per_agent_dim
                obs[off + 0] = robot_x / self._NORM_POS_X
                obs[off + 1] = robot_y / self._NORM_POS_Y
                obs[off + 2] = robot_theta / self._NORM_THETA
                obs[off + 3] = robot_vx / self._NORM_VEL
                obs[off + 4] = robot_vy / self._NORM_VEL
                obs[off + 5] = float(is_dribbling)
                obs[off + 6] = float(start_dribble_pos[0]) / self._NORM_POS_X
                obs[off + 7] = float(start_dribble_pos[1]) / self._NORM_POS_Y
                # dims 8..per_agent_dim-1 reserved → stay 0

            # --- context slots (c_max × d_ctx) ---
            # Layout: slot 0 = our goalie (unused Stage 1/2), slot 1 = opp goalie.
            # Only fill when an opponent team is configured.
            if self.opponent_team_name is not None:
                opp_pose_entries = game_state.robot_poses.get(self.opponent_team_name, [])
                opp_pose_by_id: Dict[int, Any] = {}
                for entry in opp_pose_entries:
                    if isinstance(entry, dict):
                        opp_pose_by_id.update(entry)

                # Context slot 1: first opponent robot (goalie robot_id=1 by convention).
                first_opp_id = min(opp_pose_by_id.keys()) if opp_pose_by_id else None
                if first_opp_id is not None:
                    opp_pose = opp_pose_by_id[first_opp_id]
                    opp_x = float(opp_pose[0])
                    opp_y = float(opp_pose[1])
                    opp_theta = float(np.deg2rad(opp_pose[2]))
                    prev_opp_xy = self.prev_opp_pose_by_id.get(first_opp_id)
                    if prev_opp_xy is not None:
                        opp_vx = float(opp_x - prev_opp_xy[0])
                        opp_vy = float(opp_y - prev_opp_xy[1])
                    else:
                        opp_vx = opp_vy = 0.0
                    self.prev_opp_pose_by_id[first_opp_id] = np.array([opp_x, opp_y], dtype=np.float32)

                    ctx_base = self.global_dim + self.per_agent_dim * self.a_max
                    opp_slot_off = ctx_base + 1 * self.d_ctx  # slot 1
                    obs[opp_slot_off + 0] = opp_x / self._NORM_POS_X
                    obs[opp_slot_off + 1] = opp_y / self._NORM_POS_Y
                    obs[opp_slot_off + 2] = opp_theta / self._NORM_THETA
                    obs[opp_slot_off + 3] = opp_vx / self._NORM_VEL
                    obs[opp_slot_off + 4] = opp_vy / self._NORM_VEL
                    # dims 5..d_ctx-1 reserved → stay 0

            return obs
            
        except Exception as e:
            self.logger.error(f"Error building observation: {e}")
            return np.zeros(self.obs_dim, dtype=np.float32)

    def _action_to_commands(self, action, game_state: Optional[GameState]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Convert a JAL action into simulator command strings.

        Two accepted action formats:

        1. Legacy TD3 flat Box vector, shape (9 * num_robots,):
            [0..5] logits over {goto, approach_ball, turn, kick,
                                start_dribble, stop_dribble}
            [6] goto_x_raw ∈ [-1, 1]
            [7] goto_y_raw ∈ [-1, 1]
            [8] turn_theta_raw ∈ [-1, 1]

        2. New PPO hybrid dict:
            {
              "primitive_idx": np.ndarray[num_robots] (int),
              "params": np.ndarray[num_robots * 3] (float),
                # per robot: Dx_raw, Dy_raw, Dtheta_raw ∈ [-1, 1]
            }

        The PPO format avoids the broken "continuous-logits + argmax" trick:
        the discrete primitive choice is sampled from a true Categorical at
        the policy layer; this env decode just executes whichever primitive
        was selected.

        Returns:
            commands: List of simulator command strings in robot_ids order
            action_info: Dictionary with action details (for logging/debugging)
        """
        action_types = ["goto", "approach_ball", "turn", "kick", "start_dribble", "stop_dribble"]

        # Detect format and extract per-robot (action_type, goto_x, goto_y, turn_theta).
        per_robot_decoded: List[Dict[str, Any]] = []
        is_ppo_dict = isinstance(action, dict) and "primitive_idx" in action

        if is_ppo_dict:
            # The expandable policy emits actions for all a_max agent slots, each
            # with a param_dim_max-wide param vector. Only the first num_robots
            # slots are active controlled robots (slot i ↔ robot_ids[i]); extra
            # slots are inactive/no-op. We read the env-relevant params (goto_x,
            # goto_y, turn_theta) from the first 3 entries of each slot's vector;
            # any reserved params are ignored until a future primitive uses them.
            primitive_idx_arr = np.asarray(action["primitive_idx"], dtype=np.int64).flatten()
            params_arr = np.asarray(action.get("params", []), dtype=np.float32).flatten()
            n_slots = primitive_idx_arr.shape[0]
            if n_slots < self.num_robots:
                raise ValueError(
                    f"primitive_idx has {n_slots} slots < num_robots {self.num_robots}"
                )
            params_per_slot = (params_arr.shape[0] // n_slots) if n_slots else 0
            if params_per_slot < 3:
                raise ValueError(
                    f"params has {params_arr.shape[0]} for {n_slots} slots; need >=3 per slot"
                )
            for i in range(self.num_robots):
                idx = int(primitive_idx_arr[i])
                if idx < 0 or idx >= len(action_types):
                    raise ValueError(f"primitive_idx[{i}]={idx} out of range")
                action_type = action_types[idx]
                # Honest masking sanity-check: if the policy somehow picked a
                # disabled primitive, log it and degrade to a no-op turn.
                if self.disabled_actions and action_type in self.disabled_actions:
                    self.logger.warning(
                        "PPO policy selected disabled primitive %s for robot %d; "
                        "honest masking should have blocked this — forcing turn 0.",
                        action_type, self.robot_ids[i],
                    )
                    action_type = "turn"
                    goto_x_raw, goto_y_raw, turn_theta_raw = 0.0, 0.0, 0.0
                else:
                    base = i * params_per_slot
                    goto_x_raw = float(params_arr[base + 0])
                    goto_y_raw = float(params_arr[base + 1])
                    turn_theta_raw = float(params_arr[base + 2])
                per_robot_decoded.append({
                    "action_idx": idx,
                    "action_type": action_type,
                    "goto_x_raw": goto_x_raw,
                    "goto_y_raw": goto_y_raw,
                    "turn_theta_raw": turn_theta_raw,
                    "logits": None,
                    "probs": None,
                })
        else:
            action_arr = np.asarray(action, dtype=np.float32).flatten()
            expected_dim = self.num_robots * self.action_dim_per_robot
            if action_arr.shape[0] != expected_dim:
                raise ValueError(f"Expected action dim {expected_dim}, got {action_arr.shape[0]}")
            for i in range(self.num_robots):
                base = i * self.action_dim_per_robot
                logits = np.array([
                    float(action_arr[base + 0]),
                    float(action_arr[base + 1]),
                    float(action_arr[base + 2]),
                    float(action_arr[base + 3]),
                    float(action_arr[base + 4]),
                    float(action_arr[base + 5]),
                ], dtype=np.float32)
                if self.disabled_actions:
                    for j, name in enumerate(action_types):
                        if name in self.disabled_actions:
                            logits[j] = -1e9
                logits_shifted = logits - np.max(logits)
                exp_logits = np.exp(logits_shifted)
                probs = exp_logits / np.sum(exp_logits)
                action_idx = int(np.argmax(probs))
                action_type = action_types[action_idx]
                per_robot_decoded.append({
                    "action_idx": action_idx,
                    "action_type": action_type,
                    "goto_x_raw": float(action_arr[base + 6]),
                    "goto_y_raw": float(action_arr[base + 7]),
                    "turn_theta_raw": float(action_arr[base + 8]),
                    "logits": logits.tolist(),
                    "probs": probs.tolist(),
                })

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
            decoded = per_robot_decoded[i]
            action_idx = decoded["action_idx"]
            action_type = decoded["action_type"]
            raw_action_idx = action_idx
            raw_action_type = action_type
            requested_action_type = action_type
            executed_action_type = action_type
            goto_x_raw = decoded["goto_x_raw"]
            goto_y_raw = decoded["goto_y_raw"]
            turn_theta_raw = decoded["turn_theta_raw"]

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

            can_kick = has_ball_now
            kick_fired = False
            kick_blocked_bad_aim = False
            kick_tie_break_applied = False
            approach_defer_applied = False
            kick_aim_quality: Optional[float] = None
            kick_predicted_y_at_goal_line: Optional[float] = None

            # ---- Stage 2 dribble session bookkeeping (see __init__) ----
            dribble_max_radius = float(
                getattr(self.reward_config, "dribble_max_radius", 1.0)
            )
            redribble_gap_dist = self.kickable_dist + float(
                getattr(self.reward_config, "dribble_redribble_gap_margin", 0.2)
            )
            robot_ball_dist = None
            if pose is not None and ball_pos is not None and len(ball_pos) >= 2:
                robot_ball_dist = float(np.hypot(
                    float(pose[0]) - float(ball_pos[0]),
                    float(pose[1]) - float(ball_pos[1]),
                ))
            session_active = self.dribble_session_active.get(robot_id, False)
            if session_active and not has_ball_now:
                # Ball escaped the dribbler (stolen / rolled away): session over.
                self._end_dribble_session(robot_id)
                session_active = False
            anchor = self.dribble_anchor.get(robot_id)
            anchor_dist = None
            if pose is not None and anchor is not None:
                anchor_dist = float(np.hypot(
                    float(pose[0]) - anchor[0], float(pose[1]) - anchor[1]
                ))
            # Exhaustion: an active session that reached the radius limit. The
            # session must release; start_dribble is invalid until re-approach.
            dribble_exhausted_now = bool(
                session_active
                and anchor_dist is not None
                and anchor_dist >= dribble_max_radius
            )
            if dribble_exhausted_now:
                self.dribble_exhausted[robot_id] = True
                self.awaiting_redribble_gap[robot_id] = True
            # The re-dribble gap clears once the robot visibly separates from
            # the ball after the last session ended.
            if (
                self.awaiting_redribble_gap.get(robot_id, False)
                and not session_active
                and robot_ball_dist is not None
                and robot_ball_dist > redribble_gap_dist
            ):
                self.awaiting_redribble_gap[robot_id] = False
                self.dribble_exhausted[robot_id] = False
            start_dribble_blocked = (
                action_type == "start_dribble"
                and has_ball_now
                and (
                    self.awaiting_redribble_gap.get(robot_id, False)
                    or self.dribble_exhausted.get(robot_id, False)
                )
            )
            stop_dribble_fired = False
            stop_dribble_at_limit = False

            # TD3 exposes discrete primitive choice as continuous logits, then
            # uses argmax. When turn and kick saturate at the same bound,
            # np.argmax always picks turn because it appears first. In the
            # aim-gated curriculum, a tied kick logit is deployable once the
            # robot is holding the ball and the projected shot is on target.
            if (
                self.kick_tie_break_when_aimed
                and self.kick_requires_aim
                and can_kick
                and pose is not None
                and decoded["logits"] is not None
            ):
                logits_arr = np.asarray(decoded["logits"], dtype=np.float32)
                turn_logit = float(logits_arr[2])
                kick_logit = float(logits_arr[3])
                turn_enabled = turn_logit > -1e8
                kick_enabled = kick_logit > -1e8
                if turn_enabled and kick_enabled:
                    kick_aim_quality, kick_predicted_y_at_goal_line = (
                        self._kick_aim_quality_from_pose(pose)
                    )
                    if (
                        kick_aim_quality >= self.kick_min_aim_quality
                        and kick_logit >= turn_logit - self.kick_tie_break_epsilon
                    ):
                        action_idx = 3
                        action_type = "kick"
                        requested_action_type = action_type
                        executed_action_type = action_type
                        kick_tie_break_applied = raw_action_type != "kick"

            # In approach+turn+kick stages, the approach primitive is useful
            # until possession is reached. Once the robot already has the ball,
            # a raw approach selection is a no-op ("done" -> turn 0), which can
            # trap deterministic TD3 when the newly unmasked approach logit ties
            # with turn. If turn is effectively tied, hand off to the actor's
            # turn parameter so the policy can continue aiming.
            if (
                action_type == "approach_ball"
                and self.approach_defer_when_has_ball
                and has_ball_now
                and decoded["logits"] is not None
            ):
                logits_arr = np.asarray(decoded["logits"], dtype=np.float32)
                approach_logit = float(logits_arr[1])
                turn_logit = float(logits_arr[2])
                turn_enabled = turn_logit > -1e8
                if turn_enabled and turn_logit >= approach_logit - self.approach_defer_epsilon:
                    action_idx = 2
                    action_type = "turn"
                    requested_action_type = action_type
                    executed_action_type = action_type
                    approach_defer_applied = raw_action_type == "approach_ball"

            if action_type == "kick" and can_kick and self.kick_requires_aim and pose is not None:
                if kick_aim_quality is None:
                    kick_aim_quality, kick_predicted_y_at_goal_line = self._kick_aim_quality_from_pose(pose)
                if kick_aim_quality < self.kick_min_aim_quality:
                    kick_blocked_bad_aim = True

            invalid_action_requested = (
                (action_type == "kick" and not can_kick)
                or (action_type == "start_dribble" and not has_ball_now)
                or (action_type == "stop_dribble" and not has_ball_now)
                or start_dribble_blocked
                or kick_blocked_bad_aim
            )
            if invalid_action_requested:
                invalid_action_count += 1

            if action_type == "kick":
                if not can_kick or kick_blocked_bad_aim:
                    # Use the policy's own turn_theta rather than "turn 0" so
                    # the state changes each step, allowing Q(kick) to receive
                    # proper TD targets instead of bootstrapping off itself in
                    # a stationary (s == s') loop.
                    command = f"turn {turn_theta:.2f}"
                    if kick_blocked_bad_aim:
                        executed_action_type = "turn"
                else:
                    command = "kick 100 0"
                    kick_fired = True
                    if session_active:
                        # A fired kick releases the ball, ending the session.
                        self._end_dribble_session(robot_id)
            elif action_type == "start_dribble":
                if not has_ball_now:
                    command = "turn 0"
                elif start_dribble_blocked:
                    if dribble_exhausted_now:
                        # At the radius limit with the ball still caught:
                        # force the release instead of sustaining the dribble.
                        command = "drop"
                        executed_action_type = "stop_dribble"
                        self._end_dribble_session(robot_id)
                    else:
                        # Re-catch before creating separation: refuse with a no-op.
                        command = "turn 0"
                        executed_action_type = "turn"
                else:
                    command = "catch 0"  # Start dribble
                    if not session_active and pose is not None:
                        self.dribble_session_active[robot_id] = True
                        self.dribble_anchor[robot_id] = (float(pose[0]), float(pose[1]))
                        self.dribble_exhausted[robot_id] = False
            elif action_type == "stop_dribble":
                if not has_ball_now:
                    command = "turn 0"
                else:
                    command = "drop"  # Stop dribble
                    if session_active:
                        stop_dribble_fired = True
                        stop_dribble_at_limit = bool(
                            anchor_dist is not None
                            and anchor_dist
                            >= self._STOP_DRIBBLE_RELEASE_FRACTION * dribble_max_radius
                        )
                        self._end_dribble_session(robot_id)
            elif action_type == "turn":
                command = f"turn {turn_theta:.2f}"
            elif action_type == "approach_ball":
                if pose is None or game_state is None:
                    command = "turn 0"
                else:
                    self_pose = np.array([
                        float(pose[0]),
                        float(pose[1]),
                        float(np.deg2rad(pose[2])),
                    ], dtype=np.float32)
                    command = approach_ball(
                        self_pose=self_pose,
                        game_state=game_state,
                        margin=self.kickable_dist,
                    )
                    if command == "done":
                        command = "turn 0"
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
                    "action_type": executed_action_type,
                    "requested_action_type": requested_action_type,
                    "action_idx": action_idx,
                    "raw_action_type": raw_action_type,
                    "raw_action_idx": raw_action_idx,
                    "kick_tie_break_applied": kick_tie_break_applied,
                    "approach_defer_applied": approach_defer_applied,
                    "probs": decoded["probs"],
                    "logits": decoded["logits"],
                    "goto_x": goto_x,
                    "goto_y": goto_y,
                    "turn_theta": turn_theta,
                    "has_ball_now": has_ball_now,
                    "kick_fired": kick_fired,
                    "kick_blocked_bad_aim": kick_blocked_bad_aim,
                    "kick_aim_quality": kick_aim_quality,
                    "kick_predicted_y_at_goal_line": kick_predicted_y_at_goal_line,
                    "invalid_action_requested": invalid_action_requested,
                    "dribble_session_active": self.dribble_session_active.get(robot_id, False),
                    "dribble_anchor_dist": anchor_dist,
                    "dribble_exhausted": self.dribble_exhausted.get(robot_id, False),
                    "awaiting_redribble_gap": self.awaiting_redribble_gap.get(robot_id, False),
                    "start_dribble_blocked": start_dribble_blocked,
                    "stop_dribble_fired": stop_dribble_fired,
                    "stop_dribble_at_limit": stop_dribble_at_limit,
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
        prev_ball_pos: Optional[Tuple[float, float]] = None,
        prev_facing_goal_cos: Optional[float] = None,
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

        # Stage 2: goalie y for gap-based aim gating (None in 1v0 stages) and
        # command-level dribble session state for envelope-gated dribble reward.
        goalie_pose = self._opponent_goalie_pose(current_game_state)
        anchor = self.dribble_anchor.get(robot_id)
        dribble_anchor_dist = None
        if anchor is not None:
            dribble_anchor_dist = float(np.hypot(robot_x - anchor[0], robot_y - anchor[1]))

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
            prev_ball_pos=prev_ball_pos,
            prev_facing_goal_cos=prev_facing_goal_cos,
            goalie_y=(float(goalie_pose[1]) if goalie_pose is not None else None),
            is_dribbling=bool(self.dribble_session_active.get(robot_id, False)),
            dribble_anchor_dist=dribble_anchor_dist,
        )

    @staticmethod
    def _default_reward_state(has_ball: bool) -> str:
        """Infer a default reward state when no explicit state label is provided."""

        if has_ball:
            return "secure_possession"
        return "chase_ball"

    # stop_dribble counts as a deliberate "release near the limit" (eligible
    # for stop_dribble_release_bonus) when the session covered at least this
    # fraction of dribble_max_radius.
    _STOP_DRIBBLE_RELEASE_FRACTION: float = 0.7

    def _end_dribble_session(self, robot_id: int) -> None:
        """Close a dribble session and require re-approach separation."""

        self.dribble_session_active[robot_id] = False
        self.dribble_anchor[robot_id] = None
        self.awaiting_redribble_gap[robot_id] = True

    def _opponent_goalie_pose(self, game_state) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, theta_deg) of the opponent goalie, or None.

        Stage 2 fields a single scripted goalie, so "the opponent robot
        nearest the opponent goal center" identifies it robustly without a
        config-driven goalie id; with multiple opponents this still picks the
        keeper as long as it holds its goal.
        """

        if game_state is None:
            return None
        best_pose: Optional[Tuple[float, float, float]] = None
        best_dist = float("inf")
        for team_name, entries in getattr(game_state, "robot_poses", {}).items():
            if team_name == self.team_name:
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for _rid, pose in entry.items():
                    if pose is None or len(pose) < 3:
                        continue
                    d = float(math.hypot(
                        float(pose[0]) - float(GOAL_R[0]),
                        float(pose[1]) - float(GOAL_R[1]),
                    ))
                    if d < best_dist:
                        best_dist = d
                        best_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
        return best_pose

    @staticmethod
    def _project_kick_y_at_goal_line(pose: Sequence[float]) -> Optional[float]:
        """Project a straight kick from robot pose to the opponent goal line."""

        rx, ry, theta_deg = float(pose[0]), float(pose[1]), float(pose[2])
        theta_rad = math.radians(theta_deg)
        cos_t = math.cos(theta_rad)
        if cos_t <= 1e-3:
            return None
        return float(ry + (FIELD_X[1] - rx) * (math.sin(theta_rad) / cos_t))

    def _kick_aim_quality_from_pose(self, pose: Sequence[float]) -> Tuple[float, Optional[float]]:
        """Return aim quality and projected goal-line y for a straight kick."""

        predicted_y = self._project_kick_y_at_goal_line(pose)
        return (
            aim_quality_from_prediction(
                predicted_y,
                float(getattr(self.reward_config, "goal_half_height", 5.0)),
            ),
            predicted_y,
        )

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
                    prev_ball_pos=self.prev_reward_ball_pos,
                    prev_facing_goal_cos=self.prev_reward_facing_goal_cos_by_id.get(robot_id),
                )
            except ValueError:
                continue

            reward_result = evaluate_reward(reward_inputs, config=self.reward_config)
            rewards.append(float(reward_result.reward))
            self.prev_reward_ball_dist_by_id[robot_id] = reward_result.intermediates.ball_dist
            if reward_result.intermediates.facing_goal_cos is not None:
                self.prev_reward_facing_goal_cos_by_id[robot_id] = reward_result.intermediates.facing_goal_cos
            next_ball_to_goal_dist = reward_result.intermediates.ball_to_goal_dist

        if next_ball_to_goal_dist is not None:
            self.prev_reward_ball_to_goal_dist = next_ball_to_goal_dist

        if current_game_state.ball_pos is not None:
            self.prev_reward_ball_pos = (
                float(current_game_state.ball_pos[0]),
                float(current_game_state.ball_pos[1]),
            )

        if not rewards:
            return 0.0

        return float(np.mean(rewards))

    _GOAL_HALF_HEIGHT: float = 5.0  # SSL Div B goal width=1000mm → 10 sim units → half=5.0

    def _get_curriculum_ball_pos(self) -> tuple[float, float]:
        """Curriculum: start ball near opponent goal early, move toward centre as
        episodes progress. Random kicks score frequently when the ball is near
        the goal, giving the agent the sparse goal_reward signal needed to break
        out of the approach-and-park local maximum.

        Ball must stay OUTSIDE the right penalty area (x < 35, y any) to avoid
        rcssserver's BallStuckRef: if the ball sits still for drop_ball_time=100
        cycles inside the penalty area, the server calls awardDropBall() which
        runs moveOutOfPenalty() and teleports the ball to (35, ±10) — the corner
        of the penalty area — disrupting the curriculum entirely.

        Stage 1 narrowed mode: when `self.random_ball_x` is True, the ball x is
        sampled uniformly from `self.random_ball_x_range`. When `self.random_ball_y`
        is also True, the ball y is sampled from `self.random_ball_y_range`,
        forcing the robot to actually rotate to face the goal before kicking
        (otherwise the robot spawn at theta=0 trivially aims at the goal).
        """
        if self.random_ball_x:
            x_min, x_max = self.random_ball_x_range
            x = float(np.random.uniform(x_min, x_max))
            if self.random_ball_y:
                y_min, y_max = self.random_ball_y_range
                y = float(np.random.uniform(y_min, y_max))
            else:
                y = 0.0
            return (x, y)

        ep = self.episode_num
        if ep < 50:
            return (34.0, 0.0)   # penalty spot — 11m from goal, just outside penalty area
        elif ep < 120:
            return (25.0, 0.0)   # 20m from goal
        elif ep < 220:
            return (15.0, 0.0)   # 30m from goal
        else:
            return (0.0, 0.0)    # full task — centre kickoff

    # Right-side penalty area boundary in our env coordinates. rcssserver's
    # BallStuckRef rule fires when a ball stays still inside the penalty area
    # for `drop_ball_time` cycles (default 100), then teleports it and may
    # change the playmode — corrupting any further training transitions.
    # Cutting the episode short the moment the ball enters this region (and
    # misses the goal mouth) avoids the bug entirely AND gives the model a
    # shaping signal: "kicks past x=35 must hit the goal mouth, not the sides".
    _RIGHT_PENALTY_AREA_X: float = 35.0

    # Position-jump tolerances for the teleport backstop. A ball cannot move
    # more than `ball_speed_max` units in one cycle (position advances by the
    # velocity, which the server clips to ball_speed_max). This threshold MUST
    # stay above ball_speed_max (server.conf) or legitimate kicks get flagged as
    # teleports and episodes die after one step. ball_speed_max is currently 6
    # (raised from 3 to give a ~5.75 m/s kick exit), so 8.0 leaves headroom for
    # noise/wind while still catching real server teleports (which jump the ball
    # tens of units). If you raise ball_speed_max again, raise this too.
    # The robot, under our Stage 1 action mask, cannot move at all (only kick is
    # selectable), so its tolerance stays tight.
    _BALL_TELEPORT_THRESHOLD: float = 8.0
    _ROBOT_TELEPORT_THRESHOLD: float = 1.0

    # Ball-stopped detection. Speed under this threshold for N consecutive
    # cycles signals the ball has come to rest. Combined with the robot being
    # unable to reach it (approach disabled and distance > kickable_dist), this
    # ends the episode early instead of burning the remaining max_steps.
    _BALL_STOPPED_SPEED: float = 0.05
    _BALL_STOPPED_STEPS: int = 10

    # Frozen-state detection: if the ball position AND every one of our
    # robots' poses are bit-identical (within float noise) to the previous
    # step for this many consecutive steps, the sim connection is treating
    # us to the same stale frame repeatedly (or the ball is genuinely wedged
    # with the robot unable to affect it). Runs unconditionally, unlike
    # _BALL_STOPPED_STEPS above which only ends the episode when
    # approach_ball/goto are both disabled — this is the general backstop
    # that catches the case those actions enabled but the frame still never
    # advances, which let one inference episode bank a kick_aim_bonus on
    # every step of a ~140-step frozen frame.
    _FROZEN_STATE_EPS: float = 1e-6
    _FROZEN_STATE_STEPS: int = 15
    # Heading epsilon (degrees) for the same backstop. Without this, a robot
    # that's purely turning in place (no translation) has bit-identical x/y
    # for as long as it keeps turning and gets wrongly killed as "frozen"
    # even though it's actively rotating and the sim is healthy.
    _FROZEN_STATE_THETA_EPS_DEG: float = 1e-4

    # Touchline dead-zone margin: a ball within this many units of the
    # |y|=FIELD_Y[1] touchline is treated as out-of-bounds. In a real game a
    # ball this close to the line is a kick-in/throw-in (dead, not
    # recoverable). The strict `|by| > FIELD_Y[1]` check used to terminate
    # only after the ball had fully crossed — but in stage 1.9 (with
    # approach_ball enabled) a ball that decays to rest just inside the
    # touchline triggers no termination and the episode idles to max_steps.
    # Applied to the touchline only — end-line near-misses are still handled
    # by `ball_in_penalty_off_target`, and applying a margin to the end line
    # would risk pre-empting goals.
    _TOUCHLINE_DEAD_MARGIN: float = 0.5

    # Goalie-catch detection radius. rcssserver's catch area is roughly 1.2 × 1.0
    # units; we use a generous circular threshold so a catch is reliably flagged
    # even when the ball drifts slightly after being held.
    _GOALIE_CATCH_DIST: float = 3.0

    def _own_robot_pose(self, game_state) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, theta_deg) for the first controlled robot, or None."""
        if game_state is None:
            return None
        team_pose_entries = getattr(game_state, "robot_poses", {}).get(self.team_name, [])
        for entry in team_pose_entries:
            if isinstance(entry, dict):
                for rid, pose in entry.items():
                    if rid == self.robot_ids[0]:
                        return (float(pose[0]), float(pose[1]), float(pose[2]))
        return None

    def _check_terminal(self, game_state, prev_game_state=None) -> tuple[bool, str]:
        """Return (terminated, reason) based on goal/out-of-bounds conditions.

        `prev_game_state` is the game state at the *start* of the current step
        (before the action was applied). It's used for unphysical-jump checks;
        comparing against fields like `self.prev_reward_ball_pos` doesn't work
        because those are updated by `_calculate_reward` before this method runs.
        """
        if game_state is None:
            return False, ""

        ball_pos = getattr(game_state, "ball_pos", None)
        if ball_pos is None:
            return False, ""

        bx, by = float(ball_pos[0]), float(ball_pos[1])

        if bx >= FIELD_X[1] and abs(by) < self._GOAL_HALF_HEIGHT:
            return True, "goal_scored"

        if abs(by) >= FIELD_Y[1] - self._TOUCHLINE_DEAD_MARGIN or abs(bx) > FIELD_X[1]:
            return True, "ball_out_of_bounds"

        # Ball entered the right penalty area but missed the goal mouth. Ending
        # the episode here is BOTH a bug workaround (avoids BallStuckRef) AND a
        # learning signal — see _RIGHT_PENALTY_AREA_X comment above.
        if bx >= self._RIGHT_PENALTY_AREA_X and abs(by) > self._GOAL_HALF_HEIGHT:
            return True, "ball_in_penalty_off_target"

        # Early goalie-possession detection: the rcssserver catch animation holds
        # the ball at the keeper for 1-2 cycles BEFORE free_kick_right fires.
        # Terminate in that window to prevent tug-of-war kick attempts. Only
        # fire when the keeper is deep in its goal zone (x > 38) so this check
        # doesn't trigger when the goalie rushes out to challenge during open play.
        gk_pose_early = self._opponent_goalie_pose(game_state)
        if gk_pose_early is not None:
            gk_x_e, gk_y_e, _ = gk_pose_early
            if gk_x_e > 38.0:
                early_catch_dist = float(
                    getattr(self.reward_config, "goalie_possession_dist", 2.0)
                )
                if math.hypot(bx - gk_x_e, by - gk_y_e) < early_catch_dist:
                    return True, "goalie_catch"

        # Goalie catch: free_kick_right is issued by rcssserver when the right-side
        # goalie successfully catches. Confirm with ball proximity to the goalie to
        # distinguish from other free-kick causes (e.g. fouls).
        playmode = getattr(game_state, "playmode", None)
        if playmode == "free_kick_right":
            gk_pose = self._opponent_goalie_pose(game_state)
            if gk_pose is not None:
                gk_x, gk_y, _ = gk_pose
                if math.hypot(bx - gk_x, by - gk_y) < self._GOALIE_CATCH_DIST:
                    return True, "goalie_catch"

        # Dead-ball playmodes: when rcssserver transitions out of play_on
        # (kick_in_*, corner_kick_*, goal_kick_*, foul_*, etc.) the ball is no
        # longer in play. Without approach_ball/goto enabled, the robot can't
        # do anything meaningful — keep going just wastes cycles until
        # max_steps. End immediately. The playmode is preserved in the reason
        # so the trainer logs distinguish causes.
        if playmode is not None and playmode not in ("play_on", "before_kick_off"):
            return True, f"ball_dead_{playmode}"

        # Frozen-state backstop: ball AND all our robots' poses bit-identical
        # to last step for too long means the frame isn't advancing (stale
        # sim connection, or a genuinely wedged ball). See _FROZEN_STATE_STEPS.
        if prev_game_state is not None:
            prev_ball_pos_fs = getattr(prev_game_state, "ball_pos", None)
            ball_frozen = (
                prev_ball_pos_fs is not None
                and abs(bx - float(prev_ball_pos_fs[0])) < self._FROZEN_STATE_EPS
                and abs(by - float(prev_ball_pos_fs[1])) < self._FROZEN_STATE_EPS
            )
            robots_frozen = False
            if ball_frozen:
                team_robots_fs = getattr(game_state, "robot_poses", {}).get(self.team_name, [])
                prev_team_robots_fs = getattr(prev_game_state, "robot_poses", {}).get(self.team_name, [])
                prev_pose_by_unum_fs: Dict[int, Tuple[float, float, float]] = {}
                for entry in prev_team_robots_fs:
                    for unum, pose in entry.items():
                        prev_pose_by_unum_fs[int(unum)] = (
                            float(pose[0]), float(pose[1]), float(pose[2]),
                        )
                robots_frozen = True
                any_robot_checked = False
                for robot in team_robots_fs:
                    for unum, pose in robot.items():
                        if int(unum) not in self.robot_ids:
                            continue
                        prev_pose_fs = prev_pose_by_unum_fs.get(int(unum))
                        if prev_pose_fs is None:
                            robots_frozen = False
                            continue
                        any_robot_checked = True
                        # Wrap to [-180, 180] so e.g. 179° -> -179° isn't seen as a huge jump.
                        theta_delta = (float(pose[2]) - prev_pose_fs[2] + 180.0) % 360.0 - 180.0
                        theta_delta = abs(theta_delta)
                        if (
                            abs(float(pose[0]) - prev_pose_fs[0]) >= self._FROZEN_STATE_EPS
                            or abs(float(pose[1]) - prev_pose_fs[1]) >= self._FROZEN_STATE_EPS
                            or theta_delta >= self._FROZEN_STATE_THETA_EPS_DEG
                        ):
                            robots_frozen = False
                robots_frozen = robots_frozen and any_robot_checked
            if ball_frozen and robots_frozen:
                self._frozen_state_counter += 1
            else:
                self._frozen_state_counter = 0
            if self._frozen_state_counter >= self._FROZEN_STATE_STEPS:
                return True, "frozen_state_stale_sim"

        # Ball stopped far from the robot: if the ball has come to rest (low
        # speed for several consecutive cycles) and the robot can't reach it
        # (no approach action available, robot distance > kickable_dist), the
        # episode is effectively dead. Don't burn the remaining max_steps.
        if prev_game_state is not None:
            prev_ball_pos = getattr(prev_game_state, "ball_pos", None)
            if prev_ball_pos is not None:
                pbx, pby = float(prev_ball_pos[0]), float(prev_ball_pos[1])
                ball_speed = math.hypot(bx - pbx, by - pby)
                if ball_speed < self._BALL_STOPPED_SPEED:
                    self._stopped_ball_counter += 1
                else:
                    self._stopped_ball_counter = 0
                approach_disabled = (
                    "approach_ball" in self.disabled_actions
                    and "goto" in self.disabled_actions
                )
                if (
                    self._stopped_ball_counter >= self._BALL_STOPPED_STEPS
                    and approach_disabled
                ):
                    own_pose = self._own_robot_pose(game_state)
                    if own_pose is not None:
                        rx, ry, _ = own_pose
                        if math.hypot(bx - rx, by - ry) > self.kickable_dist:
                            return True, "ball_stopped_unreachable"

        # Position-jump backstop: compare positions at the start of this step
        # (prev_game_state, pre-action) to the end (game_state, post-action).
        # An unphysical jump means rcssserver teleported the object.
        prev_ball_pos = getattr(prev_game_state, "ball_pos", None) if prev_game_state is not None else None
        if prev_ball_pos is not None:
            pbx, pby = float(prev_ball_pos[0]), float(prev_ball_pos[1])
            ball_jump = math.hypot(bx - pbx, by - pby)
            if ball_jump > self._BALL_TELEPORT_THRESHOLD:
                self.logger.warning(
                    "Ball teleport detected: prev=(%.2f, %.2f) curr=(%.2f, %.2f) "
                    "jump=%.2f", pbx, pby, bx, by, ball_jump,
                )
                return True, "ball_teleport"

        # Only iterate our own team's robots. Both teams use unum=1, so
        # iterating all teams + matching by unum would mistakenly flag the
        # opposing team's stationary player #1 (at default pose ~(20, 10))
        # as if it were our robot teleporting there every step.
        team_robots = getattr(game_state, "robot_poses", {}).get(self.team_name, [])
        prev_team_robots = (
            getattr(prev_game_state, "robot_poses", {}).get(self.team_name, [])
            if prev_game_state is not None else []
        )
        # Build a quick lookup of prev poses by unum for matching.
        prev_pose_by_unum: Dict[int, Tuple[float, float]] = {}
        for entry in prev_team_robots:
            for unum, pose in entry.items():
                prev_pose_by_unum[int(unum)] = (float(pose[0]), float(pose[1]))

        for robot in team_robots:
            for unum, pose in robot.items():
                if int(unum) in self.robot_ids:
                    rx, ry = float(pose[0]), float(pose[1])
                    if abs(rx) > FIELD_X[1] or abs(ry) > FIELD_Y[1]:
                        return True, "robot_out_of_bounds"
                    # Under our action mask (kick/turn only), the robot
                    # cannot move. Any significant displacement is a
                    # server-side teleport (e.g. dead-ball reposition).
                    if self.disabled_actions and not self._can_robot_move():
                        prev = prev_pose_by_unum.get(int(unum))
                        if prev is not None:
                            jump = math.hypot(rx - prev[0], ry - prev[1])
                            if jump > self._ROBOT_TELEPORT_THRESHOLD:
                                self.logger.warning(
                                    "Robot %s teleport detected: prev=(%.2f, %.2f) "
                                    "curr=(%.2f, %.2f) jump=%.2f",
                                    unum, prev[0], prev[1], rx, ry, jump,
                                )
                                return True, "robot_teleport"

        return False, ""

    def _can_robot_move(self) -> bool:
        """Return True if the current action mask allows any locomotion action."""
        locomotion_actions = {"goto", "approach_ball"}
        return any(a not in self.disabled_actions for a in locomotion_actions)

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

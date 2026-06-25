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
from ai_interface.utils.basic_commands import (
    goto, approach_ball, kick, dribble_to, DribbleState,
    DRIBBLE_PHASE_CARRY, DRIBBLE_PHASE_ALIGN_RELEASE,
    DRIBBLE_PHASE_GRAB, DRIBBLE_PHASE_SETTLE,
    DRIBBLE_PHASE_VERIFY,
    DRIBBLE_PHASE_RELEASE,
    DRIBBLE_PHASE_DONE,
)
from ai_interface.constants.field_constants import *
from ai_interface.constants.player_constants import *
from ai_interface.envs.reward import (
    RewardConfig,
    RewardInputs,
    aim_quality_from_prediction,
    evaluate_reward,
    extract_opponent_positions,
    goalie_gap_quality,
    post_safe_goalie_gap_quality,
    positional_gap_quality,
    reachable_gap_delta,
    lane_clear_quality,
    positional_shot_quality,
)
from networking.networker import Networker
from networking.data_utils import GameState, limit_turn_rate


class JALTeamEnv(gym.Env):
    ACTION_TYPES = ["goto", "turn", "kick", "dribble_to"]

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
        ball_action_recovery: bool = False,
        ball_claimant_robot_ids: Optional[List[int]] = None,
        ball_claimant_switch_margin: float = 0.75,
        turn_stall_limit: int = 12,
        turn_stall_displacement: float = 0.05,
        reward_config_overrides: Optional[Dict[str, float]] = None,
        some_arg=None,
        opponent_team_name: Optional[str] = None,
        num_opponents: int = 1,
        own_goalie_robot_id: Optional[int] = None,
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
        self.ball_action_recovery: bool = bool(ball_action_recovery)
        eligible_ids = self.robot_ids if ball_claimant_robot_ids is None else ball_claimant_robot_ids
        self.ball_claimant_robot_ids: List[int] = [
            int(rid) for rid in eligible_ids if int(rid) in self.robot_ids
        ]
        self.ball_claimant_switch_margin: float = max(0.0, float(ball_claimant_switch_margin))
        self.turn_stall_limit: int = max(1, int(turn_stall_limit))
        self.turn_stall_displacement: float = max(0.0, float(turn_stall_displacement))
        self.ball_claimant_id: Optional[int] = None
        self.turn_stall_steps: Dict[int, int] = {rid: 0 for rid in self.robot_ids}
        self._turn_stall_last_pose: Dict[int, Optional[Tuple[float, float]]] = {
            rid: None for rid in self.robot_ids
        }

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
        # How many opponent context slots to activate/fill (slots 1..num_opponents).
        # 1 = goalie only (stages 1-2); 2 = goalie + defender (stage 3); the
        # backbone is count-agnostic so this just toggles which context slots the
        # attention attends to, no rebuild. Capped to the available context slots.
        self.num_opponents: int = max(1, int(num_opponents))
        # Own goalie robot_id (Stage 4+): friendly HC goalie on our team, not RL-controlled.
        self.own_goalie_robot_id: Optional[int] = (
            int(own_goalie_robot_id) if own_goalie_robot_id is not None else None
        )

        # Active masks: which agent slots are controlled robots, which context
        # slots are present. Static within Stage 1 (1 robot, no context); later
        # stages / foul-randomization will vary them per episode.
        self.agent_active_mask = np.zeros(self.a_max, dtype=np.float32)
        self.agent_active_mask[: self.num_robots] = 1.0
        self.context_active_mask = np.zeros(self.c_max, dtype=np.float32)
        if self.own_goalie_robot_id is not None:
            self.context_active_mask[0] = 1.0  # slot 0 = own goalie
        if self.opponent_team_name is not None:
            # Slots 1..num_opponents = opponents (slot 1 = goalie, slot 2 = defender, …).
            for k in range(self.num_opponents):
                slot = 1 + k
                if slot < self.c_max:
                    self.context_active_mask[slot] = 1.0
        self.is_dribbling = {robot_id: False for robot_id in self.robot_ids}  # Track dribble state per robot
        self.start_dribble_pos = {robot_id: [-1.0, -1.0] for robot_id in self.robot_ids}  # Placeholder for dribble start position, can be updated in step() when dribble starts

        # Per-robot DribbleState for the dribble_to phase machine (basic_commands).
        self.dribble_states: Dict[int, DribbleState] = {rid: DribbleState() for rid in self.robot_ids}
        # Dribble session tracking derived from DribbleState phases, used by
        # reward code and info reporting.
        self.dribble_session_active: Dict[int, bool] = {rid: False for rid in self.robot_ids}
        self.dribble_anchor: Dict[int, Optional[Tuple[float, float]]] = {rid: None for rid in self.robot_ids}
        # Steps since the last dribble release per robot (phase CARRY→RELEASE).
        self.steps_since_stop_dribble: Dict[int, Optional[int]] = {rid: None for rid in self.robot_ids}
        # dribble_to reward tracking: target chosen, prev distance, action transition.
        self.dribble_to_target: Dict[int, Optional[Tuple[float, float]]] = {rid: None for rid in self.robot_ids}
        self.prev_ball_to_dribble_target_dist: Dict[int, Optional[float]] = {rid: None for rid in self.robot_ids}
        self._prev_action_was_dribble_to: Dict[int, bool] = {rid: False for rid in self.robot_ids}
        # Positional_gap_quality at the spot where the current carry segment
        # opened; used to pay the achieved-gap reward when the segment closes.
        self.dribble_carry_start_gap: Dict[int, Optional[float]] = {rid: None for rid in self.robot_ids}

        # Stage 4+: own-goalie coordination tracking
        self._prev_has_ball: Dict[int, bool] = {rid: False for rid in self.robot_ids}
        self._prev_opponent_near_ball: bool = False
        self._prev_clearance_zone_dist: Dict[int, Optional[float]] = {rid: None for rid in self.robot_ids}

        # Stage 5+: multi-robot coordination tracking
        self._prev_ball_carrier: Optional[int] = None

        # Action design per robot (8D): [goto_logit, approach_ball_logit, turn_logit, kick_logit, dribble_to_logit, goto_x, goto_y, turn_theta]
        self.action_dim_per_robot = 8
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
        # Debug/probe diagnostics for the latest real fired kick in the current
        # episode. `infer.py --debug_infer` uses these per-step fields plus the
        # final episode outcome to compute catch rate by fire distance.
        self._last_kick_probe_by_id: Dict[int, Dict[str, Any]] = {}
        # Per-robot kick macro state. With the real turn cap (20 deg/s), a
        # kick request often needs many simulator cycles of internal alignment
        # before the primitive can fire. Keep that alignment committed across
        # policy steps so a sampled dribble_to/approach does not interrupt it.
        self.kick_macro_active: Dict[int, bool] = {rid: False for rid in self.robot_ids}
        self.kick_macro_target_y: Dict[int, Optional[float]] = {rid: None for rid in self.robot_ids}
        self.kick_macro_align_steps: Dict[int, int] = {rid: 0 for rid in self.robot_ids}
        self.kick_macro_retarget_count: Dict[int, int] = {rid: 0 for rid in self.robot_ids}
        self.kick_macro_max_align_steps: int = 120

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

    def _ball_claimant(
        self,
        game_state,
        pose_by_robot_id: Optional[Dict[int, Any]] = None,
    ) -> Optional[int]:
        """Return the one controlled robot allowed to execute ball-seeking actions.

        A committed dribble owns the claim. Otherwise possession-range robots take
        priority, then the nearest configured eligible robot. The current claimant
        is retained within ``ball_claimant_switch_margin`` of the nearest candidate
        to prevent two nearby teammates from swapping ownership every simulator
        cycle. Goalkeepers or fixed support robots can be excluded through
        ``ball_claimant_robot_ids``.
        """
        if not getattr(self, "ball_action_recovery", False) or game_state is None:
            return None
        ball_pos = getattr(game_state, "ball_pos", None)
        if ball_pos is None or len(ball_pos) < 2:
            return getattr(self, "ball_claimant_id", None)
        if pose_by_robot_id is None:
            pose_by_robot_id = {}
            for entry in getattr(game_state, "robot_poses", {}).get(self.team_name, []) or []:
                if isinstance(entry, dict):
                    pose_by_robot_id.update(entry)

        eligible = getattr(self, "ball_claimant_robot_ids", self.robot_ids)
        distances = {
            rid: float(np.hypot(
                float(pose_by_robot_id[rid][0]) - float(ball_pos[0]),
                float(pose_by_robot_id[rid][1]) - float(ball_pos[1]),
            ))
            for rid in eligible
            if rid in pose_by_robot_id
        }
        if not distances:
            self.ball_claimant_id = None
            return None

        dribble_states = getattr(self, "dribble_states", {})
        kick_macro_active = getattr(self, "kick_macro_active", {})
        committed = [
            rid for rid in distances
            if (
                bool(getattr(dribble_states.get(rid), "committed", False))
                or bool(kick_macro_active.get(rid, False))
            )
        ]
        if committed:
            claimant = min(committed, key=distances.__getitem__)
            self.ball_claimant_id = claimant
            return claimant

        kickable_dist = float(getattr(self, "kickable_dist", 0.0))
        possessing = [rid for rid, dist in distances.items() if dist <= kickable_dist]
        candidates = possessing if possessing else list(distances)
        nearest = min(candidates, key=distances.__getitem__)
        current = getattr(self, "ball_claimant_id", None)
        switch_margin = float(getattr(self, "ball_claimant_switch_margin", 0.75))
        if current in candidates and distances[current] <= distances[nearest] + switch_margin:
            claimant = current
        else:
            claimant = nearest
        self.ball_claimant_id = claimant
        return claimant

    def get_primitive_valid_mask(
        self,
        game_state=None,
        num_primitives: int = 12,
    ) -> np.ndarray:
        """Build the per-slot runtime primitive mask used by PPO sampling.

        Only the claimant may approach, kick, or dribble. For the claimant,
        approach is valid only outside possession range and kick/dribble only
        inside it. A claimant that has issued too many stationary turns has turn
        temporarily masked, forcing the categorical policy to choose approach.
        Stage-disabled and reserved primitive masks are applied separately by the
        PPO agent.
        """
        mask = np.ones((self.a_max, int(num_primitives)), dtype=np.float32)
        if not getattr(self, "ball_action_recovery", False):
            return mask
        if game_state is None:
            game_state = getattr(self, "_cached_game_state", None)
        if game_state is None:
            return mask

        pose_by_robot_id: Dict[int, Any] = {}
        for entry in getattr(game_state, "robot_poses", {}).get(self.team_name, []) or []:
            if isinstance(entry, dict):
                pose_by_robot_id.update(entry)
        ball_pos = getattr(game_state, "ball_pos", None)
        claimant = self._ball_claimant(game_state, pose_by_robot_id)
        for slot, rid in enumerate(self.robot_ids):
            if slot >= self.a_max or rid not in pose_by_robot_id or ball_pos is None:
                continue
            dist = float(np.hypot(
                float(pose_by_robot_id[rid][0]) - float(ball_pos[0]),
                float(pose_by_robot_id[rid][1]) - float(ball_pos[1]),
            ))
            dribble_state = getattr(self, "dribble_states", {}).get(rid)
            if bool(getattr(self, "kick_macro_active", {}).get(rid, False)):
                # A committed kick alignment owns the ball until it fires,
                # times out, or loses possession. Do not let the sampler switch
                # to dribble_to/approach mid-aim under the 20 deg/s turn cap.
                mask[slot, :5] = 0.0
                mask[slot, 3] = 1.0
                continue
            if bool(getattr(dribble_state, "committed", False)):
                # A committed macro owns the transition. The policy may either
                # continue it or interrupt with kick; all other sampled actions
                # would be ignored and violate PPO's action/transition contract.
                mask[slot, :5] = 0.0
                mask[slot, 3] = 1.0  # kick interrupt
                mask[slot, 4] = 1.0  # parameterless continuation (target latched)
                continue
            if rid != claimant:
                mask[slot, 1] = 0.0  # approach_ball
                mask[slot, 3] = 0.0  # kick
                mask[slot, 4] = 0.0  # dribble_to
                continue
            if dist > self.kickable_dist:
                mask[slot, 3] = 0.0
                mask[slot, 4] = 0.0
            else:
                mask[slot, 1] = 0.0
            if getattr(self, "turn_stall_steps", {}).get(rid, 0) >= self.turn_stall_limit:
                mask[slot, 2] = 0.0
        return mask

    def _select_action_index(
        self,
        logits: np.ndarray,
        has_ball_now: bool,
    ) -> Tuple[int, np.ndarray]:
        """Select the highest-probability action while masking invalid ball-state actions."""
        if has_ball_now:
            allowed_indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        else:
            allowed_indices = np.array([0, 1, 4], dtype=np.int64)

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
        self._last_kick_probe_by_id = {}
        self.kick_macro_active = {rid: False for rid in self.robot_ids}
        self.kick_macro_target_y = {rid: None for rid in self.robot_ids}
        self.kick_macro_align_steps = {rid: 0 for rid in self.robot_ids}
        self.kick_macro_retarget_count = {rid: 0 for rid in self.robot_ids}

        # Clear dribble_to phase machine and session tracking state
        for rid in self.robot_ids:
            self.dribble_states[rid].reset()
        self.dribble_session_active = {rid: False for rid in self.robot_ids}
        self.dribble_anchor = {rid: None for rid in self.robot_ids}
        self.steps_since_stop_dribble = {rid: None for rid in self.robot_ids}
        self.dribble_to_target = {rid: None for rid in self.robot_ids}
        self.prev_ball_to_dribble_target_dist = {rid: None for rid in self.robot_ids}
        self._prev_action_was_dribble_to = {rid: False for rid in self.robot_ids}
        self.dribble_carry_start_gap = {rid: None for rid in self.robot_ids}
        self.ball_claimant_id = None
        self.turn_stall_steps = {rid: 0 for rid in self.robot_ids}
        self._turn_stall_last_pose = {rid: None for rid in self.robot_ids}

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
        # Stage 3: defender lane gate. The placement bonus (and dense gates) are
        # additionally scaled by how clear the shot lane is of the non-goalie
        # defender; a kick straight through the defender pays nothing and incurs a
        # one-shot penalty. Independent of the keeper gate (the two multiply).
        use_defender_gate = bool(getattr(self.reward_config, "use_defender_lane_gate", False))
        defender_lane_block_dist = float(
            getattr(self.reward_config, "defender_lane_block_dist", 2.6)
        )
        defender_lane_penalty = float(
            getattr(self.reward_config, "defender_lane_penalty", 0.0)
        )
        defender_lane_min_quality = float(
            getattr(self.reward_config, "defender_lane_min_quality", 0.3)
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
            defender_pose_kick = (
                self._opponent_defender_pose(current_game_state) if use_defender_gate else None
            )
            defender_xy_kick = (
                (float(defender_pose_kick[0]), float(defender_pose_kick[1]))
                if defender_pose_kick is not None else None
            )
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
                aim_quality, predicted_y_at_goal_line = self._kick_aim_quality_from_pose(
                    pose, kick_ball_pos
                )
                # The command-generation path may leave these as None because
                # a requested kick can first execute as an internal alignment
                # turn. Once a kick truly fires, the reward path computes the
                # authoritative shot projection; write it back into the
                # per-step info so infer.py's shot probe records the same
                # values that the reward/log line used.
                info_i["kick_aim_quality"] = aim_quality
                info_i["kick_predicted_y_at_goal_line"] = predicted_y_at_goal_line
                if kick_ball_pos is not None and len(kick_ball_pos) >= 2:
                    fire_x = float(kick_ball_pos[0])
                    fire_y = float(kick_ball_pos[1])
                else:
                    fire_x = rx
                    fire_y = ry
                if goalie_pose is not None:
                    gk_x = float(goalie_pose[0])
                    gk_y_probe = float(goalie_pose[1])
                    fire_dist_to_keeper = math.hypot(fire_x - gk_x, fire_y - gk_y_probe)
                else:
                    gk_x = None
                    gk_y_probe = None
                    fire_dist_to_keeper = None
                fire_dist_to_goal_line = max(0.0, float(FIELD_X[1]) - fire_x)
                info_i["kick_fire_x"] = fire_x
                info_i["kick_fire_y"] = fire_y
                info_i["kick_fire_dist_to_keeper"] = fire_dist_to_keeper
                info_i["kick_fire_dist_to_goal_line"] = fire_dist_to_goal_line
                info_i["kick_keeper_x"] = gk_x
                info_i["kick_keeper_y"] = gk_y_probe
                # Facing away from / parallel to the goal line: kick can
                # never cross x=FIELD_X[1]. Guaranteed miss — apply the
                # bad-aim penalty and move on.
                if predicted_y_at_goal_line is None:
                    info_i["kick_gap_quality"] = None
                    info_i["kick_gap_quality_at_fire"] = None
                    info_i["kick_keeper_zone_factor"] = self._keeper_zone_factor(
                        (fire_x, fire_y), goalie_pose
                    )
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
                # Suppress gap-scaled credit for a shot taken from inside the
                # keeper's catch zone (where gap_quality perversely peaks).
                kz = self._keeper_zone_factor((fire_x, fire_y), goalie_pose)
                # Defender lane clearance for this kick (1.0 = clear / gate off).
                lane_clear = 1.0
                if (
                    use_defender_gate
                    and defender_xy_kick is not None
                    and kick_ball_pos is not None
                    and len(kick_ball_pos) >= 2
                ):
                    lane_clear = lane_clear_quality(
                        (float(kick_ball_pos[0]), float(kick_ball_pos[1])),
                        (FIELD_X[1], predicted_y_at_goal_line),
                        defender_xy_kick,
                        defender_lane_block_dist,
                    )
                if bad_aim:
                    if bad_aim_penalty > 0.0:
                        reward -= _miss_penalty(predicted_y_at_goal_line)
                elif use_goalie_gate and gk_y is not None:
                    # Pay placement: 0 at the keeper's y, 1 at max in-mouth
                    # separation. The blend term eases transfer from the
                    # stage-1 center-aim checkpoint early in the stage.
                    gap_quality = post_safe_goalie_gap_quality(
                        predicted_y_at_goal_line, gk_y, goal_half_height,
                        float(getattr(self.reward_config, "goal_post_safety_margin", 0.0)),
                    )
                    blended_quality = (
                        (1.0 - goalie_gap_blend_center) * gap_quality
                        + goalie_gap_blend_center * aim_quality
                    )
                    reward += blended_quality * kz * lane_clear * kick_aim_weight
                    if (
                        gap_quality < goalie_gap_min_quality
                        and kick_into_keeper_penalty > 0.0
                    ):
                        # On target but into the keeper's cover.
                        reward -= kick_into_keeper_penalty
                else:
                    reward += aim_quality * kz * lane_clear * kick_aim_weight
                # Stash the goalie-gap quality of this kick so the dribble→kick
                # combo bonus below can scale by GAP (corner placement) rather
                # than center-mouth aim — a centered kick after a useless dribble
                # then earns ~0 combo. None when bad-aim or no goalie gate
                # (combo falls back to center-mouth aim in that case).
                info_i["kick_gap_quality"] = gap_quality
                info_i["kick_gap_quality_at_fire"] = gap_quality
                info_i["kick_keeper_zone_factor"] = kz
                info_i["kick_opposite_keeper_side"] = bool(
                    gk_y is not None
                    and predicted_y_at_goal_line * gk_y < 0.0
                )
                probe = {
                    "robot_id": robot_id_i,
                    "step": int(self.current_step),
                    "fire_x": fire_x,
                    "fire_y": fire_y,
                    "fire_dist_to_keeper": fire_dist_to_keeper,
                    "fire_dist_to_goal_line": fire_dist_to_goal_line,
                    "keeper_x": gk_x,
                    "keeper_y": gk_y_probe,
                    "predicted_y_at_goal_line": predicted_y_at_goal_line,
                    "aim_quality": aim_quality,
                    "bad_aim": bad_aim,
                    "gap_quality_at_fire": gap_quality,
                    "keeper_zone_factor": kz,
                    "opposite_keeper_side": info_i["kick_opposite_keeper_side"],
                    "steps_since_dribble": self.steps_since_stop_dribble.get(robot_id_i),
                }
                if robot_id_i is not None:
                    self._last_kick_probe_by_id[int(robot_id_i)] = probe
                info_i["kick_probe"] = dict(probe)
                # Stage 3: penalise an on-target kick fired through the defender.
                if (
                    not bad_aim
                    and use_defender_gate
                    and defender_xy_kick is not None
                    and lane_clear < defender_lane_min_quality
                    and defender_lane_penalty > 0.0
                ):
                    reward -= defender_lane_penalty
                # Diagnostic log: confirms predicted_y vs the actual episode
                # outcome. If most ball_in_penalty_off_target episodes show
                # high aim_quality at fire, the divergence is on the physics
                # side (ball deflection / decay), not the policy's aim.
                self.logger.info(
                    "Kick fired: robot=(%.2f, %.2f, %.1f°)  "
                    "predicted_y=%.2f  aim_quality=%.2f  bad_aim=%s  "
                    "gk_y=%s  gap_quality=%s  fire_dist_to_keeper=%s  "
                    "steps_since_dribble=%s",
                    rx, ry, theta_deg, predicted_y_at_goal_line,
                    aim_quality, bad_aim,
                    f"{gk_y:.2f}" if gk_y is not None else "N/A",
                    f"{gap_quality:.2f}" if gap_quality is not None else "N/A",
                    f"{fire_dist_to_keeper:.2f}" if fire_dist_to_keeper is not None else "N/A",
                    self.steps_since_stop_dribble.get(robot_id_i),
                )

        # Immediate signed target-quality reward. The target parameters exist
        # only on the fresh commitment transition, so reward them there rather
        # than after a potentially long GRAB/catch delay. Evaluate the endpoint
        # reachable in one legal 0.85 m segment, not the distant declared target.
        dribble_quality_weight = float(
            getattr(self.reward_config, "dribble_target_quality_weight", 0.0)
        )
        if dribble_quality_weight > 0.0 and current_game_state is not None:
            gs_ball_pos = getattr(current_game_state, "ball_pos", None)
            dq_defender_xy = (
                self._opponent_defender_pose(current_game_state)
                if use_defender_gate else None
            )
            dq_defender_pt = (
                (float(dq_defender_xy[0]), float(dq_defender_xy[1]))
                if dq_defender_xy is not None else None
            )
            for info_i in action_info.get("per_robot", []):
                if not info_i.get("dribble_committed"):
                    continue
                rid = info_i.get("robot_id")
                target = self.dribble_to_target.get(rid)
                gk_pose = self._opponent_goalie_pose(current_game_state)
                if target is None or gs_ball_pos is None or gk_pose is None:
                    self.logger.info(
                        "Dribble target committed: rid=%s gap=N/A "
                        "(target/ball/goalie missing)", rid,
                    )
                    continue
                gk_y_val = float(gk_pose[1])
                if dq_defender_pt is not None:
                    current_gap = positional_shot_quality(
                        (float(gs_ball_pos[0]), float(gs_ball_pos[1])),
                        gk_y_val, dq_defender_pt,
                        goal_half_height, defender_lane_block_dist,
                    )
                else:
                    current_gap = positional_gap_quality(
                        (float(gs_ball_pos[0]), float(gs_ball_pos[1])),
                        gk_y_val, goal_half_height,
                    )
                _keeper_delta, reachable_target = reachable_gap_delta(
                    (float(gs_ball_pos[0]), float(gs_ball_pos[1])),
                    target,
                    gk_y_val,
                    goal_half_height,
                    segment_limit=0.85,
                )
                if dq_defender_pt is not None:
                    target_gap = positional_shot_quality(
                        reachable_target, gk_y_val, dq_defender_pt,
                        goal_half_height, defender_lane_block_dist,
                    )
                else:
                    target_gap = positional_gap_quality(
                        reachable_target, gk_y_val, goal_half_height,
                    )
                raw_delta = target_gap - current_gap
                kz_tq = self._keeper_zone_factor(reachable_target, gk_pose)
                self.logger.info(
                    "Dribble target committed: rid=%s current_gap=%.3f "
                    "reachable_gap=%.3f gap_delta=%+.3f useful=%s endpoint=(%.2f,%.2f) "
                    "keeper_zone_factor=%.2f",
                    rid, current_gap, target_gap, raw_delta,
                    bool(raw_delta > 0.0),
                    reachable_target[0], reachable_target[1], kz_tq,
                )
                reward += (
                    float(np.clip(raw_delta, -1.0, 1.0))
                    * dribble_quality_weight
                    * kz_tq
                )

        # Achieved-gap reward: credit the REALIZED shooting-gap change of a
        # carry — positional_gap_quality at the ball's spot when the segment
        # CLOSES minus when it OPENED — rather than the latched hypothetical
        # target (which dribble_target_quality_weight above pays at open). A
        # carry that merely orbits the ball in place ends near where it started
        # and nets ~0 here even if its target looked good; a carry that worsens
        # the angle is penalized. The start gap is recorded on carry_started and
        # the realized delta is paid when the segment closes (stop_dribble_fired
        # = release at the segment limit, or a mid-dribble abandon).
        achieved_gap_weight = float(
            getattr(self.reward_config, "dribble_achieved_gap_weight", 0.0)
        )
        if achieved_gap_weight > 0.0 and current_game_state is not None:
            ag_ball_pos = getattr(current_game_state, "ball_pos", None)
            ag_gk_pose = self._opponent_goalie_pose(current_game_state)
            ag_defender_xy = (
                self._opponent_defender_pose(current_game_state)
                if use_defender_gate else None
            )
            ag_defender_pt = (
                (float(ag_defender_xy[0]), float(ag_defender_xy[1]))
                if ag_defender_xy is not None else None
            )
            for info_i in action_info.get("per_robot", []):
                rid = info_i.get("robot_id")
                if rid is None or ag_ball_pos is None or ag_gk_pose is None:
                    continue
                if ag_defender_pt is not None:
                    ball_gap_now = positional_shot_quality(
                        (float(ag_ball_pos[0]), float(ag_ball_pos[1])),
                        float(ag_gk_pose[1]), ag_defender_pt,
                        goal_half_height, defender_lane_block_dist,
                    )
                else:
                    ball_gap_now = positional_gap_quality(
                        (float(ag_ball_pos[0]), float(ag_ball_pos[1])),
                        float(ag_gk_pose[1]), goal_half_height,
                    )
                if info_i.get("carry_started"):
                    self.dribble_carry_start_gap[rid] = ball_gap_now
                    self.logger.info(
                        "Dribble carry opened: rid=%s current_gap=%.3f",
                        rid, ball_gap_now,
                    )
                if info_i.get("stop_dribble_fired"):
                    start_gap = self.dribble_carry_start_gap.get(rid)
                    if start_gap is not None:
                        achieved_delta = ball_gap_now - start_gap
                        kz_ag = self._keeper_zone_factor(
                            (float(ag_ball_pos[0]), float(ag_ball_pos[1])),
                            ag_gk_pose,
                        )
                        ag_bonus = achieved_delta * achieved_gap_weight * kz_ag
                        reward += ag_bonus
                        self.logger.info(
                            "Dribble achieved-gap: rid=%s start=%.3f end=%.3f "
                            "delta=%+.3f keeper_zone_factor=%.2f reward=%+.2f",
                            rid, start_gap, ball_gap_now, achieved_delta,
                            kz_ag, ag_bonus,
                        )
                    self.dribble_carry_start_gap[rid] = None

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

        # Dribble→kick combo bonus. Rewards the dribble release → kick chain
        # as a unit. Update the per-robot release age counter first,
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
        combo_quality_scale = bool(
            getattr(self.reward_config, "post_dribble_kick_quality_scale", False)
        )
        if combo_bonus > 0.0:
            for info_i in per_robot_info_list:
                rid = info_i.get("robot_id")
                if not info_i.get("kick_fired"):
                    continue
                age = self.steps_since_stop_dribble.get(rid)
                if age is not None and age <= combo_window:
                    scale = 1.0
                    if combo_quality_scale:
                        # Prefer GOALIE-GAP quality (corner placement) so the
                        # combo only pays when the dribble→kick chain produces a
                        # shot that beats the keeper — a centered kick after a
                        # useless dribble scores gap≈0 → ~0 combo. Fall back to
                        # center-mouth aim only when no gap was computed (no
                        # goalie gate); bad-aim kicks yield 0 either way.
                        gap_q = info_i.get("kick_gap_quality")
                        if gap_q is not None:
                            scale = max(0.0, float(gap_q))
                        else:
                            aim_q = info_i.get("kick_aim_quality")
                            scale = max(0.0, float(aim_q)) if aim_q is not None else 0.0
                    # Suppress the combo inside the keeper zone too (same factor
                    # applied to the kick_aim bonus for this kick).
                    scale *= float(info_i.get("kick_keeper_zone_factor", 1.0))
                    bonus = combo_bonus * scale
                    reward += bonus
                    self.logger.info(
                        "Dribble→kick combo bonus +%.1f (steps_since_stop=%d, scale=%.2f)",
                        bonus, age, scale,
                    )

        terminated, term_reason = self._check_terminal(next_game_state, prev_game_state=current_game_state)
        # Off-target terminal penalty disabled: a -3 penalty here flipped the
        # net-EV of "kick" to slightly negative once the policy was at all
        # uncertain about aim, which collapsed the policy into "turn 100%"
        # action distribution (TD3 is deterministic — once Q(turn)=0 stably
        # exceeds noisy Q(kick), kicking stops entirely). The off-target
        # exploit it was meant to address is better fixed at the shaping
        # level (aim-gate goal_progress) than at the terminal level.
        if terminated and term_reason == "ball_in_penalty_off_target":
            penalty = float(getattr(
                self.reward_config, "ball_in_penalty_off_target_penalty", 0.0
            ))
            reward -= penalty
        if terminated and term_reason == "goalie_catch":
            goal_reward = float(getattr(self.reward_config, "goal_reward", 70.0))
            reward -= goal_reward
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
        self.total_rewards += reward

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
            if self._last_kick_probe_by_id:
                for probe in self._last_kick_probe_by_id.values():
                    self.logger.info(
                        "Shot probe: outcome=%s fire=(%.2f, %.2f) "
                        "dist_to_keeper=%s dist_to_goal=%.2f gap_quality=%s "
                        "aim_quality=%.2f keeper_zone_factor=%.2f",
                        end_reason,
                        probe["fire_x"], probe["fire_y"],
                        f"{probe['fire_dist_to_keeper']:.2f}"
                        if probe.get("fire_dist_to_keeper") is not None else "N/A",
                        probe["fire_dist_to_goal_line"],
                        f"{probe['gap_quality_at_fire']:.2f}"
                        if probe.get("gap_quality_at_fire") is not None else "N/A",
                        probe["aim_quality"],
                        probe["keeper_zone_factor"],
                    )
            if terminated:
                self._cached_game_state = None

        # Update dribble_to reward tracking for next step.
        next_ball_pos = getattr(next_game_state, "ball_pos", None) if next_game_state is not None else None
        for rid in self.robot_ids:
            target = self.dribble_to_target.get(rid)
            if target is not None and next_ball_pos is not None:
                self.prev_ball_to_dribble_target_dist[rid] = float(math.hypot(
                    float(next_ball_pos[0]) - target[0],
                    float(next_ball_pos[1]) - target[1],
                ))
            else:
                self.prev_ball_to_dribble_target_dist[rid] = None
        for info_i in action_info.get("per_robot", []):
            rid = info_i.get("robot_id")
            if rid is not None:
                self._prev_action_was_dribble_to[rid] = (info_i.get("action_type") == "dribble_to")

        info = {
            "action_info": action_info,
            "reward": reward,
            "total_reward": self.total_rewards,
            "invalid_action_count": invalid_action_count,
            "termination_reason": term_reason if terminated else ("max_steps" if truncated else ""),
            "last_kick_probe": dict(next(iter(self._last_kick_probe_by_id.values()), {}))
            if (terminated or truncated) else {},
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
            # Layout: slot 0 = our goalie, slot 1 = opp goalie.
            ctx_base = self.global_dim + self.per_agent_dim * self.a_max

            # Context slot 0: own HC goalie (Stage 4+).
            if self.own_goalie_robot_id is not None:
                own_gk_pose = pose_by_robot_id.get(self.own_goalie_robot_id)
                if own_gk_pose is not None and len(own_gk_pose) >= 3:
                    gk_x = float(own_gk_pose[0])
                    gk_y = float(own_gk_pose[1])
                    gk_theta = float(np.deg2rad(own_gk_pose[2]))
                    prev_gk_xy = self.prev_opp_pose_by_id.get(-self.own_goalie_robot_id)
                    if prev_gk_xy is not None:
                        gk_vx = float(gk_x - prev_gk_xy[0])
                        gk_vy = float(gk_y - prev_gk_xy[1])
                    else:
                        gk_vx = gk_vy = 0.0
                    self.prev_opp_pose_by_id[-self.own_goalie_robot_id] = np.array([gk_x, gk_y], dtype=np.float32)

                    own_gk_off = ctx_base + 0 * self.d_ctx  # slot 0
                    obs[own_gk_off + 0] = gk_x / self._NORM_POS_X
                    obs[own_gk_off + 1] = gk_y / self._NORM_POS_Y
                    obs[own_gk_off + 2] = gk_theta / self._NORM_THETA
                    obs[own_gk_off + 3] = gk_vx / self._NORM_VEL
                    obs[own_gk_off + 4] = gk_vy / self._NORM_VEL

            # Context slots 1..num_opponents: opponent robots, ordered by robot_id
            # (slot 1 = goalie robot_id=1 by convention, slot 2 = defender id=2, …).
            # Filling additional opponents lets the attacker actually observe the
            # defender it must dribble around (stage 3). Only fill when an opponent
            # team is configured.
            if self.opponent_team_name is not None:
                opp_pose_entries = game_state.robot_poses.get(self.opponent_team_name, [])
                opp_pose_by_id: Dict[int, Any] = {}
                for entry in opp_pose_entries:
                    if isinstance(entry, dict):
                        opp_pose_by_id.update(entry)

                sorted_opp_ids = sorted(opp_pose_by_id.keys())
                for k, opp_id in enumerate(sorted_opp_ids[: self.num_opponents]):
                    slot = 1 + k
                    if slot >= self.c_max:
                        break
                    opp_pose = opp_pose_by_id[opp_id]
                    opp_x = float(opp_pose[0])
                    opp_y = float(opp_pose[1])
                    opp_theta = float(np.deg2rad(opp_pose[2]))
                    prev_opp_xy = self.prev_opp_pose_by_id.get(opp_id)
                    if prev_opp_xy is not None:
                        opp_vx = float(opp_x - prev_opp_xy[0])
                        opp_vy = float(opp_y - prev_opp_xy[1])
                    else:
                        opp_vx = opp_vy = 0.0
                    self.prev_opp_pose_by_id[opp_id] = np.array([opp_x, opp_y], dtype=np.float32)

                    opp_slot_off = ctx_base + slot * self.d_ctx
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

        1. Legacy TD3 flat Box vector, shape (8 * num_robots,):
            [0..4] logits over {goto, approach_ball, turn, kick, dribble_to}
            [5] goto_x_raw ∈ [-1, 1]
            [6] goto_y_raw ∈ [-1, 1]
            [7] turn_theta_raw ∈ [-1, 1]

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
        action_types = ["goto", "approach_ball", "turn", "kick", "dribble_to"]

        # Detect format and extract per-robot (action_type, goto_x, goto_y, turn_theta).
        per_robot_decoded: List[Dict[str, Any]] = []
        is_ppo_dict = isinstance(action, dict) and "primitive_idx" in action
        runtime_mask_applied = bool(
            action.get("runtime_mask_applied", False)
        ) if is_ppo_dict else False

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
                    "goto_x_raw": float(action_arr[base + 5]),
                    "goto_y_raw": float(action_arr[base + 6]),
                    "turn_theta_raw": float(action_arr[base + 7]),
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
        claimant_id = self._ball_claimant(game_state, pose_by_robot_id)

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
            fallback_reason: Optional[str] = None
            robot_ball_dist: Optional[float] = None
            if pose is not None and ball_pos is not None and len(ball_pos) >= 2:
                robot_ball_dist = float(np.hypot(
                    float(pose[0]) - float(ball_pos[0]),
                    float(pose[1]) - float(ball_pos[1]),
                ))
            kick_aim_quality: Optional[float] = None
            kick_predicted_y_at_goal_line: Optional[float] = None
            kick_macro_continuation = False
            kick_target_y: Optional[float] = self.kick_macro_target_y.get(robot_id)
            kick_target_angle: Optional[float] = None
            kick_target_heading_error: Optional[float] = None
            kick_retarget_applied = False
            kick_retarget_quality_before: Optional[float] = None

            # ---- dribble_to session tracking (derived from DribbleState) ----
            dribble_st = self.dribble_states.get(robot_id)
            if dribble_st is None:
                dribble_st = DribbleState()
                self.dribble_states[robot_id] = dribble_st
            prev_dribble_phase = dribble_st.phase
            anchor = self.dribble_anchor.get(robot_id)
            anchor_dist = None
            if pose is not None and anchor is not None:
                anchor_dist = float(np.hypot(
                    float(pose[0]) - anchor[0], float(pose[1]) - anchor[1]
                ))
            stop_dribble_fired = False
            stop_dribble_at_limit = False
            carry_started = False
            dribble_committed = False
            reported_dribble_phase = dribble_st.phase
            reported_catch_attempts = dribble_st.catch_attempts
            reported_verify_steps = dribble_st.verify_steps
            reported_align_steps = dribble_st.align_steps
            reported_verify_robot_moved = dribble_st.last_verify_robot_moved
            reported_verify_ball_moved = dribble_st.last_verify_ball_moved
            reported_verify_offset_change = dribble_st.last_verify_offset_change
            reported_target_heading_error = dribble_st.last_target_heading_error
            reported_dribble_target = dribble_st.target

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
                        self._kick_aim_quality_from_pose(pose, ball_pos)
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

            # Runtime safety net for stale/unmasked actions. The PPO path normally
            # prevents these selections with get_primitive_valid_mask(); these
            # guards cover the state change between sampling and execution and
            # legacy callers that do not yet pass the mask.
            recovery_enabled = bool(getattr(self, "ball_action_recovery", False))
            is_claimant = claimant_id is not None and robot_id == claimant_id
            recovery_invalid_requested = False
            if bool(self.kick_macro_active.get(robot_id, False)):
                claimant_conflict = claimant_id is not None and robot_id != claimant_id
                if claimant_conflict or not can_kick or pose is None or ball_pos is None:
                    self._reset_kick_macro(robot_id)
                    kick_target_y = None
                    fallback_reason = "kick_macro_aborted_lost_ball"
                else:
                    if action_type != "kick":
                        kick_macro_continuation = True
                        action_type = "kick"
                        action_idx = 3
                        executed_action_type = "kick"
                        fallback_reason = "kick_macro_continuation"
            if recovery_enabled and not is_claimant and action_type in (
                "approach_ball", "kick", "dribble_to"
            ):
                # A non-claimant must never be converted into another ball chaser.
                action_type = "turn"
                turn_theta = 0.0
                executed_action_type = "hold"
                fallback_reason = "non_claimant_ball_action_blocked"
                recovery_invalid_requested = True
            elif (
                recovery_enabled
                and is_claimant
                and action_type == "dribble_to"
                and robot_ball_dist is not None
                and robot_ball_dist > self.kickable_dist
            ):
                action_type = "approach_ball"
                executed_action_type = "approach_ball"
                fallback_reason = "dribble_out_of_range_to_approach"
                recovery_invalid_requested = True
            elif (
                recovery_enabled
                and is_claimant
                and action_type == "approach_ball"
                and has_ball_now
            ):
                action_type = "turn"
                executed_action_type = "turn"
                fallback_reason = "approach_complete_to_turn"
                recovery_invalid_requested = True

            # A claimant repeatedly selecting turn while stationary and still far
            # from the ball is the deterministic fixed point seen in Stage-2g
            # inference. Track actual displacement, then force one approach step
            # after a bounded run. Non-claimants are deliberately excluded: they
            # may need to hold/turn while maintaining team shape.
            turn_stall_steps = getattr(self, "turn_stall_steps", {})
            last_pose_by_id = getattr(self, "_turn_stall_last_pose", {})
            previous_xy = last_pose_by_id.get(robot_id)
            current_xy = (
                (float(pose[0]), float(pose[1])) if pose is not None else None
            )
            if (
                recovery_enabled
                and is_claimant
                and action_type == "turn"
                and raw_action_type == "turn"
                and robot_ball_dist is not None
                and robot_ball_dist > self.kickable_dist
            ):
                moved = (
                    float(np.hypot(current_xy[0] - previous_xy[0], current_xy[1] - previous_xy[1]))
                    if current_xy is not None and previous_xy is not None
                    else float("inf")
                )
                if moved <= float(getattr(self, "turn_stall_displacement", 0.05)):
                    turn_stall_steps[robot_id] = turn_stall_steps.get(robot_id, 0) + 1
                else:
                    turn_stall_steps[robot_id] = 0
                if (
                    turn_stall_steps[robot_id] >= int(getattr(self, "turn_stall_limit", 12))
                    and not runtime_mask_applied
                ):
                    action_type = "approach_ball"
                    executed_action_type = "approach_ball"
                    fallback_reason = "turn_stall_to_approach"
                    recovery_invalid_requested = True
                    turn_stall_steps[robot_id] = 0
            else:
                turn_stall_steps[robot_id] = 0
            last_pose_by_id[robot_id] = current_xy
            self.turn_stall_steps = turn_stall_steps
            self._turn_stall_last_pose = last_pose_by_id

            if action_type == "kick" and can_kick and self.kick_requires_aim and pose is not None:
                if kick_aim_quality is None:
                    kick_aim_quality, kick_predicted_y_at_goal_line = self._kick_aim_quality_from_pose(
                        pose, ball_pos
                    )
                if kick_aim_quality < self.kick_min_aim_quality:
                    kick_blocked_bad_aim = True

            invalid_action_requested = (
                (action_type == "kick" and not can_kick)
                or recovery_invalid_requested
            )
            if invalid_action_requested:
                invalid_action_count += 1

            # Dribble targets get an optional |y| clamp so the policy can only
            # aim near the goal mouth (where a carry actually opens the shooting
            # angle), not anywhere in +/-field_half_height. 0.0 = no clamp.
            # Applied to BOTH the reward-tracking target and the target handed
            # to dribble_to() below so they stay consistent.
            dribble_target_y_clip = float(
                getattr(self.reward_config, "dribble_target_y_clip", 0.0)
            )
            # Goal-RELATIVE dribble target (when dribble_fwd_max>0): carry forward
            # along the ball->goal vector so the head's neutral output (~0,0) is a
            # forward carry toward goal centre — already gap-improving — instead of
            # the absolute decode that put the neutral target at midfield (0,0)
            # (dx=45, gap~0.07 << current ~0.32), making every carry a backward dead
            # target the head never escaped (§34/§35 + 6/20 lag runs: 0% useful).
            # Falls back to the legacy absolute (goto_x, goto_y) decode when the
            # knob is 0 or the ball pose is unknown.
            dribble_fwd_max = float(getattr(self.reward_config, "dribble_fwd_max", 0.0))
            dribble_lat_max = float(getattr(self.reward_config, "dribble_lat_max", 0.0))
            if dribble_fwd_max > 0.0 and ball_pos is not None and len(ball_pos) >= 2:
                bx_t, by_t = float(ball_pos[0]), float(ball_pos[1])
                gcx, gcy = float(FIELD_X[1]), 0.0
                norm = float(np.hypot(gcx - bx_t, gcy - by_t)) or 1.0
                dirx, diry = (gcx - bx_t) / norm, (gcy - by_t) / norm
                perpx, perpy = -diry, dirx
                fwd = (goto_x_raw * 0.5 + 0.5) * dribble_fwd_max  # [-1,1] -> [0,max]
                lat = goto_y_raw * dribble_lat_max                # steer to open side
                dribble_goto_x = float(min(bx_t + dirx * fwd + perpx * lat, gcx - 1.0))
                dribble_goto_y = float(by_t + diry * fwd + perpy * lat)
            else:
                dribble_goto_x = goto_x
                dribble_goto_y = goto_y
            if dribble_target_y_clip > 0.0:
                dribble_goto_y = float(
                    np.clip(dribble_goto_y, -dribble_target_y_clip, dribble_target_y_clip)
                )
            dribble_penalty_area_y_clip = float(
                getattr(self.reward_config, "dribble_penalty_area_y_clip", 0.0)
            )
            dribble_penalty_area_guard_margin = float(
                getattr(self.reward_config, "dribble_penalty_area_guard_margin", 0.0)
            )
            dribble_penalty_guard_applied = False
            if (
                dribble_penalty_area_y_clip > 0.0
                and dribble_penalty_area_guard_margin > 0.0
            ):
                boundary_x = (
                    self._RIGHT_PENALTY_AREA_X - dribble_penalty_area_guard_margin
                )
                by_now = (
                    float(ball_pos[1])
                    if (ball_pos is not None and len(ball_pos) >= 2)
                    else 0.0
                )
                # Penalty-area ENTRY guard. Clamping only the target y is not
                # enough: a ball that is still WIDE of the goal mouth, carried
                # toward a y-clamped target inside the box, crosses the penalty
                # line (x=_RIGHT_PENALTY_AREA_X) at a wide y MID-segment and
                # terminates as ball_in_penalty_off_target before the carry can
                # pull it central (confirmed: all off-target episodes at +/-15
                # were dribble carries, no fired shot). So while the ball is wide
                # (|y| > goal mouth half), cap the target x to the boundary so the
                # carry pulls the ball toward centre OUTSIDE the box; entry past
                # the boundary resumes only once |y| is within the mouth.
                if (
                    abs(by_now) > self._GOAL_HALF_HEIGHT
                    and dribble_goto_x > boundary_x
                ):
                    dribble_goto_x = boundary_x
                    dribble_penalty_guard_applied = True
                if dribble_goto_x >= boundary_x:
                    before_y = dribble_goto_y
                    dribble_goto_y = float(
                        np.clip(
                            dribble_goto_y,
                            -dribble_penalty_area_y_clip,
                            dribble_penalty_area_y_clip,
                        )
                    )
                    if not math.isclose(before_y, dribble_goto_y, abs_tol=1e-6):
                        dribble_penalty_guard_applied = True

            # dribble_to is a MACRO: the moment the policy COMMITS to a dribble
            # (selects dribble_to while the ball is in possession range, opening the
            # GRAB phase), the env keeps driving the phase machine toward the LATCHED
            # target every step — through GRAB (align + catch), SETTLE, VERIFY
            # (probe), CARRY (directional dash), ALIGN_RELEASE, and RELEASE — until the segment completes
            # (arrival / segment limit
            # / stall) or the ball is genuinely lost, regardless of the policy's
            # per-step primitive. Only a kick may interrupt (release + shoot).
            #
            # GRAB must be inside the macro, not just CARRY/RELEASE. Opening a carry
            # needs a few uninterrupted turn-to-align steps before the catch, but the
            # stochastic policy oscillates approach_ball/turn/dribble_to at the ball,
            # and each non-dribble step's own turn knocks the heading off target, so
            # the grab almost never converged (infer trace: `catch` fired 1× in 3000
            # steps; the robot stood on the ball spinning for 100+ steps). The
            # `committed` flag (set on a real dribble_to pick, cleared on DONE/kick
            # via DribbleState.reset()) gates this so the macro only takes over AFTER
            # a genuine pick — the default GRAB phase before any pick does NOT hijack
            # ordinary approach steps.
            carry_continuation = (
                dribble_st.committed
                and dribble_st.phase in (
                    DRIBBLE_PHASE_GRAB, DRIBBLE_PHASE_SETTLE, DRIBBLE_PHASE_VERIFY,
                    DRIBBLE_PHASE_CARRY, DRIBBLE_PHASE_ALIGN_RELEASE,
                    DRIBBLE_PHASE_RELEASE
                )
                and action_type != "kick"
            )

            # Track dribble_to target for reward computation. During a carry
            # continuation the policy's per-step (Dx, Dy) is ignored downstream
            # (dribble_to() uses the latched target), so the noisy decode here is
            # overwritten with state.target inside the macro branch below.
            if action_type == "dribble_to" or carry_continuation:
                self.dribble_to_target[robot_id] = (dribble_goto_x, dribble_goto_y)
            else:
                self.dribble_to_target[robot_id] = None
                self.prev_ball_to_dribble_target_dist[robot_id] = None

            if action_type == "kick":
                # Geometric committed turn-to-align then fire, via
                # basic_commands.kick(). A fresh kick request latches a
                # keeper-away target inside the goal mouth; subsequent policy
                # steps cannot interrupt the slow capped-turn alignment until
                # the shot fires, possession is lost, or the macro times out.
                if pose is None or ball_pos is None:
                    command = "turn 0"
                    executed_action_type = "turn"
                    self._reset_kick_macro(robot_id)
                    kick_target_y = None
                else:
                    self_pose = np.array([
                        float(pose[0]),
                        float(pose[1]),
                        float(np.deg2rad(pose[2])),
                    ], dtype=np.float32)
                    ball_xy = np.array(
                        [float(ball_pos[0]), float(ball_pos[1])], dtype=np.float32
                    )
                    if not bool(self.kick_macro_active.get(robot_id, False)):
                        self.kick_macro_active[robot_id] = True
                        self.kick_macro_align_steps[robot_id] = 0
                        self.kick_macro_retarget_count[robot_id] = 0
                        self.kick_macro_target_y[robot_id] = self._keeper_away_kick_target_y(
                            game_state, ball_xy
                        )
                    kick_target_y = self.kick_macro_target_y.get(robot_id)
                    if kick_target_y is None:
                        kick_target_y = self._keeper_away_kick_target_y(game_state, ball_xy)
                        self.kick_macro_target_y[robot_id] = kick_target_y
                    kick_target_y, kick_retarget_applied, kick_retarget_quality_before = (
                        self._maybe_retarget_kick_target_y(
                            robot_id, game_state, float(kick_target_y)
                        )
                    )
                    kick_target_angle = float(
                        np.arctan2(
                            float(kick_target_y) - float(ball_xy[1]),
                            float(FIELD_X[1]) - float(ball_xy[0]),
                        )
                    )
                    kick_target_heading_error = float(
                        math.atan2(
                            math.sin(kick_target_angle - float(self_pose[2])),
                            math.cos(kick_target_angle - float(self_pose[2])),
                        )
                    )
                    command = kick(
                        self_pose, ball_xy, kick_target_angle,
                        kick_power=100.0, dribbling=True,
                    )
                    fire_ok = can_kick and not kick_blocked_bad_aim
                    if command.startswith("kick") and fire_ok:
                        kick_fired = True
                        # A fired kick releases the ball; reset dribble phase machine.
                        dribble_st.reset()
                        self.dribble_session_active[robot_id] = False
                        self.dribble_anchor[robot_id] = None
                        self._reset_kick_macro(robot_id)
                    else:
                        # Either still aligning (kick() returned a turn), or the
                        # aim/range gate vetoed the shot this step. Execute as a turn
                        # so the kick-aim one-shot only ever credits a real,
                        # well-aimed, in-range kick. A gate-vetoed "kick" holds
                        # heading (turn 0) rather than firing a bad shot.
                        if command.startswith("kick"):
                            command = "turn 0"
                            self._reset_kick_macro(robot_id)
                        executed_action_type = "turn"
                        if bool(self.kick_macro_active.get(robot_id, False)):
                            self.kick_macro_align_steps[robot_id] = (
                                self.kick_macro_align_steps.get(robot_id, 0) + 1
                            )
                            if (
                                self.kick_macro_align_steps[robot_id]
                                > self.kick_macro_max_align_steps
                            ):
                                command = "turn 0"
                                executed_action_type = "turn"
                                fallback_reason = "kick_macro_align_timeout"
                                self._reset_kick_macro(robot_id)
            elif action_type == "dribble_to" or carry_continuation:
                # A continuation (the policy picked a non-kick, non-dribble action
                # while a carry was open) is executed AS a dribble step and credited
                # as such — the macro is driving, not the sampled primitive.
                if carry_continuation and action_type != "dribble_to":
                    executed_action_type = "dribble_to"
                # A fresh dribble_to pick commits the robot to the grab: from here
                # the macro drives GRAB -> SETTLE -> VERIFY -> CARRY ->
                # ALIGN_RELEASE -> RELEASE
                # across steps even if the policy samples other primitives. Cleared
                # on DONE / kick via DribbleState.reset(). If the robot is not yet in
                # range, dribble_to() returns "done" -> phase DONE -> the session
                # ends below and the flag is reset same-step, so an out-of-range pick
                # never strands the flag.
                if action_type == "dribble_to":
                    dribble_committed = not dribble_st.committed
                    dribble_st.committed = True
                if pose is None or game_state is None or ball_pos is None:
                    command = "turn 0"
                else:
                    self_pose = np.array([
                        float(pose[0]),
                        float(pose[1]),
                        float(np.deg2rad(pose[2])),
                    ], dtype=np.float32)
                    ball_xy = np.array([float(ball_pos[0]), float(ball_pos[1])], dtype=np.float32)
                    target_xy = np.array([dribble_goto_x, dribble_goto_y], dtype=np.float32)
                    command = dribble_to(
                        self_pose=self_pose,
                        ball_pose=ball_xy,
                        target=target_xy,
                        game_state=game_state,
                        state=dribble_st,
                    )
                    # Snapshot diagnostics before DONE resets DribbleState below.
                    reported_dribble_phase = dribble_st.phase
                    reported_catch_attempts = dribble_st.catch_attempts
                    reported_verify_steps = dribble_st.verify_steps
                    reported_align_steps = dribble_st.align_steps
                    reported_verify_robot_moved = dribble_st.last_verify_robot_moved
                    reported_verify_ball_moved = dribble_st.last_verify_ball_moved
                    reported_verify_offset_change = dribble_st.last_verify_offset_change
                    reported_target_heading_error = dribble_st.last_target_heading_error
                    reported_dribble_target = dribble_st.target
                    if command == "done":
                        command = "turn 0"
                    # Update session tracking from DribbleState phase transitions.
                    # A carry is active only after VERIFY proves that catch glue
                    # transported the ball. GRAB/VERIFY are macro acquisition,
                    # not possession and must not open reward accounting.
                    carrying_phases = (DRIBBLE_PHASE_CARRY, DRIBBLE_PHASE_ALIGN_RELEASE)
                    is_carrying = dribble_st.phase in carrying_phases
                    was_carrying = prev_dribble_phase in carrying_phases
                    # A carry segment just opened (catch verified this step):
                    # gates target-quality and achieved-gap accounting on actual
                    # ball ownership rather than merely issuing `catch 0`.
                    carry_started = (
                        prev_dribble_phase != DRIBBLE_PHASE_CARRY
                        and dribble_st.phase == DRIBBLE_PHASE_CARRY
                    )
                    self.dribble_session_active[robot_id] = is_carrying
                    if dribble_st.segment_start is not None and self.dribble_anchor.get(robot_id) is None:
                        self.dribble_anchor[robot_id] = dribble_st.segment_start
                    # A carry remains active through the bounded ALIGN_RELEASE
                    # shooting-pose correction and closes only on RELEASE/DONE.
                    # This pays achieved-gap from the final aligned ball position.
                    carry_closed = (
                        prev_dribble_phase in carrying_phases
                        and dribble_st.phase in (DRIBBLE_PHASE_RELEASE, DRIBBLE_PHASE_DONE)
                    )
                    if carry_closed:
                        stop_dribble_fired = True
                        stop_dribble_at_limit = (
                            dribble_st.phase == DRIBBLE_PHASE_RELEASE
                            and dribble_st.release_at_limit
                        )
                        self.dribble_anchor[robot_id] = None
                    if not is_carrying and not was_carrying:
                        self.dribble_anchor[robot_id] = None
                    # Keep the reward's target aligned with the latched carry
                    # target, not the policy's noisy per-step (Dx, Dy).
                    if dribble_st.target is not None:
                        self.dribble_to_target[robot_id] = dribble_st.target
                    # Transport completed or aborted (arrival / stall): reset so
                    # the next dribble_to selection latches a fresh target.
                    if dribble_st.phase == DRIBBLE_PHASE_DONE:
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

            # Keep the command reported in debug traces identical to what the
            # simulator/robot receives. Serializer applies the same limiter as
            # a final safety boundary for commands from every other controller.
            command = limit_turn_rate(command)
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
                    "ball_claimant_id": claimant_id,
                    "is_ball_claimant": is_claimant,
                    "robot_ball_dist": robot_ball_dist,
                    "fallback_reason": fallback_reason,
                    "turn_stall_steps": self.turn_stall_steps.get(robot_id, 0),
                    "probs": decoded["probs"],
                    "logits": decoded["logits"],
                    "goto_x": goto_x,
                    "goto_y": goto_y,
                    "turn_theta": turn_theta,
                    "has_ball_now": has_ball_now,
                    "kick_fired": kick_fired,
                    "kick_macro_active": self.kick_macro_active.get(robot_id, False),
                    "kick_macro_continuation": kick_macro_continuation,
                    "kick_macro_align_steps": self.kick_macro_align_steps.get(robot_id, 0),
                    "kick_target_y": kick_target_y,
                    "kick_target_angle": kick_target_angle,
                    "kick_target_heading_error": kick_target_heading_error,
                    "kick_retarget_applied": kick_retarget_applied,
                    "kick_retarget_count": self.kick_macro_retarget_count.get(robot_id, 0),
                    "kick_retarget_quality_before": kick_retarget_quality_before,
                    "kick_blocked_bad_aim": kick_blocked_bad_aim,
                    "kick_aim_quality": kick_aim_quality,
                    "kick_predicted_y_at_goal_line": kick_predicted_y_at_goal_line,
                    "invalid_action_requested": invalid_action_requested,
                    "dribble_session_active": self.dribble_session_active.get(robot_id, False),
                    "dribble_anchor_dist": anchor_dist,
                    "dribble_phase": reported_dribble_phase,
                    "dribble_sim_count": getattr(game_state, "count", None),
                    "dribble_catch_attempts": reported_catch_attempts,
                    "dribble_verify_steps": reported_verify_steps,
                    "dribble_align_steps": reported_align_steps,
                    "dribble_verify_robot_moved": reported_verify_robot_moved,
                    "dribble_verify_ball_moved": reported_verify_ball_moved,
                    "dribble_verify_offset_change": reported_verify_offset_change,
                    "dribble_target": reported_dribble_target,
                    "dribble_target_heading_error": reported_target_heading_error,
                    "dribble_penalty_guard_applied": dribble_penalty_guard_applied,
                    "carry_started": carry_started,
                    "dribble_committed": dribble_committed,
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
        # Stage 4+
        is_nearest_to_ball: bool = True,
        ally_positions: Tuple[Tuple[float, float], ...] = (),
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
        # Stage 3: non-goalie defender position for lane-clearance gating.
        defender_pose = (
            self._opponent_defender_pose(current_game_state)
            if self.reward_config.use_defender_lane_gate else None
        )
        anchor = self.dribble_anchor.get(robot_id)
        dribble_anchor_dist = None
        if anchor is not None:
            dribble_anchor_dist = float(np.hypot(robot_x - anchor[0], robot_y - anchor[1]))

        # Stage 4: own goalie coordination
        own_goalie_pose = self._own_goalie_pose(current_game_state)
        own_goalie_pos: Optional[Tuple[float, float]] = None
        own_goalie_has_ball = False
        if own_goalie_pose is not None:
            own_goalie_pos = (own_goalie_pose[0], own_goalie_pose[1])
            gk_ball_dist = float(math.hypot(
                ball_x - own_goalie_pose[0], ball_y - own_goalie_pose[1]
            ))
            own_goalie_has_ball = bool(gk_ball_dist < self.reward_config.own_goalie_clearance_dist)

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
            defender_pos=(
                (float(defender_pose[0]), float(defender_pose[1]))
                if defender_pose is not None else None
            ),
            is_dribbling=bool(self.dribble_session_active.get(robot_id, False)),
            dribble_anchor_dist=dribble_anchor_dist,
            dribble_target=self.dribble_to_target.get(robot_id),
            prev_ball_to_dribble_target_dist=self.prev_ball_to_dribble_target_dist.get(robot_id),
            # Stage 4
            own_goalie_pos=own_goalie_pos,
            own_goalie_has_ball=own_goalie_has_ball,
            prev_clearance_zone_dist=self._prev_clearance_zone_dist.get(robot_id),
            prev_has_ball=self._prev_has_ball.get(robot_id, False),
            prev_opponent_near_ball=self._prev_opponent_near_ball,
            # Stage 5+
            ally_positions=ally_positions,
            is_nearest_to_ball=is_nearest_to_ball,
        )

    @staticmethod
    def _default_reward_state(has_ball: bool) -> str:
        """Infer a default reward state when no explicit state label is provided."""

        if has_ball:
            return "secure_possession"
        return "chase_ball"

    def _end_dribble_session(self, robot_id: int) -> None:
        """Close a dribble session and reset the phase machine."""

        self.dribble_session_active[robot_id] = False
        self.dribble_anchor[robot_id] = None
        self.dribble_states[robot_id].reset()

    def _reset_kick_macro(self, robot_id: int) -> None:
        """Clear the committed kick alignment state for one robot."""

        self.kick_macro_active[robot_id] = False
        self.kick_macro_target_y[robot_id] = None
        self.kick_macro_align_steps[robot_id] = 0
        self.kick_macro_retarget_count[robot_id] = 0

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

    def _keeper_away_kick_target_y(
        self,
        game_state,
        ball_pos: Optional[Sequence[float]],
    ) -> float:
        """Choose a stable, in-mouth shot target away from the keeper.

        This is intentionally deterministic and does not change the policy
        interface. The target is latched when a kick macro opens, then the robot
        aligns to that global shot line over as many capped-turn cycles as needed.
        """

        goal_half_height = float(getattr(self.reward_config, "goal_half_height", 5.0))
        safety_margin = float(getattr(self.reward_config, "goal_post_safety_margin", 0.0))
        safe_edge = max(0.0, goal_half_height - max(0.0, safety_margin))
        configured_target_y = float(
            getattr(self.reward_config, "kick_keeper_away_target_y", 0.0)
        )
        if configured_target_y > 0.0:
            target_mag = max(0.0, min(abs(configured_target_y), safe_edge))
        else:
            target_mag = max(0.0, min(safe_edge, goal_half_height * 0.8))
        if target_mag <= 1e-6:
            return 0.0

        goalie_pose = self._opponent_goalie_pose(game_state)
        goalie_y = float(goalie_pose[1]) if goalie_pose is not None else 0.0
        ball_y = float(ball_pos[1]) if ball_pos is not None and len(ball_pos) >= 2 else 0.0

        if abs(goalie_y) > 0.25:
            target_sign = -1.0 if goalie_y > 0.0 else 1.0
        elif abs(ball_y) > 0.25:
            # If the keeper is central, prefer the side opposite the current
            # ball lane. This avoids defaulting every central-keeper shot to
            # the same post and gives wide starts a sane cross-mouth target.
            target_sign = -1.0 if ball_y > 0.0 else 1.0
        else:
            target_sign = 1.0
        return float(target_sign * target_mag)

    def _maybe_retarget_kick_target_y(
        self,
        robot_id: int,
        game_state,
        current_target_y: float,
    ) -> Tuple[float, bool, Optional[float]]:
        """Optionally flip a committed kick target away from the live keeper.

        The initial target is latched for stability, but the keeper can move
        during the slow real-robot turn. This one-time guard prevents firing at
        the side the keeper has since occupied. It is disabled unless the stage
        config sets `kick_keeper_retarget_max_count > 0`.
        """

        max_count = int(getattr(self.reward_config, "kick_keeper_retarget_max_count", 0))
        if max_count <= 0:
            return current_target_y, False, None
        if self.kick_macro_retarget_count.get(robot_id, 0) >= max_count:
            return current_target_y, False, None

        goalie_pose = self._opponent_goalie_pose(game_state)
        if goalie_pose is None:
            return current_target_y, False, None
        goalie_y = float(goalie_pose[1])
        if abs(current_target_y) <= 1e-6:
            return current_target_y, False, None

        goal_half_height = float(getattr(self.reward_config, "goal_half_height", 5.0))
        safety_margin = float(getattr(self.reward_config, "goal_post_safety_margin", 0.0))
        target_mag = abs(float(current_target_y))
        candidates = [math.copysign(target_mag, current_target_y), -math.copysign(target_mag, current_target_y)]
        qualities = [
            post_safe_goalie_gap_quality(c, goalie_y, goal_half_height, safety_margin)
            for c in candidates
        ]
        current_quality = float(qualities[0])
        best_idx = int(np.argmax(np.asarray(qualities, dtype=np.float32)))
        best_target = float(candidates[best_idx])
        best_quality = float(qualities[best_idx])

        min_quality = float(
            getattr(self.reward_config, "kick_keeper_retarget_min_gap_quality", 0.0)
        )
        same_side_y = float(
            getattr(self.reward_config, "kick_keeper_retarget_same_side_y", 0.0)
        )
        min_improvement = float(
            getattr(self.reward_config, "kick_keeper_retarget_min_improvement", 0.05)
        )
        same_side_blocked = (
            same_side_y > 0.0
            and abs(goalie_y) >= same_side_y
            and current_target_y * goalie_y > 0.0
        )
        low_quality = min_quality > 0.0 and current_quality < min_quality
        if (
            best_target != current_target_y
            and (same_side_blocked or low_quality)
            and best_quality >= current_quality + min_improvement
        ):
            self.kick_macro_retarget_count[robot_id] = (
                self.kick_macro_retarget_count.get(robot_id, 0) + 1
            )
            self.kick_macro_target_y[robot_id] = best_target
            return best_target, True, current_quality
        return current_target_y, False, current_quality

    def _keeper_zone_factor(
        self,
        point: Optional[Tuple[float, float]],
        goalie_pose: Optional[Tuple[float, float, float]],
    ) -> float:
        """Suppression multiplier for gap-scaled reward inside the keeper zone.

        positional_gap_quality peaks inside the keeper's catch radius, so every
        gap-scaled dense term (kick_aim, combo, dribble target/achieved-gap) is
        maximised exactly where the keeper catches — pulling the policy to
        over-dribble to point-blank. This returns a factor that ramps from
        `keeper_zone_floor` at the keeper's position to 1.0 at
        `keeper_zone_radius` away, applied to `point` (the position whose
        gap-quality is being rewarded — the shot origin or carry endpoint).
        Returns 1.0 (no suppression) when disabled or inputs are missing.
        """
        radius = float(getattr(self.reward_config, "keeper_zone_radius", 0.0))
        if radius <= 0.0 or point is None or goalie_pose is None:
            return 1.0
        floor = float(getattr(self.reward_config, "keeper_zone_floor", 0.0))
        d = math.hypot(
            float(point[0]) - float(goalie_pose[0]),
            float(point[1]) - float(goalie_pose[1]),
        )
        if d >= radius:
            return 1.0
        return floor + (1.0 - floor) * (d / radius)
    def _opponent_defender_pose(self, game_state) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, theta_deg) of the non-goalie defender nearest the ball, or None.

        The keeper is the opponent nearest the opponent goal centre (see
        `_opponent_goalie_pose`); the defender is the remaining opponent closest to
        the ball — the one whose lane coverage actually matters for the current
        shot. Returns None when there are fewer than two opponents (stages 1-2).
        """

        if game_state is None:
            return None
        opp_poses: List[Tuple[float, float, float]] = []
        for team_name, entries in getattr(game_state, "robot_poses", {}).items():
            if team_name == self.team_name:
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for _rid, pose in entry.items():
                    if pose is None or len(pose) < 3:
                        continue
                    opp_poses.append((float(pose[0]), float(pose[1]), float(pose[2])))
        if len(opp_poses) < 2:
            return None
        keeper = min(
            opp_poses,
            key=lambda p: math.hypot(p[0] - float(GOAL_R[0]), p[1] - float(GOAL_R[1])),
        )
        rest = [p for p in opp_poses if p is not keeper]
        if not rest:
            return None
        ball_pos = getattr(game_state, "ball_pos", None)
        if ball_pos is not None and len(ball_pos) >= 2:
            bx, by = float(ball_pos[0]), float(ball_pos[1])
            return min(rest, key=lambda p: math.hypot(p[0] - bx, p[1] - by))
        return rest[0]

    def _own_goalie_pose(self, game_state) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, theta_deg) of our own HC goalie, or None."""

        if game_state is None or self.own_goalie_robot_id is None:
            return None
        for team_name, entries in getattr(game_state, "robot_poses", {}).items():
            if team_name != self.team_name:
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for rid, pose in entry.items():
                    if int(rid) == self.own_goalie_robot_id and pose is not None and len(pose) >= 3:
                        return (float(pose[0]), float(pose[1]), float(pose[2]))
        return None

    @staticmethod
    def _project_kick_y_at_goal_line(
        pose: Sequence[float],
        ball_pos: Optional[Sequence[float]] = None,
    ) -> Optional[float]:
        """Project a straight kick from the ball (the trajectory origin)."""

        rx, ry, theta_deg = float(pose[0]), float(pose[1]), float(pose[2])
        origin_x = float(ball_pos[0]) if ball_pos is not None else rx
        origin_y = float(ball_pos[1]) if ball_pos is not None else ry
        theta_rad = math.radians(theta_deg)
        cos_t = math.cos(theta_rad)
        if cos_t <= 1e-3:
            return None
        return float(origin_y + (FIELD_X[1] - origin_x) * (math.sin(theta_rad) / cos_t))

    def _kick_aim_quality_from_pose(
        self,
        pose: Sequence[float],
        ball_pos: Optional[Sequence[float]] = None,
    ) -> Tuple[float, Optional[float]]:
        """Return aim quality and projected goal-line y for a straight kick."""

        predicted_y = self._project_kick_y_at_goal_line(pose, ball_pos)
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

        # Stage 5+: pre-compute per-robot ball distances for role-gating.
        ball_x = float(current_game_state.ball_pos[0]) if current_game_state.ball_pos else 0.0
        ball_y = float(current_game_state.ball_pos[1]) if current_game_state.ball_pos else 0.0

        pose_by_robot_id: Dict[int, Any] = {}
        team_pose_entries = current_game_state.robot_poses.get(self.team_name, [])
        for entry in team_pose_entries:
            if isinstance(entry, dict):
                pose_by_robot_id.update(entry)

        robot_ball_dists: Dict[int, float] = {}
        robot_positions: Dict[int, Tuple[float, float]] = {}
        for rid in self.robot_ids:
            pose = pose_by_robot_id.get(rid)
            if pose is not None:
                rx, ry = float(pose[0]), float(pose[1])
                robot_positions[rid] = (rx, ry)
                robot_ball_dists[rid] = float(math.hypot(rx - ball_x, ry - ball_y))

        nearest_rid = min(robot_ball_dists, key=robot_ball_dists.get) if robot_ball_dists else None
        current_ball_carrier: Optional[int] = None

        for robot_id in self.robot_ids:
            try:
                is_nearest = (robot_id == nearest_rid) if nearest_rid is not None else True
                ally_pos = tuple(
                    pos for rid, pos in robot_positions.items() if rid != robot_id
                )

                reward_inputs = self.build_reward_inputs(
                    current_game_state=current_game_state,
                    robot_id=robot_id,
                    prev_ball_dist=self.prev_reward_ball_dist_by_id.get(robot_id),
                    prev_ball_to_goal_dist=self.prev_reward_ball_to_goal_dist,
                    prev_ball_pos=self.prev_reward_ball_pos,
                    prev_facing_goal_cos=self.prev_reward_facing_goal_cos_by_id.get(robot_id),
                    is_nearest_to_ball=is_nearest,
                    ally_positions=ally_pos,
                )
            except ValueError:
                continue

            reward_result = evaluate_reward(reward_inputs, config=self.reward_config)
            intermediates = reward_result.intermediates
            rewards.append(float(reward_result.reward))
            self.prev_reward_ball_dist_by_id[robot_id] = intermediates.ball_dist
            if intermediates.facing_goal_cos is not None:
                self.prev_reward_facing_goal_cos_by_id[robot_id] = intermediates.facing_goal_cos
            next_ball_to_goal_dist = intermediates.ball_to_goal_dist

            # Stage 4: update tracking state
            self._prev_has_ball[robot_id] = intermediates.has_ball
            if intermediates.own_goalie_has_ball and self.own_goalie_robot_id is not None:
                own_gk_pose = self._own_goalie_pose(current_game_state)
                if own_gk_pose is not None:
                    rx, ry = robot_positions.get(robot_id, (0.0, 0.0))
                    clearance_zone_x = 0.0
                    clearance_zone_y = own_gk_pose[1] * 0.3
                    self._prev_clearance_zone_dist[robot_id] = float(
                        math.hypot(rx - clearance_zone_x, ry - clearance_zone_y)
                    )
            else:
                self._prev_clearance_zone_dist[robot_id] = None

            # Stage 5: track ball carrier
            if intermediates.has_ball:
                current_ball_carrier = robot_id

        # Update shared tracking state
        if next_ball_to_goal_dist is not None:
            self.prev_reward_ball_to_goal_dist = next_ball_to_goal_dist

        if current_game_state.ball_pos is not None:
            self.prev_reward_ball_pos = (
                float(current_game_state.ball_pos[0]),
                float(current_game_state.ball_pos[1]),
            )

        # Stage 4: opponent-near-ball tracking for ball recovery detection
        opp_positions = extract_opponent_positions(
            robot_poses=getattr(current_game_state, "robot_poses", {}),
            team_name=self.team_name,
        )
        if opp_positions:
            nearest_opp_ball = min(
                math.hypot(ox - ball_x, oy - ball_y) for ox, oy in opp_positions
            )
            self._prev_opponent_near_ball = bool(
                nearest_opp_ball < self.reward_config.opponent_near_ball_threshold
            )
        else:
            self._prev_opponent_near_ball = False

        # Stage 5: possession transfer bonus (one-shot)
        possession_transfer_bonus = float(self.reward_config.possession_transfer_bonus)
        if (
            possession_transfer_bonus > 0.0
            and current_ball_carrier is not None
            and self._prev_ball_carrier is not None
            and current_ball_carrier != self._prev_ball_carrier
            and self._prev_ball_carrier in self.robot_ids
        ):
            for i in range(len(rewards)):
                rewards[i] += possession_transfer_bonus / max(len(rewards), 1)
        self._prev_ball_carrier = current_ball_carrier

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
    # teleports and episodes die after one step. ball_speed_max is now 4
    # (real-robot 4 m/s kick cap; 1 unit/cycle = 1 m/s), so 8.0 leaves ample
    # headroom for noise/wind while still catching real server teleports (which
    # jump the ball tens of units). If you raise ball_speed_max again, raise this too.
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
        locomotion_actions = {"goto", "approach_ball", "dribble_to"}
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

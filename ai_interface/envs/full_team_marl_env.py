"""Full 6-agent RoboCup SSL MARL environment with attention-friendly observations.

Extends the SSLHSMMultiAgentEnv pattern with:
- 57D per-agent observation (5 teammate slots + role/state encodings)
- Configurable opponent controller (None / "scripted" / frozen AttentionMAPPOAgent)
- Curriculum-aware spawn configuration
- Formation bonus rewards for 6-agent coordination

Observation layout (57D per agent):
  [0:4]   self: x, y, cos(theta), sin(theta)
  [4:10]  ball: dx, dy, dist, vx, vy, is_kickable
  [10:13] own_goal: dx, dy, dist
  [13:16] opp_goal: dx, dy, dist
  [16:20] role one-hot (STRIKER/SUPPORT/DEFENDER/GOALIE)
  [20:29] HSM state one-hot (9 states)
  [29:49] 5 nearest teammates: (dx, dy, dist, role_idx) × 5 = 20D (zero-padded)
  [49:55] 2 nearest opponents: (dx, dy, dist) × 2 = 6D (zero-padded)
  [55]    goal_threat flag
  [56]    team_possession flag
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from networking.networker import Networker
from ai_interface.constants.player_constants import KICKABLE_MARGIN, PLAYER_SIZE, BALL_SIZE
from ai_interface.envs.mappo_env import MultiAgentSoccerEnv
from ai_interface.hsm.state_machine import (
    AgentContext,
    AgentHSM,
    HSMState,
    Role,
    TeamCoordinator,
)
from ai_interface.rewards.hsm_reward import HSMReward

OBS_DIM = 57
_ROLE_ORDER = [Role.STRIKER, Role.SUPPORT, Role.DEFENDER, Role.GOALIE]
_STATE_ORDER = [
    HSMState.IDLE,
    HSMState.MOVE_TO_BALL,
    HSMState.DRIBBLE,
    HSMState.SHOOT,
    HSMState.PASS,
    HSMState.MARK,
    HSMState.BLOCK,
    HSMState.INTERCEPT,
    HSMState.GOAL_KEEP,
]

# Default role budget for full 6-agent team
DEFAULT_ROLE_BUDGET_6 = {
    Role.STRIKER: 2,
    Role.SUPPORT: 1,
    Role.DEFENDER: 2,
    Role.GOALIE: 1,
}


class FullTeamMARLEnv(gym.Env):
    """Multi-agent env for 6-robot RoboCup SSL with attention-compatible observations.

    The environment is designed to work with AttentionMAPPOAgent:
    - Fixed 57D per-agent obs with 5 teammate slots (zero-padded for N < 6)
    - Continuous 4D action per agent: [vx, vy, omega, kick_speed]
    - HSM state machine provides readable state info in observations
    - Optional opponent controller for curriculum stages

    Args:
        networker: Simulator connection.
        team_name: Name of the team being trained.
        num_agents: Number of agents on the training team (1–6).
        max_steps: Episode length cap.
        hsm_thresholds: Optional overrides for TeamCoordinator thresholds.
        forced_roles: Fixed role assignments {agent_id: Role} for curriculum stages.
        role_budget: Role count budget {Role: count}. Defaults to 6-agent budget
            when num_agents == 6, else legacy single-striker behaviour.
        opponent_controller: None = no opponents; "scripted" = naive heuristic
            opponents; an AttentionMAPPOAgent instance = self-play opponent.
        opponent_team_name: Name of the opponent team in game state.
        reward_weights: Override weights for team bonus rewards.
        spawn_config: Curriculum spawn parameters (ball_x_range, ball_y_range).
    """

    def __init__(
        self,
        networker: Networker,
        team_name: str,
        num_agents: int = 6,
        max_steps: int = 400,
        hsm_thresholds: Optional[Dict[str, float]] = None,
        forced_roles: Optional[Dict[int, Role]] = None,
        role_budget: Optional[Dict[Role, int]] = None,
        opponent_controller: Optional[Any] = None,
        opponent_team_name: Optional[str] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        spawn_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.networker = networker
        self.team_name = team_name
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.forced_roles = forced_roles or {}
        self.opponent_controller = opponent_controller
        self.opponent_team_name = opponent_team_name
        self.spawn_config = spawn_config or {}

        rw = reward_weights or {}
        self.formation_spread_weight = float(rw.get("formation_spread_weight", 0.0))
        self.striker_coord_weight = float(rw.get("striker_coord_weight", 0.0))
        self.defender_coverage_weight = float(rw.get("defender_coverage_weight", 0.0))

        self.kickable_dist = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE
        self.field_half_x = 45.0
        self.field_half_y = 30.0
        self.goal_half_y = 7.32 / 2.0  # SSL goal half-width

        # Resolve role_budget: explicit > default-for-6 > legacy
        if role_budget is not None:
            effective_budget = role_budget
        elif num_agents == 6:
            effective_budget = DEFAULT_ROLE_BUDGET_6
        else:
            effective_budget = None

        thresholds = hsm_thresholds or {}
        self.coordinator = TeamCoordinator(
            num_agents=num_agents,
            role_switch_cooldown=int(thresholds.get("role_switch_cooldown", 8)),
            possession_distance=float(thresholds.get("possession_distance", 1.35)),
            defensive_x_boundary=float(thresholds.get("defensive_x_boundary", -10.0)),
            attacking_x_boundary=float(thresholds.get("attacking_x_boundary", 10.0)),
            role_budget=effective_budget,
        )
        self.agent_hsms = [AgentHSM(i) for i in range(num_agents)]

        # Reward model reuses HSMReward over a dummy baseline env
        baseline = MultiAgentSoccerEnv(
            networker=networker,
            team_name=team_name,
            num_agents=num_agents,
            obs_dim=25,
        )
        self.reward_model = HSMReward(baseline_env=baseline)

        self.observation_space = spaces.Tuple([
            spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
            for _ in range(num_agents)
        ])
        self.action_space = spaces.Box(
            low=np.array([[-2.0, -2.0, -2.0, 0.0]] * num_agents, dtype=np.float32),
            high=np.array([[2.0, 2.0, 2.0, 1.0]] * num_agents, dtype=np.float32),
            dtype=np.float32,
        )

        self.current_step = 0
        self.prev_game_state = None
        self.current_roles: Dict[int, Role] = {i: Role.SUPPORT for i in range(num_agents)}
        self.current_states: Dict[int, HSMState] = {i: HSMState.IDLE for i in range(num_agents)}
        self._prev_robot_positions: Dict[int, Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Public setters (for trainer-side hot-swap without full env rebuild)
    # ------------------------------------------------------------------

    def set_opponent_controller(self, controller: Optional[Any]) -> None:
        self.opponent_controller = controller

    def set_forced_roles(self, forced_roles: Optional[Dict[int, Role]]) -> None:
        self.forced_roles = forced_roles or {}

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self) -> List[np.ndarray]:
        try:
            self.networker.reset_sim()
        except Exception:
            pass

        time.sleep(0.25)
        for _ in range(2):
            try:
                self.networker.get_game_state()
            except Exception:
                break
            time.sleep(0.05)

        self.current_step = 0
        self.prev_game_state = None
        self._prev_robot_positions = {}
        self.coordinator.reset()
        for hsm in self.agent_hsms:
            hsm.reset()

        game_state = self.networker.get_game_state()
        self.current_roles = self.coordinator.assign_roles(
            game_state, self.team_name, step=0, forced_roles=self.forced_roles
        )
        self.current_states = self._transition_states(game_state, self.current_roles)
        return self._build_observations(game_state, self.current_roles, self.current_states)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[List[np.ndarray], float, bool, Dict]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(self.num_agents, 4)

        game_state_before = self.networker.get_game_state()
        roles_before = self.coordinator.assign_roles(
            game_state_before, self.team_name, step=self.current_step,
            forced_roles=self.forced_roles,
        )
        states_before = self._transition_states(game_state_before, roles_before)

        # Execute opponent commands first if a controller is active
        if self.opponent_controller is not None and self.opponent_team_name:
            self._execute_opponent(game_state_before)

        # Execute training-team commands
        commands = []
        for agent_id in range(self.num_agents):
            cmd = self._action_to_command(agent_id, actions[agent_id], game_state_before)
            if cmd is not None:
                commands.append(cmd)
        if commands:
            try:
                self.networker.execute_ai_output(commands, self.team_name)
            except Exception:
                pass

        time.sleep(0.1)
        game_state = self.networker.get_game_state()

        self.current_step += 1
        roles = self.coordinator.assign_roles(
            game_state, self.team_name, step=self.current_step,
            forced_roles=self.forced_roles,
        )
        states = self._transition_states(game_state, roles)
        observations = self._build_observations(game_state, roles, states)

        # Per-agent HSM rewards (role-specific shaping)
        per_agent_rewards = []
        for agent_id in range(self.num_agents):
            r = self.reward_model.compute_role_reward(
                role=roles.get(agent_id, Role.SUPPORT),
                agent_id=agent_id,
                game_state=game_state,
                team_name=self.team_name,
                role_assignments=roles,
                actions=actions.tolist(),
                prev_game_state=self.prev_game_state,
            )
            per_agent_rewards.append(float(r))

        # Team-level formation bonuses (zero when weights are 0)
        team_bonus = self._team_bonus_rewards(game_state, roles)

        team_reward = float(np.mean(per_agent_rewards)) + team_bonus

        # Track robot positions for movement rewards next step
        self._prev_robot_positions = {
            i: pos[:2] for i, pos in self._team_positions(game_state).items()
        }
        self.prev_game_state = game_state
        self.current_roles = roles
        self.current_states = states

        goal_scored, robot_oob, ball_oob = self._termination_flags(game_state)
        done = self.current_step >= self.max_steps or goal_scored

        info = {
            "goal_scored": goal_scored,
            "robot_out_of_bounds": robot_oob,
            "ball_out_of_bounds": ball_oob,
            "roles": {i: r.value for i, r in roles.items()},
            "states": {i: s.value for i, s in states.items()},
            "role_rewards": per_agent_rewards,
            "team_bonus": team_bonus,
        }
        return observations, team_reward, done, info

    # ------------------------------------------------------------------
    # Observation building (57D per agent)
    # ------------------------------------------------------------------

    def _build_observations(
        self,
        game_state,
        roles: Dict[int, Role],
        states: Dict[int, HSMState],
        flip_x: bool = False,
    ) -> List[np.ndarray]:
        """Build 57D per-agent observations.

        Args:
            flip_x: When True, negate x-coordinates (for opponent perspective).
        """
        team_positions = self._team_positions(game_state)
        opp_positions = self._opponent_positions(game_state)
        ball_x, ball_y = (game_state.ball_pos if game_state and game_state.ball_pos else (0.0, 0.0))
        ball_vx, ball_vy = self._ball_velocity(game_state)

        if flip_x:
            ball_x, ball_vx = -ball_x, -ball_vx

        team_possession = self._team_has_possession(game_state, team_positions)
        goal_threat = (ball_x < -20.0 and abs(ball_y) < 15.0) if not flip_x else (ball_x > 20.0 and abs(ball_y) < 15.0)

        obs_list: List[np.ndarray] = []
        for agent_id in range(self.num_agents):
            raw = team_positions.get(agent_id, (0.0, 0.0, 0.0))
            rx, ry, rtheta = raw
            if flip_x:
                rx = -rx

            theta_rad = math.radians(rtheta)
            cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)

            # Ball (relative, agent-centric)
            ball_dx = ball_x - rx
            ball_dy = ball_y - ry
            ball_dist = math.hypot(ball_dx, ball_dy)
            is_kickable = 1.0 if ball_dist < self.kickable_dist else 0.0

            # Own goal (negative x half)
            own_goal_x = -self.field_half_x if not flip_x else self.field_half_x
            own_goal_dx = own_goal_x - rx
            own_goal_dy = -ry
            own_goal_dist = math.hypot(own_goal_dx, own_goal_dy)

            # Opponent goal (positive x half)
            opp_goal_x = self.field_half_x if not flip_x else -self.field_half_x
            opp_goal_dx = opp_goal_x - rx
            opp_goal_dy = -ry
            opp_goal_dist = math.hypot(opp_goal_dx, opp_goal_dy)

            # Role and state encodings
            role = roles.get(agent_id, Role.SUPPORT)
            state = states.get(agent_id, HSMState.IDLE)
            role_oh = self._role_one_hot(role)
            state_oh = self._state_one_hot(state)

            # 5 teammate slots (dx, dy, dist, role_idx), zero-padded
            teammates_raw = [
                (k, v) for k, v in team_positions.items() if k != agent_id
            ]
            teammates_raw.sort(
                key=lambda kv: math.hypot(kv[1][0] - rx, kv[1][1] - ry)
            )
            tm_feats: List[float] = []
            for tm_id, (tx, ty, _) in teammates_raw[:5]:
                tdx = (-tx if flip_x else tx) - rx
                tdy = ty - ry
                tdist = math.hypot(tdx, tdy)
                tm_role = roles.get(tm_id, Role.SUPPORT)
                tm_role_idx = float(_ROLE_ORDER.index(tm_role))
                tm_feats.extend([tdx, tdy, tdist, tm_role_idx])
            while len(tm_feats) < 20:
                tm_feats.extend([0.0, 0.0, 100.0, 0.0])
            tm_feats = tm_feats[:20]

            # 2 nearest opponents (dx, dy, dist), zero-padded
            opp_sorted = sorted(
                opp_positions.values(),
                key=lambda p: math.hypot(p[0] - rx, p[1] - ry),
            )
            opp_feats: List[float] = []
            for ox, oy, _ in opp_sorted[:2]:
                odx = (-ox if flip_x else ox) - rx
                ody = oy - ry
                odist = math.hypot(odx, ody)
                opp_feats.extend([odx, ody, odist])
            while len(opp_feats) < 6:
                opp_feats.extend([0.0, 0.0, 100.0])
            opp_feats = opp_feats[:6]

            obs = np.concatenate([
                [rx, ry, cos_t, sin_t],                # 4D self
                [ball_dx, ball_dy, ball_dist, ball_vx, ball_vy, is_kickable],  # 6D ball
                [own_goal_dx, own_goal_dy, own_goal_dist],  # 3D own goal
                [opp_goal_dx, opp_goal_dy, opp_goal_dist],  # 3D opp goal
                role_oh,                                # 4D role
                state_oh,                               # 9D state
                tm_feats,                               # 20D teammates
                opp_feats,                              # 6D opponents
                [float(goal_threat)],                   # 1D
                [float(team_possession)],               # 1D
            ], dtype=np.float32)                        # total: 57D

            assert obs.shape[0] == OBS_DIM, f"obs shape {obs.shape[0]} != {OBS_DIM}"
            obs_list.append(obs)

        return obs_list

    # ------------------------------------------------------------------
    # Opponent execution (self-play / scripted)
    # ------------------------------------------------------------------

    def _execute_opponent(self, game_state) -> None:
        """Run the opponent controller and submit its commands to the simulator."""
        ctrl = self.opponent_controller
        if ctrl is None or self.opponent_team_name is None:
            return

        opp_positions = self._raw_team_positions(game_state, self.opponent_team_name)
        if not opp_positions:
            return

        if ctrl == "scripted":
            # Naive scripted: each opponent dashes toward ball
            commands = []
            ball_x, ball_y = (game_state.ball_pos if game_state and game_state.ball_pos else (0.0, 0.0))
            for agent_id, (ox, oy, _theta) in opp_positions.items():
                dx, dy = ball_x - ox, ball_y - oy
                direction = math.atan2(dy, dx)
                commands.append(f"dash 50.0 {direction:.4f}")
            try:
                self.networker.execute_ai_output(commands, self.opponent_team_name)
            except Exception:
                pass
        else:
            # Frozen AttentionMAPPOAgent instance for self-play
            from ai_interface.algorithms.attention_mappo import AttentionMAPPOAgent
            if not isinstance(ctrl, AttentionMAPPOAgent):
                return

            n_opp = len(opp_positions)
            if n_opp == 0:
                return

            # Build mirrored observations (flip_x=True for opponent perspective)
            opp_team_positions_full = opp_positions
            # Temporarily swap perspective: reassign agent IDs 0..n_opp-1
            opp_ids_sorted = sorted(opp_team_positions_full.keys())
            opp_agent_map = {new_id: orig_id for new_id, orig_id in enumerate(opp_ids_sorted)}

            # Approximate role assignment for opponent (use default coordinator)
            opp_roles = {new_id: Role.SUPPORT for new_id in range(n_opp)}

            # Build simplified opponent obs (reuse same build logic with flip)
            opp_obs = self._build_observations_for_opponent(game_state, opp_agent_map, opp_roles)
            opp_role_list = [opp_roles[i] for i in range(n_opp)]

            opp_actions, _, _ = ctrl.select_actions(opp_obs, opp_role_list, deterministic=True)
            opp_actions = np.asarray(opp_actions, dtype=np.float32).reshape(n_opp, 4)

            commands = []
            for new_id in range(n_opp):
                orig_id = opp_agent_map[new_id]
                orig_pos = opp_team_positions_full[orig_id]
                # Convert action to command using opponent's actual position
                cmd = self._action_to_command_at_pos(opp_actions[new_id], orig_pos, game_state, flip_x=True)
                if cmd is not None:
                    commands.append(cmd)
            try:
                self.networker.execute_ai_output(commands, self.opponent_team_name)
            except Exception:
                pass

    def _build_observations_for_opponent(
        self,
        game_state,
        opp_agent_map: Dict[int, int],
        opp_roles: Dict[int, Role],
    ) -> List[np.ndarray]:
        """Build opponent observations from their perspective (x-flipped field)."""
        n_opp = len(opp_agent_map)
        # Temporarily swap team perspective
        orig_team_name = self.team_name
        # We build obs by treating opponent positions as 'team' positions with flip_x=True
        # Use a simplified approach: just return zero obs for now if structures mismatch
        obs_list = []
        ball_x, ball_y = (game_state.ball_pos if game_state and game_state.ball_pos else (0.0, 0.0))
        for new_id in range(n_opp):
            orig_id = opp_agent_map[new_id]
            raw_poses = self._raw_team_positions(game_state, self.opponent_team_name)
            pos = raw_poses.get(orig_id, (0.0, 0.0, 0.0))
            ox, oy, otheta = pos

            # Flip x for opponent-centric view
            rx, ry = -ox, oy
            theta_rad = math.radians(otheta + 180.0)
            cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)

            ball_dx = -ball_x - rx
            ball_dy = ball_y - ry
            ball_dist = math.hypot(ball_dx, ball_dy)
            is_kickable = 1.0 if ball_dist < self.kickable_dist else 0.0

            role = opp_roles.get(new_id, Role.SUPPORT)
            role_oh = self._role_one_hot(role)
            state_oh = self._state_one_hot(HSMState.IDLE)

            obs = np.concatenate([
                [rx, ry, cos_t, sin_t],
                [ball_dx, ball_dy, ball_dist, 0.0, 0.0, is_kickable],
                [0.0, 0.0, 0.0],
                [-ball_x - rx, ball_y - ry, math.hypot(-ball_x - rx, ball_y - ry)],
                role_oh,
                state_oh,
                np.zeros(20, dtype=np.float32),
                np.zeros(6, dtype=np.float32),
                [0.0],
                [0.0],
            ], dtype=np.float32)

            # Pad or trim to OBS_DIM
            if obs.shape[0] < OBS_DIM:
                obs = np.pad(obs, (0, OBS_DIM - obs.shape[0]))
            else:
                obs = obs[:OBS_DIM]
            obs_list.append(obs.astype(np.float32))
        return obs_list

    # ------------------------------------------------------------------
    # Action → simulator command
    # ------------------------------------------------------------------

    def _action_to_command(
        self, agent_id: int, action: np.ndarray, game_state
    ) -> Optional[str]:
        pos = self._team_positions(game_state).get(agent_id, (0.0, 0.0, 0.0))
        return self._action_to_command_at_pos(action, pos, game_state, flip_x=False)

    def _action_to_command_at_pos(
        self,
        action: np.ndarray,
        pos: Tuple[float, float, float],
        game_state,
        flip_x: bool = False,
    ) -> Optional[str]:
        vx, vy, omega, kick_speed = [float(x) for x in action]
        rx, ry, _ = pos
        if flip_x:
            rx = -rx

        ball_x, ball_y = (game_state.ball_pos if game_state and game_state.ball_pos else (0.0, 0.0))
        ball_dist = math.hypot(ball_x - rx, ball_y - ry)

        # Kick when near ball and kick_speed signal is strong
        if kick_speed > 0.2 and ball_dist < self.kickable_dist * 1.2:
            kick_power = float(np.clip(kick_speed * 100.0, 10.0, 100.0))
            goal_x = -self.field_half_x if flip_x else self.field_half_x
            goal_dir = math.atan2(-ry, goal_x - rx)
            return f"kick {kick_power:.2f} {goal_dir:.4f}"

        # Turn when omega is dominant and speed is low
        if abs(omega) > 0.8 and math.hypot(vx, vy) < 0.3:
            turn_power = float(np.clip(omega, -2.0, 2.0))
            return f"turn {turn_power:.4f}"

        # Dash
        speed = math.hypot(vx, vy)
        direction = math.atan2(vy, vx) if speed > 1e-6 else 0.0
        dash_power = float(np.clip(speed * 40.0, 0.0, 80.0))
        return f"dash {dash_power:.2f} {direction:.4f}"

    # ------------------------------------------------------------------
    # Team bonus rewards (new for 6-agent coordination)
    # ------------------------------------------------------------------

    def _team_bonus_rewards(self, game_state, roles: Dict[int, Role]) -> float:
        """Three formation signals that incentivize 6-agent structure."""
        if (
            self.formation_spread_weight == 0.0
            and self.striker_coord_weight == 0.0
            and self.defender_coverage_weight == 0.0
        ):
            return 0.0

        team_pos = self._team_positions(game_state)
        if len(team_pos) < 2:
            return 0.0

        positions = [(v[0], v[1]) for v in team_pos.values()]
        bonus = 0.0

        # Formation spread: reward average pairwise distance (scaled to [0,1])
        if self.formation_spread_weight > 0.0 and len(positions) >= 2:
            total_dist = 0.0
            n_pairs = 0
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    total_dist += math.hypot(
                        positions[i][0] - positions[j][0],
                        positions[i][1] - positions[j][1],
                    )
                    n_pairs += 1
            avg_dist = total_dist / max(1, n_pairs)
            spread_signal = min(1.0, avg_dist / 20.0)
            bonus += self.formation_spread_weight * spread_signal

        # Dual-striker separation: reward ~10m lateral spread between the two strikers
        if self.striker_coord_weight > 0.0:
            striker_ids = [i for i, r in roles.items() if r == Role.STRIKER]
            if len(striker_ids) >= 2:
                p1 = team_pos.get(striker_ids[0], (0.0, 0.0, 0.0))
                p2 = team_pos.get(striker_ids[1], (0.0, 0.0, 0.0))
                dist_s = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                coord_signal = max(0.0, 1.0 - abs(dist_s - 10.0) / 8.0)
                bonus += self.striker_coord_weight * coord_signal

        # Dual-defender coverage: reward when two defenders each mark different opponents
        if self.defender_coverage_weight > 0.0:
            defender_ids = [i for i, r in roles.items() if r == Role.DEFENDER]
            opp_pos = self._opponent_positions(game_state)
            opp_list = list(opp_pos.values())
            if len(defender_ids) >= 2 and len(opp_list) >= 2:
                d1 = team_pos.get(defender_ids[0], (0.0, 0.0, 0.0))
                d2 = team_pos.get(defender_ids[1], (0.0, 0.0, 0.0))
                # Check if each defender is close to a different opponent
                best_opp_d1 = min(math.hypot(d1[0] - o[0], d1[1] - o[1]) for o in opp_list)
                best_opp_d2 = min(math.hypot(d2[0] - o[0], d2[1] - o[1]) for o in opp_list)
                if best_opp_d1 < 5.0 and best_opp_d2 < 5.0:
                    bonus += self.defender_coverage_weight

        return bonus

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _termination_flags(self, game_state) -> Tuple[bool, bool, bool]:
        if game_state is None or game_state.ball_pos is None:
            return False, False, False
        ball_x, ball_y = game_state.ball_pos
        goal_scored = ball_x >= self.field_half_x and abs(ball_y) < self.goal_half_y
        ball_oob = abs(ball_y) > self.field_half_y or ball_x < -self.field_half_x
        robot_oob = any(
            abs(x) > self.field_half_x or abs(y) > self.field_half_y
            for (x, y, _) in self._team_positions(game_state).values()
        )
        return goal_scored, robot_oob, ball_oob

    # ------------------------------------------------------------------
    # HSM state transitions (mirrors ssl_hsm_env.py)
    # ------------------------------------------------------------------

    def _transition_states(
        self, game_state, roles: Dict[int, Role]
    ) -> Dict[int, HSMState]:
        team_pos = self._team_positions(game_state)
        opp_pos = self._opponent_positions(game_state)
        ball_x, ball_y = (game_state.ball_pos if game_state and game_state.ball_pos else (0.0, 0.0))
        ball_vx, ball_vy = self._ball_velocity(game_state)
        ball_speed = math.hypot(ball_vx, ball_vy)
        game_phase = self.coordinator.infer_game_phase(game_state)

        out: Dict[int, HSMState] = {}
        for agent_id in range(self.num_agents):
            rx, ry, _ = team_pos.get(agent_id, (0.0, 0.0, 0.0))
            dist_to_ball = math.hypot(ball_x - rx, ball_y - ry)
            tm_list = [(v[0], v[1]) for k, v in team_pos.items() if k != agent_id]
            opp_list = [(v[0], v[1]) for v in opp_pos.values()]
            nearest_tm = min((math.hypot(tx - rx, ty - ry) for (tx, ty) in tm_list), default=100.0)
            nearest_opp = min((math.hypot(ox - rx, oy - ry) for (ox, oy) in opp_list), default=100.0)
            ctx = AgentContext(
                dist_to_ball=dist_to_ball,
                has_possession=dist_to_ball < self.kickable_dist,
                ball_speed=ball_speed,
                ball_x=ball_x,
                ball_y=ball_y,
                self_x=rx,
                self_y=ry,
                nearest_teammate_dist=nearest_tm,
                nearest_opponent_dist=nearest_opp,
                shot_open=self._goal_lane_open((ball_x, ball_y), opp_list),
                pass_open=nearest_tm < 15.0,
                goal_threat=(ball_x < -20.0 and abs(ball_y) < 15.0),
                game_phase=game_phase,
            )
            out[agent_id] = self.agent_hsms[agent_id].transition(
                roles.get(agent_id, Role.SUPPORT), ctx
            )
        return out

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _role_one_hot(self, role: Role) -> np.ndarray:
        arr = np.zeros(len(_ROLE_ORDER), dtype=np.float32)
        arr[_ROLE_ORDER.index(role)] = 1.0
        return arr

    def _state_one_hot(self, state: HSMState) -> np.ndarray:
        arr = np.zeros(len(_STATE_ORDER), dtype=np.float32)
        arr[_STATE_ORDER.index(state)] = 1.0
        return arr

    def _team_has_possession(self, game_state, team_pos: Dict) -> bool:
        if game_state is None or game_state.ball_pos is None:
            return False
        bx, by = game_state.ball_pos
        return any(
            math.hypot(x - bx, y - by) < self.kickable_dist * 1.2
            for (x, y, _) in team_pos.values()
        )

    # ------------------------------------------------------------------
    # Game-state extraction helpers
    # ------------------------------------------------------------------

    def _team_positions(self, game_state) -> Dict[int, Tuple[float, float, float]]:
        return self._raw_team_positions(game_state, self.team_name)

    def _raw_team_positions(
        self, game_state, team_name: str
    ) -> Dict[int, Tuple[float, float, float]]:
        if game_state is None:
            return {}
        out: Dict[int, Tuple[float, float, float]] = {}
        for pose_dict in game_state.robot_poses.get(team_name, []):
            for unum, (x, y, theta) in pose_dict.items():
                out[int(unum) - 1] = (float(x), float(y), float(theta))
        return out

    def _opponent_positions(self, game_state) -> Dict[int, Tuple[float, float, float]]:
        if game_state is None:
            return {}
        out: Dict[int, Tuple[float, float, float]] = {}
        idx = 0
        for tname, poses in game_state.robot_poses.items():
            if tname == self.team_name:
                continue
            for pose_dict in poses:
                for _unum, (x, y, theta) in pose_dict.items():
                    out[idx] = (float(x), float(y), float(theta))
                    idx += 1
        return out

    def _ball_velocity(self, game_state) -> Tuple[float, float]:
        if game_state is None:
            return 0.0, 0.0
        ball_vel = getattr(game_state, "ball_vel", None)
        if ball_vel is not None:
            return float(ball_vel[0]), float(ball_vel[1])
        if (
            self.prev_game_state is not None
            and self.prev_game_state.ball_pos is not None
            and game_state.ball_pos is not None
        ):
            px, py = self.prev_game_state.ball_pos
            cx, cy = game_state.ball_pos
            return float(cx - px), float(cy - py)
        return 0.0, 0.0

    @staticmethod
    def _goal_lane_open(
        ball_pos: Tuple[float, float], opponents: List[Tuple[float, float]]
    ) -> bool:
        if not opponents:
            return True
        goal = (45.0, 0.0)
        clearance = min(
            FullTeamMARLEnv._pt_to_seg(opp, ball_pos, goal) for opp in opponents
        )
        return clearance > 1.2

    @staticmethod
    def _pt_to_seg(
        p: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:
        ax, ay = a; bx, by = b; px, py = p
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-8:
            return math.hypot(apx, apy)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        return math.hypot(px - (ax + t * abx), py - (ay + t * aby))

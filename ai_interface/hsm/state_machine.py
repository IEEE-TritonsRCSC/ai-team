"""Deterministic hierarchical state machine for role assignment and control flow."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math

# Tunable thresholds.
POSSESSION_DISTANCE = 1.35
SHOT_DISTANCE = 14.0
PASS_DISTANCE_MIN = 5.0
PASS_DISTANCE_MAX = 20.0
DEFENSIVE_X_BOUNDARY = -10.0
ATTACKING_X_BOUNDARY = 10.0
GOAL_THREAT_DISTANCE = 20.0
ROLE_SWITCH_COOLDOWN = 8
INTERCEPT_SPEED_THRESHOLD = 0.2


class Role(str, Enum):
    """Team-level role assignment from the deterministic coordinator."""

    STRIKER = "STRIKER"
    SUPPORT = "SUPPORT"
    DEFENDER = "DEFENDER"
    GOALIE = "GOALIE"


class HSMState(str, Enum):
    """Per-agent finite states used to route policy focus."""

    IDLE = "IDLE"
    MOVE_TO_BALL = "MOVE_TO_BALL"
    DRIBBLE = "DRIBBLE"
    SHOOT = "SHOOT"
    PASS = "PASS"
    MARK = "MARK"
    BLOCK = "BLOCK"
    INTERCEPT = "INTERCEPT"
    GOAL_KEEP = "GOAL_KEEP"


@dataclass
class AgentContext:
    """Context passed into state transition logic."""

    dist_to_ball: float
    has_possession: bool
    ball_speed: float
    ball_x: float
    ball_y: float
    self_x: float
    self_y: float
    nearest_teammate_dist: float
    nearest_opponent_dist: float
    shot_open: bool
    pass_open: bool
    goal_threat: bool
    game_phase: str


class AgentHSM:
    """Deterministic per-agent state machine driven by role and local context."""

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.role = Role.SUPPORT
        self.state = HSMState.IDLE

    def reset(self) -> None:
        self.state = HSMState.IDLE

    def transition(self, role: Role, context: AgentContext) -> HSMState:
        self.role = role
        if role == Role.GOALIE:
            self.state = self._goalie_transition(context)
        elif role == Role.DEFENDER:
            self.state = self._defender_transition(context)
        elif role == Role.SUPPORT:
            self.state = self._support_transition(context)
        else:
            self.state = self._striker_transition(context)
        return self.state

    def _striker_transition(self, ctx: AgentContext) -> HSMState:
        if ctx.game_phase != "play_on":
            return HSMState.IDLE
        if not ctx.has_possession:
            return HSMState.MOVE_TO_BALL
        if ctx.shot_open and ctx.dist_to_ball < POSSESSION_DISTANCE and ctx.self_x > ATTACKING_X_BOUNDARY:
            return HSMState.SHOOT
        if ctx.pass_open and PASS_DISTANCE_MIN < ctx.nearest_teammate_dist < PASS_DISTANCE_MAX:
            return HSMState.PASS
        return HSMState.DRIBBLE

    def _support_transition(self, ctx: AgentContext) -> HSMState:
        if ctx.game_phase != "play_on":
            return HSMState.IDLE
        if ctx.has_possession and ctx.pass_open:
            return HSMState.PASS
        if ctx.goal_threat:
            return HSMState.BLOCK
        if ctx.dist_to_ball < POSSESSION_DISTANCE * 1.5:
            return HSMState.MOVE_TO_BALL
        return HSMState.MARK

    def _defender_transition(self, ctx: AgentContext) -> HSMState:
        if ctx.game_phase != "play_on":
            return HSMState.IDLE
        if ctx.goal_threat:
            if ctx.ball_speed > INTERCEPT_SPEED_THRESHOLD:
                return HSMState.INTERCEPT
            return HSMState.BLOCK
        if ctx.dist_to_ball < POSSESSION_DISTANCE * 1.25:
            return HSMState.INTERCEPT
        return HSMState.MARK

    def _goalie_transition(self, ctx: AgentContext) -> HSMState:
        if ctx.goal_threat:
            if ctx.ball_speed > INTERCEPT_SPEED_THRESHOLD:
                return HSMState.INTERCEPT
            return HSMState.BLOCK
        return HSMState.GOAL_KEEP


class TeamCoordinator:
    """Assign deterministic roles to agents from global game state."""

    def __init__(
        self,
        num_agents: int,
        role_switch_cooldown: int = ROLE_SWITCH_COOLDOWN,
        possession_distance: float = POSSESSION_DISTANCE,
        defensive_x_boundary: float = DEFENSIVE_X_BOUNDARY,
        attacking_x_boundary: float = ATTACKING_X_BOUNDARY,
    ):
        self.num_agents = num_agents
        self.role_switch_cooldown = role_switch_cooldown
        self.possession_distance = possession_distance
        self.defensive_x_boundary = defensive_x_boundary
        self.attacking_x_boundary = attacking_x_boundary
        self._last_roles: Dict[int, Role] = {i: Role.SUPPORT for i in range(num_agents)}
        self._last_switch_step: Dict[int, int] = {i: -10_000 for i in range(num_agents)}

    def reset(self) -> None:
        self._last_roles = {i: Role.SUPPORT for i in range(self.num_agents)}
        self._last_switch_step = {i: -10_000 for i in range(self.num_agents)}

    def assign_roles(
        self,
        game_state,
        team_name: str,
        step: int,
        forced_roles: Optional[Dict[int, Role]] = None,
    ) -> Dict[int, Role]:
        forced_roles = forced_roles or {}
        team_positions = self._extract_team_positions(game_state, team_name)
        if not team_positions:
            return {i: forced_roles.get(i, self._last_roles.get(i, Role.SUPPORT)) for i in range(self.num_agents)}

        ball_x, ball_y = (0.0, 0.0)
        if game_state is not None and getattr(game_state, "ball_pos", None) is not None:
            ball_x, ball_y = game_state.ball_pos

        # Goalie assignment is deterministic: nearest to own goal center.
        own_goal = (-45.0, 0.0)
        goalie_id = min(team_positions, key=lambda i: self._distance(team_positions[i], own_goal))

        distances_to_ball = {
            i: self._distance(team_positions[i], (ball_x, ball_y)) for i in team_positions
        }
        sorted_by_ball = sorted(distances_to_ball, key=distances_to_ball.get)

        team_has_possession = (
            len(sorted_by_ball) > 0 and distances_to_ball[sorted_by_ball[0]] <= self.possession_distance
        )

        assignments: Dict[int, Role] = {}
        for agent_id in range(self.num_agents):
            if agent_id in forced_roles:
                assignments[agent_id] = forced_roles[agent_id]

        for agent_id in range(self.num_agents):
            if agent_id in assignments:
                continue
            if agent_id == goalie_id:
                assignments[agent_id] = Role.GOALIE
                continue

            if team_has_possession:
                # Striker is current possessor if not goalie.
                striker_candidate = sorted_by_ball[0]
                if striker_candidate == goalie_id and len(sorted_by_ball) > 1:
                    striker_candidate = sorted_by_ball[1]
                if agent_id == striker_candidate:
                    assignments[agent_id] = Role.STRIKER
                else:
                    # Players behind the ball become defenders, others support.
                    x, _ = team_positions[agent_id]
                    assignments[agent_id] = Role.DEFENDER if x < ball_x else Role.SUPPORT
            else:
                # No possession: closest non-goalie hunts as striker/interceptor.
                non_goalie = [i for i in sorted_by_ball if i != goalie_id]
                striker_candidate = non_goalie[0] if non_goalie else goalie_id
                if agent_id == striker_candidate:
                    assignments[agent_id] = Role.STRIKER
                else:
                    x, _ = team_positions[agent_id]
                    if x < self.defensive_x_boundary:
                        assignments[agent_id] = Role.DEFENDER
                    else:
                        assignments[agent_id] = Role.SUPPORT

        assignments = self._resolve_conflicts(assignments, team_positions, (ball_x, ball_y), goalie_id)
        assignments = self._apply_switch_cooldown(assignments, step, forced_roles)

        self._last_roles = dict(assignments)
        return assignments

    def infer_game_phase(self, game_state) -> str:
        playmode = getattr(game_state, "playmode", None)
        if playmode is None:
            return "play_on"
        mode = str(playmode).lower()
        if "kick_off" in mode:
            return "kick_off"
        if "goal_kick" in mode:
            return "goal_kick"
        if "corner" in mode:
            return "corner_kick"
        if "play_on" in mode:
            return "play_on"
        return "set_play"

    def _resolve_conflicts(
        self,
        assignments: Dict[int, Role],
        team_positions: Dict[int, Tuple[float, float]],
        ball_pos: Tuple[float, float],
        goalie_id: int,
    ) -> Dict[int, Role]:
        # Enforce exactly one striker among non-goalie players.
        strikers = [i for i, role in assignments.items() if role == Role.STRIKER and i != goalie_id]
        if len(strikers) != 1:
            candidates = [i for i in team_positions if i != goalie_id]
            if candidates:
                striker = min(candidates, key=lambda i: self._distance(team_positions[i], ball_pos))
                for i in candidates:
                    if assignments.get(i) == Role.STRIKER and i != striker:
                        assignments[i] = Role.SUPPORT
                assignments[striker] = Role.STRIKER

        # Guarantee at least one defender if team has >= 3 agents.
        if self.num_agents >= 3 and not any(role == Role.DEFENDER for role in assignments.values()):
            support_candidates = [i for i, role in assignments.items() if role == Role.SUPPORT and i != goalie_id]
            if support_candidates:
                defender = min(support_candidates, key=lambda i: team_positions.get(i, (999.0, 0.0))[0])
                assignments[defender] = Role.DEFENDER

        # Keep goalie unique.
        for i in assignments:
            if i != goalie_id and assignments[i] == Role.GOALIE:
                assignments[i] = Role.SUPPORT
        assignments[goalie_id] = Role.GOALIE

        return assignments

    def _apply_switch_cooldown(
        self,
        assignments: Dict[int, Role],
        step: int,
        forced_roles: Dict[int, Role],
    ) -> Dict[int, Role]:
        cooled = dict(assignments)
        for agent_id, new_role in assignments.items():
            if agent_id in forced_roles:
                continue
            old_role = self._last_roles.get(agent_id, new_role)
            if new_role == old_role:
                continue
            if step - self._last_switch_step.get(agent_id, -10_000) < self.role_switch_cooldown:
                cooled[agent_id] = old_role
            else:
                self._last_switch_step[agent_id] = step
        return cooled

    @staticmethod
    def _extract_team_positions(game_state, team_name: str) -> Dict[int, Tuple[float, float]]:
        if game_state is None:
            return {}
        team_poses = getattr(game_state, "robot_poses", {}).get(team_name, [])
        positions: Dict[int, Tuple[float, float]] = {}
        for pose_dict in team_poses:
            for unum, (x, y, _theta) in pose_dict.items():
                agent_id = int(unum) - 1
                positions[agent_id] = (float(x), float(y))
        return positions

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

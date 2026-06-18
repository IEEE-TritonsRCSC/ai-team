"""Role-specific reward shaping for HSM-MARL built on existing team rewards."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

from ai_interface.hsm.state_machine import Role
from ai_interface.envs.mappo_env import MultiAgentSoccerEnv


class HSMReward:
    """Compose existing team reward with role-specific shaping terms."""

    def __init__(self, baseline_env: Optional[MultiAgentSoccerEnv] = None):
        self.baseline_env = baseline_env

    def compute_role_reward(
        self,
        role: Role,
        agent_id: int,
        game_state,
        team_name: str,
        role_assignments: Dict[int, Role],
        actions: List[List[float]],
        prev_game_state=None,
    ) -> float:
        # Baseline team reward includes alive/progress terms that can accidentally
        # create a local optimum around idling, so keep it as a weak prior only.
        base_team_reward = 0.25 * self._baseline_team_reward(game_state, actions)

        if game_state is None:
            return base_team_reward - 0.1

        ball_x, ball_y = game_state.ball_pos or (0.0, 0.0)
        ball_pos = (ball_x, ball_y)
        team_positions = self._team_positions(game_state, team_name)
        opponent_positions = self._opponent_positions(game_state, team_name)
        self_pos = team_positions.get(agent_id, (0.0, 0.0))

        prev_pos = None
        if prev_game_state is not None:
            prev_team_positions = self._team_positions(prev_game_state, team_name)
            prev_pos = prev_team_positions.get(agent_id, None)

        dist_to_ball = self._distance(self_pos, ball_pos)

        # Command-based idle signal (prevents no-op policy from farming tiny rewards).
        cmd_idle_penalty = 0.0
        if agent_id < len(actions):
            vx, vy, omega, kick = actions[agent_id]
            cmd_mag = math.hypot(vx, vy) + 0.25 * abs(omega) + 0.5 * max(0.0, kick)
            if cmd_mag < 0.08 and dist_to_ball > 1.8:
                cmd_idle_penalty = -0.2

        movement_term = 0.0
        if prev_pos is not None:
            moved = self._distance(prev_pos, self_pos)
            # Strong anti-freeze penalty unless robot is already close enough to play the ball.
            if moved < 0.03 and dist_to_ball > 1.6:
                movement_term -= 0.25
            else:
                movement_term += min(0.12, moved * 0.25)

        if role == Role.STRIKER:
            bonus = self._striker_reward(
                self_pos=self_pos,
                prev_pos=prev_pos,
                ball_pos=ball_pos,
                opponent_positions=opponent_positions,
                prev_game_state=prev_game_state,
                game_state=game_state,
            )
        elif role == Role.SUPPORT:
            bonus = self._support_reward(
                self_pos=self_pos,
                ball_pos=ball_pos,
                team_positions=team_positions,
                opponent_positions=opponent_positions,
                role_assignments=role_assignments,
            )
        elif role == Role.DEFENDER:
            bonus = self._defender_reward(
                self_pos=self_pos,
                ball_pos=ball_pos,
                opponent_positions=opponent_positions,
                prev_game_state=prev_game_state,
                game_state=game_state,
            )
        else:
            bonus = self._goalie_reward(
                self_pos=self_pos,
                ball_pos=ball_pos,
                prev_game_state=prev_game_state,
                game_state=game_state,
            )

        total = base_team_reward + bonus + movement_term + cmd_idle_penalty
        return float(self._clip(total, -5.0, 8.0))

    def _baseline_team_reward(self, game_state, actions: List[List[float]]) -> float:
        """Reuse existing MAPPO team reward implementation as a wrapped baseline."""
        if self.baseline_env is None:
            return 0.0

        # Existing mappo env uses discrete actions; we map the continuous command intent.
        # 0 APPROACH, 1 SHOOT, 2 PASS, 3 DRIBBLE, 4 CLEAR, 5 REPOSITION
        discrete_actions = []
        for action in actions:
            vx, vy, omega, kick = action
            speed = math.hypot(vx, vy)
            if kick > 0.6:
                discrete_actions.append(1)
            elif kick > 0.2:
                discrete_actions.append(2)
            elif speed > 1.2:
                discrete_actions.append(3)
            elif abs(omega) > 0.6:
                discrete_actions.append(5)
            else:
                discrete_actions.append(0)

        return float(self.baseline_env._compute_team_reward(game_state, discrete_actions))

    def _striker_reward(
        self,
        self_pos: Tuple[float, float],
        prev_pos: Optional[Tuple[float, float]],
        ball_pos: Tuple[float, float],
        opponent_positions: List[Tuple[float, float]],
        prev_game_state,
        game_state,
    ) -> float:
        shot_quality = self._goal_lane_openness(ball_pos, opponent_positions)
        goal_scored = 4.0 if self._is_goal_scored(game_state) else 0.0
        on_target_bonus = self._ball_toward_goal_bonus(prev_game_state, game_state)
        dist_to_ball = self._distance(self_pos, ball_pos)
        possession_bonus = 0.8 if dist_to_ball < 1.4 else 0.0

        approach_term = 0.0
        if prev_pos is not None:
            prev_dist = self._distance(prev_pos, ball_pos)
            curr_dist = dist_to_ball
            approach_term = self._clip(2.4 * (prev_dist - curr_dist), -1.2, 1.2)

        distance_penalty = -0.04 * min(20.0, dist_to_ball)
        return 1.2 * shot_quality + goal_scored + on_target_bonus + possession_bonus + approach_term + distance_penalty

    def _support_reward(
        self,
        self_pos: Tuple[float, float],
        ball_pos: Tuple[float, float],
        team_positions: Dict[int, Tuple[float, float]],
        opponent_positions: List[Tuple[float, float]],
        role_assignments: Dict[int, Role],
    ) -> float:
        striker_ids = [i for i, role in role_assignments.items() if role == Role.STRIKER]
        striker_pos = team_positions.get(striker_ids[0], self_pos) if striker_ids else self_pos

        passing_lane_openness = self._segment_clearance(self_pos, striker_pos, opponent_positions)
        teammate_distance = self._distance(self_pos, striker_pos)
        teammate_distance_term = max(0.0, 1.0 - abs(teammate_distance - 7.0) / 7.0)

        # Encourage creating depth and lateral spacing around the ball.
        space_creation = min(1.0, abs(self_pos[1] - ball_pos[1]) / 10.0)
        ball_support_dist = self._distance(self_pos, ball_pos)
        ball_anchor = max(0.0, 1.0 - ball_support_dist / 14.0)
        return 1.0 * passing_lane_openness + 0.7 * teammate_distance_term + 0.6 * space_creation + 0.8 * ball_anchor

    def _defender_reward(
        self,
        self_pos: Tuple[float, float],
        ball_pos: Tuple[float, float],
        opponent_positions: List[Tuple[float, float]],
        prev_game_state,
        game_state,
    ) -> float:
        interception_success = self._interception_signal(self_pos, ball_pos, prev_game_state, game_state)
        marking_quality = 0.0
        if opponent_positions:
            nearest_opp = min(opponent_positions, key=lambda p: self._distance(self_pos, p))
            marking_quality = max(0.0, 1.0 - abs(self._distance(self_pos, nearest_opp) - 2.5) / 2.5)

        clear_distance = 0.0
        if prev_game_state is not None and game_state is not None:
            prev_ball = prev_game_state.ball_pos or (0.0, 0.0)
            curr_ball = game_state.ball_pos or (0.0, 0.0)
            clear_distance = max(0.0, curr_ball[0] - prev_ball[0]) / 3.0

        return 1.0 * interception_success + 0.9 * marking_quality + 0.7 * clear_distance

    def _goalie_reward(
        self,
        self_pos: Tuple[float, float],
        ball_pos: Tuple[float, float],
        prev_game_state,
        game_state,
    ) -> float:
        save_success = 0.0
        if prev_game_state is not None and game_state is not None:
            prev_ball = prev_game_state.ball_pos or (0.0, 0.0)
            curr_ball = game_state.ball_pos or (0.0, 0.0)
            # Reward when the ball is moved away from own goal mouth.
            if prev_ball[0] < -35.0 and curr_ball[0] > prev_ball[0]:
                save_success = 1.0

        target_x, target_y = -43.0, ball_pos[1] * 0.4
        positioning_quality = max(0.0, 1.0 - self._distance(self_pos, (target_x, target_y)) / 8.0)

        distribution_reward = 0.0
        if prev_game_state is not None and game_state is not None:
            prev_ball = prev_game_state.ball_pos or (0.0, 0.0)
            curr_ball = game_state.ball_pos or (0.0, 0.0)
            if prev_ball[0] < -30.0 and curr_ball[0] > prev_ball[0] + 1.5:
                distribution_reward = 0.8

        return 1.2 * save_success + 1.0 * positioning_quality + 0.8 * distribution_reward

    @staticmethod
    def _team_positions(game_state, team_name: str) -> Dict[int, Tuple[float, float]]:
        poses = getattr(game_state, "robot_poses", {}).get(team_name, [])
        out: Dict[int, Tuple[float, float]] = {}
        for pose_dict in poses:
            for unum, (x, y, _theta) in pose_dict.items():
                out[int(unum) - 1] = (float(x), float(y))
        return out

    @staticmethod
    def _opponent_positions(game_state, team_name: str) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for tname, poses in getattr(game_state, "robot_poses", {}).items():
            if tname == team_name:
                continue
            for pose_dict in poses:
                for _unum, (x, y, _theta) in pose_dict.items():
                    out.append((float(x), float(y)))
        return out

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _is_goal_scored(game_state) -> bool:
        if game_state is None or getattr(game_state, "ball_pos", None) is None:
            return False
        ball_x, ball_y = game_state.ball_pos
        return ball_x >= 45.0 and abs(ball_y) < 7.32 / 2

    @staticmethod
    def _goal_lane_openness(ball_pos: Tuple[float, float], opponents: List[Tuple[float, float]]) -> float:
        goal = (45.0, 0.0)
        if not opponents:
            return 1.0
        min_clearance = min(
            HSMReward._point_to_segment_distance(opp, ball_pos, goal) for opp in opponents
        )
        return max(0.0, min(1.0, min_clearance / 4.0))

    @staticmethod
    def _segment_clearance(
        start: Tuple[float, float],
        end: Tuple[float, float],
        opponents: List[Tuple[float, float]],
    ) -> float:
        if not opponents:
            return 1.0
        min_clearance = min(HSMReward._point_to_segment_distance(opp, start, end) for opp in opponents)
        return max(0.0, min(1.0, min_clearance / 3.0))

    @staticmethod
    def _point_to_segment_distance(
        p: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:
        ax, ay = a
        bx, by = b
        px, py = p
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-8:
            return math.hypot(apx, apy)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        cx = ax + t * abx
        cy = ay + t * aby
        return math.hypot(px - cx, py - cy)

    @staticmethod
    def _ball_toward_goal_bonus(prev_game_state, game_state) -> float:
        if prev_game_state is None or game_state is None:
            return 0.0
        prev_ball = prev_game_state.ball_pos or (0.0, 0.0)
        curr_ball = game_state.ball_pos or (0.0, 0.0)
        prev_goal_dist = math.hypot(45.0 - prev_ball[0], prev_ball[1])
        curr_goal_dist = math.hypot(45.0 - curr_ball[0], curr_ball[1])
        delta = prev_goal_dist - curr_goal_dist
        return max(0.0, min(2.0, delta))

    @staticmethod
    def _interception_signal(self_pos, ball_pos, prev_game_state, game_state) -> float:
        dist = math.hypot(self_pos[0] - ball_pos[0], self_pos[1] - ball_pos[1])
        base = max(0.0, 1.0 - dist / 8.0)
        if prev_game_state is None or game_state is None:
            return base
        prev_ball = prev_game_state.ball_pos or (0.0, 0.0)
        curr_ball = game_state.ball_pos or (0.0, 0.0)
        # Bonus for moving the ball away from own half danger zone.
        danger_delta = curr_ball[0] - prev_ball[0]
        return base + max(0.0, danger_delta / 4.0)

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

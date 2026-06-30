"""Hardcoded supporting attacker for hybrid Stage 4 play.

The controller is intentionally narrow: it does not replace the learned main
attacker.  It positions one teammate in a useful receive lane, tries to collect
passes that are already in flight, and finishes quickly if it gains the ball.
The geometry follows the same broad pattern as strong SSL systems: reason about
clear lanes, continuation shot quality, and role separation instead of chasing
the ball by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ai_interface.constants.field_constants import FIELD_X, FIELD_Y, GOAL_L, GOAL_R
from ai_interface.constants.player_constants import BALL_SIZE, KICKABLE_MARGIN, PLAYER_SIZE
from ai_interface.player import Player
from networking.data_utils import GameState


Point = Tuple[float, float]
Pose = Tuple[float, float, float]

KICKABLE_DISTANCE = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE
DEFAULT_GOAL_TARGETS: tuple[float, ...] = (-4.0, 0.0, 4.0)


@dataclass(frozen=True)
class SupporterConfig:
    """Tactical constants for the hardcoded supporter.

    Values are in simulator environment units.  The repository convention is
    ``10 env units = 1 real meter`` for the SSL Division B field.
    """

    field_margin: float = 2.0
    min_pass_distance: float = 8.0
    max_receive_x_abs: float = 34.0
    max_receive_y_abs: float = 14.0
    lane_block_dist: float = 2.6
    opponent_keepout: float = 3.0
    teammate_keepout: float = 5.0
    receive_arrival_margin: float = 1.2
    receive_intercept_horizon: int = 18
    receive_intercept_radius: float = 5.0
    support_hysteresis_steps: int = 10
    shoot_min_quality: float = 0.33
    shoot_min_lane_clear: float = 0.70
    return_pass_min_quality: float = 0.58
    clearout_after_touch_steps: int = 18
    pass_power: float = 65.0
    shoot_power: float = 100.0
    settle_speed: float = 75.0


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return Euclidean distance between two points."""

    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    """Return the shortest distance from ``point`` to the segment ``start``→``end``."""

    p = np.asarray(point[:2], dtype=float)
    a = np.asarray(start[:2], dtype=float)
    b = np.asarray(end[:2], dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def lane_clear_quality(
    start: Sequence[float],
    end: Sequence[float],
    blockers: Iterable[Sequence[float]],
    *,
    block_dist: float = 2.6,
) -> float:
    """Score a passing or shooting lane from 0 blocked to 1 clear."""

    blockers = list(blockers)
    if not blockers:
        return 1.0
    min_dist = min(segment_distance(blocker, start, end) for blocker in blockers)
    return float(np.clip(min_dist / max(block_dist, 1e-6), 0.0, 1.0))


def legal_support_point(point: Sequence[float], side: str, config: SupporterConfig) -> bool:
    """Return whether a receive/support point is legal and useful for SSL play."""

    x, y = float(point[0]), float(point[1])
    if x < FIELD_X[0] + config.field_margin or x > FIELD_X[1] - config.field_margin:
        return False
    if y < FIELD_Y[0] + config.field_margin or y > FIELD_Y[1] - config.field_margin:
        return False
    if abs(y) > config.max_receive_y_abs:
        return False
    if side == "left" and x > config.max_receive_x_abs:
        return False
    if side == "right" and x < -config.max_receive_x_abs:
        return False
    return True


def attack_goal_for_side(side: str) -> Point:
    """Return the opponent goal center for a team defending ``side``."""

    return (float(GOAL_R[0]), float(GOAL_R[1])) if side == "left" else (float(GOAL_L[0]), float(GOAL_L[1]))


def _extract_pose(game_state: GameState, team_name: str, robot_id: int) -> Optional[Pose]:
    """Read one robot pose from a ``GameState``.

    ``GameState`` stores headings in **degrees** (see ``sim_get_robot_poses``);
    every consumer in the env converts via ``np.deg2rad`` before doing geometry.
    The supporter's ``goto``/``kick`` helpers expect **radians**, so convert here
    once at the boundary — otherwise a degree value (e.g. 121.85) is treated as
    radians and ``goto`` emits a ~120-radian dash direction, sending the robot
    off the field.
    """

    for entry in getattr(game_state, "robot_poses", {}).get(team_name, []):
        if not isinstance(entry, dict):
            continue
        pose = entry.get(int(robot_id))
        if pose is not None and len(pose) >= 3:
            return (float(pose[0]), float(pose[1]), math.radians(float(pose[2])))
    return None


def _team_poses(game_state: GameState, team_name: str, *, exclude: set[int] | None = None) -> list[tuple[int, Pose]]:
    """Return all visible robot poses for a team, optionally excluding ids.

    Headings are converted from ``GameState`` degrees to radians (see
    ``_extract_pose``).
    """

    excluded = exclude or set()
    poses: list[tuple[int, Pose]] = []
    for entry in getattr(game_state, "robot_poses", {}).get(team_name, []):
        if not isinstance(entry, dict):
            continue
        for rid, pose in entry.items():
            if int(rid) in excluded or pose is None or len(pose) < 3:
                continue
            poses.append((int(rid), (float(pose[0]), float(pose[1]), math.radians(float(pose[2])))))
    return poses


def _estimate_ball_velocity(ball_history: Sequence[Point]) -> Point:
    """Estimate next-step ball velocity from recent position deltas."""

    if len(ball_history) < 2:
        return (0.0, 0.0)
    deltas = [
        (ball_history[i + 1][0] - ball_history[i][0], ball_history[i + 1][1] - ball_history[i][1])
        for i in range(len(ball_history) - 1)
    ]
    recent = deltas[-3:]
    weights = list(range(1, len(recent) + 1))
    denom = float(sum(weights))
    vx = sum(w * d[0] for w, d in zip(weights, recent)) / denom
    vy = sum(w * d[1] for w, d in zip(weights, recent)) / denom
    return (0.94 * vx, 0.94 * vy)


class HardcodedSupporter(Player):
    """One supporting attacker for a Stage-3-policy-plus-supporter hybrid.

    The supporter has three priorities, in order:

    1. If it controls the ball, shoot through the best legal lane or return the
       ball to the learned attacker.
    2. If the ball is already moving toward it, move to the earliest reachable
       interception point.
    3. Otherwise, hold a Sumatra-style receive lane that is separated from the
       carrier, legal, and useful for a continuation shot.
    """

    def __init__(
        self,
        teamname: str,
        unum: int,
        *,
        side: str = "left",
        main_attacker_robot_id: int = 1,
        opponent_team_name: str = "TeamB",
        opponent_goalie_robot_ids: Sequence[int] = (1,),
        config: SupporterConfig | None = None,
    ) -> None:
        super().__init__(teamname=teamname, unum=unum)
        self.side = str(side)
        self.main_attacker_robot_id = int(main_attacker_robot_id)
        self.opponent_team_name = str(opponent_team_name)
        self.opponent_goalie_robot_ids = tuple(int(rid) for rid in opponent_goalie_robot_ids)
        self.config = config or SupporterConfig()
        self._ball_history: list[Point] = []
        self._last_count: int | None = None
        self._latched_support: Point | None = None
        self._latched_until_count: int = -1
        self._last_touch_count: int = -10_000
        self.last_event: Dict[str, Any] = {"label": "hc_support_init"}

    def action(self, game_state: GameState) -> str:
        """Return one simulator command for the current world state."""

        if game_state is None or getattr(game_state, "ball_pos", None) is None:
            self._set_event("hc_support_no_state")
            return "turn 0"
        self_pose = _extract_pose(game_state, self.teamname, self.unum)
        if self_pose is None:
            self._set_event("hc_support_no_pose")
            return "turn 0"

        ball = (float(game_state.ball_pos[0]), float(game_state.ball_pos[1]))
        self._update_ball_history(int(getattr(game_state, "count", 0)), ball)

        main_pose = _extract_pose(game_state, self.teamname, self.main_attacker_robot_id)
        opponent_poses = [pose for _, pose in _team_poses(game_state, self.opponent_team_name)]
        goalie_poses = [
            pose
            for rid, pose in _team_poses(game_state, self.opponent_team_name)
            if rid in self.opponent_goalie_robot_ids
        ]

        if self._has_usable_ball(self_pose, ball):
            self._last_touch_count = int(getattr(game_state, "count", 0))
            return self._finish_or_return(self_pose, ball, main_pose, opponent_poses, goalie_poses, game_state)

        intercept = self._receive_intercept(self_pose, ball)
        if intercept is not None:
            cmd = self._move_to_receive_pose(self_pose, intercept, ball, game_state)
            self._set_event(
                "hc_receive_intercept",
                target=intercept,
                command=cmd,
                ball=ball,
            )
            return cmd

        if self._recently_touched(game_state):
            target = self._clearout_target(ball, main_pose)
            label = "hc_clearout_move"
        else:
            target = self._support_target(self_pose, ball, main_pose, opponent_poses, goalie_poses, game_state)
            label = "hc_support_move"
        cmd = self._move_to_receive_pose(self_pose, target, ball, game_state)
        self._set_event(label, target=target, command=cmd, ball=ball)
        return cmd

    def _set_event(self, label: str, **details: Any) -> None:
        """Store a compact diagnostic for the last hardcoded-supporter command."""

        self.last_event = {
            "label": str(label),
            "robot_id": int(self.unum),
            "main_attacker_robot_id": int(self.main_attacker_robot_id),
            **details,
        }

    def _update_ball_history(self, count: int, ball: Point) -> None:
        """Track recent ball samples once per simulator cycle."""

        if self._last_count == count:
            return
        self._last_count = count
        self._ball_history.append(ball)
        if len(self._ball_history) > 8:
            self._ball_history.pop(0)

    def _has_usable_ball(self, self_pose: Pose, ball: Point) -> bool:
        """Return true when the supporter can start a finish/pass action."""

        if distance(self_pose, ball) > KICKABLE_DISTANCE + 0.25:
            return False
        return True

    def _finish_or_return(
        self,
        self_pose: Pose,
        ball: Point,
        main_pose: Optional[Pose],
        opponent_poses: Sequence[Pose],
        goalie_poses: Sequence[Pose],
        game_state: GameState,
    ) -> str:
        """Shoot if possible, otherwise return the ball to the learned attacker."""

        shot_target, shot_quality = self._best_shot(ball, opponent_poses, goalie_poses)
        shot_lane = lane_clear_quality(
            ball,
            shot_target,
            opponent_poses,
            block_dist=self.config.lane_block_dist,
        )
        if shot_quality >= self.config.shoot_min_quality and shot_lane >= self.config.shoot_min_lane_clear:
            return self._kick_or_settle(
                self_pose, ball, shot_target, self.config.shoot_power, game_state,
                label_prefix="hc_shot",
                quality=shot_quality,
            )

        if main_pose is not None:
            lane_q = lane_clear_quality(ball, main_pose, opponent_poses, block_dist=self.config.lane_block_dist)
            if lane_q >= self.config.return_pass_min_quality and distance(ball, main_pose) >= self.config.min_pass_distance:
                return self._kick_or_settle(
                    self_pose, ball, main_pose[:2], self.config.pass_power, game_state,
                    label_prefix="hc_pass",
                    quality=lane_q,
                )

        staging = self._local_staging_point(ball, opponent_poses, goalie_poses)
        return self._kick_or_settle(
            self_pose, ball, staging, 35.0, game_state,
            label_prefix="hc_staging",
            quality=shot_quality,
        )

    def _kick_or_settle(
        self,
        self_pose: Pose,
        ball: Point,
        target: Sequence[float],
        power: float,
        game_state: GameState,
        *,
        label_prefix: str,
        quality: float,
    ) -> str:
        """Kick through the physical front cone, or move to a legal contact pose."""

        target_angle = math.atan2(float(target[1]) - self_pose[1], float(target[0]) - self_pose[0])
        cmd = self.kick(target_angle, self_pose, ball, kick_power=int(round(power)))
        if cmd != "failed":
            self._set_event(
                f"{label_prefix}_fired" if cmd.startswith("kick") else f"{label_prefix}_align",
                target=(float(target[0]), float(target[1])),
                quality=float(quality),
                power=float(power),
                command=cmd,
                ball=ball,
            )
            return cmd

        contact = self._contact_pose_for_target(ball, target)
        cmd = self.goto(
            contact[0],
            contact[1],
            self_pose,
            game_state,
            margin=0.18,
            theta=contact[2],
            speed=self.config.settle_speed,
            detour_margin=1.0,
        )
        self._set_event(
            f"{label_prefix}_settle",
            target=(float(target[0]), float(target[1])),
            contact_pose=contact,
            quality=float(quality),
            power=float(power),
            command=cmd,
            ball=ball,
        )
        return cmd

    def _contact_pose_for_target(self, ball: Point, target: Sequence[float]) -> Pose:
        """Return a pose behind the ball relative to the target direction."""

        dx = float(target[0]) - ball[0]
        dy = float(target[1]) - ball[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            ux, uy = (1.0 if self.side == "left" else -1.0), 0.0
        else:
            ux, uy = dx / norm, dy / norm
        stand_off = PLAYER_SIZE + BALL_SIZE + 0.08
        return (ball[0] - ux * stand_off, ball[1] - uy * stand_off, math.atan2(uy, ux))

    def _receive_intercept(self, self_pose: Pose, ball: Point) -> Optional[Point]:
        """Return an interception point if a moving ball is arriving nearby."""

        vx, vy = _estimate_ball_velocity(self._ball_history)
        speed = math.hypot(vx, vy)
        if speed < 0.08:
            return None

        best: tuple[float, Point] | None = None
        px, py = ball
        cvx, cvy = vx, vy
        for step in range(1, self.config.receive_intercept_horizon + 1):
            px += cvx
            py += cvy
            cvx *= 0.94
            cvy *= 0.94
            candidate = (px, py)
            if not legal_support_point(candidate, self.side, self.config):
                continue
            ball_time = float(step)
            robot_time = distance(self_pose, candidate) / 1.45
            miss = distance(self_pose, candidate)
            if robot_time <= ball_time + 2.0 and miss <= self.config.receive_intercept_radius:
                score = robot_time - ball_time
                if best is None or score < best[0]:
                    best = (score, candidate)
        return best[1] if best is not None else None

    def _support_target(
        self,
        self_pose: Pose,
        ball: Point,
        main_pose: Optional[Pose],
        opponent_poses: Sequence[Pose],
        goalie_poses: Sequence[Pose],
        game_state: GameState,
    ) -> Point:
        """Choose and latch a receive/support point for stable off-ball motion."""

        count = int(getattr(game_state, "count", 0))
        if self._latched_support is not None and count <= self._latched_until_count:
            if self._support_point_score(self._latched_support, self_pose, ball, main_pose, opponent_poses, goalie_poses) > 0.20:
                return self._latched_support

        target = self._best_support_candidate(self_pose, ball, main_pose, opponent_poses, goalie_poses)
        self._latched_support = target
        self._latched_until_count = count + self.config.support_hysteresis_steps
        return target

    def _best_support_candidate(
        self,
        self_pose: Pose,
        ball: Point,
        main_pose: Optional[Pose],
        opponent_poses: Sequence[Pose],
        goalie_poses: Sequence[Pose],
    ) -> Point:
        """Search a compact forward/lateral receive wedge."""

        sign = 1.0 if self.side == "left" else -1.0
        anchor = main_pose[:2] if main_pose is not None else ball
        best_score = -1e18
        best_point = self._fallback_support(ball)

        for forward in (8.0, 12.0, 16.0, 20.0):
            for lateral in (-10.0, -6.0, -3.0, 3.0, 6.0, 10.0):
                point = (ball[0] + sign * forward, ball[1] + lateral)
                point = self._clamp_support_point(point)
                if not legal_support_point(point, self.side, self.config):
                    continue
                if distance(point, ball) < self.config.min_pass_distance:
                    continue
                if main_pose is not None and distance(point, main_pose) < self.config.teammate_keepout:
                    continue
                if any(distance(point, opp) < self.config.opponent_keepout for opp in opponent_poses):
                    continue
                if distance(point, anchor) < self.config.min_pass_distance * 0.65:
                    continue

                score = self._support_point_score(point, self_pose, ball, main_pose, opponent_poses, goalie_poses)
                if score > best_score:
                    best_score = score
                    best_point = point
        return best_point

    def _support_point_score(
        self,
        point: Point,
        self_pose: Pose,
        ball: Point,
        main_pose: Optional[Pose],
        opponent_poses: Sequence[Pose],
        goalie_poses: Sequence[Pose],
    ) -> float:
        """Score a receive point by lane, continuation shot, safety, and travel."""

        passer = main_pose[:2] if main_pose is not None else ball
        pass_q = lane_clear_quality(passer, point, opponent_poses, block_dist=self.config.lane_block_dist)
        _, shot_q = self._best_shot(point, opponent_poses, goalie_poses)
        nearest_opp = min((distance(point, opp) for opp in opponent_poses), default=10.0)
        safety = float(np.clip(nearest_opp / 8.0, 0.0, 1.0))
        travel_cost = min(distance(self_pose, point) / 30.0, 1.0)
        center_bias = 1.0 - min(abs(point[1]) / self.config.max_receive_y_abs, 1.0) * 0.25
        return 0.42 * pass_q + 0.34 * shot_q + 0.16 * safety + 0.08 * center_bias - 0.18 * travel_cost

    def _best_shot(
        self,
        point: Point,
        opponent_poses: Sequence[Pose],
        goalie_poses: Sequence[Pose],
    ) -> tuple[Point, float]:
        """Return the best in-mouth shot target and its quality."""

        goal_x, _ = attack_goal_for_side(self.side)
        best_target = (goal_x, 0.0)
        best_quality = -1.0
        for gy in DEFAULT_GOAL_TARGETS:
            target = (goal_x, gy)
            lane_q = lane_clear_quality(point, target, opponent_poses, block_dist=self.config.lane_block_dist)
            goalie_gap = 1.0
            if goalie_poses:
                goalie_gap = max(min(abs(gy - goalie[1]) / 5.0, 1.0) for goalie in goalie_poses)
            distance_factor = 1.0 - min(distance(point, target) / 55.0, 0.65)
            quality = 0.55 * lane_q + 0.30 * goalie_gap + 0.15 * distance_factor
            if quality > best_quality:
                best_quality = quality
                best_target = target
        return best_target, float(np.clip(best_quality, 0.0, 1.0))

    def _local_staging_point(
        self,
        ball: Point,
        opponent_poses: Sequence[Pose],
        goalie_poses: Sequence[Pose],
    ) -> Point:
        """Pick a short legal touch point that improves the next shot."""

        sign = 1.0 if self.side == "left" else -1.0
        candidates = [
            self._clamp_support_point((ball[0] + sign * fwd, ball[1] + lat))
            for fwd in (2.0, 4.0, 6.0)
            for lat in (-3.0, 0.0, 3.0)
        ]
        legal = [p for p in candidates if legal_support_point(p, self.side, self.config)]
        if not legal:
            return self._clamp_support_point((ball[0] + sign * 4.0, ball[1]))
        return max(legal, key=lambda p: self._best_shot(p, opponent_poses, goalie_poses)[1])

    def _fallback_support(self, ball: Point) -> Point:
        """Return a safe fallback support point when no candidate scores well."""

        sign = 1.0 if self.side == "left" else -1.0
        lateral = -6.0 if ball[1] > 0.0 else 6.0
        return self._clamp_support_point((ball[0] + sign * 12.0, ball[1] + lateral))

    def _clamp_support_point(self, point: Point) -> Point:
        """Clamp a target inside field and opponent-defense-area buffers."""

        x = float(np.clip(point[0], FIELD_X[0] + self.config.field_margin, FIELD_X[1] - self.config.field_margin))
        y = float(np.clip(point[1], FIELD_Y[0] + self.config.field_margin, FIELD_Y[1] - self.config.field_margin))
        y = float(np.clip(y, -self.config.max_receive_y_abs, self.config.max_receive_y_abs))
        if self.side == "left":
            x = min(x, self.config.max_receive_x_abs)
        else:
            x = max(x, -self.config.max_receive_x_abs)
        return (x, y)

    def _move_to_receive_pose(self, self_pose: Pose, target: Point, ball: Point, game_state: GameState) -> str:
        """Move to a receive point and face the current ball/carrier direction."""

        theta = math.atan2(ball[1] - self_pose[1], ball[0] - self_pose[0])
        return self.goto(
            target[0],
            target[1],
            self_pose,
            game_state,
            margin=self.config.receive_arrival_margin,
            theta=theta,
            speed=95.0,
            detour_margin=1.2,
        )

    def _recently_touched(self, game_state: GameState) -> bool:
        """Return whether the supporter should clear out after using the ball."""

        count = int(getattr(game_state, "count", 0))
        return count - self._last_touch_count <= self.config.clearout_after_touch_steps

    def _clearout_target(self, ball: Point, main_pose: Optional[Pose]) -> Point:
        """Move away from the ball after a pass/shot so robot 1 can continue."""

        sign = 1.0 if self.side == "left" else -1.0
        base_y = ball[1]
        if main_pose is not None:
            base_y = -8.0 if main_pose[1] >= ball[1] else 8.0
        return self._clamp_support_point((ball[0] - sign * 6.0, base_y))

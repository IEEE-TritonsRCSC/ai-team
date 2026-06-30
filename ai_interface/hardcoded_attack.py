"""Hardcoded attacking bridge for hybrid Stage 4 play.

The Stage-3 PPO policy is a solo finisher.  It was trained from a narrow
attacking envelope, so the hybrid controller should not hand the ball to PPO
from arbitrary receive points.  This module owns the small deterministic bridge:
shoot immediately when the lane is good, otherwise make one short SSL-compliant
staging carry toward the Stage-3 envelope and re-evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ai_interface.constants.field_constants import FIELD_X, FIELD_Y, dt
from ai_interface.constants.player_constants import BALL_SIZE, KICKABLE_MARGIN, PLAYER_SIZE
from ai_interface.hardcoded_supporter import (
    DEFAULT_GOAL_TARGETS,
    Point,
    Pose,
    attack_goal_for_side,
    distance,
    lane_clear_quality,
)
from ai_interface.utils.algo_utils import normalize_angle
from ai_interface.utils.basic_commands import (
    DribbleState,
    approach_ball,
    ball_in_front_reception_cone,
    dribble,
    dribble_to,
    goto,
    kick,
)
from networking.data_utils import GameState, limit_turn_rate


KICKABLE_DISTANCE = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE


@dataclass(frozen=True)
class HardcodedAttackConfig:
    """Geometry and thresholds for the Stage-4 hardcoded attack bridge."""

    stage3_min_x: float = 25.0
    stage3_max_x: float = 34.0
    stage3_max_abs_y: float = 10.0
    stage3_min_quality: float = 0.25
    shoot_min_quality: float = 0.45
    shoot_min_lane_clear: float = 0.95
    advance_min_lane_clear: float = 0.85
    direct_shot_min_x: float = 22.0
    bridge_min_x: float = 17.0
    bridge_max_x: float = 36.0
    bridge_max_abs_y: float = 16.0
    advance_target_y_abs: float = 6.0
    advance_power: float = 85.0
    lane_block_dist: float = 2.6
    shot_lane_block_dist: float = 4.5
    field_margin: float = 2.0
    dribble_segment_limit: float = 8.5
    staging_forward: Tuple[float, ...] = (4.0, 6.0, 8.0)
    staging_lateral: Tuple[float, ...] = (-4.0, 0.0, 4.0)
    staging_arrival_margin: float = 0.6
    staging_speed: float = 70.0
    settle_speed: float = 75.0
    shoot_power: float = 100.0
    clearout_dist: float = 6.0


@dataclass
class HardcodedAttackStatus:
    """Result of evaluating whether PPO may safely take over."""

    ready: bool
    reason: str
    shot_quality: float
    in_stage3_x: bool
    in_stage3_y: bool
    in_reception_cone: bool
    robot_ball_dist: Optional[float]


@dataclass
class HardcodedAttackCoordinator:
    """Run hardcoded shoot/stage commands until the ball is PPO-ready."""

    team_name: str
    opponent_team_name: str
    side: str = "left"
    config: HardcodedAttackConfig = field(default_factory=HardcodedAttackConfig)
    dribble_states: Dict[int, DribbleState] = field(default_factory=dict)
    caught_for_kick: Dict[int, bool] = field(default_factory=dict)
    last_events: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        """Clear per-episode macro state."""

        self.last_events = []
        self.caught_for_kick.clear()
        for state in self.dribble_states.values():
            state.reset()

    def stage3_readiness(
        self,
        *,
        robot_id: int,
        pose_by_robot_id: Dict[int, Any],
        game_state: GameState,
        kickable_dist: float = KICKABLE_DISTANCE,
    ) -> HardcodedAttackStatus:
        """Return whether the current ball state matches Stage-3 PPO training."""

        ball_raw = getattr(game_state, "ball_pos", None)
        pose_raw = pose_by_robot_id.get(int(robot_id))
        if ball_raw is None or pose_raw is None:
            return HardcodedAttackStatus(False, "missing_state", 0.0, False, False, False, None)

        ball = (float(ball_raw[0]), float(ball_raw[1]))
        pose = _pose_rad(pose_raw)
        _, defenders, goalie_pose = self._opponents(game_state)
        shot_target, shot_quality = self.best_shot(ball, defenders, [goalie_pose] if goalie_pose else [])
        _ = shot_target

        x = ball[0] if self.side == "left" else -ball[0]
        in_stage3_x = self.config.stage3_min_x <= x <= self.config.stage3_max_x
        in_stage3_y = abs(ball[1]) <= self.config.stage3_max_abs_y
        robot_ball_dist = distance(pose, ball)
        in_cone = ball_in_front_reception_cone(
            pose,
            ball,
            kickable_tolerance=float(kickable_dist) + 0.15,
        )

        ready = (
            in_stage3_x
            and in_stage3_y
            and robot_ball_dist <= float(kickable_dist) + 0.35
            and in_cone
            and shot_quality >= self.config.stage3_min_quality
        )
        if ready:
            reason = "stage3_ready"
        elif not in_stage3_x:
            reason = "outside_stage3_x"
        elif not in_stage3_y:
            reason = "outside_stage3_y"
        elif robot_ball_dist > float(kickable_dist) + 0.35:
            reason = "ball_not_controlled"
        elif not in_cone:
            reason = "bad_reception_cone"
        else:
            reason = "low_stage3_quality"

        return HardcodedAttackStatus(
            ready=bool(ready),
            reason=reason,
            shot_quality=float(shot_quality),
            in_stage3_x=bool(in_stage3_x),
            in_stage3_y=bool(in_stage3_y),
            in_reception_cone=bool(in_cone),
            robot_ball_dist=float(robot_ball_dist),
        )

    def execute(
        self,
        game_state: Optional[GameState],
        gate_info: Dict[str, Any],
        *,
        kickable_dist: float = KICKABLE_DISTANCE,
    ) -> Tuple[Dict[int, str], list[dict[str, Any]], Dict[str, Any]]:
        """Return per-robot commands for hardcoded attack/staging control."""

        self.last_events = []
        if game_state is None or getattr(game_state, "ball_pos", None) is None:
            return {}, [], {"active": False, "reason": "missing_state"}

        owner_id = gate_info.get("carrier_id", gate_info.get("active_robot_id"))
        if owner_id is None:
            return {}, [], {"active": False, "reason": "missing_owner"}
        owner_id = int(owner_id)
        pose_by_robot_id = _pose_map(game_state, self.team_name)
        owner_pose_raw = pose_by_robot_id.get(owner_id)
        if owner_pose_raw is None:
            return {}, [], {"active": False, "reason": "missing_owner_pose", "carrier_id": owner_id}

        readiness = self.stage3_readiness(
            robot_id=owner_id,
            pose_by_robot_id=pose_by_robot_id,
            game_state=game_state,
            kickable_dist=kickable_dist,
        )
        if readiness.ready:
            event = self._event("hc_attack_handoff_ready", robot_id=owner_id, **readiness.__dict__)
            return {owner_id: "turn 0"}, [event], {
                "active": False,
                "handoff_ready": True,
                "carrier_id": owner_id,
                **readiness.__dict__,
            }

        ball = (float(game_state.ball_pos[0]), float(game_state.ball_pos[1]))
        owner_pose = _pose_rad(owner_pose_raw)
        _, defenders, goalie_pose = self._opponents(game_state)
        goalie_poses = [goalie_pose] if goalie_pose is not None else []

        owner_cmd, owner_event = self._owner_command(
            owner_id,
            owner_pose,
            ball,
            defenders,
            goalie_poses,
            game_state,
            readiness,
            kickable_dist=float(kickable_dist),
        )
        commands = {owner_id: owner_cmd}
        events = [owner_event]

        previous_passer = gate_info.get("previous_passer_id")
        if previous_passer is not None and int(previous_passer) in pose_by_robot_id:
            clear_cmd, clear_event = self._clearout_command(
                int(previous_passer),
                _pose_rad(pose_by_robot_id[int(previous_passer)]),
                ball,
                game_state,
            )
            commands[int(previous_passer)] = clear_cmd
            events.append(clear_event)

        for event in events:
            self._event(str(event["label"]), **{k: v for k, v in event.items() if k != "label"})

        return commands, events, {
            "active": True,
            "handoff_ready": False,
            "carrier_id": owner_id,
            "reason": readiness.reason,
            "shot_quality": readiness.shot_quality,
            "in_stage3_x": readiness.in_stage3_x,
            "in_stage3_y": readiness.in_stage3_y,
        }

    def best_shot(
        self,
        point: Point,
        defender_points: Sequence[Point],
        goalie_poses: Sequence[Optional[Pose]],
    ) -> tuple[Point, float]:
        """Return the best in-mouth target and quality from a point."""

        goal_x, _ = attack_goal_for_side(self.side)
        best_target = (goal_x, 0.0)
        best_quality = -1.0
        for gy in DEFAULT_GOAL_TARGETS:
            target = (goal_x, gy)
            lane_q = lane_clear_quality(point, target, defender_points, block_dist=self.config.lane_block_dist)
            goalie_gap = 1.0
            visible_goalies = [g for g in goalie_poses if g is not None]
            if visible_goalies:
                goalie_gap = max(min(abs(gy - goalie[1]) / 5.0, 1.0) for goalie in visible_goalies)
            distance_factor = 1.0 - min(distance(point, target) / 55.0, 0.65)
            quality = 0.55 * lane_q + 0.30 * goalie_gap + 0.15 * distance_factor
            if quality > best_quality:
                best_quality = quality
                best_target = target
        return best_target, float(np.clip(best_quality, 0.0, 1.0))

    def short_staging_target(
        self,
        ball: Point,
        defender_points: Sequence[Point],
        goalie_poses: Sequence[Optional[Pose]],
    ) -> tuple[Point, float]:
        """Pick the best short staging point under the SSL dribble cap."""

        sign = 1.0 if self.side == "left" else -1.0
        best_point = self._clamp_field((ball[0] + sign * 4.0, ball[1]))
        best_score = -1e18
        for forward in self.config.staging_forward:
            for lateral in self.config.staging_lateral:
                dx = sign * float(forward)
                dy = float(lateral)
                length = math.hypot(dx, dy)
                if length > self.config.dribble_segment_limit:
                    scale = self.config.dribble_segment_limit / max(length, 1e-6)
                    dx *= scale
                    dy *= scale
                point = self._clamp_field((ball[0] + dx, ball[1] + dy))
                if not self._legal_staging_point(point):
                    continue
                shot_target, shot_quality = self.best_shot(point, defender_points, goalie_poses)
                lane_q = lane_clear_quality(point, shot_target, defender_points, block_dist=self.config.lane_block_dist)
                stage_score = self._stage3_envelope_score(point)
                progress_score = self._goalward_progress_score(ball, point)
                center_score = 1.0 - min(abs(point[1]) / max(self.config.stage3_max_abs_y, 1e-6), 1.0)
                score = (
                    0.42 * shot_quality
                    + 0.30 * stage_score
                    + 0.14 * lane_q
                    + 0.10 * progress_score
                    + 0.04 * center_score
                )
                if score > best_score:
                    best_score = score
                    best_point = point
        return best_point, float(max(best_score, 0.0))

    def _owner_command(
        self,
        robot_id: int,
        pose: Pose,
        ball: Point,
        defenders: Sequence[Point],
        goalie_poses: Sequence[Optional[Pose]],
        game_state: GameState,
        readiness: HardcodedAttackStatus,
        *,
        kickable_dist: float,
    ) -> Tuple[str, Dict[str, Any]]:
        robot_ball_dist = distance(pose, ball)
        if robot_ball_dist > kickable_dist + 0.35:
            self.caught_for_kick[robot_id] = False
            cmd = approach_ball(
                pose,
                game_state,
                speed=95.0,
                obstacle_avoidance=True,
            )
            if cmd == "done":
                cmd = "turn 0"
            cmd = limit_turn_rate(cmd)
            return cmd, {
                "label": "hc_attack_recover",
                "robot_id": robot_id,
                "command": cmd,
                "reason": readiness.reason,
                "robot_ball_dist": robot_ball_dist,
            }

        shot_target, shot_quality = self.best_shot(ball, defenders, goalie_poses)
        shot_lane = lane_clear_quality(
            ball,
            shot_target,
            defenders,
            block_dist=self.config.shot_lane_block_dist,
        )
        shot_blocker = self._nearest_blocker(ball, shot_target, defenders)
        if (
            self._can_direct_shoot_from(ball)
            and shot_quality >= self.config.shoot_min_quality
            and shot_lane >= self.config.shoot_min_lane_clear
        ):
            return self._kick_or_settle(
                robot_id,
                pose,
                ball,
                shot_target,
                self.config.shoot_power,
                game_state,
                label_prefix="hc_attack_shot",
                quality=shot_quality,
                lane_clear=shot_lane,
                nearest_blocker=shot_blocker,
            )

        if not self._within_bridge_band(ball):
            target = self._advance_target(ball)
            advance_lane = lane_clear_quality(
                ball,
                target,
                defenders,
                block_dist=self.config.shot_lane_block_dist,
            )
            advance_blocker = self._nearest_blocker(ball, target, defenders)
            if advance_lane < self.config.advance_min_lane_clear:
                return self._blocked_lane_stage_command(
                    robot_id,
                    pose,
                    ball,
                    target,
                    defenders,
                    goalie_poses,
                    game_state,
                    readiness,
                    kickable_dist=kickable_dist,
                    lane_clear=advance_lane,
                    nearest_blocker=advance_blocker,
                )
            return self._kick_or_settle(
                robot_id,
                pose,
                ball,
                target,
                self.config.advance_power,
                game_state,
                label_prefix="hc_attack_advance",
                quality=shot_quality,
                lane_clear=advance_lane,
                nearest_blocker=advance_blocker,
            )

        target, target_score = self.short_staging_target(ball, defenders, goalie_poses)
        self.caught_for_kick[robot_id] = False
        state = self.dribble_states.setdefault(robot_id, DribbleState())
        state.committed = True
        cmd = dribble_to(
            pose,
            ball,
            target,
            game_state,
            state=state,
            arrival_margin=self.config.staging_arrival_margin,
            segment_limit=self.config.dribble_segment_limit,
            speed=self.config.staging_speed,
            kickable_tolerance=kickable_dist,
            stall_limit=18,
        )
        if cmd == "done":
            state.reset()
            contact = self._contact_pose_for_target(ball, target)
            cmd = goto(
                pose,
                contact[0],
                contact[1],
                game_state,
                margin=0.18,
                theta=contact[2],
                speed=self.config.settle_speed,
                detour_margin=1.0,
                obstacle_avoidance=True,
                include_ball_obstacle=False,
                include_player_obstacles=True,
            )
        cmd = limit_turn_rate(cmd if cmd != "done" else "turn 0")
        return cmd, {
            "label": "hc_attack_stage",
            "robot_id": robot_id,
            "command": cmd,
            "target": target,
            "target_score": target_score,
            "reason": readiness.reason,
            "shot_quality": shot_quality,
            "shot_lane_clear": shot_lane,
            "shot_blocked_by": shot_blocker,
            "dribble_phase": getattr(state, "phase", None),
        }

    def _kick_or_settle(
        self,
        robot_id: int,
        pose: Pose,
        ball: Point,
        target: Point,
        power: float,
        game_state: GameState,
        *,
        label_prefix: str,
        quality: float,
        lane_clear: Optional[float] = None,
        nearest_blocker: Optional[Point] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        target_angle = math.atan2(target[1] - pose[1], target[0] - pose[0])
        heading_error = float(normalize_angle(target_angle - pose[2]))
        in_mouth = ball_in_front_reception_cone(pose, ball)

        # Mirror the hardcoded pass macro: first catch/glue the ball, then rotate
        # the glued ball+robot assembly toward the target. A bare target-align
        # turn while the ball is merely touching the mouth can spin away from the
        # ball and looks like the robot is ignoring an easy grab.
        if self.caught_for_kick.get(robot_id, False):
            if in_mouth and abs(heading_error) <= math.radians(5.0):
                cmd = f"kick {float(power):.1f} 0"
                self.caught_for_kick[robot_id] = False
            else:
                cmd = f"turn {heading_error / dt}"
            cmd = limit_turn_rate(cmd)
            return cmd, {
                "label": f"{label_prefix}_fired" if cmd.startswith("kick") else f"{label_prefix}_align",
                "robot_id": robot_id,
                "command": cmd,
                "target": target,
                "quality": float(quality),
                "power": float(power),
                "heading_error": heading_error,
                "lane_clear": lane_clear,
                "nearest_blocker": nearest_blocker,
            }

        grab = dribble(pose, ball)
        if grab != "failed":
            if grab == "catch 0":
                self.caught_for_kick[robot_id] = True
            cmd = limit_turn_rate(grab)
            return cmd, {
                "label": f"{label_prefix}_grab",
                "robot_id": robot_id,
                "command": cmd,
                "target": target,
                "quality": float(quality),
                "power": float(power),
                "heading_error": heading_error,
                "lane_clear": lane_clear,
                "nearest_blocker": nearest_blocker,
            }

        self.caught_for_kick[robot_id] = False
        contact = self._contact_pose_for_target(ball, target)
        cmd = goto(
            pose,
            contact[0],
            contact[1],
            game_state,
            margin=0.18,
            theta=contact[2],
            speed=self.config.settle_speed,
            detour_margin=1.0,
            obstacle_avoidance=True,
            include_ball_obstacle=False,
            include_player_obstacles=True,
        )
        if cmd == "done":
            cmd = "turn 0"
        cmd = limit_turn_rate(cmd)
        return cmd, {
            "label": f"{label_prefix}_settle",
            "robot_id": robot_id,
            "command": cmd,
            "target": target,
            "contact_pose": contact,
            "quality": float(quality),
            "power": float(power),
            "heading_error": heading_error,
            "lane_clear": lane_clear,
            "nearest_blocker": nearest_blocker,
        }

    def _blocked_lane_stage_command(
        self,
        robot_id: int,
        pose: Pose,
        ball: Point,
        blocked_target: Point,
        defenders: Sequence[Point],
        goalie_poses: Sequence[Optional[Pose]],
        game_state: GameState,
        readiness: HardcodedAttackStatus,
        *,
        kickable_dist: float,
        lane_clear: float,
        nearest_blocker: Optional[Point],
    ) -> Tuple[str, Dict[str, Any]]:
        """Move the ball laterally/forward when an advance kick lane is blocked."""

        self.caught_for_kick[robot_id] = False
        target, target_score = self.short_staging_target(ball, defenders, goalie_poses)
        state = self.dribble_states.setdefault(robot_id, DribbleState())
        state.committed = True
        cmd = dribble_to(
            pose,
            ball,
            target,
            game_state,
            state=state,
            arrival_margin=self.config.staging_arrival_margin,
            segment_limit=self.config.dribble_segment_limit,
            speed=self.config.staging_speed,
            kickable_tolerance=kickable_dist,
            stall_limit=18,
        )
        if cmd == "done":
            state.reset()
            contact = self._contact_pose_for_target(ball, target)
            cmd = goto(
                pose,
                contact[0],
                contact[1],
                game_state,
                margin=0.18,
                theta=contact[2],
                speed=self.config.settle_speed,
                detour_margin=1.0,
                obstacle_avoidance=True,
                include_ball_obstacle=False,
                include_player_obstacles=True,
            )
        cmd = limit_turn_rate(cmd if cmd != "done" else "turn 0")
        return cmd, {
            "label": "hc_attack_blocked_lane_stage",
            "robot_id": robot_id,
            "command": cmd,
            "target": target,
            "blocked_target": blocked_target,
            "target_score": target_score,
            "reason": readiness.reason,
            "lane_clear": float(lane_clear),
            "nearest_blocker": nearest_blocker,
            "dribble_phase": getattr(state, "phase", None),
        }

    def _clearout_command(
        self,
        robot_id: int,
        pose: Pose,
        ball: Point,
        game_state: GameState,
    ) -> Tuple[str, Dict[str, Any]]:
        sign = 1.0 if self.side == "left" else -1.0
        lateral = -self.config.clearout_dist if pose[1] >= ball[1] else self.config.clearout_dist
        target = self._clamp_field((ball[0] - sign * self.config.clearout_dist, ball[1] + lateral))
        theta = math.atan2(ball[1] - pose[1], ball[0] - pose[0])
        cmd = goto(
            pose,
            target[0],
            target[1],
            game_state,
            margin=1.0,
            theta=theta,
            speed=90.0,
            detour_margin=1.2,
            obstacle_avoidance=True,
            include_ball_obstacle=False,
            include_player_obstacles=True,
        )
        if cmd == "done":
            cmd = "turn 0"
        cmd = limit_turn_rate(cmd)
        return cmd, {
            "label": "hc_attack_clearout",
            "robot_id": robot_id,
            "command": cmd,
            "target": target,
        }

    def _event(self, label: str, **details: Any) -> Dict[str, Any]:
        event = {"label": str(label), **details}
        self.last_events.append(event)
        return event

    def _opponents(
        self,
        game_state: GameState,
    ) -> Tuple[Dict[int, Any], list[Point], Optional[Pose]]:
        pose_map = _pose_map(game_state, self.opponent_team_name)
        defenders: list[Point] = []
        goalie_pose: Optional[Pose] = None
        for rid, pose in pose_map.items():
            if len(pose) < 2:
                continue
            point = (float(pose[0]), float(pose[1]))
            if int(rid) == 1 and goalie_pose is None and len(pose) >= 3:
                goalie_pose = _pose_rad(pose)
            else:
                defenders.append(point)
        return pose_map, defenders, goalie_pose

    def _contact_pose_for_target(self, ball: Point, target: Sequence[float]) -> Pose:
        dx = float(target[0]) - ball[0]
        dy = float(target[1]) - ball[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            ux, uy = (1.0 if self.side == "left" else -1.0), 0.0
        else:
            ux, uy = dx / norm, dy / norm
        stand_off = PLAYER_SIZE + BALL_SIZE + 0.08
        return (ball[0] - ux * stand_off, ball[1] - uy * stand_off, math.atan2(uy, ux))

    def _nearest_blocker(
        self,
        start: Point,
        end: Point,
        blockers: Sequence[Point],
    ) -> Optional[Point]:
        """Return the blocker nearest to a lane segment, if any."""

        if not blockers:
            return None
        sx, sy = float(start[0]), float(start[1])
        ex, ey = float(end[0]), float(end[1])
        vx, vy = ex - sx, ey - sy
        seg_len_sq = vx * vx + vy * vy
        best: Optional[Point] = None
        best_dist = float("inf")
        for blocker in blockers:
            bx, by = float(blocker[0]), float(blocker[1])
            if seg_len_sq <= 1e-9:
                dist = math.hypot(bx - sx, by - sy)
            else:
                t = max(0.0, min(1.0, ((bx - sx) * vx + (by - sy) * vy) / seg_len_sq))
                cx, cy = sx + t * vx, sy + t * vy
                dist = math.hypot(bx - cx, by - cy)
            if dist < best_dist:
                best_dist = dist
                best = (bx, by)
        return best

    def _clamp_field(self, point: Point) -> Point:
        x = float(np.clip(point[0], FIELD_X[0] + self.config.field_margin, FIELD_X[1] - self.config.field_margin))
        y = float(np.clip(point[1], FIELD_Y[0] + self.config.field_margin, FIELD_Y[1] - self.config.field_margin))
        return (x, y)

    def _legal_staging_point(self, point: Point) -> bool:
        x, y = point
        if x < FIELD_X[0] + self.config.field_margin or x > FIELD_X[1] - self.config.field_margin:
            return False
        if y < FIELD_Y[0] + self.config.field_margin or y > FIELD_Y[1] - self.config.field_margin:
            return False
        return True

    def _signed_x(self, point: Point) -> float:
        return float(point[0]) if self.side == "left" else -float(point[0])

    def _within_bridge_band(self, point: Point) -> bool:
        x = self._signed_x(point)
        return (
            x >= float(self.config.bridge_min_x)
            and x <= float(self.config.bridge_max_x)
            and abs(float(point[1])) <= float(self.config.bridge_max_abs_y)
        )

    def _can_direct_shoot_from(self, point: Point) -> bool:
        return self._signed_x(point) >= float(self.config.direct_shot_min_x)

    def _advance_target(self, ball: Point) -> Point:
        """Pick a deterministic long advance point near the Stage-3 envelope."""

        sign = 1.0 if self.side == "left" else -1.0
        signed_x = self._signed_x(ball)
        target_x_signed = max(float(self.config.stage3_min_x), signed_x + self.config.dribble_segment_limit)
        target_x_signed = min(target_x_signed, float(self.config.stage3_max_x))
        target_y = float(np.clip(ball[1], -self.config.advance_target_y_abs, self.config.advance_target_y_abs))
        return self._clamp_field((sign * target_x_signed, target_y))

    def _stage3_envelope_score(self, point: Point) -> float:
        x = point[0] if self.side == "left" else -point[0]
        if x < self.config.stage3_min_x:
            x_gap = self.config.stage3_min_x - x
        elif x > self.config.stage3_max_x:
            x_gap = x - self.config.stage3_max_x
        else:
            x_gap = 0.0
        y_gap = max(0.0, abs(point[1]) - self.config.stage3_max_abs_y)
        score = 1.0 - min(1.0, 0.07 * x_gap + 0.08 * y_gap)
        return float(np.clip(score, 0.0, 1.0))

    def _goalward_progress_score(self, start: Point, point: Point) -> float:
        sign = 1.0 if self.side == "left" else -1.0
        return float(np.clip((point[0] - start[0]) * sign / self.config.dribble_segment_limit, 0.0, 1.0))


def _pose_map(game_state: GameState, team_name: str) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    for entry in getattr(game_state, "robot_poses", {}).get(team_name, []) or []:
        if isinstance(entry, dict):
            out.update({int(k): v for k, v in entry.items()})
    return out


def _pose_rad(pose: Sequence[float]) -> Pose:
    return (float(pose[0]), float(pose[1]), math.radians(float(pose[2])))

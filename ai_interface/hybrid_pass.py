"""Hardcoded pass lifecycle for the Stage-4 hybrid fallback.

The PPO policy is treated as a solo finisher.  This module owns the brittle
team-play bridge around that policy: decide a concrete receiver target from the
env gate, align/fire the carrier pass, hold the receiver on the pass line, then
hand control back to PPO only after the receiver has stable front-cone contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ai_interface.constants.field_constants import BALL_DECAY, FIELD_X, FIELD_Y, dt
from ai_interface.constants.player_constants import BALL_SIZE, PLAYER_SIZE
from ai_interface.hardcoded_supporter import (
    KICKABLE_DISTANCE,
    Point,
    Pose,
    SupporterConfig,
    attack_goal_for_side,
    distance,
)
from ai_interface.utils.algo_utils import normalize_angle
from ai_interface.utils.basic_commands import (
    ball_in_front_reception_cone,
    dribble,
    goto,
)
from networking.data_utils import GameState, limit_turn_rate


PHASE_PREPARE = "prepare"
PHASE_RECEIVE = "receive"


@dataclass
class HybridPassConfig:
    """Constants for one hardcoded pass lifecycle.

    Units follow the rest of the environment: ``10 env units = 1 real meter`` and
    one simulator step is ``0.1s``.  The simulator's ball-speed cap is therefore
    ``4 m/s = 4 env units/step``.
    """

    receive_margin: float = 1.15
    settle_margin: float = 0.18
    contact_standoff: float = PLAYER_SIZE + BALL_SIZE + 0.08
    min_pass_power: float = 35.0
    max_pass_power: float = 100.0
    max_ball_speed: float = 4.0
    target_receive_speed: float = 1.25
    # Body-rotation budget for the carrier to face the pass lane before firing.
    # The carrier already holds the ball when a pass starts, so this phase is
    # almost pure in-place rotation, but the server caps turn rate at 2 deg/cycle
    # (MAXMOMENT, 20 deg/s @ 0.1s). kick() fires within a 5 deg tolerance, so a
    # worst-case ~180 deg correction needs ceil((180-5)/2) = 88 cycles. The old
    # value of 48 only covered ~100 deg, timing out ~half of all passes. 110
    # gives full 180 deg coverage plus margin for ball drift / minor re-approach.
    max_align_steps: int = 110
    receive_timeout_steps: int = 62
    stable_receive_frames: int = 2
    receiver_intercept_horizon: int = 22
    receiver_speed_units_per_step: float = 1.45
    carrier_clearout_dist: float = 6.0
    opponent_intercept_dist: float = KICKABLE_DISTANCE + 0.65
    attacking_min_receive_x_abs: float = 25.0
    attacking_max_receive_y_abs: float = 10.0


@dataclass
class HybridPassState:
    """Mutable state for one carrier-to-receiver pass."""

    carrier_id: int
    receiver_id: int
    target: Point
    start_count: int
    phase: str = PHASE_PREPARE
    fired_count: Optional[int] = None
    stable_frames: int = 0
    last_count: Optional[int] = None
    last_stability_count: Optional[int] = None
    ball_history: list[Point] = field(default_factory=list)
    terminal_reason: Optional[str] = None
    fired: bool = False
    # True once the carrier has caught (glued) the ball for the pass. While glued
    # the carrier can rotate to aim without the ball falling out of its mouth.
    carrier_caught: bool = False


def _pose_map(game_state: GameState, team_name: str) -> Dict[int, Any]:
    """Return robot poses keyed by unum from a GameState."""

    out: Dict[int, Any] = {}
    for entry in getattr(game_state, "robot_poses", {}).get(team_name, []) or []:
        if isinstance(entry, dict):
            out.update({int(k): v for k, v in entry.items()})
    return out


def _pose_rad(pose: Sequence[float]) -> Pose:
    """Convert a GameState pose ``(x, y, theta_deg)`` to radians."""

    return (float(pose[0]), float(pose[1]), math.radians(float(pose[2])))


def _clamp_field(point: Point, margin: float = 2.0) -> Point:
    """Clamp a point inside the playable field margin."""

    return (
        float(np.clip(point[0], FIELD_X[0] + margin, FIELD_X[1] - margin)),
        float(np.clip(point[1], FIELD_Y[0] + margin, FIELD_Y[1] - margin)),
    )


class HardcodedPassCoordinator:
    """Own the complete pass interlude between two attacking robots."""

    def __init__(
        self,
        *,
        team_name: str,
        opponent_team_name: str,
        side: str = "left",
        config: Optional[HybridPassConfig] = None,
        support_config: Optional[SupporterConfig] = None,
    ) -> None:
        self.team_name = str(team_name)
        self.opponent_team_name = str(opponent_team_name)
        self.side = str(side)
        self.config = config or HybridPassConfig()
        self.support_config = support_config or SupporterConfig()
        self.state: Optional[HybridPassState] = None
        self.last_events: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Clear any active pass."""

        self.state = None
        self.last_events = []

    def active(self) -> bool:
        """Return true while a pass interlude owns attacker control."""

        return self.state is not None

    def start(self, gate_info: Dict[str, Any], game_state: GameState) -> Optional[HybridPassState]:
        """Start a pass from claimant-follow gate information."""

        carrier_id = gate_info.get("carrier_id")
        receiver_id = gate_info.get("receiver_id")
        if carrier_id is None or receiver_id is None or game_state is None:
            return None
        poses = _pose_map(game_state, self.team_name)
        receiver_pose = poses.get(int(receiver_id))
        if receiver_pose is None:
            return None
        gate_target = gate_info.get("pass_target")
        if (
            isinstance(gate_target, (list, tuple))
            and len(gate_target) >= 2
            and self.is_attacking_receive_target((float(gate_target[0]), float(gate_target[1])))
        ):
            target = (float(gate_target[0]), float(gate_target[1]))
        else:
            ball_pos = getattr(game_state, "ball_pos", None)
            if ball_pos is None:
                target = self._clamp_receive_target((float(receiver_pose[0]), float(receiver_pose[1])))
            else:
                target = self.attacking_receive_target(
                    (float(receiver_pose[0]), float(receiver_pose[1])),
                    (float(ball_pos[0]), float(ball_pos[1])),
                    game_state,
                )
        count = int(getattr(game_state, "count", 0) or 0)
        self.state = HybridPassState(
            carrier_id=int(carrier_id),
            receiver_id=int(receiver_id),
            target=target,
            start_count=count,
        )
        return self.state

    def attacking_receive_target(
        self,
        receiver_xy: Point,
        ball: Point,
        game_state: GameState,
    ) -> Point:
        """Return the concrete final-pass target inside the attacking envelope."""

        target = self._clamp_receive_target(receiver_xy)
        # Lead pass: if a defender already sits on the ball->receiver lane, kick
        # into open space beside it and let the receiver run onto the offset
        # target instead of firing straight into the block. Computed once here so
        # the carrier's aim target remains stable during the align phase.
        return self._lead_target_around_defender(ball, target, game_state)

    def is_attacking_receive_target(self, point: Point) -> bool:
        """Return true when a target is useful as a scoring handoff, not a bailout."""

        x = float(point[0]) if self.side == "left" else -float(point[0])
        y = float(point[1])
        return (
            x >= float(self.config.attacking_min_receive_x_abs)
            and x <= float(self.support_config.max_receive_x_abs)
            and abs(y) <= float(self.config.attacking_max_receive_y_abs)
        )

    def _lead_target_around_defender(
        self, ball: Point, receiver_xy: Point, game_state: GameState
    ) -> Point:
        """Offset the pass target perpendicular to clear a blocking defender.

        Returns ``receiver_xy`` unchanged when no opponent sits on the
        ball->receiver lane. Otherwise shifts the target sideways, away from the
        nearest blocker, by the perpendicular clearance needed to open the lane
        (bounded so the receiver can still reach it). The receiver follows the
        shifted target through ``_receive_pose``.
        """

        opponents = _pose_map(game_state, self.opponent_team_name)
        if not opponents:
            return receiver_xy
        bx, by = float(ball[0]), float(ball[1])
        rx, ry = float(receiver_xy[0]), float(receiver_xy[1])
        seg = np.array([rx - bx, ry - by], dtype=float)
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            return receiver_xy
        seg_unit = seg / seg_len
        perp = np.array([-seg_unit[1], seg_unit[0]], dtype=float)  # left normal
        block_band = 2.0 * PLAYER_SIZE + 0.6  # how close to the lane counts as a block

        worst_abs: Optional[float] = None
        worst_signed = 0.0
        for pose in opponents.values():
            rel = np.array([float(pose[0]) - bx, float(pose[1]) - by], dtype=float)
            along = float(np.dot(rel, seg_unit))
            t = along / seg_len
            if t <= 0.1 or t >= 0.95:
                continue  # only opponents *between* ball and receiver block the lane
            perp_signed = float(np.dot(rel, perp))
            if abs(perp_signed) > block_band:
                continue
            if worst_abs is None or abs(perp_signed) < worst_abs:
                worst_abs = abs(perp_signed)
                worst_signed = perp_signed
        if worst_abs is None:
            return receiver_xy  # lane already clear

        # Shift to the side away from the blocker; clear it plus a margin, bounded
        # so the receiver's reposition stays feasible.
        side = -1.0 if worst_signed >= 0.0 else 1.0
        clearance = float(np.clip((block_band - worst_abs) + 2.0, 2.0, 5.0))
        lead = np.array([rx, ry], dtype=float) + perp * (side * clearance)
        return self._clamp_receive_target((float(lead[0]), float(lead[1])))

    def _clamp_receive_target(self, point: Point) -> Point:
        """Clamp pass targets to the playable support/finish band."""

        x, y = _clamp_field(point, margin=self.support_config.field_margin)
        y = float(np.clip(
            y,
            -self.config.attacking_max_receive_y_abs,
            self.config.attacking_max_receive_y_abs,
        ))
        if self.side == "left":
            x = float(np.clip(
                x,
                self.config.attacking_min_receive_x_abs,
                self.support_config.max_receive_x_abs,
            ))
        else:
            x = float(np.clip(
                x,
                -self.support_config.max_receive_x_abs,
                -self.config.attacking_min_receive_x_abs,
            ))
        return (float(x), float(y))

    def status(self, game_state: Optional[GameState], *, kickable_dist: float) -> Dict[str, Any]:
        """Inspect the active pass and report whether PPO may resume."""

        state = self.state
        if state is None:
            return {"active": False}
        if game_state is None or getattr(game_state, "ball_pos", None) is None:
            return self._terminal("missing_state")

        count = int(getattr(game_state, "count", 0) or 0)
        poses = _pose_map(game_state, self.team_name)
        receiver_pose = poses.get(state.receiver_id)
        carrier_pose = poses.get(state.carrier_id)
        if receiver_pose is None or carrier_pose is None:
            return self._terminal("missing_attacker_pose")

        ball = (float(game_state.ball_pos[0]), float(game_state.ball_pos[1]))
        self._update_ball_history(state, count, ball)
        receiver_pose_rad = _pose_rad(receiver_pose)
        receiver_dist = distance(receiver_pose_rad, ball)
        receiver_cone = ball_in_front_reception_cone(
            receiver_pose_rad,
            ball,
            kickable_tolerance=float(kickable_dist) + 0.15,
        )

        if state.phase == PHASE_RECEIVE:
            if state.last_stability_count != count:
                state.last_stability_count = count
                if receiver_dist <= float(kickable_dist) + 0.25 and receiver_cone:
                    state.stable_frames += 1
                else:
                    state.stable_frames = 0
            if state.stable_frames >= self.config.stable_receive_frames:
                return self._terminal("received", received=True)

            if self._opponent_has_intercepted(game_state, ball, receiver_dist):
                return self._terminal("intercepted")
            fired_count = state.fired_count if state.fired_count is not None else count
            if count - fired_count > self.config.receive_timeout_steps:
                return self._terminal("receive_timeout")

        if state.phase == PHASE_PREPARE and count - state.start_count > self.config.max_align_steps:
            return self._terminal("align_timeout")

        return {
            "active": True,
            "phase": state.phase,
            "carrier_id": state.carrier_id,
            "receiver_id": state.receiver_id,
            "target": state.target,
            "stable_frames": state.stable_frames,
        }

    def execute(
        self,
        game_state: Optional[GameState],
        gate_info: Dict[str, Any],
        *,
        kickable_dist: float,
    ) -> Tuple[Dict[int, str], list[dict[str, Any]], Dict[str, Any]]:
        """Return per-robot commands for the active hardcoded pass."""

        self.last_events = []
        if game_state is None:
            return {}, [], {"active": False, "reason": "missing_state"}
        state = self.state or self.start(gate_info, game_state)
        if state is None:
            return {}, [], {"active": False, "reason": "could_not_start"}

        poses = _pose_map(game_state, self.team_name)
        carrier_pose = poses.get(state.carrier_id)
        receiver_pose = poses.get(state.receiver_id)
        if carrier_pose is None or receiver_pose is None or getattr(game_state, "ball_pos", None) is None:
            info = self._terminal("missing_pose")
            return {}, self.last_events, info

        count = int(getattr(game_state, "count", 0) or 0)
        ball = (float(game_state.ball_pos[0]), float(game_state.ball_pos[1]))
        self._update_ball_history(state, count, ball)

        if state.phase == PHASE_PREPARE:
            carrier_cmd, carrier_event = self._carrier_pass_command(
                _pose_rad(carrier_pose),
                ball,
                state.target,
                game_state,
                kickable_dist=kickable_dist,
            )
            if carrier_event["label"] == "hc_pass_fired":
                state.phase = PHASE_RECEIVE
                state.fired = True
                state.fired_count = count
            receiver_cmd, receiver_event = self._receiver_hold_command(
                _pose_rad(receiver_pose),
                ball,
                state,
                game_state,
            )
        else:
            carrier_cmd, carrier_event = self._carrier_clearout_command(
                _pose_rad(carrier_pose),
                ball,
                state,
                game_state,
            )
            receiver_cmd, receiver_event = self._receiver_receive_command(
                _pose_rad(receiver_pose),
                ball,
                state,
                game_state,
                kickable_dist=kickable_dist,
            )

        events = [carrier_event, receiver_event]
        for event in events:
            self._event(event["label"], **{k: v for k, v in event.items() if k != "label"})
        commands = {
            state.carrier_id: carrier_cmd,
            state.receiver_id: receiver_cmd,
        }
        info = {
            "active": True,
            "phase": state.phase,
            "carrier_id": state.carrier_id,
            "receiver_id": state.receiver_id,
            "target": state.target,
            "fired": state.fired,
            "stable_frames": state.stable_frames,
        }
        return commands, self.last_events, info

    def consume_terminal(self) -> Optional[Dict[str, Any]]:
        """Return terminal state info and clear the lifecycle."""

        if self.state is None or self.state.terminal_reason is None:
            return None
        state = self.state
        info = {
            "reason": state.terminal_reason,
            "carrier_id": state.carrier_id,
            "receiver_id": state.receiver_id,
            "received": state.terminal_reason == "received",
        }
        self.state = None
        return info

    def _terminal(self, reason: str, *, received: bool = False) -> Dict[str, Any]:
        state = self.state
        if state is None:
            return {"active": False, "reason": reason, "received": received}
        state.terminal_reason = reason
        return {
            "active": False,
            "reason": reason,
            "received": received,
            "carrier_id": state.carrier_id,
            "receiver_id": state.receiver_id,
        }

    def _event(self, label: str, **details: Any) -> None:
        event = {"label": str(label), **details}
        self.last_events.append(event)

    def _update_ball_history(self, state: HybridPassState, count: int, ball: Point) -> None:
        if state.last_count == count:
            return
        state.last_count = count
        state.ball_history.append(ball)
        if len(state.ball_history) > 8:
            state.ball_history.pop(0)

    def _ball_velocity(self, state: HybridPassState) -> Point:
        if len(state.ball_history) < 2:
            return (0.0, 0.0)
        recent = state.ball_history[-4:]
        deltas = [
            (recent[i + 1][0] - recent[i][0], recent[i + 1][1] - recent[i][1])
            for i in range(len(recent) - 1)
        ]
        weights = list(range(1, len(deltas) + 1))
        denom = float(sum(weights))
        vx = sum(w * d[0] for w, d in zip(weights, deltas)) / denom
        vy = sum(w * d[1] for w, d in zip(weights, deltas)) / denom
        return (float(vx), float(vy))

    def _carrier_pass_command(
        self,
        carrier_pose: Pose,
        ball: Point,
        target: Point,
        game_state: GameState,
        *,
        kickable_dist: float,
    ) -> Tuple[str, Dict[str, Any]]:
        carrier_ball_dist = distance(carrier_pose, ball)
        target_angle = math.atan2(target[1] - ball[1], target[0] - ball[0])
        if carrier_ball_dist > kickable_dist + 0.35:
            # Pre-align the approach: drive to the contact pose BEHIND the ball on
            # the ball->target line (already facing the pass lane) instead of the
            # ball center. The carrier then arrives both positioned and oriented
            # for the kick, so the terminal in-place turn -- rate-capped at
            # 2 deg/cycle, which is when defenders slide onto the lane -- is as
            # short as the geometry allows instead of a full swing from a
            # ball-facing heading.
            if self.state is not None:
                self.state.carrier_caught = False
            contact = self._contact_pose_for_target(ball, target)
            cmd = goto(
                carrier_pose,
                contact[0],
                contact[1],
                game_state,
                margin=self.config.settle_margin,
                theta=contact[2],
                speed=95.0,
                detour_margin=1.0,
                obstacle_avoidance=True,
                include_ball_obstacle=False,
                include_player_obstacles=True,
            )
            if cmd == "done":
                cmd = "turn 0"
            return limit_turn_rate(cmd), {
                "label": "hc_pass_recover",
                "robot_id": self.state.carrier_id if self.state else None,
                "command": cmd,
                "target": target,
                "carrier_ball_dist": carrier_ball_dist,
            }

        # Catch-glue pass align. The old path issued a bare `turn` to aim, which
        # spun the body off the (un-glued) ball -- it fell out of the mouth, the
        # kick gate failed, and passes almost never fired. Instead: catch the ball
        # to glue it (mirrors the dribble GRAB), rotate the whole assembly toward
        # the pass lane while it stays in the mouth, then release with a real kick.
        power = self._pass_power_for_distance(distance(ball, target))
        state = self.state
        heading_error = normalize_angle(target_angle - carrier_pose[2])
        in_mouth = ball_in_front_reception_cone(carrier_pose, ball)
        fire_tol = math.radians(5.0)

        # NOTE: once caught we must NOT re-grab. A glued ball orbits with the body
        # as we turn, so calling dribble()/face-ball again would chase that orbit
        # forever (it never re-centers). The caught flag is sticky until the ball
        # actually leaves (carrier_ball_dist > kickable_dist + 0.35 -> the recover
        # branch above resets it). In-mouth can read False for a step right after
        # the catch (glue lag); that is fine, the align turn keeps the ball.
        if in_mouth and abs(heading_error) <= fire_tol:
            # Aimed with the ball in the mouth -> fire the pass.
            cmd = f"kick {power:.1f} 0"
            if state is not None:
                state.carrier_caught = False
            label = "hc_pass_fired"
        elif state is not None and state.carrier_caught:
            # Ball glued: rotate toward the pass lane. The glued ball revolves with
            # the body so it stays in the mouth. Geometric full-error turn (the
            # rate limiter only slows it, cannot break convergence).
            cmd = f"turn {heading_error / dt}"
            label = "hc_pass_align"
        else:
            # Not holding the ball yet: face it and catch to glue. dribble() returns
            # "failed" | "turn .." | "catch 0". "failed" => not yet controllable, so
            # settle onto the contact pose.
            grab = dribble(carrier_pose, ball)
            if grab == "catch 0":
                if state is not None:
                    state.carrier_caught = True
                cmd = grab
                label = "hc_pass_grab"
            elif grab.startswith("turn"):
                cmd = grab
                label = "hc_pass_grab"
            else:  # "failed"
                contact = self._contact_pose_for_target(ball, target)
                cmd = goto(
                    carrier_pose,
                    contact[0],
                    contact[1],
                    game_state,
                    margin=self.config.settle_margin,
                    theta=contact[2],
                    speed=85.0,
                    detour_margin=1.0,
                    obstacle_avoidance=True,
                    include_ball_obstacle=False,
                    include_player_obstacles=True,
                )
                if cmd == "done":
                    cmd = "turn 0"
                label = "hc_pass_settle"
        cmd = limit_turn_rate(cmd)
        return cmd, {
            "label": label,
            "robot_id": self.state.carrier_id if self.state else None,
            "command": cmd,
            "target": target,
            "power": power,
            "carrier_ball_dist": carrier_ball_dist,
            "carrier_caught": bool(state.carrier_caught) if state else False,
            "heading_error": float(heading_error),
        }

    def _receiver_hold_command(
        self,
        receiver_pose: Pose,
        ball: Point,
        state: HybridPassState,
        game_state: GameState,
    ) -> Tuple[str, Dict[str, Any]]:
        receive_pose = self._receive_pose(state.target, ball)
        cmd = goto(
            receiver_pose,
            receive_pose[0],
            receive_pose[1],
            game_state,
            margin=self.config.receive_margin,
            theta=receive_pose[2],
            speed=95.0,
            detour_margin=1.2,
            obstacle_avoidance=True,
            include_ball_obstacle=False,
            include_player_obstacles=True,
        )
        if cmd == "done":
            cmd = "turn 0"
        cmd = limit_turn_rate(cmd)
        return cmd, {
            "label": "hc_receive_hold",
            "robot_id": state.receiver_id,
            "command": cmd,
            "target": state.target,
            "receive_pose": receive_pose,
        }

    def _receiver_receive_command(
        self,
        receiver_pose: Pose,
        ball: Point,
        state: HybridPassState,
        game_state: GameState,
        *,
        kickable_dist: float,
    ) -> Tuple[str, Dict[str, Any]]:
        receiver_ball_dist = distance(receiver_pose, ball)
        if receiver_ball_dist <= kickable_dist + 0.25:
            if ball_in_front_reception_cone(receiver_pose, ball, kickable_tolerance=kickable_dist + 0.15):
                cmd = "catch 0"
                label = "hc_receive_settle"
            else:
                contact = self._contact_pose_for_target(ball, attack_goal_for_side(self.side))
                cmd = goto(
                    receiver_pose,
                    contact[0],
                    contact[1],
                    game_state,
                    margin=self.config.settle_margin,
                    theta=contact[2],
                    speed=75.0,
                    detour_margin=1.0,
                    obstacle_avoidance=True,
                    include_ball_obstacle=False,
                    include_player_obstacles=True,
                )
                if cmd == "done":
                    cmd = "turn 0"
                label = "hc_receive_reposition"
        else:
            intercept = self._intercept_point(receiver_pose, ball, state)
            theta = math.atan2(ball[1] - receiver_pose[1], ball[0] - receiver_pose[0])
            cmd = goto(
                receiver_pose,
                intercept[0],
                intercept[1],
                game_state,
                margin=0.8,
                theta=theta,
                speed=100.0,
                detour_margin=1.0,
                obstacle_avoidance=True,
                include_ball_obstacle=False,
                include_player_obstacles=True,
            )
            if cmd == "done":
                cmd = "turn 0"
            label = "hc_receive_line"
        cmd = limit_turn_rate(cmd)
        return cmd, {
            "label": label,
            "robot_id": state.receiver_id,
            "command": cmd,
            "target": state.target,
            "receiver_ball_dist": receiver_ball_dist,
        }

    def _carrier_clearout_command(
        self,
        carrier_pose: Pose,
        ball: Point,
        state: HybridPassState,
        game_state: GameState,
    ) -> Tuple[str, Dict[str, Any]]:
        sign = 1.0 if self.side == "left" else -1.0
        lateral = -self.config.carrier_clearout_dist if carrier_pose[1] >= ball[1] else self.config.carrier_clearout_dist
        target = _clamp_field((ball[0] - sign * self.config.carrier_clearout_dist, ball[1] + lateral))
        theta = math.atan2(ball[1] - carrier_pose[1], ball[0] - carrier_pose[0])
        cmd = goto(
            carrier_pose,
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
            "label": "hc_pass_clearout",
            "robot_id": state.carrier_id,
            "command": cmd,
            "target": target,
        }

    def _contact_pose_for_target(self, ball: Point, target: Sequence[float]) -> Pose:
        dx = float(target[0]) - ball[0]
        dy = float(target[1]) - ball[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            ux, uy = (1.0 if self.side == "left" else -1.0), 0.0
        else:
            ux, uy = dx / norm, dy / norm
        return (
            ball[0] - ux * self.config.contact_standoff,
            ball[1] - uy * self.config.contact_standoff,
            math.atan2(uy, ux),
        )

    def _receive_pose(self, target: Point, ball: Point) -> Pose:
        """Stand just behind the receive point while facing the incoming ball."""

        dx = target[0] - ball[0]
        dy = target[1] - ball[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            ux, uy = (1.0 if self.side == "left" else -1.0), 0.0
        else:
            ux, uy = dx / norm, dy / norm
        face = math.atan2(ball[1] - target[1], ball[0] - target[0])
        # Keep this offset modest: the 2D sim uses center distance for control,
        # while the real robot mouth is offset from center.
        offset = min(self.config.contact_standoff, 0.65)
        return (target[0] + ux * offset, target[1] + uy * offset, face)

    def _intercept_point(self, receiver_pose: Pose, ball: Point, state: HybridPassState) -> Point:
        vx, vy = self._ball_velocity(state)
        speed = math.hypot(vx, vy)
        if speed < 0.05:
            return ball
        best: tuple[float, Point] | None = None
        px, py = ball
        cvx, cvy = vx, vy
        for step in range(1, self.config.receiver_intercept_horizon + 1):
            px += cvx
            py += cvy
            cvx *= BALL_DECAY
            cvy *= BALL_DECAY
            candidate = _clamp_field((px, py))
            robot_time = distance(receiver_pose, candidate) / self.config.receiver_speed_units_per_step
            slack = robot_time - float(step)
            if robot_time <= float(step) + 2.0:
                if best is None or slack < best[0]:
                    best = (slack, candidate)
        return best[1] if best is not None else state.target

    def _opponent_has_intercepted(self, game_state: GameState, ball: Point, receiver_dist: float) -> bool:
        for pose in self._opponent_poses(game_state):
            if distance(pose, ball) <= self.config.opponent_intercept_dist and receiver_dist > distance(pose, ball) + 0.4:
                return True
        return False

    def _opponent_poses(self, game_state: GameState) -> Iterable[Pose]:
        for entry in getattr(game_state, "robot_poses", {}).get(self.opponent_team_name, []) or []:
            if not isinstance(entry, dict):
                continue
            for pose in entry.values():
                if pose is not None and len(pose) >= 3:
                    yield _pose_rad(pose)

    def _pass_power_for_distance(self, pass_distance: float) -> int:
        """Choose pass power from decaying ball travel, capped at 4 m/s."""

        best_speed = self.config.max_ball_speed
        best_score = float("inf")
        for initial_speed in np.linspace(1.2, self.config.max_ball_speed, 24):
            pos = 0.0
            speed = float(initial_speed)
            steps = 0
            while pos < pass_distance and steps < 60:
                pos += speed
                speed *= BALL_DECAY
                steps += 1
            if pos < pass_distance:
                continue
            terminal_speed = speed / BALL_DECAY if BALL_DECAY > 0.0 else speed
            score = abs(terminal_speed - self.config.target_receive_speed) + max(0, steps - 24) * 0.03
            if score < best_score:
                best_score = score
                best_speed = float(initial_speed)
        power = 100.0 * best_speed / max(self.config.max_ball_speed, 1e-6)
        return int(round(float(np.clip(power, self.config.min_pass_power, self.config.max_pass_power))))

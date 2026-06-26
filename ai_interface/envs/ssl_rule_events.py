"""Lightweight SSL rule-event detection for training environments.

The embedded simulator resolves physical overlaps, but it does not enforce SSL
rule outcomes. This module derives referee-like events from consecutive game
states so rewards can discourage illegal attacker behavior during training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from ai_interface.constants.field_constants import FIELD_X
from ai_interface.constants.player_constants import BALL_SIZE, PLAYER_SIZE


XY = Tuple[float, float]
Pose = Tuple[float, float, float]


@dataclass(frozen=True)
class SSLRuleConfig:
    robot_contact_dist: float = PLAYER_SIZE * 2.0
    ball_contact_dist: float = PLAYER_SIZE + BALL_SIZE
    crash_speed_threshold: float = 1.5
    drawn_crash_speed_diff: float = 0.3
    crash_cooldown_steps: int = 20
    attacker_crash_penalty: float = -8.0
    drawn_crash_penalty: float = -4.0
    pushing_contact_steps: int = 8
    pushing_min_step: float = 0.02
    attacker_push_penalty: float = -20.0
    no_progress_window_steps: int = 100
    no_progress_ball_displacement: float = 5.0
    no_progress_goal_progress: float = 5.0
    no_progress_attacker_penalty: float = -10.0
    dribble_limit: float = 10.0
    excessive_dribble_penalty: float = -20.0
    opponent_defense_x: float = 35.0
    defense_half_width: float = 10.0
    defense_area_touch_penalty: float = -8.0
    defense_area_touch_terminal_count: int = 0
    opponent_goalie_ids: Tuple[int, ...] = ()
    teammate_crash_penalty: float = -6.0
    teammate_crash_cooldown_steps: int = 25
    teammate_proximity_penalty: float = -0.08
    teammate_proximity_dist: float = 1.2


@dataclass(frozen=True)
class SSLRuleEvent:
    name: str
    penalty: float = 0.0
    terminal: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SSLRuleState:
    active_contacts: set[Tuple[int, int]] = field(default_factory=set)
    contact_steps: Dict[Tuple[int, int], int] = field(default_factory=dict)
    crash_cooldowns: Dict[Tuple[int, int], int] = field(default_factory=dict)
    no_progress_window: Deque[Tuple[XY, bool, bool]] = field(default_factory=deque)
    attacker_dribble_start: Optional[XY] = None
    defense_touch_cooldown: int = 0
    attacker_defense_touch_count: int = 0


def _xy(pose: Sequence[float]) -> XY:
    return (float(pose[0]), float(pose[1]))


def _dist(a: XY, b: XY) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _sub(a: XY, b: XY) -> XY:
    return (a[0] - b[0], a[1] - b[1])


def _dot(a: XY, b: XY) -> float:
    return float(a[0] * b[0] + a[1] * b[1])


def _unit(a: XY) -> XY:
    n = _dist(a, (0.0, 0.0))
    if n <= 1e-9:
        return (0.0, 0.0)
    return (a[0] / n, a[1] / n)


def _poses_by_id(game_state: Any, team_name: str) -> Dict[int, Pose]:
    poses: Dict[int, Pose] = {}
    if game_state is None:
        return poses
    for entry in getattr(game_state, "robot_poses", {}).get(team_name, []) or []:
        if not isinstance(entry, dict):
            continue
        for rid, pose in entry.items():
            if pose is not None and len(pose) >= 3:
                poses[int(rid)] = (float(pose[0]), float(pose[1]), float(pose[2]))
    return poses


def _opponent_poses(game_state: Any, own_team_name: str) -> Dict[int, Pose]:
    poses: Dict[int, Pose] = {}
    if game_state is None:
        return poses
    for team_name, entries in getattr(game_state, "robot_poses", {}).items():
        if team_name == own_team_name:
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for rid, pose in entry.items():
                if pose is not None and len(pose) >= 3:
                    poses[int(rid)] = (float(pose[0]), float(pose[1]), float(pose[2]))
    return poses


def detect_crash(
    attacker_prev: XY,
    attacker_now: XY,
    opponent_prev: XY,
    opponent_now: XY,
    config: SSLRuleConfig = SSLRuleConfig(),
) -> Optional[SSLRuleEvent]:
    """Detect SSL crash from a newly observed robot-robot collision."""

    normal = _unit(_sub(attacker_now, opponent_now))
    rel_v = _sub(_sub(attacker_now, attacker_prev), _sub(opponent_now, opponent_prev))
    projected = abs(_dot(rel_v, normal))
    if projected <= config.crash_speed_threshold:
        return None

    attacker_speed = _dist(attacker_now, attacker_prev)
    opponent_speed = _dist(opponent_now, opponent_prev)
    speed_diff = abs(attacker_speed - opponent_speed)
    if speed_diff < config.drawn_crash_speed_diff:
        return SSLRuleEvent(
            "bot_crash_drawn",
            penalty=config.drawn_crash_penalty,
            details={"projected_speed": projected, "speed_diff": speed_diff},
        )
    if attacker_speed > opponent_speed:
        return SSLRuleEvent(
            "attacker_crash",
            penalty=config.attacker_crash_penalty,
            details={"projected_speed": projected, "speed_diff": speed_diff},
        )
    return SSLRuleEvent(
        "defender_crash",
        penalty=0.0,
        details={"projected_speed": projected, "speed_diff": speed_diff},
    )


def detect_pushing(
    attacker_prev: XY,
    attacker_now: XY,
    opponent_prev: XY,
    opponent_now: XY,
    contact_steps: int,
    config: SSLRuleConfig = SSLRuleConfig(),
    attacker_id: Optional[int] = None,
    opponent_id: Optional[int] = None,
) -> Optional[SSLRuleEvent]:
    """Detect sustained pushing during contact."""

    if contact_steps < config.pushing_contact_steps:
        return None

    attacker_step = _sub(attacker_now, attacker_prev)
    opponent_step = _sub(opponent_now, opponent_prev)
    a_to_o = _unit(_sub(opponent_now, attacker_now))
    o_to_a = (-a_to_o[0], -a_to_o[1])
    attacker_toward_opponent = _dot(attacker_step, a_to_o)
    opponent_toward_opponent = _dot(opponent_step, a_to_o)
    opponent_toward_attacker = _dot(opponent_step, o_to_a)
    attacker_toward_attacker = _dot(attacker_step, o_to_a)

    attacker_push = (
        attacker_toward_opponent > config.pushing_min_step
        and opponent_toward_opponent > config.pushing_min_step
    )
    defender_push = (
        opponent_toward_attacker > config.pushing_min_step
        and attacker_toward_attacker > config.pushing_min_step
    )
    details = {
        "attacker_id": attacker_id,
        "opponent_id": opponent_id,
        "contact_steps": int(contact_steps),
        "min_contact_steps": int(config.pushing_contact_steps),
        "pushing_min_step": float(config.pushing_min_step),
        "attacker_step": attacker_step,
        "opponent_step": opponent_step,
        "attacker_pos": attacker_now,
        "opponent_pos": opponent_now,
        "attacker_toward_opponent": attacker_toward_opponent,
        "opponent_toward_opponent": opponent_toward_opponent,
        "opponent_toward_attacker": opponent_toward_attacker,
        "attacker_toward_attacker": attacker_toward_attacker,
    }
    if attacker_push and not defender_push:
        return SSLRuleEvent(
            "attacker_push_foul",
            penalty=config.attacker_push_penalty,
            terminal=True,
            details=details,
        )
    if defender_push and not attacker_push:
        return SSLRuleEvent(
            "defender_push_foul",
            penalty=0.0,
            terminal=True,
            details=details,
        )
    return None


def _goal_dist(ball: XY) -> float:
    return float(math.hypot(FIELD_X[1] - ball[0], -ball[1]))


def update_no_progress(
    window: Deque[Tuple[XY, bool, bool]],
    ball_pos: XY,
    contested: bool,
    attacker_contested: bool,
    config: SSLRuleConfig = SSLRuleConfig(),
) -> Optional[SSLRuleEvent]:
    """Update no-progress window and return a Division B forced-start event."""

    window.append((ball_pos, bool(contested), bool(attacker_contested)))
    while len(window) > config.no_progress_window_steps:
        window.popleft()
    if len(window) < config.no_progress_window_steps:
        return None
    contested_share = sum(1 for _ball, contested, _attacker in window if contested) / len(window)
    if contested_share <= 0.5:
        return None

    start_ball = window[0][0]
    end_ball = window[-1][0]
    ball_move = _dist(start_ball, end_ball)
    goal_progress = _goal_dist(start_ball) - _goal_dist(end_ball)
    if ball_move >= config.no_progress_ball_displacement:
        return None
    if abs(goal_progress) >= config.no_progress_goal_progress:
        return None

    attacker_share = sum(1 for _ball, _contested, attacker in window if attacker) / len(window)
    penalty = config.no_progress_attacker_penalty if attacker_share > 0.5 else 0.0
    return SSLRuleEvent(
        "no_progress_forced_start",
        penalty=penalty,
        terminal=True,
        details={
            "ball_move": ball_move,
            "goal_progress": goal_progress,
            "contested_share": contested_share,
            "attacker_share": attacker_share,
        },
    )


def update_excessive_dribble(
    state: SSLRuleState,
    attacker_ball_contact: bool,
    ball_pos: XY,
    config: SSLRuleConfig = SSLRuleConfig(),
) -> Optional[SSLRuleEvent]:
    if attacker_ball_contact:
        if state.attacker_dribble_start is None:
            state.attacker_dribble_start = ball_pos
        carried = _dist(ball_pos, state.attacker_dribble_start)
        if carried > config.dribble_limit:
            state.attacker_dribble_start = ball_pos
            return SSLRuleEvent(
                "attacker_excessive_dribble",
                penalty=config.excessive_dribble_penalty,
                terminal=True,
                details={"carried": carried},
            )
    else:
        state.attacker_dribble_start = None
    return None


class SSLRuleTracker:
    """Stateful SSL event detector used by JALTeamEnv."""

    def __init__(self, config: SSLRuleConfig | None = None) -> None:
        self.config = config or SSLRuleConfig()
        self.state = SSLRuleState()

    def reset(self) -> None:
        self.state = SSLRuleState()

    def update(
        self,
        prev_game_state: Any,
        game_state: Any,
        team_name: str,
        robot_ids: Iterable[int],
    ) -> List[SSLRuleEvent]:
        if prev_game_state is None or game_state is None:
            return []
        ball_pos_raw = getattr(game_state, "ball_pos", None)
        if ball_pos_raw is None or len(ball_pos_raw) < 2:
            return []
        ball_pos = (float(ball_pos_raw[0]), float(ball_pos_raw[1]))

        own_now = _poses_by_id(game_state, team_name)
        own_prev = _poses_by_id(prev_game_state, team_name)
        opp_now = _opponent_poses(game_state, team_name)
        opp_prev = _opponent_poses(prev_game_state, team_name)
        attacker_ids = [int(rid) for rid in robot_ids]
        if not attacker_ids:
            return []
        attacker_id = attacker_ids[0]
        attacker_pose_now = own_now.get(attacker_id)
        attacker_pose_prev = own_prev.get(attacker_id)
        if attacker_pose_now is None or attacker_pose_prev is None:
            return []

        events: List[SSLRuleEvent] = []
        attacker_xy_now = _xy(attacker_pose_now)
        attacker_xy_prev = _xy(attacker_pose_prev)
        attacker_ball_contact = _dist(attacker_xy_now, ball_pos) <= self.config.ball_contact_dist
        contested = False

        current_contacts: set[Tuple[int, int]] = set()
        for opp_id, opp_pose_now in opp_now.items():
            opp_pose_prev = opp_prev.get(opp_id)
            if opp_pose_prev is None:
                continue
            opp_xy_now = _xy(opp_pose_now)
            opp_xy_prev = _xy(opp_pose_prev)
            robot_contact = _dist(attacker_xy_now, opp_xy_now) <= self.config.robot_contact_dist
            shared_ball_contact = (
                attacker_ball_contact
                and _dist(opp_xy_now, ball_pos) <= self.config.ball_contact_dist
            )
            pair = (attacker_id, opp_id)
            if robot_contact or shared_ball_contact:
                contested = True
                current_contacts.add(pair)
                self.state.contact_steps[pair] = self.state.contact_steps.get(pair, 0) + 1
                if robot_contact and pair not in self.state.active_contacts:
                    cooldown = self.state.crash_cooldowns.get(pair, 0)
                    if cooldown <= 0:
                        crash = detect_crash(
                            attacker_xy_prev, attacker_xy_now,
                            opp_xy_prev, opp_xy_now,
                            self.config,
                        )
                        if crash is not None:
                            events.append(crash)
                            self.state.crash_cooldowns[pair] = self.config.crash_cooldown_steps
                pushing = detect_pushing(
                    attacker_xy_prev, attacker_xy_now,
                    opp_xy_prev, opp_xy_now,
                    self.state.contact_steps[pair],
                    self.config,
                    attacker_id=attacker_id,
                    opponent_id=opp_id,
                )
                if pushing is not None:
                    events.append(pushing)
            else:
                self.state.contact_steps[pair] = 0

        for pair, cooldown in list(self.state.crash_cooldowns.items()):
            self.state.crash_cooldowns[pair] = max(0, cooldown - 1)
        self.state.active_contacts = current_contacts

        # Teammate crash / proximity detection (Stage 4+: multi-robot)
        for i, rid_a in enumerate(attacker_ids):
            pose_a_now = own_now.get(rid_a)
            pose_a_prev = own_prev.get(rid_a)
            if pose_a_now is None or pose_a_prev is None:
                continue
            xy_a_now = _xy(pose_a_now)
            xy_a_prev = _xy(pose_a_prev)
            for rid_b in attacker_ids[i + 1:]:
                pose_b_now = own_now.get(rid_b)
                pose_b_prev = own_prev.get(rid_b)
                if pose_b_now is None or pose_b_prev is None:
                    continue
                xy_b_now = _xy(pose_b_now)
                xy_b_prev = _xy(pose_b_prev)
                teammate_dist = _dist(xy_a_now, xy_b_now)
                tm_pair = (min(rid_a, rid_b), max(rid_a, rid_b))

                if teammate_dist <= self.config.robot_contact_dist:
                    cooldown = self.state.crash_cooldowns.get(tm_pair, 0)
                    if cooldown <= 0:
                        normal = _unit(_sub(xy_a_now, xy_b_now))
                        rel_v = _sub(
                            _sub(xy_a_now, xy_a_prev),
                            _sub(xy_b_now, xy_b_prev),
                        )
                        projected = abs(_dot(rel_v, normal))
                        if projected > self.config.crash_speed_threshold:
                            events.append(SSLRuleEvent(
                                "teammate_crash",
                                penalty=self.config.teammate_crash_penalty,
                                details={"robots": tm_pair, "projected_speed": projected},
                            ))
                            self.state.crash_cooldowns[tm_pair] = self.config.teammate_crash_cooldown_steps

                if (
                    self.config.teammate_proximity_penalty < 0.0
                    and teammate_dist <= self.config.teammate_proximity_dist
                ):
                    events.append(SSLRuleEvent(
                        "teammate_proximity",
                        penalty=self.config.teammate_proximity_penalty,
                        details={"robots": tm_pair, "dist": teammate_dist},
                    ))

        dribble = update_excessive_dribble(
            self.state, attacker_ball_contact, ball_pos, self.config,
        )
        if dribble is not None:
            events.append(dribble)

        no_progress = update_no_progress(
            self.state.no_progress_window,
            ball_pos,
            contested,
            contested and attacker_ball_contact,
            self.config,
        )
        if no_progress is not None:
            events.append(no_progress)

        if self.state.defense_touch_cooldown > 0:
            self.state.defense_touch_cooldown -= 1
        elif attacker_ball_contact and (
            ball_pos[0] >= self.config.opponent_defense_x
            and abs(ball_pos[1]) <= self.config.defense_half_width
        ):
            self.state.attacker_defense_touch_count += 1
            terminal_count = int(self.config.defense_area_touch_terminal_count)
            terminal = (
                terminal_count > 0
                and self.state.attacker_defense_touch_count >= terminal_count
            )
            events.append(
                SSLRuleEvent(
                    "attacker_touched_ball_in_defense_area",
                    penalty=self.config.defense_area_touch_penalty,
                    terminal=terminal,
                    details={
                        "touch_count": self.state.attacker_defense_touch_count,
                        "terminal_count": terminal_count,
                        "ball_pos": ball_pos,
                    },
                )
            )
            self.state.defense_touch_cooldown = 20

        opponent_goalie_ids = {int(rid) for rid in self.config.opponent_goalie_ids}
        for opp_id, opp_pose in opp_now.items():
            if int(opp_id) in opponent_goalie_ids:
                continue
            opp_xy = _xy(opp_pose)
            if (
                _dist(opp_xy, ball_pos) <= self.config.ball_contact_dist
                and ball_pos[0] >= self.config.opponent_defense_x
                and abs(ball_pos[1]) <= self.config.defense_half_width
            ):
                events.append(
                    SSLRuleEvent(
                        "defender_in_defense_area",
                        penalty=0.0,
                        terminal=True,
                        details={"opponent_id": int(opp_id), "ball_pos": ball_pos, "opponent_pos": opp_xy},
                    )
                )
                break

        return events

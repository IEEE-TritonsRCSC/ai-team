# Assignee: Adnan Barwaniwala
"""
Division B FreeKick behavior (team controller).

This module decides what *our* 6 robots do during a DIRECT_FREE_* command; it
does NOT emit referee events (BALL_IN_PLAY / DOUBLE_TOUCH / fouls) — those are an
external referee's responsibility (see support/FREEKICK.md). The Division B rules
are instead obeyed as positioning/behavior constraints:

  * Attacking: take the kick, never double-touch (the kicker is demoted after the
    kick and never re-handles the ball this state), keep 0.2 m off the opponent
    defense area until the ball is in play.
  * Defending: every field robot stays >= 0.5 m off the ball and >= 0.2 m off the
    opponent defense area until the ball is in play.

Ownership is provided by the dispatcher via `free_kick_against`:
  * False -> the free kick is ours (attack)
  * True  -> the free kick is the opponent's (defend)
  * None  -> FORCE_START (out of scope here; robots hold position)

The attacking branch mirrors support/Kickoff.py, which is structurally the same
problem (take a restart shot, then chase).
"""
import math

import numpy as np

from ai_interface.constants.field_constants import FIELD_X, FIELD_Y
from ai_interface.constants.player_constants import BALL_SIZE, KICKABLE_MARGIN, PLAYER_SIZE
from ai_interface.utils.algo_utils import (
    clamp,
    compute_bisector_target,
    dist_point_to_segment,
    distance,
    face_ball_angle,
    field_units_per_meter,
    get_goal_params,
    normalize_angle,
    robot_id_and_pose,
)
from ai_interface.utils.basic_commands import goto
from networking.data_utils import GameState, TeamInfo


class AccessoryAlgo:
    """FreeKick behavior for DIRECT_FREE_BLUE / DIRECT_FREE_YELLOW."""

    GOALIE_ID = 1
    KICKER_ID = 5

    FIELD_LENGTH_M = 9.0

    # Division B rule distances (meters) — see support/FREEKICK.md.
    BALL_IN_PLAY_M = 0.05
    FREEKICK_TIMEOUT_S = 10.0
    DEFENDER_MIN_BALL_M = 0.5
    DEFENSE_AREA_MIN_M = 0.2

    # Defense-area geometry (meters) — mirrors support/BallPlacement.py.
    DEFENSE_AREA_DEPTH_M = 1.0
    DEFENSE_AREA_HALF_WIDTH_M = 1.0

    # Behavior tuning (meters).
    FIELD_MARGIN_M = 0.15
    MOVE_MARGIN_M = 0.06
    KICKER_STANDOFF_M = 0.25
    WALL_MARGIN_M = 0.12
    MARK_DIST_M = 0.6
    MIN_PASS_DIST_M = 1.0
    SHOT_CORRIDOR_M = 0.25
    PASS_CORRIDOR_M = 0.25

    MOVE_SPEED = 85.0
    SUPPORT_SPEED = 75.0
    KICK_POWER = 100.0
    PASS_POWER = 70.0
    CLEAR_POWER = 80.0
    ALIGN_TOLERANCE_RAD = math.radians(8.0)

    def __init__(self, team_infos: list[TeamInfo]):
        self.team_infos = team_infos
        u = field_units_per_meter(self.FIELD_LENGTH_M)
        self.units_per_meter = u

        self.field_margin = self.FIELD_MARGIN_M * u
        self.move_margin = self.MOVE_MARGIN_M * u
        self.ball_in_play_distance = self.BALL_IN_PLAY_M * u
        self.defender_min_ball = self.DEFENDER_MIN_BALL_M * u
        self.defense_area_min = self.DEFENSE_AREA_MIN_M * u
        self.defense_area_depth = self.DEFENSE_AREA_DEPTH_M * u
        self.defense_area_half_width = self.DEFENSE_AREA_HALF_WIDTH_M * u
        self.mark_dist = self.MARK_DIST_M * u
        self.min_pass_dist = self.MIN_PASS_DIST_M * u
        self.shot_corridor = self.SHOT_CORRIDOR_M * u
        self.pass_corridor = self.PASS_CORRIDOR_M * u

        self.kicker_standoff = max(
            self.KICKER_STANDOFF_M * u,
            PLAYER_SIZE + BALL_SIZE + KICKABLE_MARGIN + 0.1,
        )
        self.wall_standoff = self.defender_min_ball + PLAYER_SIZE + self.WALL_MARGIN_M * u
        self.kickable_distance = PLAYER_SIZE + BALL_SIZE + KICKABLE_MARGIN

        # Wired by the dispatcher before the first decide_action().
        self.attack_direction: int | None = None
        self.free_kick_against: bool | None = None

        # Per-state tracking.
        self.kicker_id: int | None = None
        self.ball_kicked = False
        self.ball_in_play = False
        self.start_ball_pos: tuple[float, float] | None = None
        self.command_start_time: float | None = None

    # ------------------------------------------------------------------ #
    #  Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def decide_action(self, game_state: GameState, teamname: str):
        """
        Decide actions for all robots on the team based on game state.

        Args:
            game_state: Current game state with ball and robot positions
            teamname: Name of the team to generate actions for

        Returns:
            One action string per robot on the team.
        """
        team_robots = game_state.robot_poses.get(teamname, [])
        ball = game_state.ball_pos
        if ball is None:
            return ["dash 0 0" for _ in team_robots]

        robot_poses = [robot_id_and_pose(robot) for robot in team_robots]
        if not robot_poses:
            return []

        now = game_state.timestamp
        if self.command_start_time is None:
            self.command_start_time = now
        if self.start_ball_pos is None:
            self.start_ball_pos = ball[:2]
        if self.attack_direction is None:
            self.attack_direction = self._infer_attack_direction(robot_poses)

        self._update_ball_in_play(ball, now)

        opponents = self._opponent_poses(game_state, teamname)

        if self.free_kick_against is None:
            # FORCE_START: ball is live immediately, no ownership, no distance
            # restrictions (§5.3.4 / §5.4 SSL rules). Both teams contest freely.
            self.ball_in_play = True
            if self._we_are_closer(robot_poses, opponents, ball):
                return self._attack(robot_poses, opponents, ball, game_state)
            return self._defend(robot_poses, opponents, ball, game_state)

        if self.free_kick_against:
            return self._defend(robot_poses, opponents, ball, game_state)
        return self._attack(robot_poses, opponents, ball, game_state)

    # ------------------------------------------------------------------ #
    #  Ball-in-play / shared state                                        #
    # ------------------------------------------------------------------ #

    def _update_ball_in_play(self, ball, now: float) -> None:
        if self.ball_in_play:
            return
        if (
            self.start_ball_pos is not None
            and distance(ball[:2], self.start_ball_pos) >= self.ball_in_play_distance
        ):
            self.ball_in_play = True
        elif (
            self.command_start_time is not None
            and (now - self.command_start_time) >= self.FREEKICK_TIMEOUT_S
        ):
            self.ball_in_play = True

    def _ball_has_moved(self, ball) -> bool:
        if self.start_ball_pos is None:
            return False
        return distance(ball[:2], self.start_ball_pos) >= self.ball_in_play_distance

    def _opponent_poses(
        self, game_state: GameState, teamname: str
    ) -> list[tuple[int, tuple[float, float, float]]]:
        opponents = []
        for name, robots in game_state.robot_poses.items():
            if name == teamname:
                continue
            opponents.extend(robot_id_and_pose(robot) for robot in robots)
        return opponents

    def _we_are_closer(self, robot_poses, opponents, ball) -> bool:
        """True if our nearest field robot is closer to the ball than any opponent."""
        goalie_id = self._goalie_id(robot_poses)
        field = [pose for unum, pose in robot_poses if unum != goalie_id]
        if not field:
            return False
        our_min = min(distance(p[:2], ball[:2]) for p in field)
        if not opponents:
            return True
        opp_min = min(distance(pose[:2], ball[:2]) for _, pose in opponents)
        return our_min <= opp_min

    def _infer_attack_direction(
        self, robot_poses: list[tuple[int, tuple[float, float, float]]]
    ) -> int:
        avg_x = sum(pose[0] for _, pose in robot_poses) / len(robot_poses)
        return 1 if avg_x < 0 else -1

    def _goalie_id(
        self, robot_poses: list[tuple[int, tuple[float, float, float]]]
    ) -> int | None:
        if any(unum == self.GOALIE_ID for unum, _ in robot_poses):
            return self.GOALIE_ID
        _, _, goal = get_goal_params(self._defended_goal_side())
        return min(robot_poses, key=lambda item: distance(item[1][:2], goal))[0]

    # ------------------------------------------------------------------ #
    #  Attacking branch (our free kick)                                   #
    # ------------------------------------------------------------------ #

    def _attack(self, robot_poses, opponents, ball, game_state: GameState):
        goalie_id = self._goalie_id(robot_poses)
        if self.kicker_id is None:
            self.kicker_id = self._kicker_id(robot_poses, goalie_id, ball)
        if self._ball_has_moved(ball):
            self.ball_kicked = True

        chaser_id = self._chaser_id(robot_poses, goalie_id, ball)
        support_targets = self._support_targets(robot_poses, goalie_id, chaser_id)

        actions = []
        for unum, pose in robot_poses:
            if unum == goalie_id:
                action = self._goalie_guard(pose, ball, game_state)
            elif unum == self.kicker_id and not self.ball_kicked:
                action = self._kicker_action(
                    pose, ball, opponents, robot_poses, goalie_id, game_state
                )
            elif unum == chaser_id and self.ball_kicked:
                action = self._chaser_action(pose, ball, game_state)
            else:
                target = support_targets.get(unum, self._fallback_support_target(unum))
                if not self.ball_in_play:
                    target = self._respect_defense_area(target)
                action = self._move_to_pose(
                    pose, target, face_ball_angle(pose, ball), game_state, self.SUPPORT_SPEED
                )
            actions.append(action)
        return actions

    def _kicker_id(self, robot_poses, goalie_id, ball) -> int | None:
        if any(unum == self.KICKER_ID and unum != goalie_id for unum, _ in robot_poses):
            return self.KICKER_ID
        candidates = [item for item in robot_poses if item[0] != goalie_id]
        if not candidates:
            return goalie_id
        return min(candidates, key=lambda item: distance(item[1][:2], ball[:2]))[0]

    def _chaser_id(self, robot_poses, goalie_id, ball) -> int | None:
        if not self.ball_kicked:
            return None
        candidates = [
            item for item in robot_poses if item[0] not in (goalie_id, self.kicker_id)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: distance(item[1][:2], ball[:2]))[0]

    def _kicker_action(self, pose, ball, opponents, robot_poses, goalie_id, game_state):
        """Shoot directly at goal when the lane is open, otherwise pass."""
        goal = self._shot_target()
        if self._corridor_clear(ball[:2], goal, opponents, self.shot_corridor):
            target, power = goal, self.KICK_POWER
        else:
            mate = self._best_pass_target(ball, robot_poses, goalie_id, opponents)
            if mate is not None:
                target, power = mate, self.PASS_POWER
            else:
                target, power = goal, self.KICK_POWER

        cmd = self._kick_toward_target(pose, ball, target, power, game_state)
        if cmd.startswith("kick "):
            self.ball_kicked = True
        return cmd

    def _best_pass_target(self, ball, robot_poses, goalie_id, opponents):
        """Most advanced teammate (toward enemy goal) with a clear passing lane."""
        candidates = []
        for unum, pose in robot_poses:
            if unum in (goalie_id, self.kicker_id):
                continue
            if distance(pose[:2], ball[:2]) < self.min_pass_dist:
                continue
            if not self._corridor_clear(ball[:2], pose[:2], opponents, self.pass_corridor):
                continue
            candidates.append((unum, pose))
        if not candidates:
            return None
        best = max(candidates, key=lambda item: self.attack_direction * item[1][0])
        return best[1][:2]

    def _chaser_action(self, pose, ball, game_state: GameState) -> str:
        if distance(pose[:2], ball[:2]) <= self.kickable_distance:
            return self._kick_toward_target(
                pose, ball, self._shot_target(), self.CLEAR_POWER, game_state
            )
        target = (ball[0] - self.attack_direction * self.kicker_standoff, ball[1])
        return self._move_to_pose(
            pose, self._clamp_to_field(target), self._goal_angle(), game_state, self.MOVE_SPEED
        )

    def _support_targets(self, robot_poses, goalie_id, chaser_id):
        support_ids = sorted(
            unum for unum, _ in robot_poses if unum not in (goalie_id, chaser_id)
        )
        targets = {}
        for index, unum in enumerate(support_ids):
            if self.ball_kicked and unum == self.kicker_id:
                # Kicker has taken the kick: retreat so it cannot double-touch.
                targets[unum] = self._kicker_release_target()
            else:
                targets[unum] = self._fallback_support_target(index)
        return targets

    def _fallback_support_target(self, index_or_unum: int) -> tuple[float, float]:
        lanes = (
            (1.15, 0.75),
            (1.15, -0.75),
            (0.25, 1.35),
            (0.25, -1.35),
            (-0.95, 0.0),
            (-1.55, 1.0),
            (-1.55, -1.0),
        )
        x_m, y_m = lanes[index_or_unum % len(lanes)]
        target = (self.attack_direction * x_m * self.units_per_meter, y_m * self.units_per_meter)
        return self._clamp_to_field(target)

    def _kicker_release_target(self) -> tuple[float, float]:
        return self._clamp_to_field(
            (-self.attack_direction * self.defender_min_ball, 1.4 * self.units_per_meter)
        )

    def _shot_target(self) -> tuple[float, float]:
        return (
            FIELD_X[1] - self.field_margin
            if self.attack_direction > 0
            else FIELD_X[0] + self.field_margin,
            0.0,
        )

    def _kick_toward_target(self, pose, ball, target, power, game_state: GameState) -> str:
        target_angle = math.atan2(target[1] - ball[1], target[0] - ball[0])
        angle_diff = normalize_angle(target_angle - pose[2])
        if abs(angle_diff) > self.ALIGN_TOLERANCE_RAD:
            return f"turn {angle_diff}"

        kick_pos = (
            ball[0] - math.cos(target_angle) * self.kicker_standoff,
            ball[1] - math.sin(target_angle) * self.kicker_standoff,
        )
        if distance(pose[:2], kick_pos) > self.move_margin:
            return self._move_to_pose(
                pose, self._clamp_to_field(kick_pos), target_angle, game_state, self.MOVE_SPEED
            )
        return f"kick {power:.1f} 0"

    # ------------------------------------------------------------------ #
    #  Defending branch (opponent's free kick)                            #
    # ------------------------------------------------------------------ #

    def _defend(self, robot_poses, opponents, ball, game_state: GameState):
        goalie_id = self._goalie_id(robot_poses)
        _, _, goal_center = get_goal_params(self._defended_goal_side())
        pose_of = {unum: pose for unum, pose in robot_poses}

        field_ids = [unum for unum, _ in robot_poses if unum != goalie_id]
        field_ids.sort(key=lambda u: distance(pose_of[u][:2], ball[:2]))
        wall_ids = field_ids[:2]
        mark_ids = field_ids[2:]

        targets: dict[int, tuple[float, float]] = {}
        wall_targets = self._wall_targets(ball, goal_center)
        for slot, unum in enumerate(wall_ids):
            targets[unum] = wall_targets[slot % len(wall_targets)]
        targets.update(self._mark_targets(mark_ids, pose_of, opponents, ball, goal_center))

        # Once the ball is in play, free-kick positioning limits are lifted:
        # the nearest field robot contests the ball directly.
        contest_id = wall_ids[0] if (self.ball_in_play and wall_ids) else None

        actions = []
        for unum, pose in robot_poses:
            if unum == goalie_id:
                actions.append(self._goalie_guard(pose, ball, game_state))
                continue
            if unum == contest_id:
                actions.append(self._chaser_action(pose, ball, game_state))
                continue
            target = targets.get(unum, self._fallback_support_target(unum))
            if not self.ball_in_play:
                target = self._enforce_ball_clearance(target, ball)
                target = self._respect_defense_area(target)
            actions.append(
                self._move_to_pose(
                    pose, target, face_ball_angle(pose, ball), game_state, self.MOVE_SPEED
                )
            )
        return actions

    def _wall_targets(self, ball, goal_center) -> list[tuple[float, float]]:
        """Two positions on the ball->our-goal line, just outside the legal ring."""
        dx, dy = goal_center[0] - ball[0], goal_center[1] - ball[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            ux, uy = (-self.attack_direction, 0.0)
        else:
            ux, uy = dx / norm, dy / norm
        base = (ball[0] + ux * self.wall_standoff, ball[1] + uy * self.wall_standoff)
        perp = (-uy, ux)
        offset = PLAYER_SIZE * 1.1
        return [
            self._clamp_to_field((base[0] + perp[0] * offset, base[1] + perp[1] * offset)),
            self._clamp_to_field((base[0] - perp[0] * offset, base[1] - perp[1] * offset)),
        ]

    def _mark_targets(self, mark_ids, pose_of, opponents, ball, goal_center):
        """Assign each remaining defender to shadow a threatening opponent."""
        opps = list(opponents)
        if len(opps) > 1:
            # Exclude the opponent kicker standing on the ball.
            kicker_opp = min(opps, key=lambda item: distance(item[1][:2], ball[:2]))
            opps = [o for o in opps if o[0] != kicker_opp[0]]

        threats = sorted(opps, key=lambda item: distance(item[1][:2], goal_center))
        targets = {}
        used: set[int] = set()
        for unum in mark_ids:
            available = [t for t in threats if t[0] not in used]
            if not available:
                targets[unum] = self._fallback_support_target(unum)
                continue
            opp = min(available, key=lambda t: distance(t[1][:2], pose_of[unum][:2]))
            used.add(opp[0])
            ox, oy = opp[1][:2]
            dx, dy = goal_center[0] - ox, goal_center[1] - oy
            norm = math.hypot(dx, dy)
            if norm < 1e-6:
                targets[unum] = self._clamp_to_field((ox, oy))
            else:
                targets[unum] = self._clamp_to_field(
                    (ox + dx / norm * self.mark_dist, oy + dy / norm * self.mark_dist)
                )
        return targets

    def _enforce_ball_clearance(self, target, ball) -> tuple[float, float]:
        """Push a target radially out so the robot's *edge* sits >= 0.5 m from
        the ball. The 0.5 m rule is measured from the nearest side of the robot
        (radius PLAYER_SIZE), so the center must clear defender_min_ball + radius."""
        min_center = self.defender_min_ball + PLAYER_SIZE
        d = distance(target, ball[:2])
        if d >= min_center:
            return target
        if d < 1e-6:
            # Degenerate: push toward our own goal.
            ang = math.pi if self.attack_direction > 0 else 0.0
            ux, uy = math.cos(ang), math.sin(ang)
        else:
            ux, uy = (target[0] - ball[0]) / d, (target[1] - ball[1]) / d
        return self._clamp_to_field(
            (ball[0] + ux * min_center, ball[1] + uy * min_center)
        )

    # ------------------------------------------------------------------ #
    #  Shared helpers                                                     #
    # ------------------------------------------------------------------ #

    def _goalie_guard(self, pose, ball, game_state: GameState) -> str:
        target = compute_bisector_target(ball[:2], pose[:2], self._defended_goal_side())
        return self._move_to_pose(
            pose, target, face_ball_angle(pose, ball), game_state, self.SUPPORT_SPEED
        )

    def _corridor_clear(self, start, end, obstacles, half_width: float) -> bool:
        """True if no obstacle (ahead of `start`) lies within `half_width` of start->end."""
        a = np.array(start[:2], dtype=float)
        b = np.array(end[:2], dtype=float)
        seg_len = distance(start, end)
        for _, pose in obstacles:
            # Only obstacles between start and end matter (closer to end than start is).
            if distance(pose[:2], end) >= seg_len:
                continue
            p = np.array(pose[:2], dtype=float)
            if dist_point_to_segment(p, a, b) <= half_width:
                return False
        return True

    def _respect_defense_area(self, target) -> tuple[float, float]:
        """Push a target out so it stays >= 0.2 m (plus robot radius) off the
        opponent defense area (the box on our attacking side)."""
        margin = self.defense_area_min + PLAYER_SIZE
        depth = self.defense_area_depth
        half_w = self.defense_area_half_width
        tx, ty = target

        if self.attack_direction > 0:
            inner_x = FIELD_X[1] - depth
            inside_x = tx >= inner_x - margin
            x_exit = (inner_x - margin) - tx  # move toward -x
        else:
            inner_x = FIELD_X[0] + depth
            inside_x = tx <= inner_x + margin
            x_exit = (inner_x + margin) - tx  # move toward +x

        if not (inside_x and abs(ty) <= half_w + margin):
            return target

        y_exit = (half_w + margin) - ty if ty >= 0 else -(half_w + margin) - ty
        if abs(x_exit) <= abs(y_exit):
            return self._clamp_to_field((tx + x_exit, ty))
        return self._clamp_to_field((tx, ty + y_exit))

    def _move_to_pose(self, pose, target, target_theta, game_state: GameState, speed: float) -> str:
        cmd = goto(
            pose,
            target[0],
            target[1],
            game_state,
            margin=self.move_margin,
            theta=target_theta,
            speed=speed,
        )
        return "dash 0 0" if cmd == "done" else cmd

    def _clamp_to_field(self, target) -> tuple[float, float]:
        return (
            clamp(target[0], FIELD_X[0] + self.field_margin, FIELD_X[1] - self.field_margin),
            clamp(target[1], FIELD_Y[0] + self.field_margin, FIELD_Y[1] - self.field_margin),
        )

    def _defended_goal_side(self) -> str:
        return "left" if self.attack_direction > 0 else "right"

    def _goal_angle(self) -> float:
        return 0.0 if self.attack_direction > 0 else math.pi

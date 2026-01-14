"""
2v1 smarter logic:
- Our team (TritonBots): Goalie + Defender
- Opponent team: Single Attacker
No dependency on intercept_demo / constants package.
"""

import math
import random
import sys
from collections import deque
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from networking.data_utils import TeamInfo, GameState  # noqa

from ai_interface.constants.field_constants import (
    FIELD_X, FIELD_Y, GOAL_L, GOAL_R, BALL_DECAY, MAX_KEEPER_OUT
)
from ai_interface.constants.player_constants import PLAYER_SIZE, BALL_SIZE


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def norm_angle(a: float) -> float:
    """normalize to [-pi, pi]"""
    return (a + math.pi) % (2 * math.pi) - math.pi


class SoccerAI:
    def __init__(self, team_info: TeamInfo):
        self.team_info = team_info
        self.goalie_ids = {}
        self.side = {}

        for i, team in enumerate(team_info):
            self.goalie_ids[team.name] = int(team.goalie_id)
            self.side[team.name] = "left" if i == 0 else "right"

        # adjust if your team name differs
        self.our_team_name = "TritonBots"

        self.ball_hist = deque(maxlen=6)
        self._last_tick = None

        # tunables
        self.goal_half_width = 6.0          # goal mouth half-width (a bit > 5)
        self.penalty_x = 18.0               # "danger zone" depth from goal line
        self.penalty_y = 20.0               # "danger zone" half height
        self.goalie_line_offset = 1.5       # stand a bit in front of goal line
        self.max_predict_steps = 60

        self.possession_dist = PLAYER_SIZE + BALL_SIZE + 0.15  # a forgiving possession radius

    def translate_ai_output(self, ai_output) -> list[str]:
        return ai_output

    # ---------- ball prediction ----------
    def _update_ball_hist(self, tick: int, ball_xy: tuple[float, float]) -> None:
        if self._last_tick != tick:
            self.ball_hist.append((float(ball_xy[0]), float(ball_xy[1])))
            self._last_tick = tick

    def _estimate_v_next(self) -> tuple[float, float]:
        """Estimate v_{t+1} from position diffs; smooth using last few diffs."""
        if len(self.ball_hist) < 2:
            return (0.0, 0.0)

        diffs = []
        for i in range(len(self.ball_hist) - 1):
            x0, y0 = self.ball_hist[i]
            x1, y1 = self.ball_hist[i + 1]
            diffs.append((x1 - x0, y1 - y0))  # approx v_t

        k = min(3, len(diffs))
        recent = diffs[-k:]
        weights = list(range(1, k + 1))  # 1,2,3 (more weight to newer)
        sw = float(sum(weights))
        vx_t = sum(w * d[0] for w, d in zip(weights, recent)) / sw
        vy_t = sum(w * d[1] for w, d in zip(weights, recent)) / sw

        # v_{t+1} = BALL_DECAY * v_t
        return (BALL_DECAY * vx_t, BALL_DECAY * vy_t)

    def _predict_positions(self, ball: tuple[float, float], v_next: tuple[float, float], steps: int):
        x, y = ball
        vx, vy = v_next
        out = []
        for _ in range(steps):
            x += vx
            y += vy
            vx *= BALL_DECAY
            vy *= BALL_DECAY
            out.append((x, y))
        return out

    def _predict_goal_cross_y(self, ball: tuple[float, float], v_next: tuple[float, float], goal_x: float):
        """simulate and find y where trajectory crosses x=goal_x"""
        prev = ball
        for p in self._predict_positions(ball, v_next, self.max_predict_steps):
            x0, y0 = prev
            x1, y1 = p
            if (goal_x - x0) * (goal_x - x1) <= 0:  # crossed or touched
                if abs(x1 - x0) < 1e-6:
                    return y1
                t = (goal_x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
            prev = p
        return None

    # ---------- geometry helpers ----------
    def _goals_for_side(self, side: str):
        defend = GOAL_L if side == "left" else GOAL_R
        attack = GOAL_R if side == "left" else GOAL_L
        return (float(defend[0]), float(defend[1])), (float(attack[0]), float(attack[1]))

    def _inside_penalty(self, x: float, y: float, side: str) -> bool:
        (gx, _), _ = self._goals_for_side(side)
        if side == "left":
            return x <= gx + self.penalty_x and abs(y) <= self.penalty_y
        else:
            return x >= gx - self.penalty_x and abs(y) <= self.penalty_y

    def _clamp_to_field(self, x: float, y: float, margin: float = 0.8) -> tuple[float, float]:
        x = clamp(x, FIELD_X[0] + margin, FIELD_X[1] - margin)
        y = clamp(y, FIELD_Y[0] + margin, FIELD_Y[1] - margin)
        return x, y

    def _has_ball(self, self_x: float, self_y: float, ball_x: float, ball_y: float) -> bool:
        return math.hypot(ball_x - self_x, ball_y - self_y) <= self.possession_dist

    # ---------- low-level motion / kick ----------
    def _move_to(self, pose: tuple[float, float, float], tx: float, ty: float,
                 power: float = 85.0, margin: float = 0.6, face_xy: tuple[float, float] | None = None) -> str:
        x, y, deg = float(pose[0]), float(pose[1]), float(pose[2])
        heading = math.radians(deg)

        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)

        if dist < margin:
            if face_xy is not None:
                fx, fy = face_xy
                desired = math.atan2(fy - y, fx - x)
                ang = norm_angle(desired - heading)
                if abs(ang) > math.radians(2.0):
                    return f"turn {ang:.4f}"
            return "dash 0 0"

        desired = math.atan2(dy, dx)
        ang = norm_angle(desired - heading)

        # if way off, turn first
        if abs(ang) > math.radians(35):
            return f"turn {ang:.4f}"

        # scale power a bit with distance to reduce overshoot
        p = min(100.0, max(35.0, power * min(1.0, dist / 6.0)))
        return f"dash {p:.1f} {ang:.4f}"

    def _kick_to(self, pose: tuple[float, float, float], tx: float, ty: float, power: float = 100.0) -> str:
        x, y, deg = float(pose[0]), float(pose[1]), float(pose[2])
        heading = math.radians(deg)
        desired = math.atan2(ty - y, tx - x)
        ang = norm_angle(desired - heading)
        return f"kick {power:.1f} {ang:.4f}"

    # ---------- roles ----------
    def _goalie_action(self, tick: int, ball: tuple[float, float], v_next: tuple[float, float],
                       pose: tuple[float, float, float], side: str) -> str:
        bx, by = float(ball[0]), float(ball[1])
        gx, gy = self._goals_for_side(side)[0]

        # keeper x-range limiter
        if side == "left":
            kx_min, kx_max = gx, gx + MAX_KEEPER_OUT
        else:
            kx_min, kx_max = gx - MAX_KEEPER_OUT, gx

        # emergency: if robot outside field, drag back in
        rx, ry = float(pose[0]), float(pose[1])
        if rx < FIELD_X[0] or rx > FIELD_X[1] or ry < FIELD_Y[0] or ry > FIELD_Y[1]:
            tx, ty = self._clamp_to_field(rx, ry)
            tx = clamp(tx, kx_min, kx_max)
            return self._move_to(pose, tx, ty, power=100.0, margin=0.2, face_xy=(bx, by))

        # possession / catch
        if self._has_ball(rx, ry, bx, by):
            # clear to the far wing / forward
            _, (ax, ay) = self._goals_for_side(side)
            wing_y = 18.0 if by < 0 else -18.0
            # a bit forward, not straight middle
            tx = 0.6 * ax
            return self._kick_to(pose, tx, wing_y, power=100.0)

        if self._inside_penalty(bx, by, side) and math.hypot(bx - rx, by - ry) < 1.2:
            return "catch 0"

        # predict shot crossing
        toward_goal = (v_next[0] < -0.05) if side == "left" else (v_next[0] > 0.05)
        cross_y = self._predict_goal_cross_y((bx, by), v_next, gx) if toward_goal else None

        if cross_y is not None:
            y_target = clamp(float(cross_y), -self.goal_half_width, self.goal_half_width)
        else:
            # mild ball-following near posts
            y_target = clamp(by * 0.5, -self.goal_half_width, self.goal_half_width)

        # default stand position
        x_target = gx + (self.goalie_line_offset if side == "left" else -self.goalie_line_offset)
        x_target = clamp(x_target, kx_min, kx_max)

        # if ball is in/near danger zone and moving toward goal, try to step to an intercept point (still clamped)
        if toward_goal and (self._inside_penalty(bx, by, side) or abs(bx - gx) < 26.0):
            preds = self._predict_positions((bx, by), v_next, 25)
            for k, (px, py) in enumerate(preds, start=1):
                if not self._inside_penalty(px, py, side):
                    continue
                px = clamp(px, kx_min, kx_max)
                py = clamp(py, FIELD_Y[0] + 0.8, FIELD_Y[1] - 0.8)
                # reachability heuristic
                if math.hypot(px - rx, py - ry) <= 1.2 * k + 0.6:
                    return self._move_to(pose, px, py, power=100.0, margin=0.5, face_xy=(bx, by))

        # otherwise slide on line
        x_target, y_target = self._clamp_to_field(x_target, y_target)
        x_target = clamp(x_target, kx_min, kx_max)
        return self._move_to(pose, x_target, y_target, power=95.0, margin=0.5, face_xy=(bx, by))

    def _defender_action(self, tick: int, ball: tuple[float, float], v_next: tuple[float, float],
                         pose: tuple[float, float, float], side: str,
                         goalie_pose: tuple[float, float, float] | None,
                         opp_attacker_pose: tuple[float, float, float] | None) -> str:
        bx, by = float(ball[0]), float(ball[1])
        (gx, gy), (ax, ay) = self._goals_for_side(side)
        rx, ry = float(pose[0]), float(pose[1])

        # 如果拿到球：优先清出去（对方只有1人，直接大脚更稳）
        if self._has_ball(rx, ry, bx, by):
            return self._kick_to(pose, ax, random.choice([-8.0, 8.0]), power=100.0)

        ball_speed = math.hypot(v_next[0], v_next[1])

        # ---- NEW: pressing trigger ----
        opp_close_to_ball = False
        if opp_attacker_pose is not None:
            ox, oy = float(opp_attacker_pose[0]), float(opp_attacker_pose[1])
            if math.hypot(ox - bx, oy - by) < 4.0:
                opp_close_to_ball = True

        ball_in_our_half = (bx < 0) if side == "left" else (bx > 0)
        toward_goal = (v_next[0] < -0.05) if side == "left" else (v_next[0] > 0.05)
        close_to_goal = abs(bx - gx) < 30.0

        threat = ball_in_our_half or toward_goal or close_to_goal or opp_close_to_ball

        # ---- dead ball near center + opponent close -> step up and pressure ----
        if ball_speed < 0.02 and abs(bx) < 8.0 and abs(by) < 8.0 and opp_close_to_ball:
            tx, ty = self._clamp_to_field(bx, by)
            # 不要冲到对方半场太深（2v1 还是要留后）
            if side == "left":
                tx = min(tx, 5.0)
            else:
                tx = max(tx, -5.0)
            return self._move_to(pose, tx, ty, power=100.0, margin=0.7, face_xy=(bx, by))

        # ---- intercept when threat ----
        if threat:
            preds = self._predict_positions((bx, by), v_next, 25)
            for k, (px, py) in enumerate(preds, start=1):
                # 别追太深
                if side == "left" and px > 12.0:
                    continue
                if side == "right" and px < -12.0:
                    continue

                px, py = self._clamp_to_field(px, py)
                if math.hypot(px - rx, py - ry) <= 1.35 * k + 0.8:
                    # 离门将太近就错开
                    if goalie_pose is not None:
                        gx2, gy2 = float(goalie_pose[0]), float(goalie_pose[1])
                        if math.hypot(px - gx2, py - gy2) < 3.0:
                            py = clamp(py + (3.0 if py < 0 else -3.0), FIELD_Y[0] + 1.0, FIELD_Y[1] - 1.0)
                    return self._move_to(pose, px, py, power=100.0, margin=0.7, face_xy=(bx, by))

            # 预测失败就直接去压迫球
            tx, ty = self._clamp_to_field(bx, by)
            if side == "left":
                tx = min(tx, 10.0)
            else:
                tx = max(tx, -10.0)
            return self._move_to(pose, tx, ty, power=95.0, margin=0.8, face_xy=(bx, by))

        # ---- not threat: block line ----
        vx, vy = bx - gx, by - gy
        d = math.hypot(vx, vy)
        if d < 1e-6:
            tx, ty = (gx + (16.0 if side == "left" else -16.0), 0.0)
        else:
            ux, uy = vx / d, vy / d
            block_dist = clamp(d * 0.35, 10.0, 20.0)
            tx, ty = gx + ux * block_dist, gy + uy * block_dist

        # 后卫活动范围（别进门将通道）
        if side == "left":
            tx = clamp(tx, gx + 7.0, 10.0)
        else:
            tx = clamp(tx, -10.0, gx - 7.0)
        ty = clamp(ty, FIELD_Y[0] + 1.0, FIELD_Y[1] - 1.0)

        if goalie_pose is not None:
            gx2, gy2 = float(goalie_pose[0]), float(goalie_pose[1])
            if math.hypot(tx - gx2, ty - gy2) < 3.0:
                ty = clamp(ty + (3.0 if ty < 0 else -3.0), FIELD_Y[0] + 1.0, FIELD_Y[1] - 1.0)

        return self._move_to(pose, tx, ty, power=80.0, margin=0.8, face_xy=(bx, by))

    def _attacker_action(self, tick: int, ball: tuple[float, float], pose: tuple[float, float, float],
                         side: str) -> str:
        bx, by = float(ball[0]), float(ball[1])
        (dgx, dgy), (ax, ay) = self._goals_for_side(side)  # attacker attacks "ax,ay"
        rx, ry = float(pose[0]), float(pose[1])

        # ---- NEW: kick-off / dead-ball forcing ----
        v_next = self._estimate_v_next()
        ball_speed = math.hypot(v_next[0], v_next[1])
        dist_to_ball = math.hypot(bx - rx, by - ry)

        # 如果球几乎静止且在中圈附近：直接冲球并踢一下，让比赛动起来
        if ball_speed < 0.02 and abs(bx) < 0.8 and abs(by) < 0.8:
            if self._has_ball(rx, ry, bx, by):
                # 小力量先开球（别一脚飞太远）
                return self._kick_to(pose, ax, 0.0, power=60.0)
            return self._move_to(pose, bx, by, power=100.0, margin=0.25, face_xy=(bx, by))

        # ---- have ball -> shoot ----
        if self._has_ball(rx, ry, bx, by):
            corner_y = random.choice([-4.5, 4.5])
            return self._kick_to(pose, ax, corner_y, power=100.0)

        # ---- NEW: far away -> chase ball directly (DON'T stop at "behind-ball" point) ----
        if dist_to_ball > 3.0:
            tx, ty = self._clamp_to_field(bx, by)
            return self._move_to(pose, tx, ty, power=100.0, margin=0.6, face_xy=(bx, by))

        # ---- close -> approach from behind ball (fine when near) ----
        to_goal_x, to_goal_y = ax - bx, ay - by
        gdist = math.hypot(to_goal_x, to_goal_y)
        if gdist < 1e-6:
            tx, ty = bx, by
        else:
            ux, uy = to_goal_x / gdist, to_goal_y / gdist
            back = self.possession_dist + 0.4
            tx, ty = bx - ux * back, by - uy * back

        tx, ty = self._clamp_to_field(tx, ty)
        return self._move_to(pose, tx, ty, power=95.0, margin=0.5, face_xy=(bx, by))

    # ---------- main entry ----------
    def decide_action(self, game_state: GameState, teamname: str):
        opp_attacker_pose = None
        for tname, plist in game_state.robot_poses.items():
            if tname != teamname and len(plist) > 0:
                d = plist[0]
                u = int(next(iter(d.keys())))
                opp_attacker_pose = d[u]
                break

        actions = []
        tick = int(game_state.count)
        bx, by = float(game_state.ball_pos[0]), float(game_state.ball_pos[1])

        self._update_ball_hist(tick, (bx, by))
        v_next = self._estimate_v_next()

        robots = game_state.robot_poses[teamname]
        side = self.side[teamname]
        goalie_id = int(self.goalie_ids[teamname])

        # build pose map for teammate reference
        pose_map = {}
        for r in robots:
            unum = int(next(iter(r.keys())))
            pose_map[unum] = r[unum]

        if teamname == self.our_team_name:
            # goalie + defender
            goalie_pose = pose_map.get(goalie_id, None)
            defender_id = None
            for u in sorted(pose_map.keys()):
                if u != goalie_id:
                    defender_id = u
                    break

            for r in robots:
                unum = int(next(iter(r.keys())))
                pose = r[unum]
                if unum == goalie_id:
                    actions.append(self._goalie_action(tick, (bx, by), v_next, pose, side))
                else:
                    actions.append(self._defender_action(tick, (bx, by), v_next, pose, side, goalie_pose, opp_attacker_pose))

            return actions

        # opponent team: only one attacker (smallest unum); others (if any) stand still
        attacker_unum = min(pose_map.keys()) if pose_map else None
        for r in robots:
            unum = int(next(iter(r.keys())))
            pose = r[unum]
            if attacker_unum is not None and unum == attacker_unum:
                actions.append(self._attacker_action(tick, (bx, by), pose, side))
            else:
                actions.append("dash 0 0")
        return actions

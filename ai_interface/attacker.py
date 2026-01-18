from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np


from ai_interface.constants.field_constants import FIELD_X, FIELD_Y, BALL_DECAY
from ai_interface.constants.player_constants import PLAYER_SIZE, BALL_SIZE, KICKABLE_MARGIN
from ai_interface.utils.algo_utils import normalize_angle
from ai_interface.player import Player


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def norm_angle(a: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class AttackerConfig:
    # When close to ball but not yet in possession, force a final step to touch the ball.
    force_contact_dist: float = 2.2
    force_contact_margin: float = 0.20

    # ---------- Kick-off / dead-ball handling ----------
    dead_ball_speed: float = 0.02
    kickoff_center_box: float = 0.8
    kickoff_touch_power: float = 60.0

    # ---------- Shooting ----------
    goal_half_width: float = 5.0
    corner_inset: float = 4.5
    shoot_range_x: float = 38.0

    # Lane blocking: defender closer than this to the shot segment => lane is blocked.
    lane_block_dist: float = 2.6

    # Score weights for choosing a corner
    w_goalie_gap: float = 3.0
    w_blocked: float = 2.2
    w_distance: float = 0.03
    w_goalie_advanced: float = 0.05

    # Score threshold to allow a shot even if slightly blocked
    shot_score_threshold: float = 5.5

    # ---------- Dribbling (one-touch) ----------
    # Prevent "slowly pushing to boundary" by not touching every tick.
    touch_cooldown: int = 5

    # Stronger forward bias reduces drifting to the sideline.
    dribble_forward: float = 4.2

    # Side step base magnitude (will be reduced near sideline)
    dribble_side_base: float = 2.2

    # How strongly to pull dribble targets toward the centerline when near sideline
    center_pull_strength: float = 0.55

    # How close to sideline counts as risky (meters)
    sideline_risk_band: float = 6.0

    # When selecting dribble targets, keep a buffer from field boundaries
    boundary_buffer: float = 2.5

    # Dribble kick power (skick)
    dribble_touch_power: float = 22.0

    # ---------- Predictive chase ----------
    chase_far_dist: float = 3.2
    predict_horizon: float = 4.0

    # ---------- Turning / alignment for the "no kick angle" rule ----------
    kick_align_tol: float = math.radians(1.0)
    turn_gain: float = 1.0


class SmartAttacker(Player):
    """
    - tick: current simulation tick
    - ball: (x, y)
    - self_pose: (x, y, heading_degrees)
    - attack_goal: (x, y) goal center that this attacker is trying to score in
    - goalie_pose / defender_pose: opponent positions to reason about (can be None)
    - game_state: passed to Player.goto() for obstacle avoidance (if your goto needs it)

    Output:
    - command string (e.g., 'dash ...', 'turn ...', 'kick ... 0', 'skick ... 0')
    """

    def __init__(self, teamname, unum, cfg: AttackerConfig | None = None):
        super().__init__(teamname=teamname, unum=unum)
        self.cfg = cfg or AttackerConfig()

        # memory
        self._ball_hist: list[tuple[float, float]] = []
        self._ball_hist_max = 6
        self._last_tick: Optional[int] = None

        self._last_touch_tick: int = -10

        self._plan_target = None
        self._plan_kind = None
        self._plan_power = None
        self._plan_expire_tick = -1

    # -------------------------------
    # Core public API
    # -------------------------------
    def step(
        self,
        tick: int,
        ball: Tuple[float, float],
        self_pose: Tuple[float, float, float],
        attack_goal: Tuple[float, float],
        goalie_pose: Optional[Tuple[float, float, float]] = None,
        defender_pose: Optional[Tuple[float, float, float]] = None,
        game_state=None,
    ) -> str:
        """Return the attacker command for this tick."""

        if self._plan_target is not None and tick <= self._plan_expire_tick:
            tx, ty = self._plan_target
            cmd = self._kick_to(self_pose, tx, ty, power=self._plan_power, kind=self._plan_kind)
            # If we actually kicked (not just turned), clear the plan
            if cmd.startswith("kick") or cmd.startswith("skick"):
                self._plan_target = None
                self._plan_kind = None
                self._plan_power = None
                self._plan_expire_tick = -1
            return cmd
        else:
            # plan expired
            self._plan_target = None
            self._plan_kind = None
            self._plan_power = None
            self._plan_expire_tick = -1

        bx, by = float(ball[0]), float(ball[1])
        rx, ry = float(self_pose[0]), float(self_pose[1])
        gx, gy = float(attack_goal[0]), float(attack_goal[1])

        self._update_ball_hist(tick, (bx, by))
        v_next = self._estimate_v_next()
        ball_speed = math.hypot(v_next[0], v_next[1])

        dist_to_ball = math.hypot(bx - rx, by - ry)

        # 0) If we are close but not yet possessing, force a final step to touch the ball.
        if dist_to_ball < self.cfg.force_contact_dist and not self.hasBall(self_pose, (bx, by)):
            return self._goto_point(bx, by, self_pose, game_state, margin=self.cfg.force_contact_margin, speed=100.0, face=(bx, by))

        # 1) Kickoff / dead-ball forcing at center
        if (
            ball_speed < self.cfg.dead_ball_speed
            and abs(bx) < self.cfg.kickoff_center_box
            and abs(by) < self.cfg.kickoff_center_box
        ):
            if self.hasBall(self_pose, (bx, by)):
                return self.face_then_kick(self_pose, target=(gx, 0.0), power=self.cfg.kickoff_touch_power, kind="kick")
            return self._goto_point(bx, by, self_pose, game_state, margin=0.25, speed=100.0, face=(bx, by))

        # 2) If we have ball: shoot or dribble
        if self.hasBall(self_pose, (bx, by)):
            return self._with_ball(tick, (bx, by), self_pose, attack_goal, goalie_pose, defender_pose)

        # 3) If we do not have ball: chase
        return self._without_ball((bx, by), v_next, self_pose, attack_goal, defender_pose, game_state)

    # -------------------------------
    # Ball model / history
    # -------------------------------
    def _update_ball_hist(self, tick: int, ball_xy: tuple[float, float]) -> None:
        if self._last_tick == tick:
            return
        self._last_tick = tick
        self._ball_hist.append(ball_xy)
        if len(self._ball_hist) > self._ball_hist_max:
            self._ball_hist.pop(0)

    def _estimate_v_next(self) -> tuple[float, float]:
        """Estimate v_{t+1} using recent position diffs and BALL_DECAY."""
        if len(self._ball_hist) < 2:
            return (0.0, 0.0)

        diffs = []
        for i in range(len(self._ball_hist) - 1):
            x0, y0 = self._ball_hist[i]
            x1, y1 = self._ball_hist[i + 1]
            diffs.append((x1 - x0, y1 - y0))

        k = min(3, len(diffs))
        recent = diffs[-k:]
        weights = list(range(1, k + 1))
        sw = float(sum(weights))

        vx_t = sum(w * d[0] for w, d in zip(weights, recent)) / sw
        vy_t = sum(w * d[1] for w, d in zip(weights, recent)) / sw

        return (BALL_DECAY * vx_t, BALL_DECAY * vy_t)

    # -------------------------------
    # Perception helpers
    # -------------------------------
    def sideline_risk(self, x: float, y: float) -> float:
        """Risk in [0,1] where 1 means close to top/bottom boundary."""
        top_gap = (FIELD_Y[1] - y)
        bot_gap = (y - FIELD_Y[0])
        gap = min(top_gap, bot_gap)
        return float(clamp(1.0 - gap / self.cfg.sideline_risk_band, 0.0, 1.0))

    @staticmethod
    def dist_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Distance from point p to segment a->b."""
        v = b - a
        vv = float(np.dot(v, v))
        if vv < 1e-9:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, v) / vv)
        t = float(np.clip(t, 0.0, 1.0))
        proj = a + t * v
        return float(np.linalg.norm(p - proj))

    # -------------------------------
    # Movement helpers
    # -------------------------------
    def _goto_point(
        self,
        tx: float,
        ty: float,
        self_pose: Tuple[float, float, float],
        game_state,
        margin: float,
        speed: float,
        face: Optional[Tuple[float, float]] = None,
    ) -> str:

        # If you have a real game_state and basic_commands.goto needs it,
        # passing it helps avoid obstacles.
        if game_state is not None:
            theta = None
            if face is not None:
                fx, fy = face
                theta = math.atan2(fy - self_pose[1], fx - self_pose[0])
            return super().goto(tx, ty, self_pose, game_state, margin=margin, theta=theta, speed=speed)

        # Fallback (works even without game_state): turn-then-dash
        sx, sy, heading = float(self_pose[0]), float(self_pose[1]), float(self_pose[2])  # heading is radians now
        dx, dy = tx - sx, ty - sy
        dist = math.hypot(dx, dy)
        if dist < margin:
            if face is not None:
                desired = math.atan2(face[1] - sy, face[0] - sx)
                ang = norm_angle(desired - heading)
                if abs(ang) > math.radians(2.0):
                    return f"turn {ang:.4f}"
            return "dash 0 0"

        desired = math.atan2(dy, dx)
        ang = norm_angle(desired - heading)
        if abs(ang) > math.radians(35):
            return f"turn {ang:.4f}"
        p = min(100.0, max(35.0, speed * min(1.0, dist / 6.0)))
        return f"dash {p:.1f} {ang:.4f}"

    # -------------------------------
    # Kicking (STRICT: must face + angle=0)
    # -------------------------------
    def face_then_kick(
        self,
        self_pose: Tuple[float, float, float],
        target: Tuple[float, float],
        power: float,
        kind: str = "kick",
    ) -> str:
        """Turn until facing target; then kick straight forward (angle=0)."""
        sx, sy, heading = float(self_pose[0]), float(self_pose[1]), float(self_pose[2])  # heading is radians
        tx, ty = float(target[0]), float(target[1])

        desired = math.atan2(ty - sy, tx - sx)
        diff = norm_angle(desired - heading)

        if abs(diff) > self.cfg.kick_align_tol:
            return f"turn {(diff * self.cfg.turn_gain):.4f}"

        if kind == "skick":
            return f"skick {power:.1f} 0"
        return f"kick {power:.1f} 0"

    def _kick_to(self, pose: tuple[float, float, float], tx: float, ty: float,
                 power: float = 100.0, kind: str = "kick") -> str:

        KICK_ALIGN_TOL = math.radians(1.0)
        TURN_GAIN = 2.0

        x, y, heading = float(pose[0]), float(pose[1]), float(pose[2])  # heading is radians
        desired = math.atan2(ty - y, tx - x)
        diff = norm_angle(desired - heading)


        if abs(diff) > KICK_ALIGN_TOL:
            return f"turn {(diff * TURN_GAIN):.4f}"

        # Only allow straight kick (angle=0)
        if kind == "skick":
            return f"skick {power:.1f} 0"
        return f"kick {power:.1f} 0"

    # -------------------------------
    # Decision logic: with ball
    # -------------------------------
    def _with_ball(
            self,
            tick: int,
            ball_xy: Tuple[float, float],
            self_pose: Tuple[float, float, float],
            attack_goal: Tuple[float, float],
            goalie_pose: Optional[Tuple[float, float, float]],
            defender_pose: Optional[Tuple[float, float, float]],
    ) -> str:
        bx, by = float(ball_xy[0]), float(ball_xy[1])
        gx, gy = float(attack_goal[0]), float(attack_goal[1])

        plan_cmd = self._run_plan(tick, self_pose)
        if plan_cmd is not None:
            return plan_cmd

        # (A) Decide the best shot corner
        corner = min(self.cfg.goal_half_width, self.cfg.corner_inset)
        targets = [
            np.array([gx, +corner], dtype=float),
            np.array([gx, -corner], dtype=float),
        ]

        ball_p = np.array([bx, by], dtype=float)

        def_p = None
        if defender_pose is not None:
            def_p = np.array([float(defender_pose[0]), float(defender_pose[1])], dtype=float)

        gk_y = float(goalie_pose[1]) if goalie_pose is not None else 0.0
        gk_x = float(goalie_pose[0]) if goalie_pose is not None else 0.0

        best_score = -1e18
        best_target = targets[0]
        best_blocked = False

        for t in targets:
            goalie_gap = abs(float(t[1]) - gk_y)

            blocked = False
            if def_p is not None:
                dseg = self.dist_point_to_segment(def_p, ball_p, t)
                blocked = dseg < self.cfg.lane_block_dist

            shot_dist = float(np.linalg.norm(t - ball_p))

            score = 0.0
            score += self.cfg.w_goalie_gap * goalie_gap
            score -= self.cfg.w_blocked * (1.0 if blocked else 0.0)
            score -= self.cfg.w_distance * shot_dist

            # If goalie is advanced (closer to ball), corner selection matters more
            if goalie_pose is not None:
                gk_ball = float(np.linalg.norm(np.array([gk_x, gk_y]) - ball_p))
                score += self.cfg.w_goalie_advanced * max(0.0, 18.0 - gk_ball)

            if score > best_score:
                best_score = score
                best_target = t
                best_blocked = blocked

        # (B) Shoot if in range and lane is acceptable
        rx = float(self_pose[0])
        in_range = abs(rx - gx) < self.cfg.shoot_range_x

        # Extra: if we are very close to goal line, shoot even if slightly blocked (be decisive)
        very_close = abs(rx - gx) < (self.cfg.shoot_range_x * 0.55)

        if (in_range and (not best_blocked or best_score > self.cfg.shot_score_threshold)) or very_close:
            tgt = (float(best_target[0]), float(best_target[1]))

            # Lock the shot for a few ticks so we will turn-then-kick reliably
            self._set_plan(tick, float(tgt[0]), float(tgt[1]), kind="kick", power=100.0, ttl=6)

            # Execute immediately (will be turn or kick)
            return self.face_then_kick(self_pose, tgt, power=100.0, kind="kick")

        # (C) Otherwise dribble: one touch with cooldown, bias toward center near sideline
        if tick - self._last_touch_tick < self.cfg.touch_cooldown:
            # After a touch, just chase/control the ball
            return self._goto_point(bx, by, self_pose, game_state=None, margin=0.1, speed=100.0, face=(bx, by))

        risk = self.sideline_risk(bx, by)

        # Dribble sign: away from defender and opposite goalie bias
        drib_sign = 1.0
        if defender_pose is not None:
            drib_sign = -1.0 if (float(defender_pose[1]) > by) else 1.0
        if goalie_pose is not None and abs(gk_y) > 1.0:
            drib_sign = -1.0 if gk_y > 0 else 1.0

        # If near boundary, override direction to pull toward center
        if risk > 0.35:
            drib_sign = -1.0 if by > 0 else 1.0

        # Compute dribble target
        to_goal = np.array([gx - bx, gy - by], dtype=float)
        n = float(np.linalg.norm(to_goal))
        if n < 1e-6:
            ux, uy = (1.0, 0.0)
        else:
            ux, uy = float(to_goal[0] / n), float(to_goal[1] / n)

        # Side vector perpendicular to attack direction
        sx, sy = -uy, ux

        forward = self.cfg.dribble_forward
        side_step = (1.0 - risk) * self.cfg.dribble_side_base * drib_sign

        # Pull y toward centerline when risky
        drib_tx = bx + ux * forward + sx * side_step
        drib_ty = (by + uy * forward + sy * side_step) * (1.0 - self.cfg.center_pull_strength * risk)

        # Keep a safe buffer from boundaries
        buf = self.cfg.boundary_buffer
        drib_tx = clamp(drib_tx, FIELD_X[0] + buf, FIELD_X[1] - buf)
        drib_ty = clamp(drib_ty, FIELD_Y[0] + buf, FIELD_Y[1] - buf)

        # Mark touch time (cooldown) AND lock plan so we complete turn->skick reliably
        self._last_touch_tick = tick
        tgt = (float(drib_tx), float(drib_ty))
        self._set_plan(tick, float(tgt[0]), float(tgt[1]), kind="skick", power=self.cfg.dribble_touch_power, ttl=4)

        return self.face_then_kick(self_pose, tgt, power=self.cfg.dribble_touch_power, kind="skick")

    # -------------------------------
    # Decision logic: without ball
    # -------------------------------
    def _without_ball(
        self,
        ball_xy: Tuple[float, float],
        v_next: Tuple[float, float],
        self_pose: Tuple[float, float, float],
        attack_goal: Tuple[float, float],
        defender_pose: Optional[Tuple[float, float, float]],
        game_state,
    ) -> str:
        bx, by = float(ball_xy[0]), float(ball_xy[1])
        gx, gy = float(attack_goal[0]), float(attack_goal[1])
        rx, ry = float(self_pose[0]), float(self_pose[1])

        dist_to_ball = math.hypot(bx - rx, by - ry)

        # Far: chase predicted position
        if dist_to_ball > self.cfg.chase_far_dist:
            px = bx + float(v_next[0]) * self.cfg.predict_horizon
            py = by + float(v_next[1]) * self.cfg.predict_horizon
            px = clamp(px, FIELD_X[0] + 0.8, FIELD_X[1] - 0.8)
            py = clamp(py, FIELD_Y[0] + 0.8, FIELD_Y[1] - 0.8)
            return self._goto_point(px, py, self_pose, game_state, margin=0.7, speed=100.0, face=(bx, by))

        # Close: approach from behind ball (relative to goal direction)
        to_goal = np.array([gx - bx, gy - by], dtype=float)
        n = float(np.linalg.norm(to_goal))
        if n < 1e-6:
            ax, ay = bx, by
        else:
            back = KICKABLE_MARGIN + 0.35
            ax = bx - float(to_goal[0] / n) * back
            ay = by - float(to_goal[1] / n) * back

        # If defender is close to ball, offset approach away from defender to avoid getting pinned
        if defender_pose is not None:
            defx, defy = float(defender_pose[0]), float(defender_pose[1])
            if math.hypot(defx - bx, defy - by) < 6.0:
                away = np.array([bx - defx, by - defy], dtype=float)
                an = float(np.linalg.norm(away))
                if an > 1e-6:
                    away = away / an
                    ax += float(away[0]) * 2.0
                    ay += float(away[1]) * 2.0

        ax = clamp(ax, FIELD_X[0] + 0.8, FIELD_X[1] - 0.8)
        ay = clamp(ay, FIELD_Y[0] + 0.8, FIELD_Y[1] - 0.8)

        # Small margin prevents "freezing" near approach point
        return self._goto_point(ax, ay, self_pose, game_state, margin=0.1, speed=95.0, face=(bx, by))

    def _set_plan(self, tick: int, tx: float, ty: float, kind: str, power: float, ttl: int = 5) -> None:
        """Lock a plan for a few ticks to avoid changing targets every frame."""
        self._plan_target = (float(tx), float(ty))
        self._plan_kind = kind
        self._plan_power = float(power)
        self._plan_expire_tick = int(tick) + int(ttl)

    def _run_plan(self, tick: int, self_pose: Tuple[float, float, float]) -> Optional[str]:
        if self._plan_target is None or tick > self._plan_expire_tick:
            self._plan_target = None
            self._plan_kind = None
            self._plan_power = 0.0
            self._plan_expire_tick = -1
            return None

        cmd = self.face_then_kick(self_pose, self._plan_target, power=self._plan_power, kind=self._plan_kind or "kick")

        # If we actually kicked, clear plan. If we are still turning, keep it.
        if isinstance(cmd, str) and (cmd.startswith("kick") or cmd.startswith("skick")):
            self._plan_target = None
            self._plan_kind = None
            self._plan_power = 0.0
            self._plan_expire_tick = -1
        return cmd



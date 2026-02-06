from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np


from ai_interface.constants.field_constants import FIELD_X, FIELD_Y, BALL_DECAY
from ai_interface.constants.player_constants import PLAYER_SIZE, BALL_SIZE, KICKABLE_MARGIN
from ai_interface.utils.algo_utils import clamp, normalize_angle, dist_point_to_segment, estimate_ball_velocity
from ai_interface.player import Player

@dataclass
class AttackerConfig:
    # When close to ball but not yet in possession, force a final step to touch the ball.
    force_contact_dist: float = 2.2
    force_contact_margin: float = 0.20

    # Back off if goalie crowds the ball
    goalie_close_backoff_dist: float = 3.0
    goalie_backoff_step: float = 10.0

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

    # How far behind the ball to stage before approaching along the shot line
    approach_back_extra: float = 1.2

    # ---------- Predictive chase ----------
    chase_far_dist: float = 3.2
    predict_horizon: float = 4.0

    # ---------- Turning / alignment for the "no kick angle" rule ----------
    kick_align_tol: float = math.radians(1.0)
    turn_gain: float = 1.0


    # ---------- Catching ----------
    catch_face_tol: float = math.radians(12.0)
    aim_align_tol: float = math.radians(8.0)


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

        self.attacker_caught = False
        self.with_ball_counter = 0
        self.with_ball_counter_thresh = 10
        self._use_deep_approach = True
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
        
        bx, by = float(ball[0]), float(ball[1])
        rx, ry = float(self_pose[0]), float(self_pose[1])
        gx, gy = float(attack_goal[0]), float(attack_goal[1])

        self._update_ball_hist(tick, (bx, by))
        if len(self._ball_hist) >= 2:
            v_next = estimate_ball_velocity(self._ball_hist, BALL_DECAY)
        else:
            v_next = (0.0, 0.0)
        ball_speed = math.hypot(v_next[0], v_next[1])

        # 1) Kickoff / dead-ball forcing at center
        if (
            ball_speed < self.cfg.dead_ball_speed
            and abs(bx) < self.cfg.kickoff_center_box
            and abs(by) < self.cfg.kickoff_center_box
        ):
            if self.attacker_caught:
                self.attacker_caught = False
                return f"kick {self.cfg.kickoff_touch_power} 0"
            kickoff_tgt = (gx, 0.0)
            cmd, self._use_deep_approach = super().approach_then_contact(ball_xy=(bx, by), target_xy=kickoff_tgt, self_pose=self_pose, game_state=game_state,
                use_deep_approach=self._use_deep_approach, approach_back_extra=self.cfg.approach_back_extra, deep_margin=0.25,)
            if cmd:
                return cmd
            pending = self._require_catch_before_kick(self_pose, (bx, by), attack_goal)
            if pending:
                return pending
            return f"kick {self.cfg.kickoff_touch_power} {0}"

        # 2) If we have ball: shoot or dribble
        has_ball = np.linalg.norm(np.array([bx - rx, by - ry], dtype=float)) - BALL_SIZE - PLAYER_SIZE <= 2

        # Once has_ball is true, keep running _with_ball for a few extra ticks even if we drift away.
        if has_ball:
            self.with_ball_counter = self.with_ball_counter_thresh + 1  # include this tick + next 10
        else:
            self._use_deep_approach = True  # reset staging when we don't have the ball

        if self.with_ball_counter > 0:
            self.with_ball_counter -= 1
            return self._with_ball(tick, (bx, by), self_pose, attack_goal, goalie_pose, defender_pose, game_state)

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

    # -------------------------------
    # Perception helpers
    # -------------------------------
    def sideline_risk(self, x: float, y: float) -> float:
        """Risk in [0,1] where 1 means close to top/bottom boundary."""
        top_gap = (FIELD_Y[1] - y)
        bot_gap = (y - FIELD_Y[0])
        gap = min(top_gap, bot_gap)
        return float(clamp(1.0 - gap / self.cfg.sideline_risk_band, 0.0, 1.0))

    def _require_catch_before_kick(
        self,
        self_pose: Tuple[float, float, float],
        ball_xy: Tuple[float, float],
        target_xy: Optional[Tuple[float, float]] = None,
    ) -> Optional[str]:
        """Ensure we catch (and face ball + shot line) before allowing a kick/skick."""
        rx, ry, heading = float(self_pose[0]), float(self_pose[1]), float(self_pose[2])
        bx, by = float(ball_xy[0]), float(ball_xy[1])

        # If already caught, do not force extra turning here.
        if self.attacker_caught:
            return None

        desired = math.atan2(by - ry, bx - rx)
        diff = normalize_angle(desired - heading)
        if abs(diff) > self.cfg.catch_face_tol:
            return f"turn {diff:.4f}"

        if target_xy is not None:
            shot_dir = math.atan2(float(target_xy[1]) - by, float(target_xy[0]) - bx)
            shot_diff = normalize_angle(shot_dir - heading)
            if abs(shot_diff) > self.cfg.aim_align_tol:
                return f"turn {shot_diff:.4f}"

        self.attacker_caught = True
        return "catch 0"

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
        diff = normalize_angle(desired - heading)

        if abs(diff) > self.cfg.kick_align_tol:
            return f"turn {(diff * self.cfg.turn_gain):.4f}"

        if self._ball_hist:
            pending = self._require_catch_before_kick(self_pose, self._ball_hist[-1], target)
            if pending:
                return pending

        if kind == "skick":
            self.attacker_caught = False
            return f"skick {power:.1f} 0"
        self.attacker_caught = False
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
            game_state
    ) -> str:
        bx, by = float(ball_xy[0]), float(ball_xy[1])
        gx, gy = float(attack_goal[0]), float(attack_goal[1])

        # (A) Decide the best shot corner (or back off if goalie crowds)
        shot_choice = self._choose_shot_target(ball_xy=ball_xy, attack_goal=attack_goal, self_pose=self_pose, goalie_pose=goalie_pose,
            defender_pose=defender_pose, game_state=game_state,)
        if isinstance(shot_choice, str):
            return shot_choice
        best_target, best_blocked, best_score = shot_choice

        # (B) Shoot if in range and lane is acceptable
        rx, ry = float(self_pose[0]), float(self_pose[1])
        in_range = abs(rx - gx) < self.cfg.shoot_range_x

        # Extra: if we are very close to goal line, shoot even if slightly blocked (be decisive)
        very_close = abs(rx - gx) < (self.cfg.shoot_range_x * 0.55)

        if (in_range and (not best_blocked or best_score > self.cfg.shot_score_threshold)) or very_close:
            tgt = (float(best_target[0]), float(best_target[1]))

            if self.attacker_caught:
                self.attacker_caught = False
                return "kick 50 0"

            # Execute immediately (will be turn or kick)
            cmd, self._use_deep_approach = super().approach_then_contact(ball_xy=(bx, by), target_xy=tgt, self_pose=self_pose, game_state=game_state, 
            use_deep_approach=self._use_deep_approach, approach_back_extra=self.cfg.approach_back_extra, deep_margin=0.3,)
            if cmd:
                return cmd
            pending = self._require_catch_before_kick(self_pose, (bx, by), tgt)
            if pending:
                return pending
            return f"kick {self.cfg.kickoff_touch_power} {0}"

        # # (C) Otherwise dribble: one touch with cooldown, bias toward center near sideline
        # if tick - self._last_touch_tick < self.cfg.touch_cooldown:
        #     # After a touch, just chase/control the ball
        #     return self._goto_point(bx, by, self_pose, game_state=None, margin=0.1, speed=100.0, face=(bx, by))

        risk = self.sideline_risk(bx, by)

        # Dribble sign: away from defender and opposite goalie bias
        drib_sign = 1.0
        if defender_pose is not None:
            drib_sign = -1.0 if (float(defender_pose[1]) > by) else 1.0
        if goalie_pose is not None:
            gk_y = float(goalie_pose[1])
            if abs(gk_y) > 1.0:
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
        
        cmd, self._use_deep_approach = super().approach_then_contact(ball_xy=(bx, by), target_xy=tgt, self_pose=self_pose, game_state=game_state,
            use_deep_approach=self._use_deep_approach, approach_back_extra=self.cfg.approach_back_extra, deep_margin=0.3,)
        if cmd:
            return cmd
        if self.attacker_caught:
            self.attacker_caught = False
            return f"skick {self.cfg.dribble_touch_power:.1f} 0"
        pending = self._require_catch_before_kick(self_pose, (bx, by), tgt)
        if pending:
            return pending
        return f"kick {self.cfg.dribble_touch_power} {0}"

    def _choose_shot_target(self, ball_xy: Tuple[float, float], attack_goal: Tuple[float, float], self_pose: Tuple[float, float, float],
        goalie_pose: Optional[Tuple[float, float, float]], defender_pose: Optional[Tuple[float, float, float]], game_state,) -> tuple[np.ndarray, bool, float] | str:
        bx, by = float(ball_xy[0]), float(ball_xy[1])
        gx = float(attack_goal[0])

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
        # If goalie is crowding the ball, back off instead of engaging.
        if goalie_pose is not None:
            gk_ball = float(np.linalg.norm(np.array([gk_x, gk_y]) - ball_p))
            if gk_ball < self.cfg.goalie_close_backoff_dist:
                away = np.array([bx - gk_x, by - gk_y], dtype=float)
                if float(np.linalg.norm(away)) < 1e-6:
                    away = np.array([1.0, 0.0], dtype=float)
                else:
                    away = away / float(np.linalg.norm(away))
                backoff_p = ball_p + away * self.cfg.goalie_backoff_step
                backoff_p[0] = clamp(backoff_p[0], FIELD_X[0] + self.cfg.boundary_buffer, FIELD_X[1] - self.cfg.boundary_buffer)
                backoff_p[1] = clamp(backoff_p[1], FIELD_Y[0] + self.cfg.boundary_buffer, FIELD_Y[1] - self.cfg.boundary_buffer)
                theta = math.atan2(by - self_pose[1], bx - self_pose[0])
                return super().goto(float(backoff_p[0]), float(backoff_p[1]), self_pose, game_state, margin=0.3, theta=theta, speed=90.0)

        best_score = -1e18
        best_target = targets[0]
        best_blocked = False

        for t in targets:
            goalie_gap = abs(float(t[1]) - gk_y)

            blocked = False
            if def_p is not None:
                dseg = dist_point_to_segment(def_p, ball_p, t)
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

        return best_target, best_blocked, best_score

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
        # If defender is close to ball, offset approach away from defender to avoid getting pinned
        if defender_pose is not None:
            defx, defy = float(defender_pose[0]), float(defender_pose[1])
            if math.hypot(defx - bx, defy - by) < 1.5:
                theta = math.atan2(by - self_pose[1], bx - self_pose[0])
                return super().goto(10, 0, self_pose, game_state, margin=0.1, theta=theta, speed=95.0)
        
        # Far: chase predicted position
        if dist_to_ball > self.cfg.chase_far_dist:
            px = bx + float(v_next[0]) * self.cfg.predict_horizon
            py = by + float(v_next[1]) * self.cfg.predict_horizon
            px = clamp(px, FIELD_X[0] + 0.8, FIELD_X[1] - 0.8)
            py = clamp(py, FIELD_Y[0] + 0.8, FIELD_Y[1] - 0.8)
            theta = math.atan2(by - self_pose[1], bx - self_pose[0])
            return super().goto(px, py, self_pose, game_state, margin=0.7, theta=theta, speed=100.0)

        # Close: approach from behind ball (relative to goal direction)
        to_goal = np.array([gx - bx, gy - by], dtype=float)
        n = float(np.linalg.norm(to_goal))
        if n < 1e-6:
            ax, ay = bx, by
        else:
            back = KICKABLE_MARGIN + 0.35
            ax = bx - float(to_goal[0] / n) * back
            ay = by - float(to_goal[1] / n) * back
        theta = math.atan2(by - self_pose[1], bx - self_pose[0])
        return super().goto(ax, ay, self_pose, game_state, margin=0.1, theta=theta, speed=95.0)

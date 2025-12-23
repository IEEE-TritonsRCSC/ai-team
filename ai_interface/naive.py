"""
Goalkeeper (unum 1):
    - Pre-attentive sweeper-keeper:
        * If very close to the ball and teammate shooter exists, do a lead-pass
          toward opponent goal.
        * If ball is moving toward our goal and near our half, predict an
          interception point along the ball trajectory on a circle around the
          goal and dash there aggressively.
        * Otherwise, stand off the goal line (7–12m inside field) and track
          ball.y along the goal mouth.

Shooter (unum 2):
    - Aggressively chases the ball.
    - When close, shoots into the goal mouth, choosing the side away from
      the opponent goalkeeper (if visible).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

from networking.data_utils import GameState


class SoccerAI:
    def __init__(self):
        # Goal mouth half height: posts at roughly y = ±goal_mouth_half_height
        self.goal_mouth_half_height = 7.0

        # Field safety bounds (avoid leaving the field too far)
        self.field_safe_x = 45.0
        self.field_safe_y = 30.0

        # Simple collision avoidance parameters
        self.avoid_distance = 2.0
        self.avoid_angle_deg = 45.0
        self.side_step_angle_deg = 60.0

        # Per-team goal lines: team_name -> (own_goal_x, opp_goal_x)
        self.team_goal_lines: Dict[str, Tuple[float, float]] = {}

        # For ball velocity estimation
        self.last_ball_pos: Optional[Tuple[float, float]] = None
        self.last_timestamp: Optional[float] = None

    # ==================================================================
    # Public interface
    # ==================================================================
    def decide_action(self, game_state: GameState, teamname: str) -> List[str]:
        actions: List[str] = []

        robots = game_state.robot_poses[teamname]
        ball_x, ball_y = self._get_ball_position(game_state)

        # Estimate ball velocity using previous frame
        timestamp = getattr(game_state, "timestamp", None)
        ball_vx, ball_vy = 0.0, 0.0
        if (
            self.last_ball_pos is not None
            and self.last_timestamp is not None
            and timestamp is not None
        ):
            dt = timestamp - self.last_timestamp
            if dt > 1e-3:
                last_bx, last_by = self.last_ball_pos
                ball_vx = (ball_x - last_bx) / dt
                ball_vy = (ball_y - last_by) / dt

        # Update memory
        self.last_ball_pos = (ball_x, ball_y)
        self.last_timestamp = timestamp

        # Own team robot poses: unum -> (x, y, theta)
        robot_info: Dict[int, Tuple[float, float, float]] = {}
        for robot in robots:
            unum, pose = self._get_robot_pose(robot)
            robot_info[unum] = pose

        # All robot positions (both teams) for collision avoidance
        all_positions: List[Tuple[float, float]] = []
        for _, robots_team in game_state.robot_poses.items():
            for r in robots_team:
                _, pose = self._get_robot_pose(r)
                all_positions.append((pose[0], pose[1]))

        # Determine own and opponent goal x
        own_goal_x, opp_goal_x = self._get_goal_lines_for_team(teamname, robot_info)

        # Shooter pose for GK passing
        shooter_pose: Optional[Tuple[float, float, float]] = robot_info.get(2)

        # Find opponent GK pose (unum == 1 on the other team)
        opp_gk_pose: Optional[Tuple[float, float, float]] = None
        for other_team, robots_team in game_state.robot_poses.items():
            if other_team == teamname:
                continue
            for r in robots_team:
                unum, pose = self._get_robot_pose(r)
                if unum == 1:
                    opp_gk_pose = pose
                    break
            if opp_gk_pose is not None:
                break

        for robot in robots:
            unum, pose = self._get_robot_pose(robot)
            x, y, theta = pose

            if unum == 1:
                # Goalkeeper
                action = self._goalkeeper_behavior(
                    x, y, theta,
                    ball_x, ball_y,
                    ball_vx, ball_vy,
                    own_goal_x,
                    opp_goal_x,
                    shooter_pose,
                    all_positions,
                )
            elif unum == 2:
                # Shooter
                opp_gk_y = opp_gk_pose[1] if opp_gk_pose is not None else 0.0
                action = self._shooter_behavior(
                    x, y, theta,
                    ball_x, ball_y,
                    opp_goal_x,
                    opp_gk_y,
                    all_positions,
                )
            else:
                # Other players idle
                action = self._cmd_dash(0.0, 0.0)

            actions.append(action)

        # Debug: print every 20 frames
        frame = getattr(game_state, "count", None)
        if frame is not None and frame % 20 == 0:
            print(f"[NAIVE] frame {frame} team {teamname} actions: {actions}")

        return actions

    def translate_ai_output(self, ai_output) -> List[str]:
        return ai_output

    # ==================================================================
    # GK behavior: 预判 + 拦截 + 主动站位
    # ==================================================================
    def _goalkeeper_behavior(
        self,
        x: float,
        y: float,
        theta: float,
        ball_x: float,
        ball_y: float,
        ball_vx: float,
        ball_vy: float,
        own_goal_x: float,
        opp_goal_x: float,
        shooter_pose: Optional[Tuple[float, float, float]],
        all_positions: List[Tuple[float, float]],
    ) -> str:
        """
        Aggressive goalkeeper:

          1. If close to the ball and shooter exists:
             -> lead-pass: pass toward a lead point in front of shooter.

          2. If ball is moving toward own goal and is not too far:
             -> compute interception point on a circle around the goal center
                and dash aggressively to that point.

          3. Otherwise:
             -> stand out from the goal line (7–12m inside field),
                with y following ball_y within goal mouth height.
        """
        # --- ① GK 已经“拿到球”：给射手做 lead-pass ---
        dist_to_ball = math.hypot(ball_x - x, ball_y - y)
        if dist_to_ball < 1.2 and shooter_pose is not None:
            sx, sy, _ = shooter_pose

            dir_to_goal = math.atan2(0.0 - sy, opp_goal_x - sx)
            lead_dist = 3.0
            lead_x = sx + lead_dist * math.cos(dir_to_goal)
            lead_y = sy + lead_dist * math.sin(dir_to_goal)

            # 防止 lead 点落在射手身后
            if (own_goal_x < opp_goal_x and lead_x < sx) or (
                own_goal_x > opp_goal_x and lead_x > sx
            ):
                lead_x, lead_y = sx, sy

            pass_angle = self._relative_angle(x, y, theta, lead_x, lead_y)
            return self._cmd_kick(80.0, pass_angle)

        # --- ② 球是否“朝着自家球门”移动？ 如果是，则尝试拦截 ---
        ball_speed = math.hypot(ball_vx, ball_vy)
        vec_goal_from_ball = (own_goal_x - ball_x, 0.0 - ball_y)
        dot_v_goal = ball_vx * vec_goal_from_ball[0] + ball_vy * vec_goal_from_ball[1]
        moving_towards_goal = (ball_speed > 0.2 and dot_v_goal > 0.0)

        dx_goal_ball = abs(ball_x - own_goal_x)

        if moving_towards_goal and dx_goal_ball < 35.0:
            # 在球门中心周围半径 r_int 的圆上做拦截
            r_int = max(8.0, min(12.0, dx_goal_ball * 0.5))  # 8~12m 之间

            ok, ix, iy = self._intercept_on_circle(
                ball_x, ball_y, ball_vx, ball_vy,
                own_goal_x, 0.0,
                r_int,
            )

            if ok:
                target_x, target_y = ix, iy
            else:
                # 拦截算不出来：直接冲球
                target_x, target_y = ball_x, ball_y

            # 边界 clamp
            target_x = max(-self.field_safe_x, min(self.field_safe_x, target_x))
            target_y = max(-self.field_safe_y, min(self.field_safe_y, target_y))

            return self._avoidant_dash(
                x, y, theta,
                target_x, target_y,
                base_power=90.0,
                all_positions=all_positions,
            )

        # --- ③ 普通情况：门前站出来 + 沿门线滑动 ---
        # own_goal_x < 0 表示左门，>0 表示右门，往场内方向站
        direction = 1.0 if own_goal_x < 0 else -1.0

        # 基础离门线距离 7m，根据球距离自家门再往前一点
        base_step = 7.0
        extra_step = 0.0
        if dx_goal_ball < 40.0:
            extra_step = (40.0 - dx_goal_ball) * 0.15  # 球越近，step 越大
            if extra_step < 0.0:
                extra_step = 0.0

        step = base_step + extra_step
        step = min(step, 12.0)  # 最多站到离门线 12m

        target_x = own_goal_x + direction * step

        # y 跟着球，但限制在门框高度范围
        target_y = max(-self.goal_mouth_half_height,
                       min(self.goal_mouth_half_height, ball_y))

        # 全场安全边界 clamp
        target_x = max(-self.field_safe_x, min(self.field_safe_x, target_x))
        target_y = max(-self.field_safe_y, min(self.field_safe_y, target_y))

        dist_to_target = math.hypot(target_x - x, target_y - y)

        # 已经在理想站位附近：微调朝向
        if dist_to_target < 0.3:
            rel_angle = self._relative_angle(x, y, theta, ball_x, ball_y)
            return self._cmd_dash(10.0, rel_angle)

        # 否则中高功率 dash 到站位
        return self._avoidant_dash(
            x, y, theta,
            target_x, target_y,
            base_power=80.0,
            all_positions=all_positions,
        )

    # ==================================================================
    # Shooter behavior: 更积极追球 + 多线路射门
    # ==================================================================
    def _shooter_behavior(
        self,
        x: float,
        y: float,
        theta: float,
        ball_x: float,
        ball_y: float,
        opp_goal_x: float,
        opp_gk_y: float,
        all_positions: List[Tuple[float, float]],
    ) -> str:
        """
        Shooter:

          - If out of field safety area: go back towards center.
          - Otherwise:
              - distance to ball > 4m: aggressive chase (fast dash)
              - 1.2m < distance <= 4m: medium power dash
              - distance <= 1.2m: shoot into goal mouth, trying to avoid opponent GK
        """
        # Out of field: return to center area
        if abs(x) > self.field_safe_x or abs(y) > self.field_safe_y:
            return self._avoidant_dash(
                x, y, theta,
                0.0, 0.0,
                base_power=90.0,
                all_positions=all_positions,
            )

        dist_to_ball = math.hypot(ball_x - x, ball_y - y)

        # Far: aggressive chase
        if dist_to_ball > 4.0:
            return self._avoidant_dash(
                x, y, theta,
                ball_x, ball_y,
                base_power=100.0,
                all_positions=all_positions,
            )

        # Medium distance: controlled approach
        if dist_to_ball > 1.2:
            return self._avoidant_dash(
                x, y, theta,
                ball_x, ball_y,
                base_power=70.0,
                all_positions=all_positions,
            )

        if opp_gk_y >= 0:
            target_goal_y = -0.7 * self.goal_mouth_half_height
        else:
            target_goal_y = 0.7 * self.goal_mouth_half_height

        target_goal_y = max(-self.goal_mouth_half_height,
                            min(self.goal_mouth_half_height, target_goal_y))

        goal_target_x = opp_goal_x
        goal_target_y = target_goal_y

        rel_angle = self._relative_angle(x, y, theta, goal_target_x, goal_target_y)
        return self._cmd_kick(100.0, rel_angle)

    def _intercept_on_circle(
        self,
        bx: float,
        by: float,
        vx: float,
        vy: float,
        gx: float,
        gy: float,
        r: float,
    ) -> Tuple[bool, float, float]:
        """
        Compute intersection point between ball trajectory and a circle.

        Ball trajectory:  B(t) = (bx, by) + t * (vx, vy), t >= 0
        Circle:          |B(t) - G|^2 = r^2, G = (gx, gy)

        Returns:
            (True, ix, iy) if there is a valid intersection t >= 0
            (False, 0.0, 0.0) otherwise
        """
        v2 = vx * vx + vy * vy
        if v2 < 1e-6:
            return False, 0.0, 0.0  # ball nearly static

        wx = bx - gx
        wy = by - gy

        # Quadratic: (v·v)t^2 + 2(w·v)t + (w·w - r^2) = 0
        a = v2
        b = 2.0 * (wx * vx + wy * vy)
        c = wx * wx + wy * wy - r * r

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return False, 0.0, 0.0

        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (a * 2.0)
        t2 = (-b + sqrt_disc) / (a * 2.0)

        ts: List[float] = []
        if t1 >= 0.0:
            ts.append(t1)
        if t2 >= 0.0:
            ts.append(t2)

        if not ts:
            return False, 0.0, 0.0

        t_hit = min(ts)
        ix = bx + vx * t_hit
        iy = by + vy * t_hit
        return True, ix, iy

    # ==================================================================
    # Collision-avoidance dash
    # ==================================================================
    def _avoidant_dash(
        self,
        x: float,
        y: float,
        theta: float,
        tx: float,
        ty: float,
        base_power: float,
        all_positions: List[Tuple[float, float]],
    ) -> str:
        """
        Dash toward (tx, ty), but:

          - If another robot is within avoid_distance directly ahead
            (within ±avoid_angle_deg), sidestep by side_step_angle_deg and reduce power.
        """
        dir_angle = math.atan2(ty - y, tx - x)

        chosen_angle = dir_angle
        chosen_power = base_power

        for (ox, oy) in all_positions:
            # skip self (rough approximation)
            if abs(ox - x) < 1e-3 and abs(oy - y) < 1e-3:
                continue

            dx = ox - x
            dy = oy - y
            dist = math.hypot(dx, dy)
            if dist < 1e-6 or dist > self.avoid_distance:
                continue

            angle_to_robot = math.atan2(dy, dx)
            diff = self._normalize_angle_rad(angle_to_robot - dir_angle)

            if abs(math.degrees(diff)) <= self.avoid_angle_deg:
                # obstacle in front: sidestep
                side = -1.0 if diff > 0 else 1.0
                offset = math.radians(self.side_step_angle_deg) * side
                chosen_angle = dir_angle + offset
                chosen_power = min(chosen_power, 80.0)
                break

        rel_angle_deg = self._normalize_angle_deg(
            math.degrees(chosen_angle) - theta
        )
        return self._cmd_dash(chosen_power, rel_angle_deg)

    # ==================================================================
    # Goal line determination
    # ==================================================================
    def _get_goal_lines_for_team(
        self,
        teamname: str,
        robot_info: Dict[int, Tuple[float, float, float]],
    ) -> Tuple[float, float]:
        """
        Determine own and opponent goal x:

          own_goal_x = GK (unum == 1) initial x
          opp_goal_x = opposite side x, used for shooting
        """
        if teamname in self.team_goal_lines:
            return self.team_goal_lines[teamname]

        if 1 in robot_info:
            x1, _, _ = robot_info[1]
            own_goal_x = x1
            if x1 <= 0:
                opp_goal_x = abs(x1) if abs(x1) > 1.0 else 10.0
            else:
                opp_goal_x = -abs(x1) if abs(x1) > 1.0 else -10.0
        else:
            # Fallback: use average x
            if robot_info:
                avg_x = sum(p[0] for p in robot_info.values()) / len(robot_info)
            else:
                avg_x = 0.0

            if avg_x <= 0:
                own_goal_x = avg_x if abs(avg_x) > 1.0 else -45.0
                opp_goal_x = abs(own_goal_x)
            else:
                own_goal_x = avg_x if abs(avg_x) > 1.0 else 45.0
                opp_goal_x = -abs(own_goal_x)

        self.team_goal_lines[teamname] = (own_goal_x, opp_goal_x)
        return own_goal_x, opp_goal_x

    # ==================================================================
    # Utility
    # ==================================================================
    def _get_ball_position(self, game_state: GameState) -> Tuple[float, float]:
        if hasattr(game_state, "ball_pos"):
            return game_state.ball_pos  # type: ignore[attr-defined]
        if hasattr(game_state, "ball_position"):
            return game_state.ball_position  # type: ignore[attr-defined]
        if hasattr(game_state, "object_positions"):
            objs = game_state.object_positions  # type: ignore[attr-defined]
            if "ball" in objs:
                return objs["ball"]
        return 0.0, 0.0

    def _get_robot_pose(self, robot_dict) -> Tuple[int, Tuple[float, float, float]]:
        unum = int(next(iter(robot_dict.keys())))
        pose = robot_dict[unum]
        if len(pose) == 2:
            x, y = pose
            theta = 0.0
        else:
            x, y, theta = pose
        return unum, (float(x), float(y), float(theta))

    def _relative_angle(
        self, x: float, y: float, theta: float, tx: float, ty: float
    ) -> float:
        global_angle = math.degrees(math.atan2(ty - y, tx - x))
        rel = global_angle - theta
        return self._normalize_angle_deg(rel)

    @staticmethod
    def _normalize_angle_deg(angle: float) -> float:
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    @staticmethod
    def _normalize_angle_rad(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    # ==================================================================
    # Command helpers
    # ==================================================================
    def _cmd_dash(self, power: float, rel_angle_deg: float) -> str:
        return f"dash {power:.1f} {rel_angle_deg:.1f}"

    def _cmd_kick(self, power: float, rel_angle_deg: float) -> str:
        return f"kick {power:.1f} {rel_angle_deg:.1f}"

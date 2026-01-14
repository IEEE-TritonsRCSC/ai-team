"""
Demo AI that uses earliest_intercept_control with a simple two-robot scenario:
- shooter (unum==1) kicks toward a receiver (unum==2)
- receiver estimates ball velocity from last k positions (k=3) and intercepts
Actions are returned in the same format as naive.py.
"""

import math
import numpy as np
import random
import time
from ai_interface.naive import SoccerAI
from ai_interface.utils.intercept import earliest_intercept_control
from ai_interface.utils.algo_utils import estimate_ball_velocity
from ai_interface.utils.basic_commands import goto, shoot_at_goal, shoot
from constants.player_constants import *
from constants.field_constants import *


class InterceptDemoAI(SoccerAI):
    def __init__(self, team_info, dt=0.1):
        super().__init__(team_info)
        self.decay = PLAYER_DECAY
        self.ball_decay = BALL_DECAY
        self.dt = dt
        self.ball_history = []
        self.shooter_id = None
        self.receiver_id = None
        self.shooter_turn = True
        self.shooter_decision = None
        self.shooter_decision_expiring = 0
        self.receiver_decision = None
        self.receiver_decision_expiring = 0
        self.decision_count = 0

    def decide_action(self, game_state, teamname: str):
        actions = []
        robots = game_state.robot_poses[teamname]
        mode = getattr(game_state, "playmode", None) or getattr(game_state, "gamemode", None)
        if mode is not None:
            print(f"Play mode: {mode}")
            if mode in ("kick_off_r", "kick_off_l"): 
                self._reset_state()
        # Assign shooter only on TritonBots, receiver only on the other team
        if teamname == "TritonBots" and self.shooter_id is None:
            unums = sorted(int(next(iter(r.keys()))) for r in robots)
            if unums:
                self.shooter_id = (teamname, unums[0])
        if teamname != "TritonBots" and self.receiver_id is None:
            unums = sorted(int(next(iter(r.keys()))) for r in robots)
            if unums:
                self.receiver_id = (teamname, unums[0])

        # record ball history
        self.ball_history.append(np.array(game_state.ball_pos[:2], dtype=float))
        if len(self.ball_history) > 10:
            self.ball_history = self.ball_history[-10:]

        # estimate ball velocity when we have k=3 samples
        ball_vel_est = np.zeros(2)
        if len(self.ball_history) >= 3:
            positions = np.stack(self.ball_history[-3:])
            ball_vel_est = estimate_ball_velocity(positions, alpha=self.ball_decay)

        pose_map = {int(next(iter(r.keys()))): r[int(next(iter(r.keys())))] for r in robots}
        receiver_pos = None
        if self.receiver_id is not None and self.receiver_id in pose_map:
            rp = pose_map[self.receiver_id]
            receiver_pos = np.array([rp[0], rp[1]], dtype=float)
        for robot in robots:
            unum = int(next(iter(robot.keys())))
            pose = robot[unum]
            self_p0 = np.array([pose[0], pose[1]], dtype=float)
            avoid_points = self._build_avoid_points(game_state, teamname, unum)
            heading = math.radians(pose[2])

            if teamname == self.shooter_id[0] and unum == self.shooter_id[1]:
                actions.append(self._action_shooter(self_p0, heading, ball_vel_est, game_state.ball_pos,
                                                   receiver_pos, avoid_points=avoid_points))
                print(f"Shooter cmd for robot {unum}: {actions[-1]}")
            elif teamname == self.receiver_id[0] and unum == self.receiver_id[1]:
                actions.append(self._action_receiver(self_p0, heading, ball_vel_est, game_state.ball_pos, avoid_points=avoid_points))
                print(f"Receiver cmd for robot {unum}: {actions[-1]}")
            else:
                # simple chase
                print("Extra robot, just stay still")
                actions.append(f"dash 0 0")
        print('----------------')
        return actions

    def _reset_state(self) -> None:
        self.ball_history = []
        self.shooter_id = None
        self.receiver_id = None
        self.shooter_turn = True
        self.shooter_decision = None
        self.shooter_decision_expiring = 0
        self.receiver_decision = None
        self.receiver_decision_expiring = 0
        self.decision_count = 0

    def _build_avoid_points(self, game_state, teamname: str, unum: int):
        avoid_points = []
        if game_state.ball_pos is not None:
            avoid_points.append((game_state.ball_pos[0], game_state.ball_pos[1], KICKABLE_MARGIN))
        for other_team, team_robots in game_state.robot_poses.items():
            for robot in team_robots:
                other_unum = int(next(iter(robot.keys())))
                if other_team == teamname and other_unum == unum:
                    continue
                pose = robot[other_unum]
                avoid_points.append((pose[0], pose[1], 1.0))
        return avoid_points

    def _receiver_region(self):
        return (FIELD_X[1] - 30, FIELD_X[1], -20, 20)

    def _inside_region(self, pos, region) -> bool:
        x, y = float(pos[0]), float(pos[1])
        xmin, xmax, ymin, ymax = region
        return xmin <= x <= xmax and ymin <= y <= ymax

    def hybrid_capture(self, self_p0, heading, ball_vel_est, ball_pos, decision, decision_expiring,
                       theta, avoid_points=None, rect_bounds=None):
        ball = np.array(ball_pos[:2], dtype=float)
        to_ball = ball - self_p0
        dist = np.linalg.norm(to_ball)
        ball_speed = np.linalg.norm(ball_vel_est)
        far_threshold = 4.0 * (PLAYER_SIZE + BALL_SIZE)
        moving_threshold = 0.2
        offset = np.array([math.cos(theta), math.sin(theta)]) * (PLAYER_SIZE + BALL_SIZE)
        target_pos = ball - offset

        if dist > far_threshold and ball_speed > moving_threshold:
            if not decision or decision_expiring <= 0:
                res = earliest_intercept_control(
                    self_p0=self_p0,
                    self_v0=np.zeros(2),
                    target_p0=ball,
                    target_v0=ball_vel_est,
                    player_decay=self.decay,
                    ball_decay=self.ball_decay,
                    vmax_self=1.05,
                    u_max=DASH_POWER_RATE * 100,
                    dt=self.dt,
                    T_hi=5.0,
                    interval=0.4,
                    rect_bounds=rect_bounds,
                )
                if res is None:
                    goto_cmd = goto(np.array([self_p0[0], self_p0[1], heading], dtype=float),
                                    target_pos[0], target_pos[1], theta=theta, avoid_points=avoid_points)
                    return goto_cmd, None, 0

                u = res["u"]
                ang = math.atan2(u[1], u[0]) - heading
                power = min(100.0, np.linalg.norm(u) / DASH_POWER_RATE)
                decision = (power, ang)
                decision_expiring = 3
                return f"dash {power:.1f} {ang}", decision, decision_expiring

            power, ang = decision
            decision_expiring -= 1
            return f"dash {power:.1f} {ang}", decision, decision_expiring

        goto_cmd = goto(np.array([self_p0[0], self_p0[1], heading], dtype=float),
                        target_pos[0], target_pos[1], theta=theta, avoid_points=avoid_points, margin=0.01)
        return goto_cmd, None, 0
    
    def _action_shooter(self, self_p0, heading, ball_vel_est, ball_pos, receiver_pos=None, avoid_points=None):
        if not self.shooter_turn:
            return "dash 0 0"
        ball = np.array(ball_pos[:2], dtype=float)
        to_ball = ball - self_p0
        dist = np.linalg.norm(to_ball)
        if self.hasBall(dist, abs(heading - math.atan2(to_ball[1], to_ball[0]))):
            cmd = shoot_at_goal([*self_p0, heading], ball, GOAL_R, kick_power=100.0)
            if cmd != "failed":
                return cmd
            # Fallback: small angle deviation toward GOAL_R
            dir_vec = np.array(GOAL_R, dtype=float) - ball
            theta = math.atan2(dir_vec[1], dir_vec[0]) + math.radians(random.uniform(-10, 10))
            self.shooter_turn = False
            rel = normalize_angle(theta - heading)
            return f"kick 100 {rel}"

        desired_theta = math.atan2(GOAL_R[1] - self_p0[1], GOAL_R[0] - self_p0[0])
        cmd, self.shooter_decision, self.shooter_decision_expiring = self.hybrid_capture(
            self_p0, heading, ball_vel_est, ball_pos,
            self.shooter_decision, self.shooter_decision_expiring,
            desired_theta,
            avoid_points=avoid_points,
        )
        return "dash 0 0" if cmd == "done" else cmd

    def _action_receiver(self, self_p0, heading, ball_vel_est, ball_pos, avoid_points=None):
        ball = np.array(ball_pos[:2], dtype=float)
        to_ball = ball - self_p0
        dist = np.linalg.norm(to_ball)
        ball_speed = np.linalg.norm(ball_vel_est)
        close_threshold = 2.0 * (PLAYER_SIZE + BALL_SIZE)
        slow_threshold = 0.2
        region = self._receiver_region()

        if not self._inside_region(ball, region):
            return "dash 0 0"

        if self.shooter_turn:
            ang = math.atan2(to_ball[1], to_ball[0])
            goto_cmd = goto(np.array([self_p0[0], self_p0[1], heading], dtype=float), 
                        GOAL_R[0] - 10, GOAL_R[1], theta=ang, avoid_points=avoid_points)
            return goto_cmd if goto_cmd != "done" else "dash 0 0"
        
        if dist <= close_threshold and ball_speed <= slow_threshold:
            kick_pos = ball + np.array([1.0, 0.0]) * (PLAYER_SIZE + BALL_SIZE)
            kick_angle = math.atan2(ball[1] - kick_pos[1], ball[0] - kick_pos[0])
            goto_cmd = goto(np.array([self_p0[0], self_p0[1], heading], dtype=float),
                            kick_pos[0], kick_pos[1], theta=kick_angle, speed=50.0,
                            avoid_points=avoid_points)
            if goto_cmd != "done":
                return goto_cmd
        
        if self.hasBall(dist, abs(heading - math.atan2(to_ball[1], to_ball[0]))):
            random_angle = math.radians(random.uniform(-30, 30))
            self.shooter_turn = True
            self.receiver_decision = None
            self.receiver_decision_expiring = 0
            return f"kick 100 {random_angle:.1f}"
        else:
            print('Ball distance:', dist, 'Ball angle:', math.degrees(math.atan2(to_ball[1], to_ball[0])), 'Heading:', math.degrees(heading))
        cmd, self.receiver_decision, self.receiver_decision_expiring = self.hybrid_capture(
            self_p0, heading, ball_vel_est, ball_pos,
            self.receiver_decision, self.receiver_decision_expiring,
            math.pi,
            avoid_points=avoid_points,
            rect_bounds=(FIELD_X[1] - 30, FIELD_X[1], -20 , 20)
        )
        return "dash 0 0" if cmd == "done" else cmd

    def hasBall(self, robot_to_ball_dist: float, angle_diff: float) -> bool:
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        return abs(robot_to_ball_dist - (PLAYER_SIZE + BALL_SIZE)) < 0.1 and abs(angle_diff) < math.radians(5)

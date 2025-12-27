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
        self.receiver_decision = None
        self.receiver_decision_expiring = 0
        self.decision_count = 0

    def decide_action(self, game_state, teamname: str):
        actions = []
        robots = game_state.robot_poses[teamname]
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
            print('Estimated ball velocity:', ball_vel_est, 'Position:', positions[-1])

        pose_map = {int(next(iter(r.keys()))): r[int(next(iter(r.keys())))] for r in robots}
        receiver_pos = None
        if self.receiver_id is not None and self.receiver_id in pose_map:
            rp = pose_map[self.receiver_id]
            receiver_pos = np.array([rp[0], rp[1]], dtype=float)
        for robot in robots:
            unum = int(next(iter(robot.keys())))
            pose = robot[unum]
            self_p0 = np.array([pose[0], pose[1]], dtype=float)
            heading = math.radians(pose[2]) # in radians

            if teamname == self.shooter_id[0] and unum == self.shooter_id[1]:
                actions.append(self._action_shooter(self_p0, heading, game_state.ball_pos, receiver_pos))
            elif teamname == self.receiver_id[0] and unum == self.receiver_id[1]:
                actions.append(self._action_receiver(self_p0, heading, ball_vel_est, game_state.ball_pos))
            else:
                # simple chase
                print("Extra robot, just stay still")
                actions.append(f"dash 0 0")
        print('----------------')
        return actions
    
    def _action_shooter(self, self_p0, heading, ball_pos, receiver_pos=None):
        if not self.shooter_turn:
            print('[Shooter] Not my turn, stay still')
            return "dash 0 0"
        ball = np.array(ball_pos[:2], dtype=float)
        to_ball = ball - self_p0
        dist = np.linalg.norm(to_ball)
        if dist < 5:
            goto_cmd = goto([*self_p0, heading], ball[0]-KICKABLE_MARGIN/2, ball[1], speed=50.0)
        else:
            goto_cmd = goto([*self_p0, heading], ball[0]-KICKABLE_MARGIN/2, ball[1], speed=100.0)
        if goto_cmd == "done":
            if self.hasBall(dist):
                # Try the robust helper first
                cmd = shoot_at_goal([*self_p0, heading], ball, GOAL_R, kick_power=100.0)
                if cmd != "failed":
                    return cmd
                # Fallback: small angle deviation toward GOAL_R
                dir_vec = np.array(GOAL_R, dtype=float) - ball
                theta = math.atan2(dir_vec[1], dir_vec[0]) + math.radians(random.uniform(-10, 10))
                self.shooter_turn = False
                return f"kick 100 {theta}"
            else:
                return "dash 0 0"
        return goto_cmd

    def _action_receiver(self, self_p0, heading, ball_vel_est, ball_pos):
        ball_p = np.array(ball_pos[:2], dtype=float)
        to_ball = ball_p - self_p0

        # If not enough history, just move toward ball
        if self.shooter_turn:
            print("[Receiver] Not my turn, go to goal")
            ang = math.atan2(to_ball[1], to_ball[0])
            goto_cmd = goto(np.array([self_p0[0], self_p0[1], heading], dtype=float), 
                        GOAL_R[0] - 10, GOAL_R[1], theta=ang)
            return goto_cmd if goto_cmd != "done" else "dash 0 0"
        
        ball = np.array(ball_pos[:2], dtype=float)
        to_ball = ball - self_p0
        dist = np.linalg.norm(to_ball)
        if self.hasBall(dist):
            random_angle = math.radians(random.uniform(-45, 45))
            self.shooter_turn = True
            self.receiver_decision = None
            self.receiver_decision_expiring = 0
            return f"kick 100 {random_angle:.1f}"

        if not self.receiver_decision or self.receiver_decision_expiring <= 0:
            time1 = time.time()
            res = earliest_intercept_control(
                self_p0=self_p0,
                self_v0=np.zeros(2),
                target_p0=ball_p,
                target_v0=ball_vel_est,
                player_decay=self.decay,
                ball_decay=self.ball_decay,
                vmax_self=1.05,
                u_max=DASH_POWER_RATE * 100,
                dt=self.dt,
                T_hi=5.0,
                interval=0.4,
                rect_bounds=(FIELD_X[1] - 30, FIELD_X[1], -20 , 20),
                # rect_bounds=(FIELD_X[0], FIELD_X[1], FIELD_Y[0], FIELD_Y[1]),
                )
            time2 = time.time()
            print(f"[Receiver] Intercept computation time: {time2 - time1:.4f} seconds")
            if res is None:
                print("[Receiver] No intercept found, just go toward the goal")
                ang = math.atan2(to_ball[1], to_ball[0])
                goto_cmd = goto(np.array([self_p0[0], self_p0[1], heading], dtype=float), 
                            GOAL_R[0] - 10, GOAL_R[1], theta=ang)
                if goto_cmd == "done":
                    return "dash 0 0"
                return goto_cmd
        

            u = res["u"]
            ang = math.atan2(u[1], u[0]) - heading
            power = min(100.0, np.linalg.norm(u) / DASH_POWER_RATE)
            self.decision_count += 1
            print(f"Intercept control output: power={power}, ang={ang}; Decision count: {self.decision_count}")
            self.receiver_decision = (power, ang)
            self.receiver_decision_expiring = 5
            return f"dash {power:.1f} {ang}"
        else:
            power, ang = self.receiver_decision
            print(f"Intercept control output: power={power}, ang={ang}; Decision count: {self.decision_count}")
            self.receiver_decision_expiring -= 1
            return f"dash {power:.1f} {ang}"

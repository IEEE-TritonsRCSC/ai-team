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
from ai_interface.attacker import SmartAttacker, AttackerConfig
from ai_interface.player import Player
from ai_interface.utils.intercept import earliest_intercept_control
from ai_interface.utils.algo_utils import estimate_ball_velocity
from ai_interface.utils.basic_commands import goto, shoot_at_goal, shoot, kick, dribble
from ai_interface.goalie import goalie_action, infer_side_from_position
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
        self.dribble_state_shooter = False
        self.dribble_state_receiver = False
        self.shooter_decision = None
        self.shooter_decision_expiring = 0
        self.receiver_decision = None
        self.receiver_decision_expiring = 0
        self.receiver_shoot_angle = 0
        self.decision_count = 0
        self.KICKABLE_RANGE = PLAYER_SIZE + KICKABLE_MARGIN + BALL_SIZE
        self.shooter_dribbling = False
        self.receiver_dribbling = False
        self._attacker_player = None
        self._smart_attacker = None
        self._attacker_cfg = AttackerConfig()

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
            heading = math.radians(pose[2])

            if teamname == self.shooter_id[0] and unum == self.shooter_id[1]:
                actions.append(self._action_shooter(self_p0, heading, ball_vel_est, game_state.ball_pos,
                                                   receiver_pos, game_state=game_state))
                print(f"Shooter cmd for robot {unum}: {actions[-1]}")
            elif teamname == self.receiver_id[0] and unum == self.receiver_id[1]:
                actions.append(self._action_receiver(self_p0, heading, ball_vel_est, game_state.ball_pos,
                                                     game_state=game_state))
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
        self._attacker_player = None
        self._smart_attacker = None

    def _receiver_region(self):
        return (FIELD_X[1] - 30, FIELD_X[1], -20, 20)

    def _inside_region(self, pos, region) -> bool:
        x, y = float(pos[0]), float(pos[1])
        xmin, xmax, ymin, ymax = region
        return xmin <= x <= xmax and ymin <= y <= ymax

    def hybrid_capture(self, self_p0, heading, ball_vel_est, ball_pos, decision, decision_expiring,
                       theta, rect_bounds=None, game_state=None, teamname: str | None = None,
                       unum: int | None = None):
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
                                    target_pos[0], target_pos[1], theta=theta,
                                    game_state=game_state)
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
                        target_pos[0], target_pos[1], theta=theta, margin=0.05,
                        game_state=game_state)
        return goto_cmd, None, 0

    def _action_shooter(self, self_p0, heading, ball_vel_est, ball_pos, receiver_pos=None,
                        game_state=None):
        #READ RECEIVER (GOALIE) POSE
        goalie_pose = None
        if self.receiver_id is not None:
            recv_team, recv_unum = self.receiver_id
            for r in game_state.robot_poses.get(recv_team, []):
                u = int(next(iter(r.keys())))
                if u == recv_unum:
                    p = r[u]
                    goalie_pose = (float(p[0]), float(p[1]), math.radians(float(p[2])))   # (x, y, heading_deg)
                    break
        # Ensure attacker module is initialized once
        if self._attacker_player is None or self._smart_attacker is None:
            # In your current InterceptDemoAI, shooter_id is (teamname, unum).
            # We wrap that unum with Player to reuse goto() defaults.
            teamname = self.shooter_id[0] if self.shooter_id is not None else "TritonBots"
            unum = self.shooter_id[1] if self.shooter_id is not None else 1
            self._attacker_player = Player(teamname=teamname, unum=unum, dribbling=False, is_goalie=False)
            self._smart_attacker = SmartAttacker(self._attacker_player, self._attacker_cfg)

        # Convert pose format for SmartAttacker: (x, y, heading_deg)
        self_pose = (float(self_p0[0]), float(self_p0[1]), float(heading))
        ball_xy = (float(ball_pos[0]), float(ball_pos[1]))

        # Attacker is TritonBots -> attacks right goal (GOAL_R)
        attack_goal = (float(GOAL_R[0]), float(GOAL_R[1]))

        # We do NOT touch goalie code, so we pass None for goalie/defender info here.
        # (If you later want attacker to "see" goalie, we can pass receiver pose,
        # but that would be a non-attacker structural change.)
        cmd = self._smart_attacker.step(
            tick=int(getattr(game_state, "count", 0)),
            ball=ball_xy,
            self_pose=self_pose,
            attack_goal=attack_goal,
            goalie_pose=goalie_pose,
            defender_pose=None,
            game_state=game_state
        )
        return cmd


    def _action_receiver(self, self_p0, heading, ball_vel_est, ball_pos,
                         game_state=None,):
        ball = np.array(ball_pos[:2], dtype=float)
        to_ball = ball - self_p0
        dist = np.linalg.norm(to_ball)
        ball_speed = np.linalg.norm(ball_vel_est)
        close_threshold = 2.0 * (PLAYER_SIZE + BALL_SIZE)
        slow_threshold = 0.2
        region = self._receiver_region()
        if self.receiver_shoot_angle == 0:
                self.receiver_shoot_angle = math.radians(random.uniform(-30, 30))
        if not self._inside_region(ball, region):
            return "dash 0 0"

        if self.shooter_turn:
            goalie_pos = (float(self_p0[0]), float(self_p0[1]))
            goalie_pose = (float(self_p0[0]), float(self_p0[1]), math.degrees(heading))
            has_ball = self.hasBall(dist, abs(heading - math.atan2(to_ball[1], to_ball[0])))
            side = infer_side_from_position(goalie_pos)
            print('-----------------------------GOALIE ACTION-----------------------------')
            goalie_cmd = goalie_action(
                ball_pos=(float(ball[0]), float(ball[1])),
                goalie_pose=goalie_pose,
                has_ball=has_ball,
                goalie_to_ball_dist=dist,
                side=side,
                charge_distance=10,
                game_state=game_state
            )
            return goalie_cmd
        
        if dist <= close_threshold and ball_speed <= slow_threshold:
            kick_pos = ball + np.array([np.cos(self.receiver_shoot_angle), np.sin(self.receiver_shoot_angle)]) * (PLAYER_SIZE + BALL_SIZE)
            kick_angle = math.atan2(ball[1] - kick_pos[1], ball[0] - kick_pos[0])
            goto_cmd = goto(np.array([self_p0[0], self_p0[1], heading], dtype=float),
                            kick_pos[0], kick_pos[1], theta=kick_angle, speed=50.0,
                            game_state=game_state)
            if goto_cmd != "done":
                return goto_cmd
        
        if self.hasBall(dist, abs(heading - math.atan2(to_ball[1], to_ball[0]))):
            cmd = kick([*self_p0, heading], ball, kick_angle, dribbling=self.receiver_dribbling)
            # receiver_cmd = f"kick {100} {self.receiver_shoot_angle}"
            if "kick" in cmd:
                self.shooter_turn = True
                self.receiver_decision = None
                self.receiver_decision_expiring = 0
                self.receiver_shoot_angle = 0
                self.receiver_dribbling = False
                return cmd
            elif "catch" in cmd:
                self.receiver_dribbling = True
                return cmd
            elif cmd == "failed":
                print('[Warning] Receiver kick failed')
                return "dash 0 0"
            else:
                return cmd
        else:
            print('Ball distance:', dist, 'Ball angle:', math.degrees(math.atan2(to_ball[1], to_ball[0])), 'Heading:', math.degrees(heading))
        cmd, self.receiver_decision, self.receiver_decision_expiring = self.hybrid_capture(
            self_p0, heading, ball_vel_est, ball_pos,
            self.receiver_decision, self.receiver_decision_expiring,
            math.pi,
            rect_bounds=(FIELD_X[1] - 30, FIELD_X[1], -20 , 20),
            game_state=game_state, teamname=self.receiver_id[0]
        )
        return "dash 0 0" if cmd == "done" else cmd

    def hasBall(self, robot_to_ball_dist: float, angle_diff: float) -> bool:
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        return abs(robot_to_ball_dist - (PLAYER_SIZE + BALL_SIZE)) < 0.1 and abs(angle_diff) < math.radians(5)

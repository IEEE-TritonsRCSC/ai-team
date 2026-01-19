"""
Demo AI that uses earliest_intercept_control with a simple two-robot scenario:
- shooter (unum==1) kicks toward a receiver (unum==2)
- receiver estimates ball velocity from last k positions (k=3) and intercepts
Actions are returned in the same format as naive.py.
"""

import math
import numpy as np
import random
from ai_interface.naive import SoccerAI
from ai_interface.player import Player
from ai_interface.utils.intercept import earliest_intercept_control
from ai_interface.utils.algo_utils import estimate_ball_velocity
from constants.player_constants import *
from constants.field_constants import *

dt = 0.1  # time step for discrete dynamics

class InterceptPlayer(Player):
    def __init__(self, teamname: str | None = None, unum: int | None = None,
                 decay: float = PLAYER_DECAY, ball_decay: float = BALL_DECAY, dt: float = dt, is_goalie: bool = False) -> None:
        super().__init__(teamname=teamname, unum=unum, is_goalie=is_goalie)
        self.teamname = teamname
        self.unum = unum
        self.decay = decay
        self.ball_decay = ball_decay
        self.dt = dt
        self.decision = None
        self.decision_expiring = 0
        self.shoot_angle = 0
        self.kickable_range = PLAYER_SIZE + KICKABLE_MARGIN + BALL_SIZE

    def reset(self) -> None:
        self.decision = None
        self.decision_expiring = 0
        self.shoot_angle = 0
        self.dribbling = False

    def set_identity(self, teamname: str | None, unum: int | None) -> None:
        self.teamname = teamname
        self.unum = unum

    def kick(self, target_angle: float,
             self_pose: list | tuple,
             ball_pose: list | tuple,
             kick_power: int | None = None,) -> str:
        cmd = super().kick(target_angle, self_pose, ball_pose, kick_power)
        if "kick" in cmd:
            self.decision = None
            self.decision_expiring = 0
            self.shoot_angle = 0
        return cmd

    def hybrid_capture(self, self_p0, heading, ball_vel_est, ball_pos, theta,
                       rect_bounds=None, game_state=None):
        ball = np.array(ball_pos[:2], dtype=float)
        to_ball = ball - self_p0
        dist = np.linalg.norm(to_ball)
        ball_speed = np.linalg.norm(ball_vel_est)
        far_threshold = 4.0 * (PLAYER_SIZE + BALL_SIZE)
        moving_threshold = 0.2
        offset = np.array([math.cos(theta), math.sin(theta)]) * (PLAYER_SIZE + BALL_SIZE)
        target_pos = ball - offset

        if dist > far_threshold and ball_speed > moving_threshold:
            if not self.decision or self.decision_expiring <= 0:
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
                    return self.goto(
                        target_pos[0],
                        target_pos[1],
                        [self_p0[0], self_p0[1], heading],
                        game_state,
                        theta=theta,
                    )

                u = res["u"]
                ang = math.atan2(u[1], u[0]) - heading
                power = min(100.0, np.linalg.norm(u) / DASH_POWER_RATE)
                self.decision = (power, ang)
                self.decision_expiring = 3
                return f"dash {power:.1f} {ang}"

            power, ang = self.decision
            self.decision_expiring -= 1
            return f"dash {power:.1f} {ang}"
        return self.goto(
            target_pos[0],
            target_pos[1],
            [self_p0[0], self_p0[1], heading],
            game_state,
            theta=theta,
            margin=0.05,
        )

class InterceptDemoAI(SoccerAI):
    def __init__(self, team_info):
        super().__init__(team_info)
        self.decay = PLAYER_DECAY
        self.ball_decay = BALL_DECAY
        self.ball_history = []
        self.shooter_id = None
        self.receiver_id = None
        self.shooter_turn = True
        self.shooter = InterceptPlayer(decay=self.decay, ball_decay=self.ball_decay, is_goalie=False)
        self.receiver = InterceptPlayer(decay=self.decay, ball_decay=self.ball_decay, is_goalie=True)

    def decide_action(self, game_state, teamname: str):
        actions = []
        robots = game_state.robot_poses[teamname]
        mode = getattr(game_state, "playmode", None) or getattr(game_state, "gamemode", None)
        if mode is not None:
            print(f"Play mode: {mode}")
            if mode in ("kick_off_r", "kick_off_l"): 
                self._reset_state()
        # Assign shooter only on TritonBots, receiver only on the other team
        if teamname != "TritonBots" and self.shooter_id is None:
            unums = sorted(int(next(iter(r.keys()))) for r in robots)
            if unums:
                self.shooter_id = (teamname, unums[0])
                self.shooter.set_identity(teamname, unums[0])
        if teamname == "TritonBots" and self.receiver_id is None:
            unums = sorted(int(next(iter(r.keys()))) for r in robots)
            if unums:
                self.receiver_id = (teamname, unums[0])
                self.receiver.set_identity(teamname, unums[0])

        # record ball history
        self.ball_history.append(np.array(game_state.ball_pos[:2], dtype=float))
        if len(self.ball_history) > 10:
            self.ball_history = self.ball_history[-10:]

        # estimate ball velocity when we have k=3 samples
        ball_vel_est = np.zeros(2)
        if len(self.ball_history) >= 3:
            positions = np.stack(self.ball_history[-3:])
            ball_vel_est = estimate_ball_velocity(positions, alpha=self.ball_decay)

        for robot in robots:
            unum = int(next(iter(robot.keys())))
            pose = robot[unum]
            self_p0 = np.array([pose[0], pose[1]], dtype=float)
            heading = math.radians(pose[2])
            if mode is not None and (mode == "goal_l" or mode == "goal_r" or mode.startswith("kick_off")):
                actions.append("dash 0 0")
            else:
                if teamname == self.shooter_id[0] and unum == self.shooter_id[1]:
                    actions.append(self._action_shooter(self_p0, heading, ball_vel_est, game_state.ball_pos,
                                                    self.shooter, game_state=game_state))
                    print(f"Shooter cmd for robot {unum}: {actions[-1]}")
                elif teamname == self.receiver_id[0] and unum == self.receiver_id[1]:
                    actions.append(self._action_receiver(self_p0, heading, ball_vel_est, game_state.ball_pos,
                                                        self.receiver, game_state=game_state))
                    print(f"Receiver cmd for robot {unum}: {actions[-1]}")
                else:
                    # simple chase
                    print("Extra robot, just stay still")
                    actions.append(f"dash 0 0")
        print('----------------')
        return actions

    def _reset_state(self) -> None:
        self.ball_history = []
        self.shooter_turn = True
        self.shooter.reset()
        self.receiver.reset()

    def _receiver_region(self):
        return (FIELD_X[0], FIELD_X[0] + 45, -30, 30)

    def _inside_region(self, pos, region) -> bool:
        x, y = float(pos[0]), float(pos[1])
        xmin, xmax, ymin, ymax = region
        return xmin <= x <= xmax and ymin <= y <= ymax

    def _action_shooter(self, self_p0, heading, ball_vel_est, ball_pos, player: InterceptPlayer,
                        game_state=None):
        if not self.shooter_turn:
            print('Waiting for receiver to catch')
            return "dash 0 0"
        ball = np.array(ball_pos[:2], dtype=float)
        if player.hasBall([*self_p0, heading], ball_pos):
            print('FLAG2')
            cmd = player.shoot_at_goal(GOAL_L, [*self_p0, heading], ball, kick_power=100)
            if "kick" in cmd:
                self.shooter_turn = False
            elif cmd == "failed":
                print('[Warning] Shooter kick failed')
            return cmd
        desired_theta = math.atan2(GOAL_L[1] - self_p0[1], GOAL_L[0] - self_p0[0])
        print('FLAG3')
        cmd = player.hybrid_capture(
            self_p0,
            heading,
            ball_vel_est,
            ball_pos,
            desired_theta,
            game_state=game_state,
        )
        return "dash 0 0" if cmd == "done" else cmd

    def _action_receiver(self, self_p0, heading, ball_vel_est, ball_pos, player: InterceptPlayer,
                         game_state=None, close_threshold=10.0, slow_threshold=0.5):
        ball = np.array(ball_pos[:2], dtype=float)
        to_ball = ball - self_p0
        dist = np.linalg.norm(to_ball)
        ball_speed = np.linalg.norm(ball_vel_est)
        region = self._receiver_region()
        if player.shoot_angle == 0:
            player.shoot_angle = math.radians(random.uniform(-30, 30)) + np.pi
        if not self._inside_region(ball, region):
            print('FLAG1')
            return "dash 0 0"

        if self.shooter_turn:
            ang = math.atan2(to_ball[1], to_ball[0])
            goto_cmd = player.goto(
                GOAL_L[0] + 10,
                GOAL_L[1],
                [self_p0[0], self_p0[1], heading],
                game_state,
                theta=ang,
                margin=0.5,
            )
            return goto_cmd if goto_cmd != "done" else "dash 0 0"
        
        kick_angle = None
        if dist <= close_threshold and ball_speed <= slow_threshold:
            kick_pos = ball + np.array([np.cos(player.shoot_angle), np.sin(player.shoot_angle)]) * (PLAYER_SIZE + BALL_SIZE)
            kick_angle = math.atan2(ball[1] - kick_pos[1], ball[0] - kick_pos[0])
            cmd = player.goto(
                kick_pos[0],
                kick_pos[1],
                [self_p0[0], self_p0[1], heading],
                game_state,
                theta=kick_angle,
                speed=50.0,
            )
            if cmd != "done":
                return cmd
        
        if player.hasBall([*self_p0, heading], ball_pos):
            if kick_angle is None:
                kick_pos = ball + np.array([np.cos(player.shoot_angle), np.sin(player.shoot_angle)]) * (PLAYER_SIZE + BALL_SIZE)
                kick_angle = math.atan2(ball[1] - kick_pos[1], ball[0] - kick_pos[0])
            cmd = player.kick(kick_angle, [*self_p0, heading], ball)
            if "kick" in cmd:
                self.shooter_turn = True
                return cmd
            elif cmd == "failed":
                print('[Warning] Receiver kick failed')
            return cmd
        else:
            print('Ball distance:', dist, 'Ball angle:', math.degrees(math.atan2(to_ball[1], to_ball[0])), 'Heading:', math.degrees(heading))
        cmd = player.hybrid_capture(
            self_p0,
            heading,
            ball_vel_est,
            ball_pos,
            math.pi,
            rect_bounds=(FIELD_X[0], FIELD_X[0]+30, -20 , 20),
            game_state=game_state,
        )
        return cmd

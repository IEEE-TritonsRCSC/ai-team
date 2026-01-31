import math
import numpy as np
from typing import Tuple, Optional, List
from ai_interface.constants.field_constants import (
    MAX_KEEPER_OUT,
    PLAYER_DECAY,
    BALL_DECAY,
    dt,
    FIELD_X,
    FIELD_Y,
)
from ai_interface.constants.player_constants import DASH_POWER_RATE
from ai_interface.utils.intercept import earliest_intercept_control
from ai_interface.utils.algo_utils import (
    estimate_ball_velocity,
    normalize_angle,
    get_side,
    get_goal_params,
    compute_bisector_target,
)
from ai_interface.player import Player

class Goalie(Player):
    """
    Goalkeeper class that inherits from Player.
    """
    
    def __init__(self,
                 teamname: str,
                 unum: int,
                 side: str = None,
                 charge_distance: float = 15,
                 dribbling: bool = False) -> None:
        super().__init__(teamname, unum, dribbling=dribbling, is_goalie=True)
        self.side = side
        self.charge_distance = charge_distance

        self.ball_history: List[np.ndarray] = []
        self.goalie_history: List[np.ndarray] = []
        self.max_history_length = 5

        self.threat_speed_threshold = 0.5
        self.threat_distance_max = 20.0
        self.goal_mouth_padding = 2.0
        self.catch_distance = 1.2
        self.alignment_angle_threshold = math.radians(5.0)
        self.max_player_speed = 1.05
        self.max_dash_power = 100.0
    
    def _is_ball_threat(self, ball_pos: Tuple[float, float], 
                       ball_vel: np.ndarray,
                       side: Optional[str] = None) -> bool:
        """
        True only if all three conditions hold:
        1. Ball speed > threat_speed_threshold
        2. Ball within threat_distance_max (default 20 m) of our goal line
        3. Ball shot toward the goal: velocity toward goal and trajectory
           would hit the goal mouth (posts ± goal_mouth_padding)
        """
        side = side if side is not None else self.side
        if side is None:
            side = "left" if ball_pos[0] < 0 else "right"

        ball_speed = float(np.linalg.norm(ball_vel))
        if ball_speed < self.threat_speed_threshold:
            return False

        post_top, post_bottom, goal_center = get_goal_params(side)
        goal_x = float(goal_center[0])

        dist_to_goal_line = abs(ball_pos[0] - goal_x)
        if dist_to_goal_line > self.threat_distance_max:
            return False

        if side == "left":
            if ball_vel[0] >= -0.3:
                return False
        else:
            if ball_vel[0] <= 0.3:
                return False

        bx, by = ball_pos[0], ball_pos[1]
        vx, vy = float(ball_vel[0]), float(ball_vel[1])
        if abs(vx) < 1e-9:
            return False
        t = (goal_x - bx) / vx
        if t < 0:
            return False
        y_at_goal = by + t * vy

        post_y_lo = min(post_top[1], post_bottom[1]) - self.goal_mouth_padding
        post_y_hi = max(post_top[1], post_bottom[1]) + self.goal_mouth_padding
        if not (post_y_lo <= y_at_goal <= post_y_hi):
            return False

        return True
    
    def _find_nearest_teammate(self, goalie_pos: Tuple[float, float],
                              game_state) -> Optional[Tuple[float, float]]:
        return super().find_nearest_teammate(goalie_pos, game_state)

    def action(self, ball_pos: Tuple[float, float],
               goalie_pose: Tuple[float, float, float],
               has_ball: bool = None,
               goalie_to_ball_dist: float = None,
               game_state=None) -> str:
        """
        Compute goalkeeper action command using enhanced logic:
        1. Always stays at angular bisector when not threatened
        2. Uses intercept control when ball is shot toward goal
        3. Catches and clears/passes when in possession
        """
        goalie_pos_2d = (goalie_pose[0], goalie_pose[1])
        goalie_pose_rad = (goalie_pose[0], goalie_pose[1], math.radians(goalie_pose[2]))

        self.ball_history.append(np.array(ball_pos, dtype=float))
        if len(self.ball_history) > self.max_history_length:
            self.ball_history = self.ball_history[-self.max_history_length:]

        self.goalie_history.append(np.array(goalie_pos_2d, dtype=float))
        if len(self.goalie_history) > self.max_history_length:
            self.goalie_history = self.goalie_history[-self.max_history_length:]

        if goalie_to_ball_dist is None:
            goalie_to_ball_dist = math.hypot(ball_pos[0] - goalie_pose[0],
                                            ball_pos[1] - goalie_pose[1])

        if has_ball is None:
            has_ball = super().hasBall(goalie_pose_rad, ball_pos, check_angle=False)

        side = self.side if self.side is not None else get_side(goalie_pos_2d)

        # ========== PRIORITY 1: Ball Possession and Clearing ==========
        if has_ball:
            teammate_pos = super().find_nearest_teammate(goalie_pos_2d, game_state)
            if teammate_pos is not None:
                return super().pass_to_teammate(
                    teammate_pos,
                    goalie_pose_rad,
                    ball_pos,
                )
            else:
                return super().kick(
                    target_angle=0,
                    self_pose=goalie_pose_rad,
                    ball_pose=ball_pos,
                    kick_power=100,
                    allow_dribble=False,
                    game_state=game_state,
                )
        
        # ========== PRIORITY 2: Catch Attempt (close to ball) ==========
        if goalie_to_ball_dist < self.catch_distance:
            to_ball_angle = math.atan2(ball_pos[1] - goalie_pose[1],
                                      ball_pos[0] - goalie_pose[0])
            angle_diff = normalize_angle(to_ball_angle - math.radians(goalie_pose[2]))

            if abs(angle_diff) < self.alignment_angle_threshold:
                return "catch 0"
            return super().goto(
                ball_pos[0], ball_pos[1], goalie_pose_rad, game_state,
                margin=0.1, theta=to_ball_angle, speed=50.0,
            )

        # ========== PRIORITY 3: Intercept Mode (ball shot toward goal) ==========
        if len(self.ball_history) >= 2:
            try:
                ball_vel = estimate_ball_velocity(
                    np.stack(self.ball_history[-3:] if len(self.ball_history) >= 3 else self.ball_history),
                    BALL_DECAY,
                )
            except Exception:
                ball_vel = np.zeros(2)
        else:
            ball_vel = np.zeros(2)

        is_threat = self._is_ball_threat(ball_pos, ball_vel, side)

        if is_threat and len(self.ball_history) >= 2:
            try:
                goalie_vel = super().estimate_velocity(goalie_pos_2d, self.goalie_history, dt=dt)

                self_p0 = np.array(goalie_pos_2d, dtype=float)
                self_v0 = goalie_vel
                target_p0 = np.array([ball_pos[0], ball_pos[1]], dtype=float)
                target_v0 = ball_vel
                
                _, _, goal_center = get_goal_params(side)
                if side == "left":
                    rect_bounds = (FIELD_X[0], goal_center[0] + MAX_KEEPER_OUT, 
                                  FIELD_Y[0], FIELD_Y[1])
                else:
                    rect_bounds = (goal_center[0] - MAX_KEEPER_OUT, FIELD_X[1],
                                  FIELD_Y[0], FIELD_Y[1])

                u_max = self.max_dash_power * DASH_POWER_RATE
                intercept_result = earliest_intercept_control(
                    self_p0=self_p0,
                    self_v0=self_v0,
                    target_p0=target_p0,
                    target_v0=target_v0,
                    player_decay=PLAYER_DECAY,
                    ball_decay=BALL_DECAY,
                    vmax_self=self.max_player_speed,
                    u_max=u_max,
                    interval=0.5,
                    T_hi=10.0,
                    tol=0.1,
                    dt=dt,
                    rect_bounds=rect_bounds
                )
                
                if intercept_result is not None:
                    intercept_pos = intercept_result['intercept']
                    intercept_angle = intercept_result['angle']

                    goto_cmd = super().goto(
                        float(intercept_pos[0]), float(intercept_pos[1]),
                        goalie_pose_rad, game_state,
                        margin=0.2, theta=float(intercept_angle), speed=100.0,
                    )
                    return goto_cmd if goto_cmd != "done" else "dash 0 0"
            except Exception as e:
                print(f"Intercept calculation failed: {e}")
                pass
        
        # ========== PRIORITY 4: Default - Angular Bisector Positioning ==========
        target_pos = compute_bisector_target(ball_pos, goalie_pos_2d, side)
        tx, ty = target_pos
        gx, gy = goalie_pos_2d[0], goalie_pos_2d[1]

        dist_to_target = math.hypot(tx - gx, ty - gy)
        facing_ball = math.atan2(ball_pos[1] - gy, ball_pos[0] - gx)

        if dist_to_target < 0.5:
            goto_cmd = super().goto(
                tx, ty, goalie_pose_rad, game_state,
                margin=0.5, theta=facing_ball, speed=0.0,
            )
            return goto_cmd if goto_cmd != "done" else "dash 0 0"

        goto_cmd = super().goto(
            tx, ty, goalie_pose_rad, game_state,
            margin=0.1, speed=100.0,
        )
        return goto_cmd if goto_cmd != "done" else "dash 0 0"


# TODO : Backward compatibility: Will be removed in the future. Used for testing purposes.
def goalie_action(ball_pos: Tuple[float, float],
                  goalie_pose: Tuple[float, float, float],
                  has_ball: bool,
                  goalie_to_ball_dist: float,
                  side: str = None,
                  charge_distance: float = 15,
                  game_state=None) -> str:
    """
    Compute goalkeeper action command using angle-bisector positioning.
    
    This is a backward-compatibility wrapper around the Goalie class.
    """
    goalie_pos = (goalie_pose[0], goalie_pose[1])
    inferred_side = get_side(goalie_pos) if side is None else side
    
    goalie = Goalie(
        teamname="temp",
        unum=1,
        side=inferred_side,
        charge_distance=charge_distance
    )
    
    return goalie.action(
        ball_pos=ball_pos,
        goalie_pose=goalie_pose,
        has_ball=has_ball,
        goalie_to_ball_dist=goalie_to_ball_dist,
        game_state=game_state
    )

import math
import numpy as np
from scipy.optimize import least_squares

from ai_interface.naive import SoccerAI
from ai_interface.utils.basic_commands import goto
from ai_interface.utils.algo_utils import estimate_ball_velocity, normalize_angle
from ai_interface.constants.field_constants import BALL_DECAY
from ai_interface.constants.player_constants import (
    BALL_SIZE,
    KICKABLE_MARGIN,
    PLAYER_SIZE,
)

EPS = 1e-9


def _simulate_discrete(p0, v0, u, decay, vmax, n_steps):
    p = np.asarray(p0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    u = np.asarray(u, dtype=float)

    pos_hist = np.empty((n_steps, p.shape[0]), dtype=float)
    vel_hist = np.empty((n_steps, v.shape[0]), dtype=float)
    pos_hist[0] = p
    vel_hist[0] = v

    for i in range(1, n_steps):
        v = decay * (v + u)
        speed = np.linalg.norm(v)
        if speed > vmax + EPS:
            v = v * (vmax / (speed + EPS))
        p = p + v
        pos_hist[i] = p
        vel_hist[i] = v

    return pos_hist, vel_hist




def _init_fit_stats():
    return {"sse": 0.0, "sum": 0.0, "sum_sq": 0.0, "n": 0}


def _accumulate_fit_stats(stats, positions, velocities, u, decay, vmax) -> None:
    pos_hat, vel_hat = _simulate_discrete(
        positions[0], velocities[0], u, decay, vmax, positions.shape[0]
    )
    residual = (vel_hat - velocities).reshape(-1)
    stats["sse"] += float(np.sum(residual * residual))
    v_flat = velocities.reshape(-1)
    stats["sum"] += float(np.sum(v_flat))
    stats["sum_sq"] += float(np.sum(v_flat * v_flat))
    stats["n"] += int(v_flat.size)


def _accumulate_variable_u_stats(stats, positions, powers, headings,
                                 decay, dash_rate, vmax) -> None:
    steps = min(positions.shape[0] - 1, powers.shape[0], headings.shape[0])
    if steps < 2:
        return
    pos_trim = positions[:steps + 1]
    vel = pos_trim[1:] - pos_trim[:-1]
    power_steps = powers[:steps]
    heading_steps = headings[:steps]
    residuals = []
    targets = []
    for i in range(steps - 1):
        u_vec = dash_rate * power_steps[i] * heading_steps[i]
        v_pred = decay * vel[i] + u_vec
        speed = np.linalg.norm(v_pred)
        if speed > vmax + EPS:
            v_pred = v_pred * (vmax / (speed + EPS))
        residuals.append(v_pred - vel[i + 1])
        targets.append(vel[i + 1])
    if not residuals:
        return
    residual = np.concatenate(residuals).reshape(-1)
    target = np.concatenate(targets).reshape(-1)
    stats["sse"] += float(np.sum(residual * residual))
    stats["sum"] += float(np.sum(target))
    stats["sum_sq"] += float(np.sum(target * target))
    stats["n"] += int(target.size)


def _finalize_fit_stats(stats):
    n = stats["n"]
    if n <= 0:
        return None
    sse = stats["sse"]
    mean = stats["sum"] / n
    sst = stats["sum_sq"] - (stats["sum"] * mean)
    rmse = math.sqrt(sse / n)
    if sst <= EPS:
        r2 = None
    else:
        r2 = 1.0 - (sse / sst)
    return {"rmse": rmse, "r2": r2, "n": n}


class ParamEstimatorAI(SoccerAI):
    """
    Single-robot AI for simulator data collection.
    Ball mode: go behind the ball, kick with angle 0, then wait until the ball stops.
    Player mode: dash with fixed powers for 20 steps, wait until stopped, then turn pi.
    """

    def __init__(
        self,
        team_info,
        mode: str,
        stop_speed_thresh: float = 0.02,
        stop_steps: int = 3,
        kick_power: int = 60,
        ball_decay: float = BALL_DECAY,
        traj_start_delay: int = 3,
    ):
        super().__init__(team_info)
        self.mode = mode
        self.stop_speed_thresh = float(stop_speed_thresh)
        self.stop_steps = int(stop_steps)
        self.kick_power = int(kick_power)
        self.ball_decay = float(ball_decay)
        self.traj_start_delay = int(traj_start_delay)
        self.wait_for_stop = False
        self.ball_moved_since_kick = False
        self.ball_still_steps = 0
        self.last_ball_pos = None
        self.ball_trajectories = []
        self.current_traj = []
        self.traj_pending_steps = 0
        self.traj_active = False
        self.traj_stationary_steps = 0
        self.player_powers = [100, 60, 30]
        self.player_dash_steps = 20
        self.player_run_index = 0
        self.player_dash_steps_left = self.player_dash_steps
        self.player_phase = "turn"
        self.player_turn_target = None
        self.player_trajectories = []
        self.player_current_positions = []
        self.player_current_powers = []
        self.player_current_headings = []
        self.player_traj_active = False
        self.player_still_steps = 0
        self.player_moved_since_start = False

    def _update_ball_stop_state(self, ball_pos: tuple[float, float]) -> None:
        if self.last_ball_pos is None:
            self.last_ball_pos = np.array(ball_pos, dtype=float)
            return

        current = np.array(ball_pos, dtype=float)
        delta = np.linalg.norm(current - self.last_ball_pos)
        moving = delta > self.stop_speed_thresh
        self.last_ball_pos = current

        if not self.wait_for_stop:
            self.ball_still_steps = 0
            self.ball_moved_since_kick = False
            return

        if moving:
            self.ball_moved_since_kick = True
            self.ball_still_steps = 0
            return

        if self.ball_moved_since_kick:
            self.ball_still_steps += 1

        if self.ball_moved_since_kick and self.ball_still_steps >= self.stop_steps:
            self.wait_for_stop = False
            self.ball_moved_since_kick = False
            self.ball_still_steps = 0

    def _update_ball_trajectory(self, ball_pos: tuple[float, float]) -> None:
        if self.traj_pending_steps > 0:
            self.traj_pending_steps -= 1
            if self.traj_pending_steps == 0:
                self.current_traj = []
                self.traj_active = True
                self.traj_stationary_steps = 0

        if not self.traj_active:
            return

        self.current_traj.append(np.array(ball_pos[:2], dtype=float))
        if len(self.current_traj) < 2:
            return

        window = np.stack(self.current_traj[-3:])
        vel_est = estimate_ball_velocity(window, alpha=self.ball_decay)
        speed = np.linalg.norm(vel_est)
        if speed <= self.stop_speed_thresh:
            self.traj_stationary_steps += 1
        else:
            self.traj_stationary_steps = 0

        if self.traj_stationary_steps >= self.stop_steps:
            self.ball_trajectories.append(np.stack(self.current_traj))
            self.traj_active = False
            self.current_traj = []
            self.traj_stationary_steps = 0

    def decide_action(self, game_state, teamname: str):
        if self.mode == "player":
            return self._decide_action_player(game_state, teamname)
        return self._decide_action_ball(game_state, teamname)

    def _decide_action_ball(self, game_state, teamname: str):
        actions = []
        ball_pos = game_state.ball_pos
        team_robots = game_state.robot_poses.get(teamname, [])
        if ball_pos is None or not team_robots:
            return actions

        self._update_ball_stop_state(ball_pos)
        self._update_ball_trajectory(ball_pos)
        if self.wait_for_stop:
            actions.append("turn 0")
            return actions

        ball = np.array(ball_pos[:2], dtype=float)
        if not np.isclose(np.linalg.norm(ball), 0.0, atol=1.0):
            desired_theta = math.atan2(-ball[1], -ball[0])
        else:
            desired_theta = 0.0
        print(f"Ball at {ball}, desired theta: {math.degrees(desired_theta):.1f} deg")
        offset = np.array([math.cos(desired_theta), math.sin(desired_theta)]) * (PLAYER_SIZE + BALL_SIZE)
        target_pos = ball - offset

        robot = team_robots[0]
        unum = int(next(iter(robot.keys())))
        robot_pose = robot[unum]

        robot_pos = (robot_pose[0], robot_pose[1])
        robot_to_ball_dist = self.get_dist(ball_pos, robot_pos)

        heading = normalize_angle(math.radians(robot_pose[2]))
        if robot_to_ball_dist < (KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE) and np.isclose(
            heading, desired_theta, atol=math.radians(10.0)
        ):
            actions.append(f"kick {self.kick_power} 0")
            self.wait_for_stop = True
            self.ball_moved_since_kick = False
            self.ball_still_steps = 0
            self.traj_pending_steps = self.traj_start_delay
            self.traj_active = False
            self.current_traj = []
            self.traj_stationary_steps = 0
            return actions
        else:
            print(
                f"Robot at {robot_pos} with heading {math.degrees(heading):.1f} deg, moving to {target_pos}"
            )

        goto_cmd = goto(
            np.array([robot_pose[0], robot_pose[1], heading], dtype=float),
            target_pos[0],
            target_pos[1],
            theta=desired_theta,
            margin=0.05,
            detour_margin=0.9,
            game_state=game_state,
        )
        actions.append("dash 0 0" if goto_cmd == "done" else goto_cmd)
        return actions

    def _start_player_trajectory(self) -> None:
        self.player_traj_active = True
        self.player_current_positions = []
        self.player_current_powers = []
        self.player_current_headings = []
        self.player_still_steps = 0
        self.player_moved_since_start = False

    def _record_player_step(
        self,
        position: tuple[float, float],
        power: float,
        heading_vec: np.ndarray,
        check_stop: bool,
    ) -> None:
        if not self.player_traj_active:
            return
        pos_arr = np.array(position[:2], dtype=float)
        self.player_current_positions.append(pos_arr)
        self.player_current_powers.append(float(power))
        self.player_current_headings.append(np.array(heading_vec, dtype=float))
        if len(self.player_current_positions) < 2:
            return
        delta = self.player_current_positions[-1] - self.player_current_positions[-2]
        speed = np.linalg.norm(delta)
        if speed > self.stop_speed_thresh:
            self.player_moved_since_start = True
            self.player_still_steps = 0
            return
        if check_stop and self.player_moved_since_start:
            self.player_still_steps += 1

    def _finish_player_trajectory(self) -> None:
        if not self.player_traj_active:
            return
        if self.player_current_positions:
            positions = np.stack(self.player_current_positions)
        else:
            positions = np.empty((0, 2))
        if self.player_current_powers:
            powers = np.array(self.player_current_powers, dtype=float)
        else:
            powers = np.empty((0,))
        if self.player_current_headings:
            headings = np.stack(self.player_current_headings)
        else:
            headings = np.empty((0, 2))
        self.player_trajectories.append(
            {"positions": positions, "powers": powers, "headings": headings}
        )
        self.player_traj_active = False
        self.player_current_positions = []
        self.player_current_powers = []
        self.player_current_headings = []
        self.player_still_steps = 0
        self.player_moved_since_start = False

    def _decide_action_player(self, game_state, teamname: str):
        actions = []
        team_robots = game_state.robot_poses.get(teamname, [])
        if not team_robots:
            return actions

        robot = team_robots[0]
        unum = int(next(iter(robot.keys())))
        robot_pose = robot[unum]
        position = (robot_pose[0], robot_pose[1])
        heading = normalize_angle(math.radians(robot_pose[2]))

        if self.player_phase == "done":
            actions.append("dash 0 0")
            return actions

        if self.player_phase == "turn":
            if self.player_turn_target is None:
                self.player_turn_target = normalize_angle(heading + math.pi)
            angle_diff = normalize_angle(self.player_turn_target - heading)
            if abs(angle_diff) <= math.radians(5.0):
                self.player_phase = "dash"
                self.player_turn_target = None
                self.player_dash_steps_left = self.player_dash_steps
                actions.append("dash 0 0")
                return actions
            actions.append(f"turn {angle_diff * 10}")
            return actions

        if self.player_phase == "dash":
            if self.player_run_index >= len(self.player_powers):
                self.player_phase = "done"
                actions.append("dash 0 0")
                return actions
            if self.player_dash_steps_left == self.player_dash_steps and not self.player_traj_active:
                self._start_player_trajectory()
            power = self.player_powers[self.player_run_index]
            heading_vec = np.array([math.cos(heading), math.sin(heading)])
            self._record_player_step(position, power, heading_vec, check_stop=False)
            actions.append(f"dash {power} 0")
            self.player_dash_steps_left -= 1
            if self.player_dash_steps_left <= 0:
                self.player_phase = "coast"
            return actions

        if self.player_phase == "coast":
            heading_vec = np.zeros(2)
            self._record_player_step(position, 0.0, heading_vec, check_stop=True)
            if self.player_moved_since_start and self.player_still_steps >= self.stop_steps:
                self._finish_player_trajectory()
                if self.player_run_index < len(self.player_powers) - 1:
                    self.player_run_index += 1
                    self.player_phase = "turn"
                    self.player_turn_target = normalize_angle(heading + math.pi)
                else:
                    self.player_phase = "done"
                actions.append("dash 0 0")
                return actions
            actions.append("dash 0 0")
            return actions

        actions.append("dash 0 0")
        return actions

    def estimate(self):
        if self.mode == "player":
            return self._estimate_player_decay()
        if self.mode != "ball":
            print(f"Unknown mode for estimation: {self.mode}")
            return None
        return self._estimate_ball_decay()

    def _estimate_ball_decay(self):
        trajectories = self.ball_trajectories
        if not trajectories:
            print("No trajectories available for estimation.")
            return None
        print(f"Estimating ball_decay from {len(trajectories)} trajectories.")
        v_prev_list = []
        v_next_list = []
        step_count = 0
        for traj in trajectories:
            positions = np.asarray(traj, dtype=float)
            if positions.shape[0] < 3:
                continue
            velocities = positions[1:] - positions[:-1]
            v_prev = velocities[:-1]
            v_next = velocities[1:]
            if v_prev.size == 0:
                continue
            v_prev_list.append(v_prev)
            v_next_list.append(v_next)
            step_count += v_prev.shape[0]

        if not v_prev_list:
            print("Insufficient motion to estimate ball_decay.")
            return None

        v_prev = np.concatenate(v_prev_list, axis=0)
        v_next = np.concatenate(v_next_list, axis=0)
        denom = float(np.sum(v_prev * v_prev))
        if denom > EPS:
            init_decay = float(np.sum(v_next * v_prev) / denom)
        else:
            init_decay = 0.9
        init_decay = float(np.clip(init_decay, EPS, 1.0 - EPS))

        def residual(params):
            decay = params[0]
            return (v_next - decay * v_prev).reshape(-1)

        res = least_squares(
            residual,
            x0=np.array([init_decay], dtype=float),
            bounds=(np.array([0.0]), np.array([1.0])),
        )
        decay = float(res.x[0])
        print(f"Estimated ball_decay: {decay:.3f} from {step_count} steps.")
        fit_stats = _init_fit_stats()
        for traj in trajectories:
            positions = np.asarray(traj, dtype=float)
            if positions.shape[0] < 3:
                continue
            pos_aligned = positions[1:]
            vel_aligned = positions[1:] - positions[:-1]
            if pos_aligned.shape[0] < 2:
                continue
            _accumulate_fit_stats(
                fit_stats,
                pos_aligned,
                vel_aligned,
                u=np.zeros(2),
                decay=decay,
                vmax=np.inf,
            )
        metrics = _finalize_fit_stats(fit_stats)
        if metrics is None:
            print("Ball fit: insufficient data.")
        else:
            r2 = metrics["r2"]
            r2_str = "nan" if r2 is None else f"{r2:.4f}"
            print(f"Ball fit: rmse={metrics['rmse']:.3f}, r2={r2_str}, n={metrics['n']}")
        return {"decay": decay}

    def _estimate_player_decay(self):
        trajectories = self.player_trajectories
        if not trajectories:
            print("No player trajectories available for estimation.")
            return None
        print(f"Estimating player_decay from {len(trajectories)} trajectories.")
        numer = 0.0
        denom = 0.0
        coast_segments = []
        for traj in trajectories:
            positions = np.asarray(traj.get("positions", []), dtype=float)
            powers = np.asarray(traj.get("powers", []), dtype=float)
            if positions.shape[0] < 3 or powers.shape[0] < 2:
                continue
            steps = min(positions.shape[0] - 1, powers.shape[0])
            power_steps = powers[:steps]
            zero_mask = np.isclose(power_steps, 0.0, atol=EPS)
            idx = 0
            while idx < steps:
                if not zero_mask[idx]:
                    idx += 1
                    continue
                start = idx
                while idx < steps and zero_mask[idx]:
                    idx += 1
                end = idx - 1
                if end - start + 1 < 2:
                    continue
                pos_seg = positions[start:end + 2]
                if pos_seg.shape[0] < 3:
                    continue
                pos_aligned = pos_seg[1:]
                vel_aligned = pos_seg[1:] - pos_seg[:-1]
                if pos_aligned.shape[0] < 2:
                    continue
                coast_segments.append((pos_aligned, vel_aligned))
                v_prev = vel_aligned[:-1]
                v_next = vel_aligned[1:]
                numer += float(np.sum(v_next * v_prev))
                denom += float(np.sum(v_prev * v_prev))

        if denom <= EPS:
            print("Insufficient coast motion to estimate player_decay.")
            return None

        decay = numer / denom
        print(
            f"Estimated player_decay from coast: {decay:.3f}"
        )
        coast_stats = _init_fit_stats()
        for pos_aligned, vel_aligned in coast_segments:
            _accumulate_fit_stats(
                coast_stats,
                pos_aligned,
                vel_aligned,
                u=np.zeros(2),
                decay=decay,
                vmax=np.inf,
            )
        coast_metrics = _finalize_fit_stats(coast_stats)
        if coast_metrics is None:
            print("Player coast fit: insufficient data.")
        else:
            r2 = coast_metrics["r2"]
            r2_str = "nan" if r2 is None else f"{r2:.4f}"
            print(
                "Player coast fit: rmse="
                f"{coast_metrics['rmse']:.3f}, r2={r2_str}, n={coast_metrics['n']}"
            )

        dash_rate_init = None
        dash_rate_samples = []
        for traj in trajectories:
            positions = np.asarray(traj.get("positions", []), dtype=float)
            powers = np.asarray(traj.get("powers", []), dtype=float)
            headings = np.asarray(traj.get("headings", []), dtype=float)
            if positions.shape[0] < 3 or powers.shape[0] < 2 or headings.shape[0] < 2:
                continue
            steps = min(positions.shape[0] - 1, powers.shape[0], headings.shape[0])
            if steps < 2:
                continue
            pos_trim = positions[:steps + 1]
            vel = pos_trim[1:] - pos_trim[:-1]
            power_steps = powers[:steps]
            heading_steps = headings[:steps]
            for i in range(1, steps):
                if power_steps[i - 1] <= EPS:
                    continue
                dir_vec = heading_steps[i - 1]
                if np.linalg.norm(dir_vec) <= EPS:
                    continue
                approx = np.dot((vel[i] / decay - vel[i - 1]), dir_vec) / power_steps[i - 1]
                if approx > EPS:
                    dash_rate_samples.append(float(approx))

        if dash_rate_samples:
            dash_rate_init = float(np.median(dash_rate_samples))
        else:
            raise ValueError("Insufficient dash data to estimate dash_power_rate.")

        def residual(params):
            dash_rate = params[0]
            res_parts = []
            for traj in trajectories:
                positions = np.asarray(traj.get("positions", []), dtype=float)
                powers = np.asarray(traj.get("powers", []), dtype=float)
                headings = np.asarray(traj.get("headings", []), dtype=float)
                if positions.shape[0] < 3 or powers.shape[0] < 2 or headings.shape[0] < 2:
                    continue
                steps = min(positions.shape[0] - 1, powers.shape[0], headings.shape[0])
                if steps < 2:
                    continue
                pos_trim = positions[:steps + 1]
                vel = pos_trim[1:] - pos_trim[:-1]
                power_steps = powers[:steps]
                heading_steps = headings[:steps]
                for i in range(1, steps):
                    # vel[i] = p[i+1] - p[i] = vel[i-1] * decay + dash_rate * u
                    u_vec = dash_rate * power_steps[i - 1] * heading_steps[i - 1]
                    v_pred = vel[i - 1] * decay + u_vec
                    res_parts.append(v_pred - vel[i])
            if not res_parts:
                return np.array([0.0])
            return np.concatenate(res_parts)

        res = least_squares(
            residual,
            x0=np.array([dash_rate_init], dtype=float),
            bounds=(np.array([EPS]), np.array([np.inf])),
        )

        dash_rate_est = float(res.x[0])
        print(
            f"Estimated dash_power_rate: {dash_rate_est:.3f}."
        )
        player_stats = _init_fit_stats()
        for traj in trajectories:
            positions = np.asarray(traj.get("positions", []), dtype=float)
            powers = np.asarray(traj.get("powers", []), dtype=float)
            headings = np.asarray(traj.get("headings", []), dtype=float)
            if positions.shape[0] < 3 or powers.shape[0] < 2 or headings.shape[0] < 2:
                continue
            _accumulate_variable_u_stats(
                player_stats,
                positions,
                powers,
                headings,
                decay=decay,
                dash_rate=dash_rate_est,
                vmax=np.inf,
            )
        player_metrics = _finalize_fit_stats(player_stats)
        if player_metrics is None:
            print("Player fit: insufficient data.")
        else:
            r2 = player_metrics["r2"]
            r2_str = "nan" if r2 is None else f"{r2:.4f}"
            print(
                "Player fit: rmse="
                f"{player_metrics['rmse']:.3f}, r2={r2_str}, n={player_metrics['n']}"
            )
        return {"decay": decay, "dash_power_rate": dash_rate_est}

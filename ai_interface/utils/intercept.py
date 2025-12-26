import numpy as np
import time
from scipy.optimize import least_squares, brentq
from scipy.optimize import differential_evolution

EPS = 1e-9

def integrate_dynamics(p0, v0, u, decay, vmax, T, dt, return_hist=False):
    """
    Discrete-time integration using capped velocity and constant control u.
    Matches: v <- decay * (v + u); p <- p + v * dt, with speed cap.
    """
    p = np.asarray(p0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    u = np.asarray(u, dtype=float)

    if T <= 0 or dt <= 0:
        return (p, v) if not return_hist else (np.array([0.0]), np.stack([p]), np.stack([v]))

    n_steps = int(np.ceil(T / dt))
    if return_hist:
        times = np.linspace(0.0, n_steps * dt, n_steps + 1)
        pos_hist = np.empty((n_steps + 1, 2), dtype=float)
        vel_hist = np.empty((n_steps + 1, 2), dtype=float)
        pos_hist[0] = p
        vel_hist[0] = v

    for i in range(1, n_steps + 1):
        v = decay * (v + u)
        speed = np.linalg.norm(v)
        if speed > vmax + EPS:
            v = v * (vmax / (speed + EPS))
        p = p + v * dt
        if return_hist:
            pos_hist[i] = p
            vel_hist[i] = v

    if return_hist:
        return times, pos_hist, vel_hist
    return p, v

def solve_u_for_time(T, self_p0, self_v0, pT_target, decay_self, vmax_self, u_max, dt):
    d = pT_target - self_p0
    n = np.linalg.norm(d)
    u_init = (d / (n + EPS)) * (u_max - EPS)
    u_init_polar = np.array([np.linalg.norm(u_init), np.arctan2(u_init[1], u_init[0])])

    def residual(u_vec):
        r = np.clip(u_vec[0], 0.0, u_max - EPS)
        theta = u_vec[1]
        u_cart = np.array([r * np.cos(theta), r * np.sin(theta)])
        pT_self, _ = integrate_dynamics(self_p0, self_v0, u_cart, decay_self, vmax_self, T, dt)
        return pT_self - pT_target

    lb = np.array([0.0, -np.pi - EPS])
    ub = np.array([+u_max - EPS, np.pi + EPS])

    res = least_squares(residual, x0=u_init_polar, bounds=(lb, ub), xtol=1e-2, ftol=1e-2, gtol=1e-2)
    r_star, theta_star = res.x
    r_star = np.clip(r_star, 0.0, u_max)
    u_star = np.array([r_star * np.cos(theta_star), r_star * np.sin(theta_star)])
    miss = np.linalg.norm(res.fun)
    return u_star, miss

def earliest_intercept_control(self_p0, self_v0,
                               target_p0, target_v0,
                               player_decay, ball_decay,
                               vmax_self,
                               u_max, interval=1.0, T_hi=10.0, tol=1e-3, dt=0.1,
                               rect_bounds=None):
    """
    Intercept using discrete dynamics for both self and target.
    player_beta/ball_beta are decay rates mapped to decay via exp(-beta*dt).
    """
    t0 = time.time()

    # Precompute target trajectory once with discrete dynamics
    Ts = np.linspace(0.0, T_hi, int(T_hi / interval))
    t_precompute_start = time.time()
    times_target, pos_target, _ = integrate_dynamics(
        target_p0, target_v0, np.zeros(2), ball_decay, vmax_self, T_hi, dt, return_hist=True
    )
    target_cache = {float(t): p for t, p in zip(times_target, pos_target)}
    t_precompute_end = time.time()

    # Restrict to times when target inside rectangle
    if rect_bounds is None:
        mask_inside = np.ones_like(times_target, dtype=bool)
    else:
        xmin, xmax, ymin, ymax = rect_bounds
        mask_inside = (
            (pos_target[:, 0] >= xmin - EPS) & (pos_target[:, 0] <= xmax + EPS) &
            (pos_target[:, 1] >= ymin - EPS) & (pos_target[:, 1] <= ymax + EPS)
        )
    # Reachability pre-filter: distance must be within a conservative bound
    dist = np.linalg.norm(pos_target - self_p0, axis=1)
    reach = vmax_self * times_target + np.linalg.norm(self_v0) * dt / max(EPS, 1 - player_decay)
    mask_reach = dist <= reach + tol

    Ts_inside = [float(t) for t, m_rect, m_reach in zip(times_target, mask_inside, mask_reach) if m_rect and m_reach and t <= T_hi]
    if not Ts_inside:
        return None

    t_bracket_start = time.time()
    # helper to interpolate target position
    def interp_target(T):
        T = float(T)
        if T <= times_target[0]:
            return pos_target[0]
        if T >= times_target[-1]:
            return pos_target[-1]
        px = np.interp(T, times_target, pos_target[:, 0])
        py = np.interp(T, times_target, pos_target[:, 1])
        return np.array([px, py])

    best = None
    for T in Ts_inside:
        pT_target = target_cache.get(T, interp_target(T))
        u_star, miss = solve_u_for_time(
            T, self_p0, self_v0, pT_target, player_decay, vmax_self, u_max, dt
        )
        if best is None or miss < best[1]:
            best = (T, miss, u_star, pT_target)
        if miss <= tol:
            T_feas = T
            break
    else:
        return None
    t_bracket_end = time.time()

    # refine between adjacent inside times if needed
    def g(T):
        pT_target = interp_target(T)
        if rect_bounds is not None:
            if not ((xmin - EPS) <= pT_target[0] <= (xmax + EPS) and (ymin - EPS) <= pT_target[1] <= (ymax + EPS)):
                return np.inf
        u_star, miss = solve_u_for_time(
            T, self_p0, self_v0, pT_target, player_decay, vmax_self, u_max, dt
        )
        return miss - tol

    t_refine_start = time.time()
    Th = T_feas
    idx = Ts_inside.index(T_feas)
    Tl = Ts_inside[idx - 1] if idx > 0 else max(0.01, Th - interval)
    while Tl > 0.01 and g(Tl) <= 0:
        Th = Tl
        idx = Ts_inside.index(Th)
        if idx == 0:
            break
        Tl = Ts_inside[idx - 1]

    
    T_star = brentq(g, Tl, Th, xtol=1e-2)
    pT_target = interp_target(T_star)
    u_star, miss = solve_u_for_time(
        T_star, self_p0, self_v0, pT_target, player_decay, vmax_self, u_max, dt
    )
    t_refine_end = time.time()

    print(f"[timing] decay:{t_refine_end - t0:.4f}s precompute:{t_precompute_end - t_precompute_start:.4f}s")
    print(f"bracket:{t_bracket_end - t_bracket_start:.4f}s refine:{t_refine_end - t_refine_start:.4f}s")
    angle = np.arctan2(u_star[1], u_star[0])
    acc = np.linalg.norm(u_star)
    return dict(T=T_star, u=u_star, angle=angle, acc=acc, miss=miss, intercept=pT_target)

def earliest_intercept_control_const_friction(self_p0: np.ndarray, target_p0: np.ndarray, target_v0: np.ndarray, 
                                 target_acc: float, vmax_self: float, rect_bounds=None):
    """
    Calculate the interception position and required velocity for a player to intercept a moving target.
    Args:
        self_p0 (np.ndarray): The player's current position [x, y] (theta ignored).
        target_p0 (np.ndarray): The target's current position as [x, y].
        target_v0 (np.ndarray): The target's current velocity as [vx, vy].
        target_acc (float): The target's acceleration magnitude (colinear with velocity).
        vmax_self (float): The player's maximum speed.
        time_limit (float or 'stop'): Maximum time to intercept. If 'stop', intercept before target stops.
    Returns:
        intercept_pos (np.ndarray): The position where interception occurs as [x, y].
        required_velocity (np.ndarray): The required velocity vector for interception as [vx, vy].
        interception_time (float): The time until interception occurs.
        None, None, None if interception is not possible within constraints.
    Raises:
        ValueError: If time_limit is invalid.
    """
    self_pos = np.array(self_p0, dtype=float)
    target_pos = np.array(target_p0, dtype=float)
    target_vel = np.array(target_v0, dtype=float)

    def inside_rect(p):
        if rect_bounds is None:
            return True
        xmin, xmax, ymin, ymax = rect_bounds
        return (xmin - 1e-9) <= p[0] <= (xmax + 1e-9) and (ymin - 1e-9) <= p[1] <= (ymax + 1e-9)
    
    # Calculate acceleration vector (colinear with velocity)
    vel_magnitude = np.linalg.norm(target_vel)
    if vel_magnitude > 1e-6:
        vel_direction = target_vel / vel_magnitude
        target_acc_vec = target_acc * vel_direction
        if target_acc < -1e-6:
            time_limit = vel_magnitude / -target_acc  # Time to stop
        else:
            time_limit = float('inf')  # No time limit if accelerating or constant speed
    else:
        target_acc_vec = np.zeros(2)
        # Just go to target
        direction_to_target = target_pos - self_pos
        distance_to_target = np.linalg.norm(direction_to_target)
        if not inside_rect(target_pos):
            return None, None, None
        if distance_to_target < 1e-6:
            return target_pos, np.zeros(2), 0.0
        required_velocity = (direction_to_target / distance_to_target) * vmax_self
        return target_pos, required_velocity, distance_to_target / vmax_self

    # Solve for optimal direction angle that minimizes interception time
    # For each direction theta, calculate time to interception with fixed player speed
    def time_to_interception(theta) -> float:
        """Calculate time to interception for a given direction angle theta.
        Returns inf if no feasible interception exists within max_speed constraint.
        """
        # Player velocity direction (unit vector)
        player_dir = np.array([np.cos(theta), np.sin(theta)])
        
        # Set up the interception equation:
        # Player position: self_pos + player_dir * speed * t
        # Target position: target_pos + target_vel * t + 0.5 * target_acc_vec * t^2
        # At interception: player_pos = target_pos
        A = np.array([player_dir, -target_vel]).T
        if np.linalg.matrix_rank(A) < 2:
            return float('inf')  # No valid interception time as directions are colinear
        intersection_parametrized = np.linalg.solve(A, target_pos - self_p0)
        if intersection_parametrized[1] < 0:
            return float('inf')  # No valid interception time as time cannot be negative
        intersection = self_p0 + player_dir * intersection_parametrized[0]
        distance = np.linalg.norm(intersection - target_pos)
        if distance < 1e-6:
            time_to_impact = 0.0  # Already at interception point
        else:
            roots = np.roots([0.5 * target_acc, np.linalg.norm(target_vel), -distance])
            roots = roots[np.isreal(roots)].real  # Keep only real roots
            if len(roots) == 0 or np.all(roots <= 0):
                return float('inf')  # No valid interception time as no positive real roots
            time_to_impact = max(roots) if min(roots) <= 0 else min(roots)
            assert time_to_impact > 0, "Time to impact should be positive"
        if time_to_impact * vmax_self < np.linalg.norm(intersection - self_pos):
            return float('inf')  # Cannot reach interception point in time
        intercept_pos = 0.5 * target_acc_vec * time_to_impact**2 + target_vel * time_to_impact + target_pos
        if not inside_rect(intercept_pos):
            return float('inf')
        return float(time_to_impact)
        
    
    # Find optimal direction angle using optimization
    try:
        def objective_for_de(x):
            return time_to_interception(x[0])
        
        result = differential_evolution(
            objective_for_de,
            bounds=[(0, np.pi)],
            seed=42,
            maxiter=50,
            popsize=20 
        )
        
        if result.success and result.fun < time_limit:
            optimal_theta = result.x[0]
            optimal_time = result.fun
        else:
            return None, None, None
    except Exception as e:
        print(e)
        return None, None, None
    
    # Calculate interception position
    intercept_pos = 0.5 * target_acc_vec * optimal_time**2 + target_vel * optimal_time + target_pos
    # Calculate final interception using optimal direction
    player_vel_dir = np.array([np.cos(optimal_theta), np.sin(optimal_theta)])
    player_intersection_dist = np.linalg.norm(intercept_pos - self_pos)
    required_speed = min(vmax_self, player_intersection_dist / optimal_time)
    required_velocity = required_speed * player_vel_dir * np.sign(np.dot(player_vel_dir, intercept_pos - self_pos))
    return intercept_pos, required_velocity, optimal_time

## ------------------------------Usage1---------------------------------- ##

# decay = 0.4
# ball_decay = 0.94
# power_rate = 0.006
# dt = 0.1

# max_speed_self = 1.05
# max_power_self = 100.0

# self_pos = np.array([0.0, 0.0])
# self_vel = np.array([0.0, 2.0])

# target_pos = np.array([3.0, 1.0])
# target_vel = np.array([-2.0, 1.0])

# res = earliest_intercept_control(
#     self_p0=self_pos,
#     self_v0=self_vel,
#     target_p0=target_pos,
#     target_v0=target_vel,
#     player_decay=decay,
#     ball_decay=ball_decay,
#     vmax_self=max_speed_self,
#     u_max=max_power_self * power_rate,  # u has units of vel increment per step
#     T_hi=20.0,
#     dt=dt,
#     rect_bounds=rect_bounds,
# )

# print("=== Intercept result ===")
# print(f"T*           = {res['T']:.3f} s")
# print(f"acc          = {res['acc']:.3f}")
# print(f"power (sim)  = {res['acc'] / power_rate:.2f}")
# print(f"angle (deg)  = {np.degrees(res['angle']):.2f}")
# print(f"intercept    = {res['intercept']}")

## ------------------------------Usage2---------------------------------- ##

# self_p0 = np.array([0.0, 0.0])
# target_p0 = np.array([10.0, 5.0])  # Ball position
# target_v0 = np.array([-3.0, -1.0])  # Ball moving towards goal
# target_acc = -2.0  # Ball decelerating due to friction
# vmax_self = 8.0
# rect_bounds = (-2.0, 12.0, -2.0, 8.0)

# intercept_pos, required_vel, interception_time = earliest_intercept_control_const_friction(self_p0, target_p0, target_v0, target_acc, vmax_self, rect_bounds=rect_bounds)
# if intercept_pos is not None:
#     print(f"Player at: {self_p0}")
#     print(f"Ball at: {target_p0}, velocity: {target_v0}, deceleration: {target_acc}")
#     print(f"Interception position: [{intercept_pos[0]:.3f}, {intercept_pos[1]:.3f}]")
#     print(f"Required velocity: [{required_vel[0]:.3f}, {required_vel[1]:.3f}]")
#     print(f"Speed: {np.linalg.norm(required_vel):.3f}")
#     print(f"Interception time: {interception_time:.3f}s")

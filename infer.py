"""
Inference entry point using the simulator via Networker, with modular RL components.

Loads a saved model and runs deterministic actions inside the appropriate
environment. Supports:
  - Hierarchical PPO    (--trainer hier_ppo)
  - Discrete PPO        (--trainer discrete_ppo)
  - Stable Baselines3   (--trainer sb3_ppo)
  - Attention MAPPO     (--trainer attention_mappo)

Usage examples:
    python infer.py --trainer hier_ppo --model_path models/hier_ppo_policy.pth
    python infer.py --trainer discrete_ppo --model_path models/discrete_ppo_policy.pth
    python infer.py --trainer discreteq_learning --model_path models/qlearning_policy.pth
    python infer.py --trainer sb3_ppo --model_path models/sb3_ppo_policy.zip
    python infer.py --trainer attention_mappo --model_path models/attention_mappo/attention_mappo_stage1_final.pth \
        --config configs/attention_mappo_config.json --stage stage1
"""
import argparse
import datetime as _dt
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from networking.networker import TeamInfo, Networker


# ---------------------------------------------------------------------------
# Inference logging helpers (shared by TD3 / TD3+HER paths)
# ---------------------------------------------------------------------------

_KICK_RE = re.compile(
    r"Kick fired:.*?aim_quality=([\d.]+)\s+bad_aim=(True|False)"
)


class _KickAimCollector(logging.Handler):
    """Captures per-kick aim_quality from env 'Kick fired:' log lines."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.kicks: list[tuple[float, bool]] = []

    def emit(self, record):
        m = _KICK_RE.search(record.getMessage())
        if m:
            self.kicks.append((float(m.group(1)), m.group(2) == "True"))


def _make_infer_log_dir(model_path: str, log_root: str = "infer_logs") -> Path:
    """Create infer_logs/<timestamp>_<model_basename>/ and return its Path."""
    stem = Path(model_path).stem  # e.g. stage1_9_approach_turn_kick_steps180000
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(log_root) / f"{ts}_{stem}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _setup_inference_logging(
    log_dir: Path,
    debug: bool = False,
) -> tuple[_KickAimCollector, Path]:
    """Route env logs to stdout + file, and attach a kick-aim collector.

    The env emits useful per-episode signals (kick aim, action distribution,
    termination reason) via Python ``logging``. Without a configured handler
    those lines vanish, which is why inference output looked empty before.

    Returns the kick collector and the path to the on-disk log file.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    has_stream = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, (_KickAimCollector, logging.FileHandler))
        for h in root.handlers
    )
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)

    log_file = log_dir / "infer_log.log"
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Silence the networker's chatty per-frame debug output unless --debug_infer
    if not debug:
        logging.getLogger("networking").setLevel(logging.WARNING)

    collector = _KickAimCollector()
    root.addHandler(collector)
    return collector, log_file


def _format_pct(num: int, denom: int) -> str:
    if denom <= 0:
        return "—"
    return f"{100.0 * num / denom:.1f}%"


_SUMMARY_LOG = logging.getLogger("infer.summary")


def _print_window_summary(
    *,
    label: str,
    outcomes: list[str],
    rewards: list[float],
    kicks: list[tuple[float, bool]],
    actions: Counter,
) -> dict:
    """Log a compact aggregate for a window of episodes.

    Goes through the ``infer.summary`` logger so it shows up in both stdout
    and the on-disk log. Returns a dict of the stats for downstream JSON dump.
    """
    n = len(outcomes)
    if n == 0:
        return {"label": label, "n": 0}
    goals = sum(1 for o in outcomes if o == "goal_scored")
    breakdown = Counter(outcomes)
    avg_r = sum(rewards) / n if rewards else 0.0
    if kicks:
        bad = sum(1 for _, b in kicks if b)
        aim_avg = sum(q for q, _ in kicks) / len(kicks)
        kick_str = (
            f"{len(kicks)} kicks  aim={aim_avg:.3f}  bad-aim={_format_pct(bad, len(kicks))}"
        )
    else:
        bad = 0
        aim_avg = 0.0
        kick_str = "0 kicks"
    act_total = sum(actions.values())
    if act_total > 0:
        top = ", ".join(
            f"{k}={100 * v // act_total}%"
            for k, v in actions.most_common()
        )
    else:
        top = "—"
    breakdown_str = ", ".join(f"{k}={v}" for k, v in breakdown.most_common())
    _SUMMARY_LOG.info(
        "[%s] %d eps  goal_rate=%s  avg_reward=%.1f | outcomes: %s | "
        "kicks: %s | actions: %s",
        label, n, _format_pct(goals, n), avg_r, breakdown_str, kick_str, top,
    )
    return {
        "label": label,
        "n": n,
        "goals": goals,
        "goal_rate": goals / n,
        "avg_reward": avg_r,
        "outcomes": dict(breakdown),
        "kicks": len(kicks),
        "aim_avg": aim_avg,
        "bad_aim_rate": (bad / len(kicks)) if kicks else None,
        "actions": dict(actions),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_team_config(file_path: str) -> list[TeamInfo]:
    with open(file_path, 'r') as f:
        config = json.load(f)["teams"]

    if len(config) != 2:
        raise ValueError("Team configuration must contain exactly two teams.")

    team1_info, team2_info = config
    if len(team1_info) != 3 or len(team2_info) != 3:
        raise ValueError("Each team configuration must have name, n_players, and goalie_id.")

    return [TeamInfo(*team1_info), TeamInfo(*team2_info)]


def _resolve_device() -> torch.device:
    """Pick the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Per-trainer inference helpers
# ---------------------------------------------------------------------------

def _run_hier_ppo(args, networker: Networker, team_name: str):
    """Run inference with the Hierarchical PPO agent."""
    from ai_interface.envs.curriculum_ppo import CurriculumSoccerEnv
    from ai_interface.algorithms.hier_ppo import PPOAgent

    device = _resolve_device()

    env = CurriculumSoccerEnv(
        networker=networker,
        team_name=team_name,
        obs_dim=args.obs_dim,
    )

    agent = PPOAgent(obs_dim=args.obs_dim, device=device)
    agent.load(args.model_path)
    agent.model.eval()

    obs = env.reset()
    for step in range(int(args.steps)):
        state = torch.as_tensor(np.array(obs, dtype=np.float32), device=device)
        with torch.no_grad():
            high_logits, low_mean, _low_std, _value = agent.model(state)

        # Deterministic: argmax for high-level, mean for low-level
        high_action = int(torch.argmax(high_logits, dim=-1).item())
        low_action = low_mean.cpu().numpy()

        action = {"high_level": high_action, "low_level": low_action}
        obs, _reward, done, _info = env.step(action)
        if done:
            obs = env.reset()

    print(f"Hierarchical PPO inference complete — {args.steps} steps.")


def _run_discrete_ppo(args, networker: Networker, team_name: str):
    """Run inference with the Discrete PPO agent."""
    from ai_interface.envs.discrete_ppo import SimplifiedSoccerEnv
    from ai_interface.algorithms.discrete_ppo import DiscretePPOAgent

    device = _resolve_device()

    env = SimplifiedSoccerEnv(
        networker=networker,
        team_name=team_name,
        obs_dim=args.obs_dim,
    )

    agent = DiscretePPOAgent(obs_dim=args.obs_dim, num_actions=5, device=device)
    agent.load(args.model_path)
    agent.model.eval()

    obs = env.reset()
    for step in range(int(args.steps)):
        state = torch.as_tensor(np.array(obs, dtype=np.float32), device=device)
        with torch.no_grad():
            logits, _value = agent.model(state)

        # Deterministic: argmax over discrete actions
        action = int(torch.argmax(logits, dim=-1).item())
        obs, _reward, done, _info = env.step(action)
        if done:
            obs = env.reset()

    print(f"Discrete PPO inference complete — {args.steps} steps.")


def _run_discreteq_learning(args, networker: Networker, team_name: str):
    """Run inference with the Discrete Q-Learning agent."""
    from ai_interface.envs.discrete_simple import SimpleDiscreteEnv
    from ai_interface.algorithms.discrete_qlearning import QLearningAgent

    device = _resolve_device()

    env = SimpleDiscreteEnv(
        networker=networker,
        team_name=team_name,
        obs_dim=args.obs_dim,
    )

    agent = QLearningAgent(obs_dim=args.obs_dim, num_actions=5, device=device)
    agent.load(args.model_path)

    obs = env.reset()
    for step in range(int(args.steps)):
        state = np.array(obs, dtype=np.float32)
        action = agent.select_action(state, eval_mode=True)
        obs, _reward, done, _info = env.step(action)
        if done:
            obs = env.reset()

    print(f"Discrete Q-Learning inference complete — {args.steps} steps.")


def _run_sb3_ppo(args, networker: Networker, team_name: str):
    """Run inference with Stable Baselines3 PPO."""
    from ai_interface.envs.sim_env import SimulatorEnv
    from ai_interface.trainers.sb3_trainer import AI_Trainer

    env = SimulatorEnv(networker, team_name)
    trainer = AI_Trainer(env=env)
    trainer.load_model(args.model_path)

    obs = env.reset()
    for step in range(int(args.steps)):
        state = np.array(obs, dtype=np.float32)
        action = trainer.eval_step(state)
        obs, _reward, done, _info = env.step(action)
        if done:
            obs = env.reset()

    print(f"SB3 PPO inference complete — {args.steps} steps.")


def _run_td3_jal(args, networker: Networker, team_name: str):
    """Run inference with TD3 JAL (centralized team policy)."""
    from ai_interface.algorithms.td3_jal import TD3JALAlgorithm
    from ai_interface.envs.JAL_env import JALTeamEnv

    device = _resolve_device()
    num_robots = int(args.num_robots)
    robot_ids = list(range(1, num_robots + 1))

    env = JALTeamEnv(
        networker=networker,
        team_name=team_name,
        robot_ids=robot_ids,
        obs_dim_per_robot=int(args.obs_dim_per_robot),
        non_robot_obs_dim=int(args.non_robot_obs_dim),
        max_steps=int(args.steps_per_episode),
        debug=bool(args.debug_infer),
    )

    model = TD3JALAlgorithm.load(args.model_path, env=env, device=str(device))
    model.model.set_parameters(model.model.get_parameters())  # Ensure model is ready

    obs, _ = env.reset()
    total_reward = 0.0
    episode_count = 0

    for step in range(int(args.steps)):
        # Deterministic inference: use mean from policy (no exploration noise)
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(
                f"Episode {episode_count + 1}: Reward={total_reward:.2f}, "
                f"Steps={step + 1}"
            )
            episode_count += 1
            total_reward = 0.0
            obs, _ = env.reset()

    print(f"TD3 JAL inference complete — {episode_count} episodes, {args.steps} total steps.")


def _run_td3_jal_her(args, networker: Networker, team_name: str):
    """Run inference with TD3+HER (Dict observation space).

    Loads env settings (disabled_actions, spawn flags, randomization, reward
    overrides) from the training config so the inference env matches what the
    model was trained on. Specify which stage's settings to use via --stage.
    """
    from stable_baselines3 import TD3
    from ai_interface.envs.JAL_her_env import JALHEREnv

    device = _resolve_device()

    with open(args.config, "r") as f:
        config = json.load(f)
    stage_config = config["curriculum"][args.stage]

    num_robots = int(stage_config.get("num_robots", 1))
    robot_ids = stage_config.get("robot_ids", list(range(1, num_robots + 1)))

    reward_overrides = stage_config.get("reward_config_overrides")
    env = JALHEREnv(
        networker=networker,
        team_name=team_name,
        robot_ids=robot_ids,
        obs_dim_per_robot=int(config.get("obs_dim_per_robot", 8)),
        non_robot_obs_dim=int(config.get("non_robot_obs_dim", 4)),
        max_steps=int(config.get("max_steps", 300)),
        her_distance_threshold=float(config.get("her_distance_threshold", 0.1)),
        disabled_actions=list(stage_config.get("disabled_actions", [])),
        spawn_robot_at_ball=bool(stage_config.get("spawn_robot_at_ball", False)),
        spawn_offset_behind_ball=float(stage_config.get("spawn_offset_behind_ball", 1.0)),
        random_ball_x=bool(stage_config.get("random_ball_x", False)),
        random_ball_x_range=tuple(stage_config.get("random_ball_x_range", [5.0, 30.0])),
        random_ball_y=bool(stage_config.get("random_ball_y", False)),
        random_ball_y_range=tuple(stage_config.get("random_ball_y_range", [-3.0, 3.0])),
        random_spawn_theta=bool(stage_config.get("random_spawn_theta", False)),
        random_spawn_theta_range_deg=tuple(stage_config.get("random_spawn_theta_range_deg", [-45.0, 45.0])),
        spawn_theta_relative_to_goal=bool(stage_config.get("spawn_theta_relative_to_goal", False)),
        spawn_theta_min_abs_deg=float(stage_config.get("spawn_theta_min_abs_deg", 0.0)),
        kick_requires_aim=bool(stage_config.get("kick_requires_aim", False)),
        kick_min_aim_quality=float(stage_config.get("kick_min_aim_quality", 0.0)),
        kick_tie_break_when_aimed=bool(stage_config.get("kick_tie_break_when_aimed", False)),
        kick_tie_break_epsilon=float(stage_config.get("kick_tie_break_epsilon", 0.0)),
        approach_defer_when_has_ball=bool(stage_config.get("approach_defer_when_has_ball", False)),
        approach_defer_epsilon=float(stage_config.get("approach_defer_epsilon", 0.0)),
        reward_config_overrides=dict(reward_overrides) if reward_overrides else None,
        debug=bool(args.debug_infer),
    )

    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else _make_infer_log_dir(args.model_path)
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    collector, log_file = _setup_inference_logging(
        log_dir=log_dir, debug=bool(args.debug_infer)
    )
    _SUMMARY_LOG.info("Inference log dir: %s", log_dir)

    print(f"Loading TD3+HER model from {args.model_path} (stage={args.stage})")
    model = TD3.load(args.model_path, env=env, device=str(device))

    obs, _ = env.reset()
    total_reward = 0.0
    episode_count = 0
    goal_count = 0
    elapsed_steps = 0

    outcomes: list[str] = []
    episode_rewards: list[float] = []
    actions_total: Counter = Counter()
    actions_window: Counter = Counter()
    requested_actions_total: Counter = Counter()
    requested_actions_window: Counter = Counter()
    invalid_action_total = 0
    invalid_action_window = 0
    blocked_bad_aim_total = 0
    blocked_bad_aim_window = 0
    action_samples: list[dict] = []
    last_kick_idx = 0  # marker into collector.kicks for window slicing

    WINDOW = 20
    eval_noise_std = float(getattr(args, "td3_eval_noise_std", 0.0) or 0.0)
    eval_logit_noise_std = getattr(args, "td3_eval_logit_noise_std", None)
    eval_param_noise_std = getattr(args, "td3_eval_param_noise_std", None)
    use_split_eval_noise = eval_logit_noise_std is not None or eval_param_noise_std is not None
    rng = np.random.default_rng(0)

    def _add_eval_noise(action):
        action_arr = np.asarray(action, dtype=np.float32)
        if use_split_eval_noise:
            per_robot_dim = int(getattr(env, "action_dim_per_robot", 9))
            logit_sigma = float(eval_logit_noise_std if eval_logit_noise_std is not None else eval_noise_std)
            param_sigma = float(eval_param_noise_std if eval_param_noise_std is not None else eval_noise_std)
            per_robot_sigma = np.array(
                [logit_sigma] * 6 + [param_sigma] * (per_robot_dim - 6),
                dtype=np.float32,
            )
            sigma = np.tile(per_robot_sigma, action_arr.size // per_robot_dim).reshape(action_arr.shape)
        elif eval_noise_std > 0.0:
            sigma = np.full(action_arr.shape, eval_noise_std, dtype=np.float32)
        else:
            return action
        return np.clip(action_arr + rng.normal(0.0, sigma, size=action_arr.shape), -1.0, 1.0).astype(np.float32)

    for step in range(int(args.steps)):
        action, _state = model.predict(obs, deterministic=True)
        action = _add_eval_noise(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        elapsed_steps += 1

        invalid_this_step = 0
        blocked_this_step = 0
        if isinstance(info, dict):
            ai = info.get("action_info") or {}
            atype = ai.get("action_type")
            if atype:
                actions_total[atype] += 1
                actions_window[atype] += 1
            invalid_this_step = int(ai.get("invalid_action_count", 0) or 0)
            invalid_action_total += invalid_this_step
            invalid_action_window += invalid_this_step
            for robot_info in ai.get("per_robot", []) or []:
                requested = robot_info.get("requested_action_type") or robot_info.get("action_type")
                if requested:
                    requested_actions_total[requested] += 1
                    requested_actions_window[requested] += 1
                if robot_info.get("kick_blocked_bad_aim", False):
                    blocked_this_step += 1
                if len(action_samples) < 20:
                    action_samples.append({
                        "step": elapsed_steps,
                        "raw": robot_info.get("raw_action_type"),
                        "requested": requested,
                        "executed": robot_info.get("action_type"),
                        "command": robot_info.get("command"),
                        "invalid": bool(robot_info.get("invalid_action_requested", False)),
                        "kick_tie_break_applied": bool(robot_info.get("kick_tie_break_applied", False)),
                        "approach_defer_applied": bool(robot_info.get("approach_defer_applied", False)),
                        "kick_blocked_bad_aim": bool(robot_info.get("kick_blocked_bad_aim", False)),
                        "kick_aim_quality": robot_info.get("kick_aim_quality"),
                        "kick_predicted_y_at_goal_line": robot_info.get("kick_predicted_y_at_goal_line"),
                        "logits": robot_info.get("logits"),
                        "turn_theta": robot_info.get("turn_theta"),
                    })
            blocked_bad_aim_total += blocked_this_step
            blocked_bad_aim_window += blocked_this_step

        if terminated or truncated:
            episode_count += 1
            reason = (
                info.get("termination_reason")
                if isinstance(info, dict) else None
            )
            if not reason:
                reason = "max_steps" if truncated else "unknown"
            outcomes.append(reason)
            episode_rewards.append(total_reward)
            if reason == "goal_scored":
                goal_count += 1

            if episode_count % WINDOW == 0:
                window_kicks = collector.kicks[last_kick_idx:]
                last_kick_idx = len(collector.kicks)
                _print_window_summary(
                    label=f"eps {episode_count - WINDOW + 1}-{episode_count}",
                    outcomes=outcomes[-WINDOW:],
                    rewards=episode_rewards[-WINDOW:],
                    kicks=window_kicks,
                    actions=actions_window,
                )
                _SUMMARY_LOG.info(
                    "[eps %d-%d] requested: %s | invalid_actions=%d | blocked_bad_aim=%d",
                    episode_count - WINDOW + 1,
                    episode_count,
                    " ".join(f"{k}={v}" for k, v in sorted(requested_actions_window.items())) or "—",
                    invalid_action_window,
                    blocked_bad_aim_window,
                )
                actions_window = Counter()
                requested_actions_window = Counter()
                invalid_action_window = 0
                blocked_bad_aim_window = 0

            total_reward = 0.0
            obs, _ = env.reset()

    # Final summary
    _SUMMARY_LOG.info("=" * 72)
    _SUMMARY_LOG.info(
        "TD3+HER inference complete — %d episodes, %d steps (%s)",
        episode_count, elapsed_steps, args.model_path,
    )
    _SUMMARY_LOG.info("=" * 72)
    all_stats = _print_window_summary(
        label="ALL",
        outcomes=outcomes,
        rewards=episode_rewards,
        kicks=collector.kicks,
        actions=actions_total,
    )
    _SUMMARY_LOG.info(
        "[ALL] requested: %s | invalid_actions=%d | blocked_bad_aim=%d",
        " ".join(f"{k}={v}" for k, v in sorted(requested_actions_total.items())) or "—",
        invalid_action_total,
        blocked_bad_aim_total,
    )
    last_stats = None
    if len(outcomes) > 100:
        last_stats = _print_window_summary(
            label="last 100",
            outcomes=outcomes[-100:],
            rewards=episode_rewards[-100:],
            kicks=[],   # per-kick slicing across windows is approximate; skip here
            actions=Counter(),
        )

    summary_payload = {
        "model_path": args.model_path,
        "stage": args.stage,
        "config_path": args.config,
        "requested_steps": int(args.steps),
        "elapsed_steps": elapsed_steps,
        "episodes": episode_count,
        "all": all_stats,
        "last_100": last_stats,
        "requested_actions": dict(requested_actions_total),
        "executed_actions": dict(actions_total),
        "invalid_action_total": invalid_action_total,
        "blocked_bad_aim_total": blocked_bad_aim_total,
        "td3_eval_noise_std": eval_noise_std,
        "td3_eval_logit_noise_std": eval_logit_noise_std,
        "td3_eval_param_noise_std": eval_param_noise_std,
        "action_samples": action_samples,
    }
    summary_file = log_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(summary_payload, f, indent=2, default=str)
    _SUMMARY_LOG.info("Saved summary to %s", summary_file)
    _SUMMARY_LOG.info("Full log: %s", log_file)


def _run_ppo_jal(args, networker: Networker, team_name: str):
    """Run inference with the hybrid-PPO JAL policy (PPOJALAgent + JALTeamEnv).

    Loads env settings (disabled_actions, spawn flags, randomization, reward
    overrides) from the training config so the inference env matches what the
    model was trained on. Specify which stage's settings to use via --stage.

    Actions are sampled deterministically (argmax primitive + Gaussian mean),
    matching the policy the curriculum trainer learned.
    """
    from ai_interface.algorithms.ppo_jal import PPOJALAgent, PRIMITIVE_NAMES
    from ai_interface.envs.JAL_env import JALTeamEnv
    from ai_interface.trainers.policy_control import GoalieCommandProvider

    device = _resolve_device()

    with open(args.config, "r") as f:
        config = json.load(f)
    stage_config = config["curriculum"][args.stage]

    num_robots = int(stage_config.get("num_robots", 1))
    robot_ids = stage_config.get("robot_ids", list(range(1, num_robots + 1)))
    reward_overrides = stage_config.get("reward_config_overrides")

    # Mirror PPOJALCurriculumTrainer.setup_environment exactly so the obs the
    # policy sees at inference matches training (same spawn / reward / mask).
    env = JALTeamEnv(
        networker=networker,
        team_name=team_name,
        robot_ids=robot_ids,
        obs_dim_per_robot=int(config.get("obs_dim_per_robot", 8)),
        non_robot_obs_dim=int(config.get("non_robot_obs_dim", 4)),
        a_max=int(config.get("a_max", 5)),
        c_max=int(config.get("c_max", 7)),
        global_dim=int(config.get("global_dim", 6)),
        per_agent_dim=int(config.get("per_agent_dim", 10)),
        d_ctx=int(config.get("d_ctx", 7)),
        max_steps=int(config.get("max_steps", 300)),
        debug=bool(args.debug_infer),
        invalid_action_penalty=float(stage_config.get("invalid_action_penalty", 0.2)),
        disabled_actions=list(stage_config.get("disabled_actions", [])),
        spawn_robot_at_ball=bool(stage_config.get("spawn_robot_at_ball", False)),
        spawn_offset_behind_ball=float(stage_config.get("spawn_offset_behind_ball", 1.0)),
        random_ball_x=bool(stage_config.get("random_ball_x", False)),
        random_ball_x_range=tuple(stage_config.get("random_ball_x_range", [5.0, 30.0])),
        random_ball_y=bool(stage_config.get("random_ball_y", False)),
        random_ball_y_range=tuple(stage_config.get("random_ball_y_range", [-3.0, 3.0])),
        random_spawn_theta=bool(stage_config.get("random_spawn_theta", False)),
        random_spawn_theta_range_deg=tuple(stage_config.get("random_spawn_theta_range_deg", [-45.0, 45.0])),
        reward_config_overrides=dict(reward_overrides) if reward_overrides else None,
    )

    disabled_actions = list(stage_config.get("disabled_actions", []))

    # Build scripted opponent controller if the stage config defines one
    # (e.g. GoalieCommandProvider for stage2_goalie). Mirrors training behaviour.
    opp_controller = None
    opp_team_name = None
    aux_specs = stage_config.get("aux_team_policies", [])
    if aux_specs and isinstance(aux_specs[0], dict):
        spec = aux_specs[0]
        if str(spec.get("controller_type", "")).lower() == "goalie":
            robot_id = int(spec.get("robot_ids", [1])[0])
            side = str(spec.get("side", "right"))
            opp_team_name = str(spec.get("team_name", ""))
            opp_controller = GoalieCommandProvider(
                team_name=opp_team_name, robot_id=robot_id, side=side
            )

    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else _make_infer_log_dir(args.model_path)
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    collector, log_file = _setup_inference_logging(
        log_dir=log_dir, debug=bool(args.debug_infer)
    )
    _SUMMARY_LOG.info("Inference log dir: %s", log_dir)

    # Build hparams from model_params so the network shape (encoder_hidden) and
    # agent config match the checkpoint, then load weights.
    model_params = config.get("model_params", {})
    hparams: dict = {}
    for k in [
        "gamma", "gae_lambda", "clip_range", "target_kl", "n_epochs",
        "minibatch_size", "rollout_size",
        "learning_rate_initial", "learning_rate_final",
        "ent_coef_initial", "ent_coef_final",
        "vf_coef", "max_grad_norm",
        "value_clip_range", "advantage_clip", "entropy_tripwire",
        "kl_lr_halve_factor", "feature_dim", "num_heads",
    ]:
        if k in model_params:
            hparams[k] = model_params[k]

    obs_dim = int(env.observation_space.shape[0])
    agent = PPOJALAgent(
        obs_dim=obs_dim,
        num_robots=num_robots,
        a_max=int(config.get("a_max", 5)),
        c_max=int(config.get("c_max", 7)),
        global_dim=int(config.get("global_dim", 6)),
        per_agent_dim=int(config.get("per_agent_dim", 10)),
        d_ctx=int(config.get("d_ctx", 7)),
        num_primitives=int(config.get("num_primitives", 8)),
        param_dim=int(config.get("param_dim", 5)),
        device=device,
        hparams=hparams,
    )
    print(f"Loading PPO JAL model from {args.model_path} (stage={args.stage})")
    agent.load(args.model_path)
    agent.model.eval()

    # Param-active mask (matches training): Dx,Dy ← goto, Dtheta ← turn.
    param_active_mask = np.zeros(agent.param_dim, dtype=np.float32)
    if "goto" not in disabled_actions:
        param_active_mask[0] = 1.0
        param_active_mask[1] = 1.0
    if "turn" not in disabled_actions:
        param_active_mask[2] = 1.0

    obs, info = env.reset()
    agent_mask = info.get("agent_active_mask")
    context_mask = info.get("context_active_mask")
    total_reward = 0.0
    episode_count = 0
    elapsed_steps = 0
    ep_step = 0

    outcomes: list[str] = []
    episode_rewards: list[float] = []
    actions_total: Counter = Counter()
    actions_window: Counter = Counter()
    last_kick_idx = 0
    WINDOW = 20

    # Per-step diagnostic trace (debug only) — one JSON line per step capturing
    # the turn-vs-kick decision so we can confirm/deny the "continuous turning"
    # fixed-point hypothesis: when has_ball is True, is turn_logit >= kick_logit
    # and is turn_theta ~ 0 (the policy fine-tuning an angle that's already
    # correct, especially near the goal center line where ball_y ~ 0)?
    trace_file = None
    if bool(args.debug_infer):
        trace_path = log_dir / "step_trace.jsonl"
        trace_file = trace_path.open("w")
        _SUMMARY_LOG.info("Per-step debug trace: %s", trace_path)

    for step in range(int(args.steps)):
        action, _transition = agent.sample_action(
            obs=obs,
            disabled_actions=disabled_actions,
            agent_active_mask=agent_mask,
            context_active_mask=context_mask,
            param_active_mask=param_active_mask,
            deterministic=not bool(args.ppo_stochastic),
        )
        # Deployment-friendly variant: deterministic argmax primitive, but inject
        # Gaussian noise on the param MEAN to supply the fine turn corrections the
        # bang-bang mean cannot. Only meaningful when not already fully stochastic.
        if not bool(args.ppo_stochastic) and float(args.ppo_param_noise_std) > 0.0:
            params = np.asarray(action["params"], dtype=np.float32)
            params = params + np.random.normal(
                0.0, float(args.ppo_param_noise_std), size=params.shape
            ).astype(np.float32)
            action["params"] = np.clip(params, -1.0, 1.0)
        for p in np.asarray(action["primitive_idx"]).reshape(-1):
            name = PRIMITIVE_NAMES[int(p)]
            actions_total[name] += 1
            actions_window[name] += 1

        if (
            opp_controller is not None
            and opp_team_name
            and getattr(env, "_cached_game_state", None) is not None
        ):
            try:
                opp_cmds = opp_controller.predict_commands(env._cached_game_state)
                networker.execute_ai_output(opp_cmds, opp_team_name)
            except Exception:
                pass

        obs, reward, terminated, truncated, info = env.step(action)
        if isinstance(info, dict):
            agent_mask = info.get("agent_active_mask", agent_mask)
            context_mask = info.get("context_active_mask", context_mask)
        total_reward += float(reward)
        elapsed_steps += 1
        ep_step += 1

        if trace_file is not None and isinstance(info, dict):
            ai = info.get("action_info") or {}
            # Pull post-step robot pose (degrees) + ball pos from the env's
            # cached game state so we can measure how many degrees a "turn X"
            # command actually rotates the body, and the robot->goal heading
            # error. Goal center is at (+FIELD_X, 0) for the left team.
            gs = getattr(env, "_cached_game_state", None)
            pose_by_id = {}
            ball_xy = None
            if gs is not None:
                for entry in getattr(gs, "robot_poses", {}).get(team_name, []) or []:
                    if isinstance(entry, dict):
                        pose_by_id.update(entry)
                bp = getattr(gs, "ball_pos", None)
                if bp is not None:
                    ball_xy = [float(bp[0]), float(bp[1])]
            for ri, robot_info in enumerate(ai.get("per_robot", []) or []):
                logits = robot_info.get("logits")
                turn_logit = kick_logit = None
                if logits is not None and len(logits) > 3:
                    turn_logit = float(logits[2])
                    kick_logit = float(logits[3])
                rx = ry = theta_deg = head_err_goal = head_err_ball = None
                pose = pose_by_id.get(robot_info.get("robot_id"))
                if pose is not None:
                    rx, ry, theta_deg = float(pose[0]), float(pose[1]), float(pose[2])
                    gx, gy = 52.5, 0.0  # right goal center (left team attacks +x)
                    ang_to_goal = math.degrees(math.atan2(gy - ry, gx - rx))
                    head_err_goal = (ang_to_goal - theta_deg + 180.0) % 360.0 - 180.0
                    if ball_xy is not None:
                        ang_to_ball = math.degrees(math.atan2(ball_xy[1] - ry, ball_xy[0] - rx))
                        head_err_ball = (ang_to_ball - theta_deg + 180.0) % 360.0 - 180.0
                trace_file.write(json.dumps({
                    "ep": episode_count,
                    "ep_step": ep_step,
                    "robot": ri,
                    "primitive": robot_info.get("action_type"),
                    "has_ball": bool(robot_info.get("has_ball_now", False)),
                    "turn_theta": robot_info.get("turn_theta"),
                    "theta_deg": theta_deg,
                    "head_err_goal": head_err_goal,
                    "head_err_ball": head_err_ball,
                    "rx": rx, "ry": ry, "ball": ball_xy,
                    "turn_logit": turn_logit,
                    "kick_logit": kick_logit,
                    "kick_margin": (kick_logit - turn_logit)
                    if (turn_logit is not None and kick_logit is not None) else None,
                    "kick_fired": bool(robot_info.get("kick_fired", False)),
                    "kick_aim_quality": robot_info.get("kick_aim_quality"),
                    "command": robot_info.get("command"),
                }, default=str) + "\n")

        if terminated or truncated:
            episode_count += 1
            ep_step = 0
            reason = info.get("termination_reason") if isinstance(info, dict) else None
            if not reason:
                reason = "max_steps" if truncated else "unknown"
            outcomes.append(reason)
            episode_rewards.append(total_reward)

            if episode_count % WINDOW == 0:
                window_kicks = collector.kicks[last_kick_idx:]
                last_kick_idx = len(collector.kicks)
                _print_window_summary(
                    label=f"eps {episode_count - WINDOW + 1}-{episode_count}",
                    outcomes=outcomes[-WINDOW:],
                    rewards=episode_rewards[-WINDOW:],
                    kicks=window_kicks,
                    actions=actions_window,
                )
                actions_window = Counter()

            total_reward = 0.0
            obs, info = env.reset()
            agent_mask = info.get("agent_active_mask", agent_mask)
            context_mask = info.get("context_active_mask", context_mask)

    # Final summary.
    _SUMMARY_LOG.info("=" * 72)
    _SUMMARY_LOG.info(
        "PPO JAL inference complete — %d episodes, %d steps (%s)",
        episode_count, elapsed_steps, args.model_path,
    )
    _SUMMARY_LOG.info("=" * 72)
    all_stats = _print_window_summary(
        label="ALL",
        outcomes=outcomes,
        rewards=episode_rewards,
        kicks=collector.kicks,
        actions=actions_total,
    )
    last_stats = None
    if len(outcomes) > 100:
        last_stats = _print_window_summary(
            label="last 100",
            outcomes=outcomes[-100:],
            rewards=episode_rewards[-100:],
            kicks=[],
            actions=Counter(),
        )

    summary_payload = {
        "model_path": args.model_path,
        "stage": args.stage,
        "config_path": args.config,
        "requested_steps": int(args.steps),
        "elapsed_steps": elapsed_steps,
        "episodes": episode_count,
        "all": all_stats,
        "last_100": last_stats,
        "executed_actions": dict(actions_total),
    }
    summary_file = log_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(summary_payload, f, indent=2, default=str)
    _SUMMARY_LOG.info("Saved summary to %s", summary_file)
    _SUMMARY_LOG.info("Full log: %s", log_file)
    if trace_file is not None:
        trace_file.close()


def _run_attention_mappo(args, networker: Networker, team_name: str):
    """Run inference with the Attention MAPPO agent (multi-agent, role-based).

    Loads curriculum stage settings from the config to reconstruct the
    FullTeamMARLEnv with matching num_agents, forced_roles, reward_weights,
    and spawn config. Actions are selected deterministically (Gaussian mean).
    """
    from ai_interface.algorithms.attention_mappo import (
        AttentionMAPPOAgent, AttentionMAPPOConfig,
    )
    from ai_interface.envs.full_team_marl_env import FullTeamMARLEnv
    from ai_interface.hsm.state_machine import Role

    device = _resolve_device()

    with open(args.config, "r") as f:
        config = json.load(f)
    stage_config = config["curriculum"][args.stage]

    num_agents = int(stage_config.get("num_agents", 6))
    max_steps = int(stage_config.get("max_steps", config.get("max_steps", 100)))
    obs_dim = int(config.get("obs_dim", 57))

    role_map = {
        "STRIKER": Role.STRIKER,
        "SUPPORT": Role.SUPPORT,
        "DEFENDER": Role.DEFENDER,
        "GOALIE": Role.GOALIE,
    }
    forced_roles = {
        int(k): role_map.get(str(v).upper(), Role.SUPPORT)
        for k, v in stage_config.get("forced_roles", {}).items()
    }
    reward_weights = stage_config.get("reward_weights", {})
    spawn_config = stage_config.get("spawn_config", {})
    hsm_thresholds = config.get("hsm_thresholds", {})

    env = FullTeamMARLEnv(
        networker=networker,
        team_name=team_name,
        num_agents=num_agents,
        max_steps=max_steps,
        hsm_thresholds=hsm_thresholds,
        forced_roles=forced_roles,
        reward_weights=reward_weights,
        spawn_config=spawn_config,
    )
    env.set_forced_roles(forced_roles)

    algo_cfg_raw = config.get("algorithm", {})
    algo_config = AttentionMAPPOConfig(
        gamma=float(algo_cfg_raw.get("gamma", 0.99)),
        gae_lambda=float(algo_cfg_raw.get("gae_lambda", 0.95)),
        lr_actor=float(algo_cfg_raw.get("lr_actor", 3e-4)),
        lr_critic=float(algo_cfg_raw.get("lr_critic", 1e-3)),
        lr_encoder=float(algo_cfg_raw.get("lr_encoder", 5e-4)),
        clip_eps=float(algo_cfg_raw.get("clip_eps", 0.2)),
        entropy_coef=float(algo_cfg_raw.get("entropy_coef", 0.01)),
        value_coef=float(algo_cfg_raw.get("value_coef", 0.5)),
        max_grad_norm=float(algo_cfg_raw.get("max_grad_norm", 0.5)),
        ppo_epochs=int(algo_cfg_raw.get("ppo_epochs", 4)),
        rollout_length=int(algo_cfg_raw.get("rollout_length", 2048)),
        minibatch_size=int(algo_cfg_raw.get("minibatch_size", 256)),
        min_std=float(algo_cfg_raw.get("min_std", 0.1)),
        agent_embed_dim=int(algo_cfg_raw.get("agent_embed_dim", 128)),
        num_attn_heads=int(algo_cfg_raw.get("num_attn_heads", 4)),
        num_attn_layers=int(algo_cfg_raw.get("num_attn_layers", 2)),
        actor_hidden=(
            int(algo_cfg_raw.get("actor_hidden", [256, 256])[0]),
            int(algo_cfg_raw.get("actor_hidden", [256, 256])[1]),
        ),
    )

    agent = AttentionMAPPOAgent(
        num_agents=num_agents,
        obs_dim=obs_dim,
        config=algo_config,
        device=device,
    )

    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else _make_infer_log_dir(args.model_path)
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    collector, log_file = _setup_inference_logging(
        log_dir=log_dir, debug=bool(args.debug_infer)
    )
    _SUMMARY_LOG.info("Inference log dir: %s", log_dir)

    print(f"Loading Attention MAPPO model from {args.model_path} (stage={args.stage})")
    agent.load(args.model_path, weights_only=True)

    observations = env.reset()
    episode_count = 0
    elapsed_steps = 0
    total_reward = 0.0

    outcomes: list[str] = []
    episode_rewards: list[float] = []
    actions_total: Counter = Counter()
    actions_window: Counter = Counter()
    last_kick_idx = 0
    WINDOW = 20

    for step in range(int(args.steps)):
        current_roles = [
            env.current_roles.get(i, Role.SUPPORT)
            for i in range(num_agents)
        ]

        actions_arr, _logprobs, _value = agent.select_actions(
            observations, current_roles, deterministic=True,
        )

        from ai_interface.hsm.state_machine import HSMState as _HSMState
        for a in actions_arr.flatten():
            name = list(_HSMState)[int(a)].value
            actions_total[name] += 1
            actions_window[name] += 1

        next_obs, reward, done, info = env.step(actions_arr)
        total_reward += float(reward)
        elapsed_steps += 1

        if done:
            episode_count += 1
            reason = info.get("termination_reason") if isinstance(info, dict) else None
            if not reason:
                reason = "max_steps" if not info.get("goal_scored", False) else "goal_scored"
            outcomes.append(reason)
            episode_rewards.append(total_reward)

            if episode_count % WINDOW == 0:
                window_kicks = collector.kicks[last_kick_idx:]
                last_kick_idx = len(collector.kicks)
                _print_window_summary(
                    label=f"eps {episode_count - WINDOW + 1}-{episode_count}",
                    outcomes=outcomes[-WINDOW:],
                    rewards=episode_rewards[-WINDOW:],
                    kicks=window_kicks,
                    actions=actions_window,
                )
                actions_window = Counter()

            total_reward = 0.0
            observations = env.reset()
        else:
            observations = next_obs

    _SUMMARY_LOG.info("=" * 72)
    _SUMMARY_LOG.info(
        "Attention MAPPO inference complete — %d episodes, %d steps (%s)",
        episode_count, elapsed_steps, args.model_path,
    )
    _SUMMARY_LOG.info("=" * 72)
    all_stats = _print_window_summary(
        label="ALL",
        outcomes=outcomes,
        rewards=episode_rewards,
        kicks=collector.kicks,
        actions=actions_total,
    )
    last_stats = None
    if len(outcomes) > 100:
        last_stats = _print_window_summary(
            label="last 100",
            outcomes=outcomes[-100:],
            rewards=episode_rewards[-100:],
            kicks=[],
            actions=Counter(),
        )

    summary_payload = {
        "model_path": args.model_path,
        "stage": args.stage,
        "config_path": args.config,
        "requested_steps": int(args.steps),
        "elapsed_steps": elapsed_steps,
        "episodes": episode_count,
        "all": all_stats,
        "last_100": last_stats,
    }
    summary_file = log_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(summary_payload, f, indent=2, default=str)
    _SUMMARY_LOG.info("Saved summary to %s", summary_file)
    _SUMMARY_LOG.info("Full log: %s", log_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_TRAINER_RUNNERS = {
    "hier_ppo": _run_hier_ppo,
    "discrete_ppo": _run_discrete_ppo,
    "discreteq_learning": _run_discreteq_learning,
    "sb3_ppo": _run_sb3_ppo,
    "td3_jal": _run_td3_jal,
    "td3_jal_her": _run_td3_jal_her,
    "ppo_jal": _run_ppo_jal,
    "attention_mappo": _run_attention_mappo,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a saved RL policy in simulator"
    )
    parser.add_argument(
        "--trainer",
        choices=list(_TRAINER_RUNNERS.keys()),
        required=True,
        help="Which trainer/algorithm the saved model was trained with",
    )
    parser.add_argument("--team_config", type=str, default="team_config.json",
                        help="Path to team configuration JSON file")
    parser.add_argument("--env", choices=[
        "sim-only", "sim-embedded", "sim-mixed", "field-practice", "field-tournament"
    ], default="sim-only", help="Environment mode for Networker")
    parser.add_argument("--team", type=str, default="TritonBots",
                        help="Team name to control")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model (.pth for hier/discrete PPO, .zip for SB3/TD3)")
    parser.add_argument("--obs_dim", type=int, default=18,
                        help="Observation dimension (used by hier_ppo and discrete_ppo)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of inference steps to run")
    parser.add_argument("--num_robots", type=int, default=1,
                        help="Number of robots (for td3_jal)")
    parser.add_argument("--obs_dim_per_robot", type=int, default=8,
                        help="Observation dimension per robot (for td3_jal)")
    parser.add_argument("--non_robot_obs_dim", type=int, default=4,
                        help="Non-robot (global) observation dimension (for td3_jal)")
    parser.add_argument("--steps_per_episode", type=int, default=200,
                        help="Max steps per episode (for td3_jal)")
    parser.add_argument("--debug_infer", action="store_true",
                        help="Enable debug logging during inference")
    parser.add_argument("--config", type=str, default="configs/td3_jal_her_config.json",
                        help="Path to TD3+HER training config (for td3_jal_her stage settings)")
    parser.add_argument("--stage", type=str, default="stage1_9_approach_turn_kick",
                        help="Curriculum stage whose env settings to mirror (for td3_jal_her)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory to write infer_log.log + summary.json. "
                             "Defaults to infer_logs/<timestamp>_<model_stem>/")
    parser.add_argument("--td3_eval_noise_std", type=float, default=0.0,
                        help="Optional Gaussian action noise for TD3+HER inference. "
                             "Default 0 keeps deterministic evaluation.")
    parser.add_argument("--td3_eval_logit_noise_std", type=float, default=None,
                        help="Optional TD3+HER eval noise for primitive logits only.")
    parser.add_argument("--td3_eval_param_noise_std", type=float, default=None,
                        help="Optional TD3+HER eval noise for continuous params only.")
    parser.add_argument("--ppo_stochastic", action="store_true",
                        help="PPO JAL: sample primitive + params (deterministic=False), "
                             "exactly as during training. Confirms the saturated-mean "
                             "turn diagnosis vs the default argmax+mean inference.")
    parser.add_argument("--ppo_param_noise_std", type=float, default=0.3,
                        help="PPO JAL: keep argmax primitive but add Gaussian noise of this "
                             "std to the deterministic param mean (then re-clamp to [-1,1]). "
                             "Supplies the fine turn corrections the saturated bang-bang mean "
                             "cannot, breaking the continuous-turning loop (see CHANGES.md #26). "
                             "Default 0.3 (validated: 94.4%% goal, 1.4%% timeout vs 73.6%%/19%% "
                             "at 0.0). Set 0 for pure deterministic mean (debug only).")
    parser.add_argument("--sim_host", type=str, default="127.0.0.1",
                        help="rcssserver host (default 127.0.0.1). Use to target "
                             "a remote sim or a non-default loopback alias.")
    parser.add_argument("--sim_player_port", type=int, default=6000,
                        help="rcssserver player port (default 6000). Override "
                             "to run in parallel against a second sim, e.g. 7000.")
    parser.add_argument("--sim_trainer_port", type=int, default=6001,
                        help="rcssserver trainer/monitor port (default 6001). "
                             "Override to match a second sim, e.g. 7001.")

    args = parser.parse_args()

    # Setup networker
    team_infos = load_team_config(args.team_config)
    team_name = args.team or team_infos[0].name
    networker = Networker(
        team_infos, args.env,
        sim_host=args.sim_host,
        sim_player_port=args.sim_player_port,
        sim_trainer_port=args.sim_trainer_port,
    )

    try:
        runner = _TRAINER_RUNNERS[args.trainer]
        runner(args, networker, team_name)
    finally:
        try:
            networker.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

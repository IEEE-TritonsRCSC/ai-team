"""
Inference entry point using the simulator via Networker, with modular RL components.

Loads a saved model and runs deterministic actions inside the appropriate
environment. Supports:
  - Hierarchical PPO  (--trainer hier_ppo)
  - Discrete PPO      (--trainer discrete_ppo)
  - Stable Baselines3 (--trainer sb3_ppo)

Usage examples:
    python infer.py --trainer hier_ppo --model_path models/hier_ppo_policy.pth
    python infer.py --trainer discrete_ppo --model_path models/discrete_ppo_policy.pth
    python infer.py --trainer discreteq_learning --model_path models/qlearning_policy.pth
    python infer.py --trainer sb3_ppo --model_path models/sb3_ppo_policy.zip
"""
import argparse
import datetime as _dt
import json
import logging
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
    last_kick_idx = 0  # marker into collector.kicks for window slicing

    WINDOW = 20

    for step in range(int(args.steps)):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        elapsed_steps += 1

        if isinstance(info, dict):
            ai = info.get("action_info") or {}
            atype = ai.get("action_type")
            if atype:
                actions_total[atype] += 1
                actions_window[atype] += 1

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
                actions_window = Counter()

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

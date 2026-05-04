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
import json
from pathlib import Path

import numpy as np
import torch

from networking.networker import TeamInfo, Networker


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_TRAINER_RUNNERS = {
    "hier_ppo": _run_hier_ppo,
    "discrete_ppo": _run_discrete_ppo,
    "discreteq_learning": _run_discreteq_learning,
    "sb3_ppo": _run_sb3_ppo,
    "td3_jal": _run_td3_jal,
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
        "sim-only", "sim-mixed", "field-practice", "field-tournament"
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

    args = parser.parse_args()

    # Setup networker
    team_infos = load_team_config(args.team_config)
    team_name = args.team or team_infos[0].name
    networker = Networker(team_infos, args.env)

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

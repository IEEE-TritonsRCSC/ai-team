"""
Training entry point using the simulator via Networker, with modular RL components.

Mirrors the structure of `__main__.py` but focuses on RL training with
`SimulatorEnv` and `AI_Trainer`.
"""
import argparse
import json
from pathlib import Path

from networking.networker import TeamInfo, Networker
from ai_interface.utils.sim_env import SimulatorEnv
from ai_interface.utils.trainer import AI_Trainer


def load_team_config(file_path: str) -> list[TeamInfo]:
    with open(file_path, 'r') as f:
        config = json.load(f)["teams"]

    if len(config) != 2:
        raise ValueError("Team configuration must contain exactly two teams.")

    team1_info, team2_info = config
    if len(team1_info) != 3 or len(team2_info) != 3:
        raise ValueError("Each team configuration must have name, n_players, and goalie_id.")

    return [TeamInfo(*team1_info), TeamInfo(*team2_info)]


def make_env(networker: Networker, team_name: str) -> SimulatorEnv:
    return SimulatorEnv(networker, team_name)


def main():
    parser = argparse.ArgumentParser(description="Train RL policy in simulator")
    parser.add_argument("--team_config", type=str, default="team_config.json",
                        help="Path to team configuration JSON file")
    parser.add_argument("--env", choices=[
        "sim-only", "sim-mixed", "field-practice", "field-tournament"
    ], default="sim-only", help="Environment mode for Networker")
    parser.add_argument("--team", type=str, default="TritonBots",
                        help="Team name to control (default: our team from config). In non-tournament modes you may pick either.")
    parser.add_argument("--timesteps", type=int, default=10000,
                        help="Total timesteps to train")
    parser.add_argument("--save_path", type=str, default="models/ppo_policy.zip",
                        help="Where to save the trained model")
    parser.add_argument("--train_batch_size", type=int, default=2048,
                        help="Transitions to buffer before learn() is called")
    parser.add_argument("--learn_batch_timesteps", type=int, default=2048,
                        help="Timesteps to pass to model.learn per batch")

    args = parser.parse_args()

    team_infos = load_team_config(args.team_config)
    # Choose team: default to first team unless overridden
    team_name = args.team or team_infos[0].name

    networker = Networker(team_infos, args.env)
    env = make_env(networker, team_name)

    trainer = AI_Trainer(
        env=env,
        model_params={},
        model_class=None,  # default SB3 PPO
        train_batch_size=args.train_batch_size,
        learn_timesteps_per_batch=args.learn_batch_timesteps,
    )

    # Minimal rollout loop to let SB3 collect steps; trainer buffers are used for custom algos
    # For SB3 PPO, calling learn will internally sample from env, so we just call learn in chunks
    total = int(args.timesteps)
    remaining = total
    while remaining > 0:
        chunk = min(args.learn_batch_timesteps, remaining)
        trainer.model.learn(total_timesteps=chunk)
        remaining -= chunk

    # Ensure save directory exists
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_path))

    # Clean shutdown for networker if needed
    try:
        networker.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()

"""
Inference entry point using the simulator via Networker, with modular RL components.

Loads a saved model and runs deterministic actions inside `SimulatorEnv`.
"""
import argparse
import json
from pathlib import Path

import numpy as np

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
    parser = argparse.ArgumentParser(description="Run inference with a saved RL policy in simulator")
    parser.add_argument("--team_config", type=str, default="team_config.json",
                        help="Path to team configuration JSON file")
    parser.add_argument("--env", choices=[
        "sim-only", "sim-mixed", "field-practice", "field-tournament"
    ], default="sim-only", help="Environment mode for Networker")
    parser.add_argument("--team", type=str, default="TritonBots",
                        help="Team name to control (default: our team from config). In non-tournament modes you may pick either.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model (e.g., models/ppo_policy.zip)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of inference steps to run")

    args = parser.parse_args()

    team_infos = load_team_config(args.team_config)
    team_name = args.team or team_infos[0].name

    networker = Networker(team_infos, args.env)
    env = make_env(networker, team_name)

    trainer = AI_Trainer(env=env)
    # Load SB3-style model
    trainer.load_model(args.model_path)

    obs = env.reset()
    for _ in range(int(args.steps)):
        # Ensure observation is numpy array
        state = np.array(obs, dtype=np.float32)
        action = trainer.eval_step(state)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    try:
        networker.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()

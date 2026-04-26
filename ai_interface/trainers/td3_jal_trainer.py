"""
TD3 JAL (Joint-Action Learners) Trainer for RoboCup Soccer.

This trainer keeps JAL as a centralized team policy where one network outputs
one concatenated action vector for all controlled robots.
"""

import gymnasium as gym
from pathlib import Path
from typing import Any, Dict
import torch

from ai_interface.algorithms.td3_jal import TD3JALAlgorithm
from ai_interface.envs.JAL_env import JALTeamEnv
from ai_interface.trainers.base_trainer import BaseTrainer
from networking.networker import Networker, TeamInfo


class TD3JALTrainer(BaseTrainer):
    """Trainer for TD3 with team-level Joint-Action Learning (JAL)."""

    def __init__(self, config: Dict[str, Any], log_dir: str = None, device=None):
        super().__init__(config, log_dir, algorithm_name="td3_jal")
        self.networker = None
        self.env = None
        self.model = None
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_dim_per_robot = 6  # [goto_logit, kick_logit, dribble_logit, goto_x, goto_y, goto_theta]
        self.num_robots = int(config.get("num_robots", 1))
        self.total_action_dim = self.num_robots * self.action_dim_per_robot
        self.logger.info(
            "TD3 JAL Trainer initialized: %d robots, %dD action space",
            self.num_robots,
            self.total_action_dim,
        )

    def setup_environment(self) -> gym.Env:
        """Setup centralized team JAL environment."""
        team_infos = self._load_team_config(self.config["team_config"])
        team_name = self.config.get("team_name") or team_infos[0].name

        sim_host, sim_player_port, sim_trainer_port = self._sim_endpoint_for_env(0)
        self.networker = Networker(
            team_infos,
            self.config.get("env_mode", "sim-only"),
            sim_host=sim_host,
            sim_player_port=sim_player_port,
            sim_trainer_port=sim_trainer_port,
        )

        robot_ids = self.config.get("robot_ids")
        if robot_ids is None:
            robot_ids = list(range(1, int(self.config.get("num_robots", 1)) + 1))

        self.env = JALTeamEnv(
            networker=self.networker,
            team_name=team_name,
            robot_ids=robot_ids,
            obs_dim_per_robot=int(self.config.get("obs_dim_per_robot", 5)),
            non_robot_obs_dim=int(self.config.get("non_robot_obs_dim", 4)),
            max_steps=int(self.config.get("max_steps", 200)),
            debug=bool(self.config.get("debug", False)),
        )

        self.num_robots = len(robot_ids)
        self.total_action_dim = self.num_robots * self.action_dim_per_robot

        self.logger.info(
            "JALTeamEnv created - Team: %s, robots=%s, obs_dim=%d, action_dim=%d, Endpoint: %s:%d/%d",
            team_name,
            robot_ids,
            int(self.env.observation_space.shape[0]),
            int(self.env.action_space.shape[0]),
            sim_host,
            sim_player_port,
            sim_trainer_port,
        )
        return self.env

    def setup_model(self, env: gym.Env):
        """Setup TD3 model via algorithm module wrapper."""
        self.logger.info("Setting up TD3 model on device: %s", self.device)
        model_params = self.config.get("model_params", {})

        learning_rate = model_params.get("learning_rate", 1e-3)
        buffer_size = model_params.get("buffer_size", 100_000)
        batch_size = model_params.get("batch_size", 64)
        gamma = model_params.get("gamma", 0.9)
        tau = model_params.get("tau", 0.01)
        policy_delay = model_params.get("policy_delay", 2)
        target_policy_noise = model_params.get("target_policy_noise", 0.2)
        target_noise_clip = model_params.get("target_noise_clip", 0.5)
        action_noise_std = model_params.get("action_noise_std", 0.05)
        network_type = model_params.get("network_type", "mlp")
        network_kwargs = model_params.get("network_kwargs", {})
        policy_kwargs = model_params.get("policy_kwargs", {"net_arch": [64, 48, 32]})

        self.logger.info("TD3 Hyperparameters:")
        self.logger.info("  Learning rate: %s", learning_rate)
        self.logger.info("  Buffer size: %s", f"{buffer_size:,}")
        self.logger.info("  Batch size: %s", batch_size)
        self.logger.info("  Gamma (discount): %s", gamma)
        self.logger.info("  Tau (target update): %s", tau)
        self.logger.info("  Policy delay: %s", policy_delay)
        self.logger.info("  Target policy noise: %s", target_policy_noise)
        self.logger.info("  Action noise std: %s", action_noise_std)
        self.logger.info("  Network type: %s", network_type)
        self.logger.info("  Network kwargs: %s", network_kwargs)
        self.logger.info("  Network architecture: %s", policy_kwargs.get("net_arch"))

        if env is None:
            self.logger.warning("Environment is None - model creation skipped")
            self.model = None
            return

        load_path = self.config.get("load_model")
        if load_path:
            self.logger.info("Loading model from %s", load_path)
            self.model = TD3JALAlgorithm.load(
                load_path,
                env=env,
                device=str(self.device),
            )
            self.logger.info("TD3 JAL model setup complete")
            return

        self.model = TD3JALAlgorithm(
            env=env,
            policy="MlpPolicy",
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=int(model_params.get("learning_starts", 1000)),
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=int(model_params.get("train_freq", 1)),
            gradient_steps=int(model_params.get("gradient_steps", 1)),
            action_noise_std=action_noise_std,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            network_type=network_type,
            network_kwargs=network_kwargs,
            policy_kwargs=policy_kwargs,
            verbose=int(model_params.get("verbose", 1)),
            device=str(self.device),
            tensorboard_log=str(self.run_dir / "tensorboard"),
        )

        self.logger.info("TD3 JAL model setup complete")

    def train(self):
        """Execute TD3 training loop in timestep chunks."""
        if self.env is None:
            self.setup_environment()

        if self.model is None:
            self.setup_model(self.env)

        if self.env is None or self.model is None:
            self.logger.error("Cannot train: Environment or model failed to initialize")
            return

        total_timesteps = int(self.config.get("timesteps", 400_000))
        learn_batch_timesteps = int(self.config.get("learn_batch_timesteps", 2048))
        save_interval = int(self.config.get("save_interval", 10_000))

        self.logger.info("Starting TD3 JAL training for %s timesteps", f"{total_timesteps:,}")
        self.logger.info("Learning in batches of %s timesteps", f"{learn_batch_timesteps:,}")
        self.logger.info("Saving checkpoints every %s timesteps", f"{save_interval:,}")

        remaining = total_timesteps
        timesteps_trained = 0
        next_save_timesteps = save_interval

        while remaining > 0:
            chunk = min(learn_batch_timesteps, remaining)
            self.model.learn(total_timesteps=chunk, reset_num_timesteps=False)

            remaining -= chunk
            timesteps_trained += chunk
            self.training_metrics["total_timesteps"] = timesteps_trained

            self.logger.info(
                "Progress: %s/%s timesteps (%.1f%%)",
                f"{timesteps_trained:,}",
                f"{total_timesteps:,}",
                100 * timesteps_trained / total_timesteps,
            )

            save_name = Path(self.config.get("save_path", "models/td3_jal_policy.zip")).name
            while remaining > 0 and timesteps_trained >= next_save_timesteps:
                checkpoint_path = self._get_checkpoint_path(save_name, next_save_timesteps)
                self.save_model(str(checkpoint_path))
                self.logger.info("Checkpoint saved: %s", checkpoint_path)
                next_save_timesteps += save_interval

            if remaining == 0:
                final_path = self.model_dir / save_name
                self.save_model(str(final_path))
                self.logger.info("Final model saved: %s", final_path)

        self.logger.info("Training complete!")
        self.logger.info("Total timesteps: %s", f"{timesteps_trained:,}")
        self.logger.info("Model saved to: %s", self.model_dir)

    def save_model(self, path: str):
        """Save the trained TD3 model."""
        if self.model is None:
            raise RuntimeError("Model not initialized - cannot save")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        self.logger.info("Model saved to %s", save_path)

    def load_model(self, path: str):
        """Load a pre-trained TD3 model."""
        if self.env is None:
            raise RuntimeError("Environment must be initialized before loading model")

        self.model = TD3JALAlgorithm.load(path, env=self.env, device=str(self.device))
        self.logger.info("Model loaded from %s", path)

    def cleanup(self):
        """Cleanup resources after training."""
        super().cleanup()
        if self.networker:
            try:
                if hasattr(self.networker, "shutdown") and callable(self.networker.shutdown):
                    self.networker.shutdown()
                elif hasattr(self.networker, "disconnect_from_sim") and callable(self.networker.disconnect_from_sim):
                    self.networker.disconnect_from_sim()
            except Exception as e:
                self.logger.error("Error during networker shutdown: %s", e)

    def _load_team_config(self, file_path: str) -> list[TeamInfo]:
        """Load team configuration from JSON file."""
        import json

        with open(file_path, "r") as f:
            config = json.load(f)["teams"]

        if len(config) != 2:
            raise ValueError("Team configuration must contain exactly two teams.")

        team1_info, team2_info = config
        if len(team1_info) != 3 or len(team2_info) != 3:
            raise ValueError("Each team configuration must have name, n_players, and goalie_id.")

        return [TeamInfo(*team1_info), TeamInfo(*team2_info)]

    def _get_checkpoint_path(self, filename: str, timesteps: int) -> Path:
        """Generate checkpoint path with timesteps inside model_dir."""
        p = Path(filename)
        return self.model_dir / f"{p.stem}_steps{timesteps}{p.suffix}"

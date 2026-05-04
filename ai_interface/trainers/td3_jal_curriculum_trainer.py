"""
TD3 JAL Curriculum Trainer - Expandable networks for multi-robot learning.

Supports curriculum learning stages where robots are progressively added,
with the neural network expanding to accommodate new robots while retaining
learned weights for existing robots.
"""

import gymnasium as gym
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import numpy as np

from ai_interface.algorithms.td3_jal import TD3JALAlgorithm
from ai_interface.algorithms.td3_jal_expandable import ExpandableJALBackbone
from ai_interface.envs.JAL_env import JALTeamEnv
from ai_interface.trainers.base_trainer import BaseTrainer
from networking.networker import Networker, TeamInfo


class TD3JALCurriculumTrainer(BaseTrainer):
    """Trainer for TD3-JAL with curriculum learning and expandable networks."""

    def __init__(self, config: Dict[str, Any], log_dir: str = None, device=None):
        super().__init__(config, log_dir, algorithm_name="td3_jal_curriculum")
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.networker: Optional[Networker] = None
        self.env: Optional[JALTeamEnv] = None
        self.model: Optional[TD3JALAlgorithm] = None
        self.backbone: Optional[ExpandableJALBackbone] = None

        self.curriculum = config.get("curriculum", {})
        self.action_dim_per_robot = 8
        self.logger.info(f"TD3 JAL Curriculum Trainer initialized")

    def setup_environment(self, num_robots: int, robot_ids: List[int]) -> gym.Env:
        """Setup JAL environment for given number of robots."""
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

        self.env = JALTeamEnv(
            networker=self.networker,
            team_name=team_name,
            robot_ids=robot_ids,
            obs_dim_per_robot=int(self.config.get("obs_dim_per_robot", 8)),
            non_robot_obs_dim=int(self.config.get("non_robot_obs_dim", 4)),
            max_steps=int(self.config.get("max_steps", 200)),
            debug=bool(self.config.get("debug", False)),
        )

        self.logger.info(
            "JAL environment setup - Team: %s, robots=%s, obs_dim=%d, action_dim=%d",
            team_name,
            robot_ids,
            int(self.env.observation_space.shape[0]),
            int(self.env.action_space.shape[0]),
        )
        return self.env

    def setup_model(self, env: gym.Env, num_robots: int):
        """Setup TD3 model with expandable backbone."""
        self.logger.info("Setting up TD3-JAL model on device: %s", self.device)
        model_params = self.config.get("model_params", {})
        network_params = self.config.get("network_params", {})

        # Create expandable backbone
        max_robots = int(network_params.get("max_robots", 3))
        self.backbone = ExpandableJALBackbone(
            global_dim=int(self.config.get("non_robot_obs_dim", 4)),
            per_robot_dim=int(self.config.get("obs_dim_per_robot", 8)),
            action_dim_per_robot=self.action_dim_per_robot,
            max_robots=max_robots,
            feature_dim=int(network_params.get("feature_dim", 64)),
            num_heads=int(network_params.get("num_heads", 4)),
        )

        learning_rate = model_params.get("learning_rate", 1e-3)
        buffer_size = model_params.get("buffer_size", 100_000)
        batch_size = model_params.get("batch_size", 64)
        gamma = model_params.get("gamma", 0.9)
        tau = model_params.get("tau", 0.01)
        policy_delay = model_params.get("policy_delay", 2)
        target_policy_noise = model_params.get("target_policy_noise", 0.2)
        target_noise_clip = model_params.get("target_noise_clip", 0.5)
        action_noise_std = model_params.get("action_noise_std", 0.05)

        self.logger.info("TD3 Hyperparameters: lr=%s, buffer=%s, batch=%s, gamma=%s", 
                         learning_rate, buffer_size, batch_size, gamma)

        load_path = self.config.get("load_model")
        if load_path:
            self.logger.info("Loading model from %s", load_path)
            self.model = TD3JALAlgorithm.load(
                load_path,
                env=env,
                device=str(self.device),
            )
            self.logger.info("TD3 JAL model loaded")
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
            network_type="mlp",
            policy_kwargs=model_params.get("policy_kwargs", {"net_arch": [64, 48, 32]}),
            verbose=int(model_params.get("verbose", 1)),
            device=str(self.device),
            tensorboard_log=str(self.run_dir / "tensorboard"),
        )
        self.logger.info("TD3 JAL model setup complete")

    def train(self):
        """Execute curriculum learning stages."""
        if not self.curriculum:
            self.logger.error("No curriculum defined in config")
            return

        stages = sorted(self.curriculum.items())
        for stage_name, stage_config in stages:
            self._run_stage(stage_name, stage_config)

        self.logger.info("Curriculum training complete!")

    def _run_stage(self, stage_name: str, stage_config: Dict[str, Any]):
        """Run a single curriculum stage."""
        num_robots = int(stage_config.get("num_robots", 1))
        robot_ids = stage_config.get("robot_ids", list(range(1, num_robots + 1)))
        timesteps = int(stage_config.get("timesteps", 200000))

        self.logger.info(
            "Starting %s with %d robots %s",
            stage_name,
            num_robots,
            robot_ids,
        )

        # Setup environment
        if self.env is None or self.env.num_robots != num_robots:
            self.env = self.setup_environment(num_robots, robot_ids)

        # Expand model if needed or setup new
        if self.model is None:
            self.setup_model(self.env, num_robots)
        elif self.env.num_robots > (self.model.model.env.num_robots if hasattr(self.model.model, 'env') else 1):
            # Expand network for new robots
            self.logger.info("Expanding network to support %d robots", num_robots)
            if self.backbone:
                self.backbone.expand_robots(num_robots)
            # Reinitialize environment with expanded model
            self.model = None
            self.setup_model(self.env, num_robots)

        # Training loop for this stage
        learn_batch_timesteps = int(self.config.get("learn_batch_timesteps", 2048))
        save_interval = int(self.config.get("save_interval", 10000))
        checkpoint_dir = Path(self.config.get("save_path", "models/td3_jal_curriculum"))

        remaining = timesteps
        stage_timesteps = 0
        next_save_timesteps = save_interval

        self.logger.info("Training %s for %s timesteps", stage_name, f"{timesteps:,}")

        while remaining > 0:
            chunk = min(learn_batch_timesteps, remaining)
            self.model.learn(total_timesteps=chunk, reset_num_timesteps=False)

            remaining -= chunk
            stage_timesteps += chunk
            self.training_metrics["total_timesteps"] = stage_timesteps

            self.logger.info(
                "%s progress: %s/%s timesteps (%.1f%%)",
                stage_name,
                f"{stage_timesteps:,}",
                f"{timesteps:,}",
                100 * stage_timesteps / timesteps,
            )

            # Save checkpoints
            while remaining > 0 and stage_timesteps >= next_save_timesteps:
                checkpoint_path = checkpoint_dir / f"{stage_name}_steps{next_save_timesteps}.zip"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                self.save_model(str(checkpoint_path))
                self.logger.info("Checkpoint saved: %s", checkpoint_path)
                next_save_timesteps += save_interval

        # Save stage completion checkpoint
        stage_ckpt = checkpoint_dir / f"{stage_name}_complete.zip"
        stage_ckpt.parent.mkdir(parents=True, exist_ok=True)
        self.save_model(str(stage_ckpt))
        self.logger.info("%s complete - model saved: %s", stage_name, stage_ckpt)

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

    def _load_team_config(self, file_path: str) -> List[TeamInfo]:
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

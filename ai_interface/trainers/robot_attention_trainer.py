"""
Stable Baselines3 PPO trainer for RobotAttentionEnv.

This trainer wires RobotAttentionEnv to the attention-based feature extractor
defined in ai_interface.algorithms.robot_attention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
import torch
from stable_baselines3 import PPO

from ai_interface.algorithms.base import AlgorithmBase
from ai_interface.algorithms.robot_attention import (
    MultiRobotFeatureExtractor,
    SB3MultiRobotFeatureExtractor,
)
from ai_interface.envs.RobotAttentionEnv import RobotAttentionEnv
from ai_interface.trainers.base_trainer import BaseTrainer
from networking.networker import Networker, TeamInfo


class RobotAttentionTrainer(BaseTrainer):
    """Trainer for PPO with RobotAttentionEnv and attention feature backbone."""

    def __init__(
        self,
        config: Dict[str, Any],
        log_dir: str = None,
        device=None,
        actor_net_arch: Optional[Sequence[int]] = None,
        encoder_path: Optional[str] = None,
        action_head_path: Optional[str] = None,
    ):
        super().__init__(config, log_dir, algorithm_name="robot_attention")
        self.networker = None
        self.env = None
        self.model = None
        self.team_infos: list[TeamInfo] = []
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.actor_net_arch = (
            [int(width) for width in actor_net_arch]
            if actor_net_arch is not None
            else None
        )
        self.encoder_path = encoder_path or self.config.get("encoder_path")
        self.action_head_path = action_head_path or self.config.get("action_head_path")

    def setup_environment(self) -> gym.Env:
        """Setup RobotAttentionEnv."""

        team_infos = self._load_team_config(self.config["team_config"])
        self.team_infos = team_infos
        team_name = self.config.get("team_name") or team_infos[0].name
        feature_params = dict(self.config.get("feature_extractor_params", {}))
        time_window = int(feature_params.get("time_window", 1))

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
            robot_ids = list(range(1, int(self.config.get("num_robots", 6)) + 1))

        self.env = RobotAttentionEnv(
            networker=self.networker,
            team_name=team_name,
            robot_ids=robot_ids,
            max_steps=int(self.config.get("max_steps", 200)),
            debug=bool(self.config.get("debug", False)),
            time_window=time_window,
        )

        self.logger.info(
            "RobotAttentionEnv created - Team: %s, obs_shape=%s, action_shape=%s, robots=%s, Endpoint: %s:%d/%d",
            team_name,
            tuple(self.env.observation_space.shape),
            tuple(self.env.action_space.shape),
            robot_ids,
            sim_host,
            sim_player_port,
            sim_trainer_port,
        )
        return self.env

    def setup_model(self, env: gym.Env):
        """Setup PPO with the attention feature extractor."""

        if env is None:
            raise ValueError("env must not be None")

        model_params = dict(self.config.get("model_params", {}))
        model_class = self.config.get("model_class", PPO)

        if "device" not in model_params:
            model_params["device"] = str(self.device)

        feature_params = dict(self.config.get("feature_extractor_params", {}))
        hidden_dim = int(feature_params.get("hidden_dim", 64))
        num_heads = int(feature_params.get("num_heads", 4))
        num_attention_layers = int(feature_params.get("num_attention_layers", 1))
        time_window = int(feature_params.get("time_window", 1))

        backbone = MultiRobotFeatureExtractor(
            obs_dim=int(self.env.obs_dim_per_robot),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_attention_layers=num_attention_layers,
        )

        user_policy_kwargs = dict(model_params.pop("policy_kwargs", {}))
        net_arch = user_policy_kwargs.get("net_arch")
        if self.actor_net_arch is not None:
            if net_arch is None:
                user_policy_kwargs["net_arch"] = {
                    "pi": list(self.actor_net_arch),
                    "vf": list(self.actor_net_arch),
                }
            elif isinstance(net_arch, dict):
                user_policy_kwargs["net_arch"] = {
                    **net_arch,
                    "pi": list(self.actor_net_arch),
                }
            else:
                user_policy_kwargs["net_arch"] = {
                    "pi": list(self.actor_net_arch),
                    "vf": list(net_arch),
                }

        merged_policy_kwargs = {
            "features_extractor_class": SB3MultiRobotFeatureExtractor,
            "features_extractor_kwargs": {
                "feature_extractor": backbone,
                "num_robots": int(self.env.num_robots),
                "obs_dim": int(self.env.obs_dim_per_robot),
                "time_window": time_window,
            },
            **user_policy_kwargs,
        }
        model_params["policy_kwargs"] = merged_policy_kwargs

        self.logger.info("Using device for model: %s", self.device)

        try:
            is_algo_base = issubclass(model_class, AlgorithmBase)
        except Exception:
            is_algo_base = False

        assert is_algo_base or issubclass(
            model_class, PPO
        ), "Model class must be a subclass of AlgorithmBase or PPO"

        if is_algo_base:
            self.model = model_class(env, model_params)
        else:
            self.model = model_class("MlpPolicy", env, **model_params)

        self._load_pretrained_modules()
        self._attach_opponent_model()

        self.logger.info(
            "Model setup complete - Class: %s, hidden_dim=%d, num_heads=%d, attention_layers=%d, time_window=%d, actor_net_arch=%s, encoder_path=%s, action_head_path=%s",
            model_class.__name__,
            hidden_dim,
            num_heads,
            num_attention_layers,
            time_window,
            self.actor_net_arch,
            self.encoder_path,
            self.action_head_path,
        )

        if self.config.get("load_model"):
            self.load_model(self.config["load_model"])

    def train(self):
        """Execute the SB3 PPO training loop."""

        if self.env is None:
            self.setup_environment()
        if self.model is None:
            self.setup_model(self.env)

        print("Starting training with config:", self.config)  # Debug print for config visibility

        total_timesteps = int(self.config.get("timesteps", 10000))
        learn_batch_timesteps = int(self.config.get("learn_batch_timesteps", 2048))
        save_interval = int(self.config.get("save_interval", 5000))

        self.logger.info(
            "Starting robot-attention training for %d total timesteps",
            total_timesteps,
        )

        remaining = total_timesteps
        timesteps_trained = 0

        while remaining > 0:
            chunk = min(learn_batch_timesteps, remaining)
            self.model.learn(total_timesteps=chunk)

            remaining -= chunk
            timesteps_trained += chunk
            self.training_metrics["total_timesteps"] = timesteps_trained

            self.logger.info(
                "Trained %d/%d timesteps", timesteps_trained, total_timesteps
            )

            if timesteps_trained % save_interval == 0 or remaining == 0:
                save_name = Path(
                    self.config.get("save_path", "models/robot_attention_policy.zip")
                ).name
                if remaining > 0:
                    checkpoint_path = self._get_checkpoint_path(
                        save_name, timesteps_trained
                    )
                    self.save_model(str(checkpoint_path))
                    self.logger.info("Model checkpoint saved to %s", checkpoint_path)
                else:
                    final_path = self.model_dir / save_name
                    self.save_model(str(final_path))
                    self.logger.info("Final model saved to %s", final_path)

    def save_model(self, path: str):
        """Save the trained model."""

        if self.model is None:
            raise RuntimeError("Model not initialized")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, "save"):
            self.model.save(str(save_path))
        else:
            save_fn = getattr(self.model, "save_state", None)
            if callable(save_fn):
                save_fn(str(save_path))
            else:
                raise RuntimeError("Model does not implement a save method")

    def load_model(self, path: str):
        """Load a pre-trained model."""

        if self.env is None:
            raise RuntimeError("Environment must be setup before loading model")

        model_class = self.config.get("model_class", PPO)

        try:
            is_algo_base = issubclass(model_class, AlgorithmBase)
        except Exception:
            is_algo_base = False
        
        assert is_algo_base or issubclass(model_class, PPO), "Model class must be a subclass of AlgorithmBase or PPO"

        self.model = model_class.load(path, env=self.env)
        self._attach_opponent_model()
        self.logger.info("Model loaded from %s", path)

    def cleanup(self):
        """Cleanup resources after training."""

        super().cleanup()
        if self.networker:
            try:
                self.networker.shutdown()
            except Exception as e:
                self.logger.error(f"Error during networker shutdown: {e}")

    def _load_pretrained_modules(self) -> None:
        """Load optional pretrained feature extractor and action head weights."""

        if self.model is None:
            raise RuntimeError("Model must be initialized before loading pretrained modules")

        policy = getattr(self.model, "policy", None)
        if policy is None:
            raise RuntimeError("Model does not expose a policy module")

        if self.encoder_path:
            self._load_module_state_dict(
                module=policy.feature_extractor,
                checkpoint_path=self.encoder_path,
                module_name="feature_extractor",
            )

        if self.action_head_path:
            self._load_module_state_dict(
                module=policy.action_net,
                checkpoint_path=self.action_head_path,
                module_name="action_net",
            )

    def _load_module_state_dict(
        self, module: torch.nn.Module, checkpoint_path: str, module_name: str
    ) -> None:
        """Load a state dict into one policy submodule."""

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint and isinstance(
                checkpoint["model_state_dict"], dict
            ):
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            raise ValueError(
                f"Unsupported checkpoint format in {module_name} path={checkpoint_path}"
            )

        module.load_state_dict(state_dict)
        self.logger.info("Loaded pretrained %s weights from %s", module_name, checkpoint_path)

    def _load_team_config(self, file_path: str) -> list[TeamInfo]:
        """Load team configuration from JSON file."""

        import json

        with open(file_path, "r") as f:
            config = json.load(f)["teams"]

        if len(config) != 2:
            raise ValueError("Team configuration must contain exactly two teams.")

        team1_info, team2_info = config
        if len(team1_info) != 3 or len(team2_info) != 3:
            raise ValueError(
                "Each team configuration must have name, n_players, and goalie_id."
            )

        return [TeamInfo(*team1_info), TeamInfo(*team2_info)]

    def _attach_opponent_model(self) -> None:
        """Point the env opponent branch at this trainer's current model."""

        if self.env is None or self.model is None or len(self.team_infos) < 2:
            return

        self.env.opponent_model = self.model
        self.env.opponent_name = self.team_infos[1].name

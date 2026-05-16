"""Trainer for single-agent HSM + SB3 PPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .base_trainer import BaseTrainer
from .parallel_envs import make_networked_env_factory, make_vec_env
from ai_interface.envs.hsm_sb3_env import HSMSingleAgentEnv
from networking.networker import Networker, TeamInfo


class _SB3LoggingCallback(BaseCallback):
    """Emit episode and update/loss logs in the project's trainer style."""

    def __init__(self, trainer: "HSMSB3PPOTrainer"):
        super().__init__()
        self.trainer = trainer
        self.episode_count = int(trainer.training_metrics.get("episodes", 0))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode") if isinstance(info, dict) else None
            if ep is None:
                continue

            self.episode_count += 1
            reward = float(ep.get("r", 0.0))
            length = int(ep.get("l", 0))
            self.trainer.log_episode(self.episode_count, reward, length)
        return True


class HSMSB3PPOTrainer(BaseTrainer):
    """Single-agent hierarchical state machine training with SB3 PPO."""

    def __init__(self, config: Dict[str, Any], log_dir: str = None, device=None):
        super().__init__(config, log_dir, algorithm_name="hsm_sb3_ppo")
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.networker: Networker | None = None
        self.env: gym.Env | None = None
        self.model: PPO | None = None
        self._callback: _SB3LoggingCallback | None = None
        self.num_envs = 1

    def setup_environment(self) -> gym.Env:
        team_infos = self._load_team_config(self.config["team_config"])
        team_name = self.config.get("team_name") or team_infos[0].name

        self.num_envs = max(1, int(self.config.get("num_envs", 1)))
        env_mode = self.config.get("env_mode", "sim-only")
        env_kwargs = {
            "obs_dim": int(self.config.get("obs_dim", 28)),
            "max_steps": int(self.config.get("max_steps", 220)),
            "unum": int(self.config.get("unum", 1)),
            "debug": bool(self.config.get("debug_env", False)),
        }
        env_fns = []

        for env_idx in range(self.num_envs):
            sim_host, sim_player_port, sim_trainer_port = self._sim_endpoint_for_env(env_idx)
            env_fns.append(
                make_networked_env_factory(
                    HSMSingleAgentEnv,
                    team_infos,
                    env_mode,
                    team_name,
                    sim_host,
                    sim_player_port,
                    sim_trainer_port,
                    env_kwargs=env_kwargs,
                    monitor=True,
                )
            )
            self.logger.info(
                "Env %d connected to %s:%d/%d",
                env_idx,
                sim_host,
                sim_player_port,
                sim_trainer_port,
            )

        self.env = make_vec_env(
            env_fns,
            backend=self.config.get("parallel_backend", "subproc"),
            start_method=self.config.get("vec_env_start_method"),
        )
        self.logger.info(
            "HSM-SB3 environment setup complete - team=%s unum=%d parallel_envs=%d",
            team_name,
            int(self.config.get("unum", 1)),
            self.num_envs,
        )
        return self.env

    def setup_model(self, env: gym.Env):
        model_params = dict(self.config.get("model_params", {}))
        model_params.setdefault("learning_rate", 3e-4)
        model_params.setdefault("n_steps", 2048)
        model_params.setdefault("batch_size", 64)
        model_params.setdefault("n_epochs", 10)
        model_params.setdefault("gamma", 0.99)
        model_params.setdefault("gae_lambda", 0.95)
        model_params.setdefault("clip_range", 0.2)
        model_params.setdefault("ent_coef", 0.01)
        model_params.setdefault("vf_coef", 0.5)
        model_params.setdefault("max_grad_norm", 0.5)

        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            device=str(self.device),
            **model_params,
        )
        self._callback = _SB3LoggingCallback(self)
        self.logger.info("HSM-SB3 PPO model initialized on device=%s", self.device)

        if self.config.get("load_model"):
            self.logger.info("Loading model from %s", self.config["load_model"])
            self.load_model(self.config["load_model"])

    def train(self):
        if self.env is None:
            self.setup_environment()
        if self.model is None:
            self.setup_model(self.env)

        total_timesteps = int(self.config.get("timesteps", 200000))
        learn_batch_timesteps = int(self.config.get("learn_batch_timesteps", 4096))
        save_interval = int(self.config.get("save_interval", 50000))

        self.logger.info(
            "Starting HSM-SB3 PPO training for %d timesteps (chunk=%d)",
            total_timesteps,
            learn_batch_timesteps,
        )

        remaining = total_timesteps
        trained = 0
        last_updates = int(getattr(self.model, "_n_updates", 0))

        while remaining > 0:
            chunk = min(learn_batch_timesteps, remaining)
            self.model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                callback=self._callback,
            )

            current_updates = int(getattr(self.model, "_n_updates", last_updates))
            if current_updates > last_updates:
                stats = self.model.logger.name_to_value

                def _get(*keys: str, default: float = 0.0) -> float:
                    for key in keys:
                        if key in stats:
                            return float(stats[key])
                    return float(default)

                policy_loss = _get("train/policy_gradient_loss", "train/policy_loss")
                value_loss = _get("train/value_loss")
                entropy_loss = _get("train/entropy_loss")
                total_loss = _get("train/loss")
                # SB3 tracks entropy as a loss term (typically negative).
                entropy = -entropy_loss

                self.logger.info(
                    "PPO update #%d after %d steps - Policy loss: %.4f, Value loss: %.4f, Entropy: %.4f",
                    current_updates,
                    chunk,
                    policy_loss,
                    value_loss,
                    entropy,
                )
                self.log_metrics(
                    {
                        "update": current_updates,
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "entropy": entropy,
                        "total_loss": total_loss,
                    }
                )
                last_updates = current_updates

            remaining -= chunk
            trained += chunk
            self.training_metrics["total_timesteps"] = trained

            self.logger.info("Trained %d/%d timesteps", trained, total_timesteps)

            if trained % save_interval == 0 or remaining == 0:
                save_name = Path(self.config.get("save_path", "models/hsm_sb3_ppo_policy.zip")).name
                if remaining > 0:
                    checkpoint_path = self._get_checkpoint_path(save_name, trained)
                    self.save_model(str(checkpoint_path))
                    self.logger.info("Checkpoint saved: %s", checkpoint_path)
                else:
                    final_path = self.model_dir / save_name
                    self.save_model(str(final_path))
                    self.logger.info("Final model saved: %s", final_path)

        self.plot_training()

    def save_model(self, path: str):
        if self.model is None:
            raise RuntimeError("Model not initialized")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))

    def load_model(self, path: str):
        if self.env is None:
            self.setup_environment()
        self.model = PPO.load(path, env=self.env, device=str(self.device))
        self.logger.info("Loaded model from %s", path)

    def cleanup(self):
        super().cleanup()
        if self.env is not None and hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception as e:
                self.logger.error("Error during env shutdown: %s", e)
        if self.networker:
            try:
                self.networker.shutdown()
            except Exception as e:
                self.logger.error("Error during network shutdown: %s", e)

    @staticmethod
    def _load_team_config(file_path: str) -> List[TeamInfo]:
        with open(file_path, "r", encoding="utf-8") as f:
            config = json.load(f)["teams"]

        if len(config) != 2:
            raise ValueError("Team configuration must contain exactly two teams.")

        team1_info, team2_info = config
        if len(team1_info) != 3 or len(team2_info) != 3:
            raise ValueError("Each team configuration must have name, n_players, and goalie_id.")

        return [TeamInfo(*team1_info), TeamInfo(*team2_info)]

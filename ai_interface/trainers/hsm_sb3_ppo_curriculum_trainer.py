"""Trainer for HSM SB3 PPO with curriculum learning support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .base_trainer import BaseTrainer
from ai_interface.envs.hsm_sb3_env import HSMSingleAgentEnv
from networking.networker import Networker, TeamInfo


class _SB3LoggingCallback(BaseCallback):
    """Emit episode and update/loss logs in the project's trainer style."""

    def __init__(self, trainer: "HSMSB3PPOCurriculumTrainer"):
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


class HSMSB3PPOCurriculumTrainer(BaseTrainer):
    """Single-agent HSM training with SB3 PPO and curriculum learning."""

    def __init__(self, config: Dict[str, Any], log_dir: str = None, device=None):
        super().__init__(config, log_dir, algorithm_name="hsm_sb3_ppo_curriculum")
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.networker: Optional[Networker] = None
        self.env: Optional[HSMSingleAgentEnv] = None
        self.model: Optional[PPO] = None
        self._callback: Optional[_SB3LoggingCallback] = None

        self.curriculum = config.get("curriculum", {})
        self.logger.info("HSM SB3 PPO Curriculum Trainer initialized")

    def setup_environment(self, unum: int = 1) -> gym.Env:
        """Setup HSM SB3 environment."""
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

        self.env = HSMSingleAgentEnv(
            networker=self.networker,
            team_name=team_name,
            obs_dim=int(self.config.get("obs_dim", 28)),
            max_steps=int(self.config.get("max_steps", 220)),
            unum=int(unum),
            debug=bool(self.config.get("debug_env", False)),
        )
        # Monitor injects episode reward/length into info for callback logging.
        self.env = Monitor(self.env)
        self.logger.info(
            "HSM-SB3 environment setup complete - team=%s unum=%d endpoint=%s:%d/%d",
            team_name,
            int(unum),
            sim_host,
            sim_player_port,
            sim_trainer_port,
        )
        return self.env

    def setup_model(self, env: gym.Env):
        """Setup PPO model."""
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
        unum = int(stage_config.get("unum", 1))
        timesteps = int(stage_config.get("timesteps", 200000))
        opponent_difficulty = stage_config.get("opponent_difficulty", "none")

        self.logger.info(
            "Starting %s: unum=%d, opponent=%s",
            stage_name,
            unum,
            opponent_difficulty,
        )

        # Setup environment
        if self.env is None or (hasattr(self.env, 'env') and self.env.env.unum != unum):
            self.env = self.setup_environment(unum)

        # Setup model (load if it exists from previous stage, else create new)
        if self.model is None:
            self.setup_model(self.env)

        # Training loop for this stage
        learn_batch_timesteps = int(self.config.get("learn_batch_timesteps", 4096))
        save_interval = int(self.config.get("save_interval", 50000))
        checkpoint_dir = Path(self.config.get("save_path", "models/hsm_sb3_ppo_curriculum"))

        remaining = timesteps
        stage_timesteps = 0
        next_save_timesteps = save_interval

        self.logger.info("Training %s for %s timesteps", stage_name, f"{timesteps:,}")

        while remaining > 0:
            chunk = min(learn_batch_timesteps, remaining)
            self.model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                callback=self._callback,
            )

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
        """Save the trained PPO model."""
        if self.model is None:
            raise RuntimeError("Model not initialized - cannot save")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        self.logger.info("Model saved to %s", save_path)

    def load_model(self, path: str):
        """Load a pre-trained PPO model."""
        if self.env is None:
            raise RuntimeError("Environment must be initialized before loading model")

        self.model = PPO.load(path, env=self.env, device=str(self.device))
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
        with open(file_path, "r") as f:
            config = json.load(f)["teams"]

        if len(config) != 2:
            raise ValueError("Team configuration must contain exactly two teams.")

        team1_info, team2_info = config
        if len(team1_info) != 3 or len(team2_info) != 3:
            raise ValueError("Each team configuration must have name, n_players, and goalie_id.")

        return [TeamInfo(*team1_info), TeamInfo(*team2_info)]

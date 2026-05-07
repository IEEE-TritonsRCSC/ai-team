"""
Stable Baselines3 PPO Trainer implementation.

This trainer wraps the existing SB3 trainer with the new base trainer interface.
"""
import gymnasium as gym
from typing import Dict, Any
from pathlib import Path
import torch

from .base_trainer import BaseTrainer
from .parallel_envs import make_networked_env_factory, make_vec_env
from ai_interface.envs.sim_env import SimulatorEnv
from ai_interface.algorithms.base import AlgorithmBase
from networking.networker import TeamInfo
from stable_baselines3 import PPO


class SB3PPOTrainer(BaseTrainer):
    """Trainer for Stable Baselines3 PPO algorithm."""
    
    def __init__(self, config: Dict[str, Any], log_dir: str = None, device=None):
        super().__init__(config, log_dir, algorithm_name="sb3_ppo")
        self.networker = None
        self.env = None
        self.model = None
        self.device = device
        self.num_envs = 1
    
    def setup_environment(self) -> gym.Env:
        """Setup the simulator environment."""
        # Load team configuration
        team_infos = self._load_team_config(self.config["team_config"])
        team_name = self.config.get("team_name") or team_infos[0].name

        self.num_envs = max(1, int(self.config.get("num_envs", 1)))
        env_mode = self.config.get("env_mode", "sim-only")
        env_fns = []

        for env_idx in range(self.num_envs):
            sim_host, sim_player_port, sim_trainer_port = self._sim_endpoint_for_env(env_idx)
            env_fns.append(
                make_networked_env_factory(
                    SimulatorEnv,
                    team_infos,
                    env_mode,
                    team_name,
                    sim_host,
                    sim_player_port,
                    sim_trainer_port,
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
            "Environment setup complete - Team: %s, Parallel envs: %d",
            team_name, self.num_envs
        )
        return self.env
    
    def setup_model(self, env: gym.Env):
        """Setup the SB3 PPO model."""
        model_params = self.config.get("model_params", {})
        model_class = self.config.get("model_class", PPO)
        self.logger.info(f"Using device for model: {str(self.device)}")
        if "device" not in model_params:
            model_params = dict(model_params)
            model_params["device"] = str(self.device)
        
        # Determine if it's a custom algorithm or SB3
        try:
            is_algo_base = issubclass(model_class, AlgorithmBase)
        except Exception:
            is_algo_base = False
        
        if is_algo_base:
            # AlgorithmBase implementations may expect (env, model_params)
            self.model = model_class(env, model_params)
        else:
            # Assume SB3-style: (policy_str, env, **params)
            self.model = model_class("MlpPolicy", env, **model_params)
        
        self.logger.info(f"Model setup complete - Class: {model_class.__name__}")
        
        # Load pre-trained model if specified
        if self.config.get("load_model"):
            self.load_model(self.config["load_model"])
    
    def train(self):
        """Execute the SB3 PPO training loop."""
        if self.env is None:
            self.setup_environment()
        if self.model is None:
            self.setup_model(self.env)
        
        total_timesteps = self.config.get("timesteps", 10000)
        learn_batch_timesteps = self.config.get("learn_batch_timesteps", 2048)
        save_interval = self.config.get("save_interval", 5000)
        
        self.logger.info(f"Starting training for {total_timesteps} total timesteps...")
        
        # Training loop - SB3 handles the episode structure internally
        remaining = total_timesteps
        timesteps_trained = 0
        
        while remaining > 0:
            chunk = min(learn_batch_timesteps, remaining)
            
            # Train for this chunk
            self.model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            
            remaining -= chunk
            timesteps_trained += chunk
            self.training_metrics["total_timesteps"] = timesteps_trained
            
            # Log progress
            self.logger.info(f"Trained {timesteps_trained}/{total_timesteps} timesteps")
            
            # Save model periodically
            if timesteps_trained % save_interval == 0 or remaining == 0:
                save_name = Path(self.config.get("save_path", "models/sb3_ppo_policy.zip")).name
                if remaining > 0:  # Checkpoint save
                    checkpoint_path = self._get_checkpoint_path(save_name, timesteps_trained)
                    self.save_model(str(checkpoint_path))
                    self.logger.info(f"Model checkpoint saved to {checkpoint_path}")
                else:  # Final save
                    final_path = self.model_dir / save_name
                    self.save_model(str(final_path))
                    self.logger.info(f"Final model saved to {final_path}")
    
    def save_model(self, path: str):
        """Save the trained SB3 model."""
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
        """Load a pre-trained SB3 model."""
        if self.model is None and self.env is None:
            raise RuntimeError("Environment must be setup before loading model")
        
        model_class = self.config.get("model_class", PPO)
        
        try:
            is_algo_base = issubclass(model_class, AlgorithmBase)
        except Exception:
            is_algo_base = False
        
        if is_algo_base:
            # AlgorithmBase implementations should provide a `load` classmethod
            self.model = model_class.load(path, env=self.env)
        else:
            # Assume SB3-style
            self.model = model_class.load(path, env=self.env)
        
        self.logger.info(f"Model loaded from {path}")
    
    def cleanup(self):
        """Cleanup resources after training."""
        super().cleanup()
        if self.env is not None and hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception as e:
                self.logger.error(f"Error during env shutdown: {e}")
        if self.networker:
            try:
                self.networker.shutdown()
            except Exception as e:
                self.logger.error(f"Error during networker shutdown: {e}")
    
    def _load_team_config(self, file_path: str) -> list[TeamInfo]:
        """Load team configuration from JSON file."""
        import json
        
        with open(file_path, 'r') as f:
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

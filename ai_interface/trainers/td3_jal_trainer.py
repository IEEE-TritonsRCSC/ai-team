"""
TD3 JAL (Joint-Action Learners) Trainer for RoboCup Soccer.

This trainer implements the JAL approach from the paper using TD3 (Twin Delayed DDPG)
instead of vanilla DDPG for improved stability and performance.

JAL treats multiple robots as a single agent with concatenated action spaces:
- 1 robot → 6D action space [robot1_actions(6D),]
- Each robot: [kick_logit, dash_logit, turn_logit, dash_power, dash_angle, turn_angle]

Key differences from DDPG (used in original paper):
- TD3 uses twin Q-networks to reduce overestimation bias
- Delayed policy updates for stability
- Target policy smoothing to reduce variance
"""

import gymnasium as gym
from typing import Dict, Any
from pathlib import Path
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from ai_interface.trainers.base_trainer import BaseTrainer
from networking.networker import Networker, TeamInfo
from ai_interface.envs.td3_jal_env import TD3JALEnv

class TD3JALTrainer(BaseTrainer):
    """
    Trainer for TD3 with Joint-Action Learning (JAL).
    
    This trainer coordinates 2 robots as a single agent with a 10-dimensional
    continuous action space, following the JAL approach from the research paper.
    """
    
    def __init__(self, config: Dict[str, Any], log_dir: str = None, device=None):
        """
        Initialize TD3 JAL trainer.
        
        Args:
            config: Configuration dictionary containing:
                - team_config: Path to team config JSON
                - env_mode: Environment mode (sim-only, etc.)
                - team_name: Name of team to control
                - timesteps: Total training timesteps
                - save_interval: Save model every N timesteps
                - obs_dim: Observation dimension 
                - num_robots: Number of robots in JAL
                - model_params: TD3-specific hyperparameters
            log_dir: Directory for logs
            device: Torch device (cuda/cpu)
        """
        super().__init__(config, log_dir)
        self.networker = None   # setup in the setup_environment() method
        self.env = None     # setup in the setup_environment() method
        self.model = None
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # JAL-specific configuration
        self.num_robots = config.get("num_robots", 1)  # For the 1v0 scenario
        self.action_dim_per_robot = 6  # [kick_logit, dash_logit, turn_logit, dash_pow, dash_ang, turn_ang]
        self.total_action_dim = self.num_robots * self.action_dim_per_robot # 1 * 6 = 6
        self.logger.info(f"TD3 JAL Trainer initialized: {self.num_robots} robots, {self.total_action_dim}D action space")
    
    
    def setup_environment(self) -> gym.Env:
        """
        Setup the JAL environment for TD3 training.
        
        The environment will be implemented separately (assumed for now).
        It should return:
        - Observation: 10D state vector (as in paper) ## Might need to change for 1v0
        - Action: 6D continuous vector (1 robot × 6 actions each)
        - Reward: From calculate_reward() function (implemented by teammate)
        
        Returns:
            JAL environment instance
        """        
        team_infos = self._load_team_config(self.config["team_config"])
        team_name = self.config.get("team_name") or team_infos[0].name

        self.networker = Networker(team_infos, self.config.get("env_mode", "sim-only"))

        self.env = TD3JALEnv(
            networker=self.networker,
            team_name=team_name,
            robot_id=1,
            num_robots=self.config.get("num_robots", 1),
            obs_dim=self.config.get("obs_dim", 10),
            max_steps=self.config.get("max_steps", 200),
        )

        self.logger.info(f"TD3JALEnv created - Team: {team_name}, obs_dim: {self.config.get('obs_dim', 10)}, action_dim: {self.total_action_dim}")
        return self.env
    
    
    def setup_model(self, env: gym.Env):
        """
        Setup the TD3 model with JAL-appropriate hyperparameters.
        
        TD3 hyperparameters based on the paper and best practices:
        - Learning rate: 1e-3 (same as DDPG in paper)
        - Buffer size: 100K (same as paper)
        - Batch size: 64 (same as paper)
        - Gamma: 0.9 (same as paper)
        - Tau: 0.01 (same as paper for soft target updates)
        - Policy delay: 2 (TD3-specific, delays policy updates)
        - Target policy noise: 0.2 (TD3-specific, for smoothing)
        - Action noise: Gaussian with std=0.05 (similar to paper's ε=0.5)
        
        Args:
            env: The JAL environment instance
        """
        self.logger.info(f"Setting up TD3 model on device: {self.device}")
        
        # Get hyperparameters from config or use paper defaults
        model_params = self.config.get("model_params", {})
        
        # TD3 default hyperparameters (matching paper where possible)
        learning_rate = model_params.get("learning_rate", 1e-3)  # Same as DDPG in paper
        buffer_size = model_params.get("buffer_size", 100_000)  # Same as paper
        batch_size = model_params.get("batch_size", 64)  # Same as paper
        gamma = model_params.get("gamma", 0.9)  # Same as paper
        tau = model_params.get("tau", 0.01)  # Same as paper (soft target update)
        
        # TD3-specific parameters (not in original DDPG paper)
        policy_delay = model_params.get("policy_delay", 2)  # Update policy every 2 critic updates
        # These are used to add noise to the target actor's predictions which are used to update the critic's weights.
        target_policy_noise = model_params.get("target_policy_noise", 0.2)  # Noise for target smoothing
        target_noise_clip = model_params.get("target_noise_clip", 0.5)  # Clip target noise
        
        # Action noise (exploration)
        # Paper used ε-greedy with ε=0.5, we use Gaussian noise with std=0.05
        action_noise_std = model_params.get("action_noise_std", 0.05)
        
        # Network architecture from paper: [64, 48, 32]
        # In SB3, we specify this via policy_kwargs
        policy_kwargs = model_params.get("policy_kwargs", {
            "net_arch": [64, 48, 32]  # Same as paper
        })
        
        # Create action noise for exploration
        # TD3 uses Ornstein-Uhlenbeck or Gaussian noise
        # This is used while actually taking actions in the simulator evironment
        n_actions = self.total_action_dim  # 6 for 1 robot
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=action_noise_std * np.ones(n_actions)
        ) 

        self.logger.info("TD3 Hyperparameters:")
        self.logger.info(f"  Learning rate: {learning_rate}")
        self.logger.info(f"  Buffer size: {buffer_size:,}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Gamma (discount): {gamma}")
        self.logger.info(f"  Tau (target update): {tau}")
        self.logger.info(f"  Policy delay: {policy_delay}")
        self.logger.info(f"  Target policy noise: {target_policy_noise}")
        self.logger.info(f"  Action noise std: {action_noise_std}")
        self.logger.info(f"  Network architecture: {policy_kwargs['net_arch']}")
        
        # Create TD3 model
        # NOTE: env is None for now (placeholder), will be actual JAL environment later
        if env is None:
            self.logger.warning("Environment is None - model creation skipped")
            self.logger.info("When environment is ready, TD3 will be initialized with:")
            self.logger.info(f"  TD3('MlpPolicy', env, learning_rate={learning_rate}, ...)")
            self.model = None
            return
        
        self.model = TD3(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=1000,  # Start learning after 1000 steps (warm-up)
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=1,  # Update at every step
            gradient_steps=1,  # One gradient step per update
            action_noise=action_noise,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=str(self.device),
            tensorboard_log=str(self.run_dir / "tensorboard")
        )

        # An MLP policy is just a regular feed-forward neural network. It takes the current 
        # game state numbers (the observation vector) as input, passes them through a few 
        # fully-connected layers, and outputs a continuous action vector. In TD3, that 
        # action vector is our 6 values per robot (3 logits for kick/dash/turn + 3 raw parameters). 
        # Then the environment decodes those outputs into an actual command.
        
        # Load pre-trained model if specified
        if self.config.get("load_model"):
            self.logger.info(f"Loading model from {self.config['load_model']}")
            self.load_model(self.config["load_model"])
        
        self.logger.info("TD3 JAL model setup complete")
    
    
    def train(self):
        """
        Execute TD3 training loop.
        
        Unlike PPO (which uses episodes), TD3 uses timesteps.
        SB3's TD3.learn() handles the training loop internally.
        """
        if self.env is None:
            self.setup_environment()
        
        if self.model is None:
            self.setup_model(self.env)

        if self.env is None or self.model is None:
            self.logger.error("Cannot train: Environment or model failed to initialize")
            return
        
        total_timesteps = self.config.get("timesteps", 400_000)  # Paper used ~300K-400K
        learn_batch_timesteps = self.config.get("learn_batch_timesteps", 2048)
        save_interval = self.config.get("save_interval", 10_000)
        
        self.logger.info(f"Starting TD3 JAL training for {total_timesteps:,} timesteps")
        self.logger.info(f"Learning in batches of {learn_batch_timesteps:,} timesteps")
        self.logger.info(f"Saving checkpoints every {save_interval:,} timesteps")
        
        # Training loop - TD3 handles episode structure internally
        remaining = total_timesteps
        timesteps_trained = 0
        next_save_timesteps = save_interval
        
        while remaining > 0:
            chunk = min(learn_batch_timesteps, remaining)
            
            # Train for this chunk
            self.model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            
            remaining -= chunk
            timesteps_trained += chunk
            self.training_metrics["total_timesteps"] = timesteps_trained
            
            # Log progress
            self.logger.info(f"Progress: {timesteps_trained:,}/{total_timesteps:,} timesteps ({100*timesteps_trained/total_timesteps:.1f}%)")
            
            # Save checkpoints whenever we cross the next save threshold.
            save_name = Path(self.config.get("save_path", "models/td3_jal_policy.zip")).name
            
            # If a large chunk skips over multiple save thresholds, save each one.
            while remaining > 0 and timesteps_trained >= next_save_timesteps:
                checkpoint_path = self._get_checkpoint_path(save_name, next_save_timesteps)
                self.save_model(str(checkpoint_path))
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")
                next_save_timesteps += save_interval
            
            # Always save a final model at the end.
            if remaining == 0:
                final_path = self.model_dir / save_name
                self.save_model(str(final_path))
                self.logger.info(f"Final model saved: {final_path}")
        
        # Plot training progress (if we tracked episode rewards)
        # Note: For TD3, we'd need to add custom callbacks to track episode rewards
        # This is a TODO for later
        self.logger.info("Training complete!")
        self.logger.info(f"Total timesteps: {timesteps_trained:,}")
        self.logger.info(f"Model saved to: {self.model_dir}")
    
    
    def save_model(self, path: str):
        """
        Save the trained TD3 model.
        
        Args:
            path: Path where the model should be saved (*.zip for SB3)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized - cannot save")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # SB3 models save as .zip files
        self.model.save(str(save_path))
        self.logger.info(f"Model saved to {save_path}")
    
    
    def load_model(self, path: str):
        """
        Load a pre-trained TD3 model.
        
        Args:
            path: Path to the model to load (*.zip for SB3)
        """
        if self.env is None:
            raise RuntimeError("Environment must be initialized before loading model")
        
        self.model = TD3.load(path, env=self.env, device=str(self.device))
        self.logger.info(f"Model loaded from {path}")
    
    
    def cleanup(self):
        """Cleanup resources after training."""
        super().cleanup()
        if self.networker:
            try:
                self.networker.shutdown()
            except Exception as e:
                self.logger.error(f"Error during networker shutdown: {e}")
    

    # Figure out how this will work in the 1v0 scenario. Check if there any other changes that need to be made for 1v0?
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


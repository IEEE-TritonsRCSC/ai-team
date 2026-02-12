"""
Discrete PPO Trainer implementation for simplified soccer environment.

This trainer handles discrete PPO training using the simplified action space
with the SimplifiedSoccerEnv environment.
"""
import gymnasium as gym
import torch
from typing import Dict, Any
from pathlib import Path
import json

from .base_trainer import BaseTrainer
from ai_interface.envs.discrete_ppo import SimplifiedSoccerEnv
from ai_interface.algorithms.discrete_ppo import DiscretePPOAgent
from networking.networker import Networker, TeamInfo


class DiscretePPOTrainer(BaseTrainer):
    """Trainer for discrete PPO algorithm with simplified action space."""
    
    def __init__(self, config: Dict[str, Any], log_dir: str = None, device="cpu"):
        super().__init__(config, log_dir, algorithm_name="discrete_ppo")
        self.networker = None
        self.env = None
        self.agent = None
        self.device = device
    
    def setup_environment(self) -> gym.Env:
        """Setup the simplified soccer environment."""
        # Load team configuration
        team_infos = self._load_team_config(self.config["team_config"])
        team_name = self.config.get("team_name") or team_infos[0].name
        
        # Setup networker
        self.networker = Networker(team_infos, self.config.get("env_mode", "sim-only"))
        
        # Create environment
        self.env = SimplifiedSoccerEnv(
            networker=self.networker,
            team_name=team_name,
            obs_dim=self.config.get("obs_dim", 18)
        )
        
        self.logger.info(f"Environment setup complete - Team: {team_name}, Obs dim: {self.config.get('obs_dim', 18)}")
        self.logger.info(f"Action space: Discrete(5) - APPROACH_BALL, SHOOT_GOAL, DRIBBLE_FORWARD, CLEAR_BALL, REPOSITION")
        return self.env
    
    def setup_model(self, env: gym.Env):
        """Setup the discrete PPO agent."""
        self.logger.info(f"Using device: {self.device}")

        self.agent = DiscretePPOAgent(
            obs_dim=self.config.get("obs_dim", 18),
            num_actions=5,  # 5 discrete actions
            device=self.device
        )
        
        # Load pre-trained model if specified
        if self.config.get("load_model"):
            self.logger.info(f"Loading model from {self.config['load_model']}")
            self.agent.load(self.config["load_model"])
        
        self.logger.info("Discrete PPO Agent setup complete")
    
    def train(self):
        """Execute the discrete PPO training loop."""
        if self.env is None:
            self.setup_environment()
        if self.agent is None:
            self.setup_model(self.env)
        
        episodes = self.config.get("episodes", 2000)
        max_steps = self.config.get("max_steps", 200)
        save_interval = self.config.get("save_interval", 100)
        
        from ai_interface.algorithms.discrete_ppo import BATCH_SIZE
        self.logger.info(
            f"Starting training for {episodes} episodes  "
            f"(PPO updates every {BATCH_SIZE} steps)..."
        )
        
        total_steps_since_update = 0
        n_updates = 0
        
        # Track action distribution for debugging
        action_counts = {i: 0 for i in range(5)}
        action_names = ["APPROACH_BALL", "SHOOT_GOAL", "DRIBBLE_FORWARD", "CLEAR_BALL", "REPOSITION"]

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_actions = []
            
            for step in range(max_steps):
                # Select action (returns integer 0-4)
                action = self.agent.select_action(state)
                episode_actions.append(action)
                action_counts[action] += 1
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store reward and mask in agent buffer
                self.agent.store_reward_mask(reward, 1.0 - float(done))
                
                episode_reward += reward
                state = next_state
                total_steps_since_update += 1
                
                if done:
                    break
            
            # Log episode with action distribution
            self.log_episode(episode + 1, episode_reward, step + 1)
            
            # Log action distribution every 10 episodes
            if (episode + 1) % 10 == 0:
                total_actions = sum(action_counts.values())
                action_dist = {
                    action_names[i]: f"{100*count/total_actions:.1f}%"
                    for i, count in action_counts.items()
                }
                self.logger.info(f"Action distribution (last 10 ep): {action_dist}")
                action_counts = {i: 0 for i in range(5)}  # Reset
            
            # Run PPO update when buffer is full
            if self.agent.batch_ready:
                losses = self.agent.update()
                n_updates += 1
                
                self.logger.info(
                    f"PPO update #{n_updates} after {total_steps_since_update} steps - "
                    f"Policy loss: {losses['policy_loss']:.4f}, "
                    f"Value loss: {losses['value_loss']:.4f}, "
                    f"Entropy: {losses['entropy']:.4f}"
                )
                total_steps_since_update = 0
                
                # Log losses
                self.log_metrics({
                    "update": n_updates,
                    "policy_loss": losses['policy_loss'],
                    "value_loss": losses['value_loss'],
                    "entropy": losses['entropy'],
                    "total_loss": losses['total_loss']
                })
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                save_name = Path(self.config.get("save_path", "models/discrete_ppo_policy.pth")).name
                checkpoint_path = self._get_checkpoint_path(save_name, episode + 1)
                self.save_model(str(checkpoint_path))
                self.logger.info(f"Model checkpoint saved to {checkpoint_path}")
        
        # Flush any remaining transitions
        if len(self.agent.memory) > 0:
            losses = self.agent.update(force=True)
            if losses:
                n_updates += 1
                self.logger.info(
                    f"Final PPO update #{n_updates} (flushed remaining buffer) - "
                    f"Policy loss: {losses['policy_loss']:.4f}, "
                    f"Value loss: {losses['value_loss']:.4f}"
                )
            else:
                self.logger.info("No remaining transitions to flush.")

        # Final save
        final_save_name = Path(self.config.get("save_path", "models/discrete_ppo_policy.pth")).name
        final_save_path = self.model_dir / final_save_name
        self.save_model(str(final_save_path))
        self.logger.info(f"Final model saved to {final_save_path}")

        # Plot training progress
        self.plot_training()
    
    def save_model(self, path: str):
        """Save the trained discrete PPO agent."""
        if self.agent is None:
            raise RuntimeError("Agent not initialized")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(save_path))
    
    def load_model(self, path: str):
        """Load a pre-trained discrete PPO agent."""
        if self.agent is None:
            self.setup_model(self.env)
        self.agent.load(path)
    
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
                self.logger.error(f"Error during networker shutdown: {e}")
    
    def _load_team_config(self, file_path: str) -> list[TeamInfo]:
        """Load team configuration from JSON file."""
        with open(file_path, 'r') as f:
            config = json.load(f)["teams"]
        
        if len(config) != 2:
            raise ValueError("Team configuration must contain exactly two teams.")
        
        team1_info, team2_info = config
        if len(team1_info) != 3 or len(team2_info) != 3:
            raise ValueError("Each team configuration must have name, n_players, and goalie_id.")
        
        return [TeamInfo(*team1_info), TeamInfo(*team2_info)]
    



# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of DiscretePPOTrainer.
    
    To run:
        python -m ai_interface.trainers.discrete_ppo_trainer
    """
    import torch
    
    # Configuration
    config = {
        "team_config": "configs/team_config.json",
        "team_name": "YourTeam",
        "env_mode": "sim-only",
        "obs_dim": 18,
        "episodes": 2000,
        "max_steps": 200,
        "save_interval": 100,
        "save_path": "models/discrete_ppo/policy.pth"
    }
    
    # Create trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = DiscretePPOTrainer(
        config=config,
        log_dir="train_logs/discrete_ppo",
        device=device
    )
    
    # Train
    try:
        trainer.train()
    finally:
        trainer.cleanup()
    
    print(f"\nTraining complete! Logs saved to: {trainer.run_dir}")
    print(f"Final model saved to: {config['save_path']}")
    print(f"Reward plot: {trainer.run_dir}/reward_plot.png")
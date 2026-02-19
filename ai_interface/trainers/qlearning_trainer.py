"""
Q-Learning Trainer for Minimal Soccer Environment.

Trains agent using raw simulator commands:
- kick 100 0
- dash 100 0  
- turn 100
- turn -100
- dash 50 90
"""

import gymnasium as gym
import torch
from typing import Dict, Any
from pathlib import Path
import json

from .base_trainer import BaseTrainer
from ai_interface.envs.minimal_soccer_env import MinimalSoccerEnv
from ai_interface.algorithms.q_learning import QLearningAgent
from networking.networker import Networker, TeamInfo


class QLearningTrainer(BaseTrainer):
    """Trainer for Q-Learning with minimal discrete actions."""
    
    def __init__(self, config: Dict[str, Any], log_dir: str = None, device=None):
        super().__init__(config, log_dir)
        self.networker = None
        self.env = None
        self.agent = None
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def setup_environment(self) -> gym.Env:
        """Setup minimal soccer environment."""
        team_infos = self._load_team_config(self.config["team_config"])
        team_name = self.config.get("team_name") or team_infos[0].name
        
        self.networker = Networker(team_infos, self.config.get("env_mode", "sim-only"))
        
        self.env = MinimalSoccerEnv(
            networker=self.networker,
            team_name=team_name,
            obs_dim=self.config.get("obs_dim", 8)
        )
        
        self.logger.info(f"Minimal environment setup - Team: {team_name}, Obs dim: 8")
        self.logger.info(f"Actions: KICK_FORWARD, DASH_FORWARD, TURN_RIGHT, TURN_LEFT, DASH_SIDE")
        return self.env
    
    def setup_model(self, env: gym.Env):
        """Setup Q-Learning agent."""
        self.logger.info(f"Using device: {self.device}")
        
        self.agent = QLearningAgent(
            obs_dim=self.config.get("obs_dim", 8),
            num_actions=5,
            device=self.device,
            lr=self.config.get("lr", 1e-3),
            gamma=self.config.get("gamma", 0.99),
            epsilon_start=self.config.get("epsilon_start", 1.0),
            epsilon_end=self.config.get("epsilon_end", 0.01),
            epsilon_decay=self.config.get("epsilon_decay", 0.995),
            buffer_size=self.config.get("buffer_size", 10000),
            batch_size=self.config.get("batch_size", 64)
        )
        
        if self.config.get("load_model"):
            self.logger.info(f"Loading model from {self.config['load_model']}")
            self.agent.load(self.config["load_model"])
        
        self.logger.info(f"Q-Learning Agent setup complete")
        self.logger.info(f"Epsilon: {self.agent.epsilon:.3f} (will decay to {self.agent.epsilon_end})")
    
    def train(self):
        """Execute Q-Learning training loop."""
        if self.env is None:
            self.setup_environment()
        if self.agent is None:
            self.setup_model(self.env)
        
        episodes = self.config.get("episodes", 2000)
        max_steps = self.config.get("max_steps", 200)
        save_interval = self.config.get("save_interval", 100)
        update_frequency = self.config.get("update_frequency", 4)  # Update every N steps
        
        self.logger.info(
            f"Starting Q-Learning training for {episodes} episodes  "
            f"(Updates every {update_frequency} steps)..."
        )
        
        # Track action distribution
        action_counts = {i: 0 for i in range(5)}
        action_names = self.env.action_names
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_actions = []
            
            for step in range(max_steps):
                # Select action (epsilon-greedy)
                action = self.agent.select_action(state)
                episode_actions.append(action)
                action_counts[action] += 1
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Update Q-network
                if step % update_frequency == 0:
                    losses = self.agent.update()
                    if losses and (episode + 1) % 10 == 0 and step % 40 == 0:
                        # Log updates occasionally
                        self.log_metrics({
                            "update": self.agent.updates,
                            "loss": losses['loss'],
                            "epsilon": losses['epsilon'],
                            "q_mean": losses['q_mean']
                        })
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Log episode
            goal_str = " GOAL!" if info.get("goal_scored") else ""
            self.logger.info(
                f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                f"Length={step + 1}, Epsilon={self.agent.epsilon:.3f}{goal_str}"
            )
            self.log_episode(episode + 1, episode_reward, step + 1)
            
            # Log action distribution every 10 episodes
            if (episode + 1) % 10 == 0:
                total_actions = sum(action_counts.values())
                if total_actions > 0:
                    action_dist = {
                        action_names[i]: f"{100*count/total_actions:.1f}%"
                        for i, count in action_counts.items()
                    }
                    self.logger.info(f"Action distribution: {action_dist}")
                action_counts = {i: 0 for i in range(5)}
            
            # Save periodically
            if (episode + 1) % save_interval == 0:
                save_path = self.config.get("save_path", "models/q_learning.pth")
                checkpoint_path = self._get_checkpoint_path(save_path, episode + 1)
                self.save_model(str(checkpoint_path))
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Final save
        final_save_path = self.config.get("save_path", "models/q_learning.pth")
        self.save_model(final_save_path)
        self.logger.info(f"Final model saved to {final_save_path}")
        
        # Plot
        self.plot_training()
    
    def save_model(self, path: str):
        """Save Q-Learning agent."""
        if self.agent is None:
            raise RuntimeError("Agent not initialized")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(save_path))
    
    def load_model(self, path: str):
        """Load Q-Learning agent."""
        if self.agent is None:
            self.setup_model(self.env)
        self.agent.load(path)
    
    def cleanup(self):
        """Cleanup resources."""
        super().cleanup()
        if self.networker:
            try:
                if hasattr(self.networker, "shutdown"):
                    self.networker.shutdown()
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")
    
    def _load_team_config(self, file_path: str) -> list[TeamInfo]:
        """Load team configuration from JSON."""
        with open(file_path, 'r') as f:
            config = json.load(f)["teams"]
        
        if len(config) != 2:
            raise ValueError("Team configuration must contain exactly two teams.")
        
        team1_info, team2_info = config
        return [TeamInfo(*team1_info), TeamInfo(*team2_info)]
    
    def _get_checkpoint_path(self, base_path: str, episode: int) -> Path:
        """Generate checkpoint path with episode number."""
        save_path = Path(base_path)
        return save_path.parent / f"{save_path.stem}_ep{episode}{save_path.suffix}"


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of QLearningTrainer.
    """
    import torch
    
    config = {
        "team_config": "configs/team_config.json",
        "team_name": "YourTeam",
        "env_mode": "sim-only",
        "obs_dim": 8,
        "episodes": 2000,
        "max_steps": 200,
        "save_interval": 100,
        "save_path": "models/q_learning/agent.pth",
        
        # Q-Learning hyperparameters
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "buffer_size": 10000,
        "batch_size": 64,
        "update_frequency": 4
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = QLearningTrainer(
        config=config,
        log_dir="train_logs/q_learning",
        device=device
    )
    
    try:
        trainer.train()
    finally:
        trainer.cleanup()
    
    print(f"\nQ-Learning training complete!")
    print(f"Logs: {trainer.run_dir}")
    print(f"Model: {config['save_path']}")
"""
Hierarchical PPO Trainer implementation.

This trainer handles hierarchical PPO training using the custom hier_ppo algorithm
with the SoccerEnv environment that supports hierarchical actions.
"""
import gymnasium as gym
import torch
from typing import Dict, Any
from pathlib import Path

from .base_trainer import BaseTrainer
from ai_interface.envs.ppo_env import SoccerEnv
from ai_interface.algorithms.hier_ppo import PPOAgent
from networking.networker import Networker, TeamInfo


class HierarchicalPPOTrainer(BaseTrainer):
    """Trainer for hierarchical PPO algorithm."""
    
    def __init__(self, config: Dict[str, Any], log_dir: str = None, device=None):
        super().__init__(config, log_dir)
        self.networker = None
        self.env = None
        self.agent = None
        self.device = device
    
    def setup_environment(self) -> gym.Env:
        """Setup the hierarchical soccer environment."""
        # Load team configuration
        team_infos = self._load_team_config(self.config["team_config"])
        team_name = self.config.get("team_name") or team_infos[0].name
        
        # Setup networker
        self.networker = Networker(team_infos, self.config.get("env_mode", "sim-only"))
        
        # Create environment
        self.env = SoccerEnv(
            networker=self.networker,
            team_name=team_name,
            obs_dim=self.config.get("obs_dim", 18)  # Updated to 18 for new obs space
        )
        
        self.logger.info(f"Environment setup complete - Team: {team_name}, Obs dim: {self.config.get('obs_dim', 18)}")
        return self.env
    
    def setup_model(self, env: gym.Env):
        """Setup the hierarchical PPO agent."""
        # Determine device (use trainer.device if provided)
        self.logger.info(f"Using device: {self.device}")

        self.agent = PPOAgent(obs_dim=self.config.get("obs_dim", 18), device=self.device)  # Updated to 18
        
        # Load pre-trained model if specified
        if self.config.get("load_model"):
            self.logger.info(f"Loading model from {self.config['load_model']}")
            self.agent.load(self.config["load_model"])
        
        self.logger.info("PPO Agent setup complete")
    
    def train(self):
        """Execute the hierarchical PPO training loop."""
        if self.env is None:
            self.setup_environment()
        if self.agent is None:
            self.setup_model(self.env)
        
        episodes = self.config.get("episodes", 1000)
        max_steps = self.config.get("max_steps", 200)
        save_interval = self.config.get("save_interval", 100)
        
        from ai_interface.algorithms.hier_ppo import BATCH_SIZE
        self.logger.info(
            f"Starting training for {episodes} episodes  "
            f"(PPO updates every {BATCH_SIZE} steps)..."
        )
        
        total_steps_since_update = 0
        n_updates = 0

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Select action using agent (stores state/action in agent memory)
                action = self.agent.select_action(state)
                
                # Take step in environment
                next_state, reward, done, _ = self.env.step(action)
                
                # Store reward and mask directly in agent buffer
                self.agent.store_reward_mask(reward, 1.0 - float(done))
                
                episode_reward += reward
                state = next_state
                total_steps_since_update += 1
                
                if done:
                    break
            
            # Log episode (avg reward per 10 ep is logged automatically)
            self.log_episode(episode + 1, episode_reward, step + 1)
            
            # Run PPO update only after collecting enough transitions
            if self.agent.batch_ready:
                self.agent.update()
                n_updates += 1
                self.logger.info(
                    f"PPO update #{n_updates} after {total_steps_since_update} steps"
                )
                total_steps_since_update = 0
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                save_path = self.config.get("save_path", "models/hier_ppo_policy.pth")
                checkpoint_path = self._get_checkpoint_path(save_path, episode + 1)
                self.save_model(str(checkpoint_path))
                self.logger.info(f"Model checkpoint saved to {checkpoint_path}")
        
        # Flush any remaining transitions
        if len(self.agent.memory) > 0:
            self.agent.update()
            n_updates += 1
            self.logger.info(f"Final PPO update #{n_updates} (flushed remaining buffer)")

        # Final save
        final_save_path = self.config.get("save_path", "models/hier_ppo_policy.pth")
        self.save_model(final_save_path)
        self.logger.info(f"Final model saved to {final_save_path}")

        # ---- Plot average reward per 10 episodes ----
        self.plot_training()
    
    def save_model(self, path: str):
        """Save the trained PPO agent."""
        if self.agent is None:
            raise RuntimeError("Agent not initialized")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(save_path))
    
    def load_model(self, path: str):
        """Load a pre-trained PPO agent."""
        if self.agent is None:
            self.setup_model(self.env)
        self.agent.load(path)
    
    def cleanup(self):
        """Cleanup resources after training."""
        super().cleanup()
        if self.networker:
            try:
                # Prefer the high-level shutdown API; fall back to older disconnect methods
                if hasattr(self.networker, "shutdown") and callable(self.networker.shutdown):
                    self.networker.shutdown()
                elif hasattr(self.networker, "disconnect_from_sim") and callable(self.networker.disconnect_from_sim):
                    self.networker.disconnect_from_sim()
                else:
                    # Try commander/game watcher shutdown hooks where available
                    try:
                        if hasattr(self.networker, "commander") and hasattr(self.networker.commander, "disconnect_from_sim"):
                            self.networker.commander.disconnect_from_sim()
                    except Exception:
                        pass
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
        
        # TeamInfo requires name, n_players, and goalie_id
        return [TeamInfo(*team1_info), TeamInfo(*team2_info)]
    
    def _get_checkpoint_path(self, base_path: str, episode: int) -> Path:
        """Generate checkpoint path with episode number."""
        save_path = Path(base_path)
        return save_path.parent / f"{save_path.stem}_ep{episode}{save_path.suffix}"
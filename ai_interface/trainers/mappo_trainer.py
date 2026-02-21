"""
MAPPO Trainer for multi-agent soccer teams.

Trains teams of robots to coordinate using Multi-Agent PPO.
"""

import gymnasium as gym
import torch
from typing import Dict, Any
from pathlib import Path
import json

from .base_trainer import BaseTrainer
from ai_interface.envs.mappo_env import MultiAgentSoccerEnv
from ai_interface.algorithms.mappo import MAPPOAgent
from networking.networker import Networker, TeamInfo


class MAPPOTrainer(BaseTrainer):
    """Trainer for multi-agent PPO (MAPPO) algorithm."""
    
    def __init__(self, config: Dict[str, Any], log_dir: str = None, device=None):
        super().__init__(config, log_dir, algorithm_name="mappo")
        self.networker = None
        self.env = None
        self.agent = None
        self.num_agents = config.get("num_agents", 3)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def setup_environment(self) -> gym.Env:
        """Setup the multi-agent soccer environment."""
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
        
        self.env = MultiAgentSoccerEnv(
            networker=self.networker,
            team_name=team_name,
            num_agents=self.num_agents,
            obs_dim=self.config.get("obs_dim", 25)
        )
        
        self.logger.info(
            "Multi-agent environment setup - Team: %s, Agents: %d, Endpoint: %s:%d/%d",
            team_name, self.num_agents, sim_host, sim_player_port, sim_trainer_port
        )
        self.logger.info(f"Action space: 6 discrete actions per agent (APPROACH, SHOOT, PASS, DRIBBLE, CLEAR, REPOSITION)")
        return self.env
    
    def setup_model(self, env: gym.Env):
        """Setup the MAPPO agent."""
        self.logger.info(f"Using device: {self.device}")
        
        self.agent = MAPPOAgent(
            num_agents=self.num_agents,
            obs_dim=self.config.get("obs_dim", 25),
            num_actions=6,  # 6 actions including PASS
            device=self.device
        )
        
        if self.config.get("load_model"):
            self.logger.info(f"Loading model from {self.config['load_model']}")
            self.agent.load(self.config["load_model"])
        
        self.logger.info(f"MAPPO Agent setup complete - {self.num_agents} actors, 1 centralized critic")
    
    def train(self):
        """Execute MAPPO training loop."""
        if self.env is None:
            self.setup_environment()
        if self.agent is None:
            self.setup_model(self.env)
        
        episodes = self.config.get("episodes", 2000)
        max_steps = self.config.get("max_steps", 400)
        save_interval = self.config.get("save_interval", 100)
        
        from ai_interface.algorithms.mappo import BATCH_SIZE
        self.logger.info(
            f"Starting MAPPO training for {episodes} episodes  "
            f"(Updates every {BATCH_SIZE} steps)..."
        )
        
        total_steps_since_update = 0
        n_updates = 0
        
        # Track per-agent action statistics
        action_counts = {i: {j: 0 for j in range(6)} for i in range(self.num_agents)}
        action_names = ["APPROACH", "SHOOT", "PASS", "DRIBBLE", "CLEAR", "REPOSITION"]
        
        for episode in range(episodes):
            observations = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # All agents select actions
                actions = self.agent.select_actions(observations)
                
                # Track action distribution
                for agent_idx, action in enumerate(actions):
                    action_counts[agent_idx][action] += 1
                
                # Environment step (all agents act simultaneously)
                next_observations, reward, done, info = self.env.step(actions)
                
                # Store shared team reward
                self.agent.store_reward_mask(reward, 1.0 - float(done))
                
                episode_reward += reward
                observations = next_observations
                total_steps_since_update += 1
                
                if done:
                    break
            
            # Log episode
            goal_str = " GOAL!" if info.get("goal_scored") else ""
            self.logger.info(
                f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                f"Length={step + 1}{goal_str}"
            )
            self.log_episode(episode + 1, episode_reward, step + 1)
            
            # Log action distribution every 10 episodes
            if (episode + 1) % 10 == 0:
                for agent_idx in range(self.num_agents):
                    total = sum(action_counts[agent_idx].values())
                    if total > 0:
                        dist = {
                            action_names[j]: f"{100*count/total:.1f}%"
                            for j, count in action_counts[agent_idx].items()
                        }
                        self.logger.info(f"Agent {agent_idx} actions: {dist}")
                
                # Reset counts
                action_counts = {i: {j: 0 for j in range(6)} for i in range(self.num_agents)}
            
            # MAPPO update
            if self.agent.batch_ready:
                losses = self.agent.update()
                n_updates += 1
                
                self.logger.info(
                    f"MAPPO update #{n_updates} after {total_steps_since_update} steps - "
                    f"Actor: {losses['actor_loss']:.4f}, "
                    f"Critic: {losses['critic_loss']:.4f}, "
                    f"Entropy: {losses['entropy']:.4f}"
                )
                total_steps_since_update = 0
                
                self.log_metrics({
                    "update": n_updates,
                    "actor_loss": losses['actor_loss'],
                    "critic_loss": losses['critic_loss'],
                    "entropy": losses['entropy']
                })
            
            # Save periodically
            if (episode + 1) % save_interval == 0:
                save_name = Path(self.config.get("save_path", "models/mappo_team.pth")).name
                checkpoint_path = self._get_checkpoint_path(save_name, episode + 1)
                self.save_model(str(checkpoint_path))
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Final update and save
        if len(self.agent.memory) > 0:
            losses = self.agent.update()
            n_updates += 1
            self.logger.info(f"Final update #{n_updates}")
        
        final_save_name = Path(self.config.get("save_path", "models/mappo_team.pth")).name
        final_save_path = self.model_dir / final_save_name
        self.save_model(str(final_save_path))
        self.logger.info(f"Final model saved to {final_save_path}")
        
        # Plot training
        self.plot_training()
    
    def save_model(self, path: str):
        """Save the MAPPO agents."""
        if self.agent is None:
            raise RuntimeError("Agent not initialized")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(save_path))
    
    def load_model(self, path: str):
        """Load pre-trained MAPPO agents."""
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
        if len(team1_info) != 3 or len(team2_info) != 3:
            raise ValueError("Each team must have name, n_players, and goalie_id.")
        
        return [TeamInfo(*team1_info), TeamInfo(*team2_info)]
    



# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of MAPPOTrainer for 3v3 soccer.
    """
    import torch
    
    config = {
        "team_config": "configs/team_config.json",
        "team_name": "YourTeam",
        "env_mode": "sim-only",
        "num_agents": 3,  # 3v3 soccer
        "obs_dim": 25,
        "episodes": 3000,  # More episodes for multi-agent
        "max_steps": 400,  # Longer episodes
        "save_interval": 100,
        "save_path": "models/mappo/team.pth"
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = MAPPOTrainer(
        config=config,
        log_dir="train_logs/mappo",
        device=device
    )
    
    try:
        trainer.train()
    finally:
        trainer.cleanup()
    
    print(f"\nTraining complete!")
    print(f"Logs: {trainer.run_dir}")
    print(f"Model: {config['save_path']}")

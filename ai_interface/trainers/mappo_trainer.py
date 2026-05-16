"""
MAPPO Trainer for multi-agent soccer teams.

Trains teams of robots to coordinate using Multi-Agent PPO.
"""

import gymnasium as gym
import torch
from typing import Dict, Any
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

from .base_trainer import BaseTrainer
from ai_interface.envs.mappo_env import MultiAgentSoccerEnv
from ai_interface.algorithms.mappo import MAPPOAgent
from networking.networker import Networker, TeamInfo


class MAPPOTrainer(BaseTrainer):
    """Trainer for multi-agent PPO (MAPPO) algorithm."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        log_dir: str = None,
        device=None,
        algorithm_name: str = "mappo",
    ):
        super().__init__(config, log_dir, algorithm_name=algorithm_name)
        self.networker = None
        self.networkers = []
        self.env = None
        self.envs = []
        self.agent = None
        self.num_agents = config.get("num_agents", 3)
        self.num_envs = 1
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def setup_environment(self) -> gym.Env:
        """Setup the multi-agent soccer environment."""
        team_infos = self._load_team_config(self.config["team_config"])
        team_name = self.config.get("team_name") or team_infos[0].name

        self.num_envs = max(1, int(self.config.get("num_envs", 1)))
        self.networkers = []
        self.envs = []

        for env_idx in range(self.num_envs):
            sim_host, sim_player_port, sim_trainer_port = self._sim_endpoint_for_env(env_idx)
            networker = Networker(
                team_infos,
                self.config.get("env_mode", "sim-only"),
                sim_host=sim_host,
                sim_player_port=sim_player_port,
                sim_trainer_port=sim_trainer_port,
            )
            env = MultiAgentSoccerEnv(
                networker=networker,
                team_name=team_name,
                num_agents=self.num_agents,
                obs_dim=self.config.get("obs_dim", 25)
            )
            self.networkers.append(networker)
            self.envs.append(env)
            self.logger.info(
                "Env %d connected to %s:%d/%d",
                env_idx,
                sim_host,
                sim_player_port,
                sim_trainer_port,
            )

        self.networker = self.networkers[0]
        self.env = self.envs[0]
        self.logger.info(
            "Multi-agent environment setup - Team: %s, Agents: %d, Parallel envs: %d",
            team_name, self.num_agents, self.num_envs
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
        
        def _maybe_update():
            nonlocal n_updates, total_steps_since_update
            if not self.agent.batch_ready:
                return
            losses = self.agent.update()
            if not losses:
                return
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

        def _log_action_distribution():
            nonlocal action_counts
            for agent_idx in range(self.num_agents):
                total = sum(action_counts[agent_idx].values())
                if total > 0:
                    dist = {
                        action_names[j]: f"{100*count/total:.1f}%"
                        for j, count in action_counts[agent_idx].items()
                    }
                    self.logger.info(f"Agent {agent_idx} actions: {dist}")
            action_counts = {i: {j: 0 for j in range(6)} for i in range(self.num_agents)}

        def _save_checkpoint(episode_num: int):
            save_name = Path(self.config.get("save_path", "models/mappo_team.pth")).name
            checkpoint_path = self._get_checkpoint_path(save_name, episode_num)
            self.save_model(str(checkpoint_path))
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        if self.num_envs <= 1:
            for episode in range(episodes):
                observations = self.env.reset()
                episode_reward = 0
                info = {}

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

                    _maybe_update()

                # Log episode
                goal_str = " GOAL!" if info.get("goal_scored") else ""
                self.logger.info(
                    f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                    f"Length={step + 1}{goal_str}"
                )
                self.log_episode(episode + 1, episode_reward, step + 1)

                # Log action distribution every 10 episodes
                if (episode + 1) % 10 == 0:
                    _log_action_distribution()

                _maybe_update()

                # Save periodically
                if (episode + 1) % save_interval == 0:
                    _save_checkpoint(episode + 1)
        else:
            self.logger.info(f"Parallel rollout enabled with {self.num_envs} simulator instances")
            episode_counter = 0

            with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
                while episode_counter < episodes:
                    batch_envs = min(self.num_envs, episodes - episode_counter)
                    observations = [self.envs[i].reset() for i in range(batch_envs)]
                    episode_rewards = [0.0 for _ in range(batch_envs)]
                    episode_lengths = [0 for _ in range(batch_envs)]
                    done_flags = [False for _ in range(batch_envs)]
                    infos = [{} for _ in range(batch_envs)]

                    for _ in range(max_steps):
                        active_envs = [i for i in range(batch_envs) if not done_flags[i]]
                        if not active_envs:
                            break

                        actions_by_env = [None for _ in range(batch_envs)]
                        for env_idx in active_envs:
                            actions = self.agent.select_actions(observations[env_idx])
                            actions_by_env[env_idx] = actions
                            for agent_idx, action in enumerate(actions):
                                action_counts[agent_idx][action] += 1

                        futures = {
                            env_idx: executor.submit(self.envs[env_idx].step, actions_by_env[env_idx])
                            for env_idx in active_envs
                        }

                        for env_idx in active_envs:
                            next_observations, reward, done, info = futures[env_idx].result()
                            self.agent.store_reward_mask(reward, 1.0 - float(done))
                            episode_rewards[env_idx] += reward
                            episode_lengths[env_idx] += 1
                            observations[env_idx] = next_observations
                            done_flags[env_idx] = bool(done)
                            infos[env_idx] = info
                            total_steps_since_update += 1

                        _maybe_update()

                    for env_idx in range(batch_envs):
                        episode_counter += 1
                        goal_str = " GOAL!" if infos[env_idx].get("goal_scored") else ""
                        length = max(1, episode_lengths[env_idx])
                        self.logger.info(
                            f"Episode {episode_counter}: Reward={episode_rewards[env_idx]:.2f}, "
                            f"Length={length}{goal_str}"
                        )
                        self.log_episode(episode_counter, episode_rewards[env_idx], length)

                        if episode_counter % 10 == 0:
                            _log_action_distribution()

                        if episode_counter % save_interval == 0:
                            _save_checkpoint(episode_counter)
        
        # Final update and save
        if len(self.agent.memory) > 0:
            losses = self.agent.update(force=True)
            if losses:
                n_updates += 1
                self.logger.info(f"Final update #{n_updates}")
            else:
                self.logger.info("No remaining transitions to flush.")
        
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
        networkers = self.networkers or ([self.networker] if self.networker else [])
        seen = set()
        for networker in networkers:
            if networker is None:
                continue
            key = id(networker)
            if key in seen:
                continue
            seen.add(key)
            try:
                if hasattr(networker, "shutdown"):
                    networker.shutdown()
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

"""
Hierarchical PPO Trainer implementation.

This trainer handles hierarchical PPO training using the custom hier_ppo algorithm
with the SoccerEnv environment that supports hierarchical actions.
"""
import gymnasium as gym
import torch
from typing import Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .base_trainer import BaseTrainer
from ai_interface.envs.ppo_env import SoccerEnv
from ai_interface.envs.curriculum_ppo import CurriculumSoccerEnv
from ai_interface.algorithms.hier_ppo import PPOAgent
from networking.networker import Networker, TeamInfo


class HierarchicalPPOTrainer(BaseTrainer):
    """Trainer for hierarchical PPO algorithm."""
    
    def __init__(self, config: Dict[str, Any], log_dir: str = None, device=None):
        super().__init__(config, log_dir, algorithm_name="hier_ppo")
        self.networker = None
        self.networkers = []
        self.env = None
        self.envs = []
        self.num_envs = 1
        self.agent = None
        self.device = device
    
    def setup_environment(self) -> gym.Env:
        """Setup the hierarchical soccer environment."""
        # Load team configuration
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
            env = CurriculumSoccerEnv(
                networker=networker,
                team_name=team_name,
                obs_dim=self.config.get("obs_dim", 18),
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
            "Environment setup complete - Team: %s, Obs dim: %s, Parallel envs: %d",
            team_name,
            self.config.get("obs_dim", 18),
            self.num_envs,
        )
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

        if self.num_envs <= 1:
            for episode in range(episodes):
                state = self.env.reset()
                episode_reward = 0

                for step in range(max_steps):
                    action = self.agent.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.agent.store_reward_mask(reward, 1.0 - float(done))

                    episode_reward += reward
                    state = next_state
                    total_steps_since_update += 1

                    if done:
                        break

                self.log_episode(episode + 1, episode_reward, step + 1)

                if self.agent.batch_ready:
                    self.agent.update()
                    n_updates += 1
                    self.logger.info(
                        f"PPO update #{n_updates} after {total_steps_since_update} steps"
                    )
                    total_steps_since_update = 0

                if (episode + 1) % save_interval == 0:
                    save_name = Path(self.config.get("save_path", "models/hier_ppo_policy.pth")).name
                    checkpoint_path = self._get_checkpoint_path(save_name, episode + 1)
                    self.save_model(str(checkpoint_path))
                    self.logger.info(f"Model checkpoint saved to {checkpoint_path}")
        else:
            self.logger.info(f"Parallel rollout enabled with {self.num_envs} simulator instances")
            episode_counter = 0
            with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
                while episode_counter < episodes:
                    batch_envs = min(self.num_envs, episodes - episode_counter)
                    states = [self.envs[i].reset() for i in range(batch_envs)]
                    episode_rewards = [0.0 for _ in range(batch_envs)]
                    episode_lengths = [0 for _ in range(batch_envs)]
                    done_flags = [False for _ in range(batch_envs)]

                    for _ in range(max_steps):
                        active_envs = [i for i in range(batch_envs) if not done_flags[i]]
                        if not active_envs:
                            break

                        actions = [None for _ in range(batch_envs)]
                        transitions = [None for _ in range(batch_envs)]
                        for env_idx in active_envs:
                            action, transition = self.agent.sample_action(states[env_idx])
                            actions[env_idx] = action
                            transitions[env_idx] = transition

                        futures = {
                            env_idx: executor.submit(self.envs[env_idx].step, actions[env_idx])
                            for env_idx in active_envs
                        }

                        for env_idx in active_envs:
                            next_state, reward, done, _ = futures[env_idx].result()
                            self.agent.append_transition(
                                transitions[env_idx], reward, 1.0 - float(done)
                            )
                            episode_rewards[env_idx] += reward
                            episode_lengths[env_idx] += 1
                            states[env_idx] = next_state
                            done_flags[env_idx] = bool(done)
                            total_steps_since_update += 1

                        if self.agent.batch_ready:
                            self.agent.update()
                            n_updates += 1
                            self.logger.info(
                                f"PPO update #{n_updates} after {total_steps_since_update} steps"
                            )
                            total_steps_since_update = 0

                    for env_idx in range(batch_envs):
                        episode_counter += 1
                        self.log_episode(
                            episode_counter,
                            episode_rewards[env_idx],
                            max(1, episode_lengths[env_idx]),
                        )

                        if episode_counter % save_interval == 0:
                            save_name = Path(self.config.get("save_path", "models/hier_ppo_policy.pth")).name
                            checkpoint_path = self._get_checkpoint_path(save_name, episode_counter)
                            self.save_model(str(checkpoint_path))
                            self.logger.info(f"Model checkpoint saved to {checkpoint_path}")
        
        # Flush any remaining transitions
        if len(self.agent.memory) > 0:
            self.agent.update()
            n_updates += 1
            self.logger.info(f"Final PPO update #{n_updates} (flushed remaining buffer)")

        # Final save
        final_save_name = Path(self.config.get("save_path", "models/hier_ppo_policy.pth")).name
        final_save_path = self.model_dir / final_save_name
        self.save_model(str(final_save_path))
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
                if hasattr(networker, "shutdown") and callable(networker.shutdown):
                    networker.shutdown()
                elif hasattr(networker, "disconnect_from_sim") and callable(networker.disconnect_from_sim):
                    networker.disconnect_from_sim()
                else:
                    try:
                        if hasattr(networker, "commander") and hasattr(networker.commander, "disconnect_from_sim"):
                            networker.commander.disconnect_from_sim()
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
    

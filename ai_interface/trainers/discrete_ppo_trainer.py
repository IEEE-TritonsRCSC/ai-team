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
from concurrent.futures import ThreadPoolExecutor

from .base_trainer import BaseTrainer
from ai_interface.envs.discrete_ppo import SimplifiedSoccerEnv
from ai_interface.envs.discrete_curriculum_ppo import DiscreteCurriculumEnv
from ai_interface.algorithms.discrete_ppo import DiscretePPOAgent
from networking.networker import Networker, TeamInfo


class DiscretePPOTrainer(BaseTrainer):
    """Trainer for discrete PPO algorithm with simplified action space."""
    
    def __init__(self, config: Dict[str, Any], log_dir: str = None, device="cpu"):
        super().__init__(config, log_dir, algorithm_name="discrete_ppo")
        self.networker = None
        self.networkers = []
        self.env = None
        self.envs = []
        self.num_envs = 1
        self.agent = None
        self.device = device
        # Track the source checkpoint so we never overwrite it
        self._source_checkpoint: str | None = config.get("load_model")
    
    def setup_environment(self) -> gym.Env:
        """Setup the soccer environment (standard or curriculum)."""
        # Load team configuration
        team_infos = self._load_team_config(self.config["team_config"])
        team_name = self.config.get("team_name") or team_infos[0].name

        self.num_envs = max(1, int(self.config.get("num_envs", 1)))
        self.networkers = []
        self.envs = []

        use_curriculum = self.config.get("curriculum", False)
        obs_dim = self.config.get("obs_dim", 18)

        for env_idx in range(self.num_envs):
            sim_host, sim_player_port, sim_trainer_port = self._sim_endpoint_for_env(env_idx)
            networker = Networker(
                team_infos,
                self.config.get("env_mode", "sim-only"),
                sim_host=sim_host,
                sim_player_port=sim_player_port,
                sim_trainer_port=sim_trainer_port,
            )

            if use_curriculum:
                start_phase = self.config.get("start_phase", 0)
                env = DiscreteCurriculumEnv(
                    networker=networker,
                    team_name=team_name,
                    obs_dim=obs_dim,
                    start_phase=start_phase,
                )
            else:
                env = SimplifiedSoccerEnv(
                    networker=networker,
                    team_name=team_name,
                    obs_dim=obs_dim,
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
        if use_curriculum:
            start_phase = self.config.get("start_phase", 0)
            self.logger.info(
                f"Curriculum environment setup - Team: {team_name}, "
                f"Obs dim: {obs_dim}, Start phase: {start_phase}, "
                f"Parallel envs: {self.num_envs}"
            )
        else:
            self.logger.info(
                f"Standard environment setup - Team: {team_name}, Obs dim: {obs_dim}, "
                f"Parallel envs: {self.num_envs}"
            )

        self.logger.info("Action space: Discrete(5) - APPROACH_BALL, SHOOT_GOAL, DRIBBLE_FORWARD, CLEAR_BALL, REPOSITION")
        return self.env
    
    def setup_model(self, env: gym.Env):
        """Setup the discrete PPO agent, optionally loading a checkpoint."""
        self.logger.info(f"Using device: {self.device}")

        self.agent = DiscretePPOAgent(
            obs_dim=self.config.get("obs_dim", 18),
            num_actions=5,  # 5 discrete actions
            device=self.device
        )
        
        # Load pre-trained model if specified
        load_path = self.config.get("load_model")
        if load_path:
            self.logger.info(f"Loading checkpoint from: {load_path}")
            self.agent.load(load_path)
            self.logger.info(
                f"Checkpoint loaded successfully. Original file will NOT be "
                f"modified — all new saves go to: {self.model_dir}/"
            )
        
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

        if self.num_envs <= 1:
            for episode in range(episodes):
                state = self.env.reset()
                episode_reward = 0

                for step in range(max_steps):
                    action = self.agent.select_action(state)
                    action_counts[action] += 1

                    next_state, reward, done, info = self.env.step(action)
                    self.agent.store_reward_mask(reward, 1.0 - float(done))

                    episode_reward += reward
                    state = next_state
                    total_steps_since_update += 1

                    if done:
                        break

                self.log_episode(episode + 1, episode_reward, step + 1)

                if (episode + 1) % 10 == 0:
                    total_actions = sum(action_counts.values())
                    if total_actions > 0:
                        action_dist = {
                            action_names[i]: f"{100*count/total_actions:.1f}%"
                            for i, count in action_counts.items()
                        }
                        self.logger.info(f"Action distribution (last 10 ep): {action_dist}")
                    action_counts = {i: 0 for i in range(5)}

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

                    self.log_metrics({
                        "update": n_updates,
                        "policy_loss": losses['policy_loss'],
                        "value_loss": losses['value_loss'],
                        "entropy": losses['entropy'],
                        "total_loss": losses['total_loss']
                    })

                if (episode + 1) % save_interval == 0:
                    save_name = Path(self.config.get("save_path", "models/discrete_ppo_policy.pth")).name
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
                            action_counts[action] += 1

                        futures = {
                            env_idx: executor.submit(self.envs[env_idx].step, actions[env_idx])
                            for env_idx in active_envs
                        }

                        for env_idx in active_envs:
                            next_state, reward, done, info = futures[env_idx].result()
                            self.agent.append_transition(
                                transitions[env_idx], reward, 1.0 - float(done)
                            )
                            episode_rewards[env_idx] += reward
                            episode_lengths[env_idx] += 1
                            states[env_idx] = next_state
                            done_flags[env_idx] = bool(done)
                            total_steps_since_update += 1

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

                            self.log_metrics({
                                "update": n_updates,
                                "policy_loss": losses['policy_loss'],
                                "value_loss": losses['value_loss'],
                                "entropy": losses['entropy'],
                                "total_loss": losses['total_loss']
                            })

                    for env_idx in range(batch_envs):
                        episode_counter += 1
                        self.log_episode(
                            episode_counter,
                            episode_rewards[env_idx],
                            max(1, episode_lengths[env_idx]),
                        )

                        if episode_counter % 10 == 0:
                            total_actions = sum(action_counts.values())
                            if total_actions > 0:
                                action_dist = {
                                    action_names[i]: f"{100*count/total_actions:.1f}%"
                                    for i, count in action_counts.items()
                                }
                                self.logger.info(f"Action distribution (last 10 ep): {action_dist}")
                            action_counts = {i: 0 for i in range(5)}

                        if episode_counter % save_interval == 0:
                            save_name = Path(self.config.get("save_path", "models/discrete_ppo_policy.pth")).name
                            checkpoint_path = self._get_checkpoint_path(save_name, episode_counter)
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
        """Save the trained discrete PPO agent.
        
        Raises an error if the target path matches the source checkpoint
        to prevent accidental overwrites.
        """
        if self.agent is None:
            raise RuntimeError("Agent not initialized")
        
        # Safety: never overwrite the source checkpoint
        save_path = Path(path).resolve()
        if self._source_checkpoint:
            source_path = Path(self._source_checkpoint).resolve()
            if save_path == source_path:
                raise RuntimeError(
                    f"Refusing to overwrite source checkpoint: {source_path}. "
                    f"Choose a different save path."
                )
        
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

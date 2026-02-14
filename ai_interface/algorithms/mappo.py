"""
Multi-Agent PPO (MAPPO) for soccer team coordination.

MAPPO extends PPO to multiple agents that:
1. Share observations about teammates and opponents
2. Learn coordinated behaviors (passing, positioning, shooting)
3. Use centralized training, decentralized execution (CTDE)

Key differences from single-agent PPO:
- Each agent has its own actor (policy) but shares a critic (value function)
- Critic sees global state (all agents), actors only see local observations
- Agents learn to coordinate through shared rewards and global value estimation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Hyperparameters
GAMMA = 0.99
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3  # Critic learns faster (sees more info)
EPS_CLIP = 0.2
K_EPOCHS = 4
ENTROPY_COEFF = 0.01
MAX_GRAD_NORM = 0.5
BATCH_SIZE = 2048
NUM_ACTIONS = 6  # Expanded from 5: APPROACH, SHOOT, PASS, DRIBBLE, CLEAR, REPOSITION


class Actor(nn.Module):
    """
    Individual agent's policy network (decentralized execution).
    Takes local observation, outputs action probabilities.
    """
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    
    def forward(self, obs):
        return self.network(obs)


class CentralizedCritic(nn.Module):
    """
    Centralized value function (centralized training).
    Takes global state (all agents' observations), outputs state value.
    
    This is key to MAPPO: critic sees everything during training,
    but actors only see local observations during execution.
    """
    def __init__(self, global_obs_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, global_obs):
        return self.network(global_obs)


class MAPPOAgent:
    """
    Multi-Agent PPO coordinator.
    
    Manages multiple agents that:
    - Each have their own actor (policy)
    - Share a centralized critic (value function)
    - Learn coordinated behaviors through shared experiences
    """
    
    def __init__(self, num_agents, obs_dim, num_actions=NUM_ACTIONS, device=None):
        """
        Args:
            num_agents: Number of agents on the team (e.g., 3 for 3v3)
            obs_dim: Observation dimension per agent
            num_actions: Number of discrete actions per agent
            device: torch device
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.device = device if device else torch.device("cpu")
        
        # Create actor for each agent
        self.actors = nn.ModuleList([
            Actor(obs_dim, num_actions).to(self.device)
            for _ in range(num_agents)
        ])
        
        # Shared centralized critic (sees all agents' observations)
        global_obs_dim = obs_dim * num_agents
        self.critic = CentralizedCritic(global_obs_dim).to(self.device)
        
        # Separate optimizers for actors and critic
        self.actor_optimizer = optim.Adam(self.actors.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        # Memory buffers
        self.memory = []
        self.rewards = []
        self.masks = []
    
    @property
    def batch_ready(self) -> bool:
        return len(self.memory) >= BATCH_SIZE
    
    def store_reward_mask(self, reward: float, mask: float):
        """Store shared team reward and done mask."""
        self.rewards.append(reward)
        self.masks.append(mask)
    
    def select_actions(self, observations):
        """
        Select actions for all agents.
        
        Args:
            observations: List of observations, one per agent
                         Each obs is shape (obs_dim,)
        
        Returns:
            List of actions (integers), one per agent
        """
        actions = []
        logprobs = []
        
        # Convert observations to tensors
        obs_tensors = [
            torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            for obs in observations
        ]
        
        # Global observation for critic (concatenate all local obs)
        global_obs = torch.cat(obs_tensors, dim=-1)
        
        with torch.no_grad():
            # Critic evaluates global state
            value = self.critic(global_obs)
            
            # Each actor selects action based on local observation
            for i, (actor, obs) in enumerate(zip(self.actors, obs_tensors)):
                logits = actor(obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
                
                actions.append(action.item())
                logprobs.append(logprob)
        
        # Store in memory
        self.memory.append({
            "observations": [obs.detach() for obs in obs_tensors],
            "global_obs": global_obs.detach(),
            "actions": [a for a in actions],  # Store as list of ints
            "logprobs": logprobs,  # List of tensors
            "value": value.detach()
        })
        
        return actions
    
    def compute_advantages(self, rewards, masks, values):
        """Compute GAE advantages."""
        advantages = []
        gae = 0
        values = values + [0]
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * values[i+1] * masks[i] - values[i]
            gae = delta + GAMMA * 0.95 * masks[i] * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self):
        """Run MAPPO update for all agents."""
        if len(self.memory) < BATCH_SIZE:
            return {}
        
        rewards = self.rewards
        masks = self.masks
        
        # Extract data from memory
        batch_size = len(self.memory)
        
        # For each agent, collect their observations and actions
        all_obs = [[] for _ in range(self.num_agents)]
        all_actions = [[] for _ in range(self.num_agents)]
        all_old_logprobs = [[] for _ in range(self.num_agents)]
        
        global_obs_list = []
        old_values_list = []
        
        for mem in self.memory:
            # Store per-agent data
            for i in range(self.num_agents):
                all_obs[i].append(mem["observations"][i])
                all_actions[i].append(mem["actions"][i])
                all_old_logprobs[i].append(mem["logprobs"][i])
            
            # Global observations and values
            global_obs_list.append(mem["global_obs"])
            old_values_list.append(mem["value"])
        
        # Convert to tensors
        # Per-agent tensors: [batch_size, obs_dim/action_dim]
        obs_tensors = [
            torch.stack(all_obs[i]).to(self.device)
            for i in range(self.num_agents)
        ]
        action_tensors = [
            torch.tensor(all_actions[i], dtype=torch.long, device=self.device)
            for i in range(self.num_agents)
        ]
        old_logprob_tensors = [
            torch.stack(all_old_logprobs[i]).to(self.device)
            for i in range(self.num_agents)
        ]
        
        # Global tensors
        global_obs = torch.stack(global_obs_list).to(self.device)
        old_values = torch.stack(old_values_list).squeeze().to(self.device)
        
        # Compute advantages
        advantages = torch.FloatTensor(
            self.compute_advantages(rewards, masks, old_values.detach().cpu().tolist())
        ).to(self.device)
        returns = advantages + old_values
        
        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Clip advantages
        advantages = torch.clamp(advantages, -5.0, 5.0)
        
        # MAPPO update for K epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(K_EPOCHS):
            # ============================================================
            # Update Centralized Critic
            # ============================================================
            values = self.critic(global_obs).squeeze()
            critic_loss = (returns - values).pow(2).mean()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
            self.critic_optimizer.step()
            
            # ============================================================
            # Update Each Actor (Decentralized Policies)
            # ============================================================
            actor_losses = []
            entropies = []
            
            for i in range(self.num_agents):
                # Re-evaluate agent i's actions
                logits = self.actors[i](obs_tensors[i])
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(action_tensors[i])
                entropy = dist.entropy()
                
                # PPO clipped objective
                ratio = torch.exp(logprobs - old_logprob_tensors[i])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                actor_losses.append(actor_loss)
                entropies.append(entropy.mean())
            
            # Combined actor loss (sum over all agents)
            total_actor_loss_epoch = sum(actor_losses)
            total_entropy_epoch = sum(entropies) / self.num_agents
            
            # Actor update with entropy bonus
            actor_objective = total_actor_loss_epoch - ENTROPY_COEFF * total_entropy_epoch
            
            self.actor_optimizer.zero_grad()
            actor_objective.backward()
            nn.utils.clip_grad_norm_(self.actors.parameters(), MAX_GRAD_NORM)
            self.actor_optimizer.step()
            
            # Track metrics
            total_actor_loss += total_actor_loss_epoch.item()
            total_critic_loss += critic_loss.item()
            total_entropy += total_entropy_epoch.item()
        
        # Clear buffers
        self.memory = []
        self.rewards = []
        self.masks = []
        
        return {
            "actor_loss": total_actor_loss / K_EPOCHS,
            "critic_loss": total_critic_loss / K_EPOCHS,
            "entropy": total_entropy / K_EPOCHS
        }
    
    def save(self, filepath: str):
        """Save all actors and critic."""
        torch.save({
            'actors_state_dict': [actor.state_dict() for actor in self.actors],
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load all actors and critic."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors_state_dict'][i])
        
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        for actor in self.actors:
            actor.eval()
        self.critic.eval()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of MAPPO for 3v3 soccer.
    """
    import torch
    
    # Create MAPPO agent for 3 robots
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mappo = MAPPOAgent(
        num_agents=3,
        obs_dim=18,  # Same observation space as before
        num_actions=6,  # Added PASS action
        device=device
    )
    
    # Simulated training loop
    for episode in range(10):
        # Reset environment (multi-agent)
        observations = [
            np.random.randn(18).astype(np.float32)  # Agent 1 obs
            for _ in range(3)  # 3 agents
        ]
        
        episode_reward = 0
        
        for step in range(200):
            # All agents select actions simultaneously
            actions = mappo.select_actions(observations)
            
            # Environment step (would return next_obs for each agent)
            # next_observations, reward, done, info = env.step(actions)
            
            # For demo: simulate
            next_observations = [np.random.randn(18).astype(np.float32) for _ in range(3)]
            reward = np.random.randn()
            done = (step == 199)
            
            # Store shared team reward
            mappo.store_reward_mask(reward, 1.0 - float(done))
            
            episode_reward += reward
            observations = next_observations
            
            if done:
                break
        
        # Update when batch is ready
        if mappo.batch_ready:
            losses = mappo.update()
            print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                  f"Actor loss={losses['actor_loss']:.4f}, "
                  f"Critic loss={losses['critic_loss']:.4f}")
    
    # Save trained agents
    mappo.save("models/mappo_team.pth")
    print("MAPPO team saved!")
"""
Simple Discrete PPO for Simplified Soccer Environment

Much simpler than hierarchical PPO - only learns discrete action selection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
GAMMA = 0.99
LR = 1e-4  # Reduced from 3e-4 to prevent value explosion
EPS_CLIP = 0.2
K_EPOCHS = 4
ENTROPY_COEFF = 0.01
MAX_GRAD_NORM = 0.5
BATCH_SIZE = 2048
NUM_ACTIONS = 5  # 5 discrete actions


class ActorCritic(nn.Module):
    """Simple actor-critic for discrete actions."""
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_head = nn.Linear(128, num_actions)
        
        # Value head (critic)
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.encoder(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


class DiscretePPOAgent:
    """PPO agent for discrete action spaces."""
    
    def __init__(self, obs_dim, num_actions=NUM_ACTIONS, device=None):
        self.device = device if device else torch.device("cpu")
        self.model = ActorCritic(obs_dim, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        
        self.memory = []
        self.rewards = []
        self.masks = []
    
    @property
    def batch_ready(self) -> bool:
        return len(self.memory) >= BATCH_SIZE
    
    def store_reward_mask(self, reward: float, mask: float):
        self.rewards.append(reward)
        self.masks.append(mask)
    
    def select_action(self, state):
        """Select an action using the policy."""
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits, value = self.model(state)
        
        # Sample action from categorical distribution
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        
        # Store in memory
        self.memory.append({
            "state": state.detach(),
            "action": action.detach(),
            "logprob": logprob.detach(),
            "value": value.detach()
        })
        
        return action.item()  # Return as integer
    
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
        """Run PPO update."""
        if len(self.memory) < BATCH_SIZE:
            return {}
        
        rewards = self.rewards
        masks = self.masks
        
        # Prepare tensors
        states = torch.stack([m['state'] for m in self.memory]).to(self.device)
        actions = torch.stack([m['action'] for m in self.memory]).to(self.device)
        old_logprobs = torch.stack([m['logprob'] for m in self.memory]).to(self.device)
        old_values = torch.stack([m['value'] for m in self.memory]).squeeze().to(self.device)
        
        # Compute advantages
        advantages = torch.FloatTensor(
            self.compute_advantages(rewards, masks, old_values.detach().cpu().tolist())
        ).to(self.device)
        returns = advantages + old_values
        
        # Diagnostic: Check for value explosion
        max_value = old_values.abs().max().item()
        max_return = returns.abs().max().item()
        if max_value > 1000 or max_return > 1000:
            print(f"WARNING: Value explosion detected!")
            print(f"  Max old value: {max_value:.2f}")
            print(f"  Max return: {max_return:.2f}")
            print(f"  Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
        
        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Clip advantages to prevent catastrophic updates
        advantages = torch.clamp(advantages, -5.0, 5.0)
        
        # PPO update epochs
        for epoch in range(K_EPOCHS):
            # Re-evaluate actions
            logits, values = self.model(states)
            
            dist = Categorical(logits=logits)
            logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Ratio for clipping
            ratio = torch.exp(logprobs - old_logprobs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with clipping (prevents value explosion)
            # Clip the value predictions to stay near old values
            values_pred = values.squeeze()
            values_clipped = old_values + torch.clamp(
                values_pred - old_values,
                -EPS_CLIP,
                EPS_CLIP
            )
            
            # Compute value loss for both clipped and unclipped
            value_loss_unclipped = (returns - values_pred).pow(2)
            value_loss_clipped = (returns - values_clipped).pow(2)
            
            # Take the max (more conservative, prevents explosion)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss - ENTROPY_COEFF * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()
        
        # Clear buffers
        self.memory = []
        self.rewards = []
        self.masks = []
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
            "max_value": old_values.abs().max().item(),
            "max_return": returns.abs().max().item(),
            "mean_value": old_values.mean().item(),
            "mean_return": returns.mean().item()
        }
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()
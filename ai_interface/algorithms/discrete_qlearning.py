"""
Q-Learning for Minimal Soccer Environment.

Classic tabular Q-Learning with function approximation (neural network).
Simpler than PPO - good for learning basic command sequences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class QNetwork(nn.Module):
    """Simple Q-network for discrete actions."""
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for Q-Learning."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store transition."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class QLearningAgent:
    """
    Q-Learning agent with experience replay.
    
    Simpler than PPO:
    - One Q-network (estimates action values)
    - Epsilon-greedy exploration
    - Experience replay
    - Target network for stability
    """
    
    def __init__(self, obs_dim, num_actions=5, device=None,
                 lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        """
        Args:
            obs_dim: Observation dimension
            num_actions: Number of discrete actions (5)
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_size: Replay buffer capacity
            batch_size: Batch size for updates
        """
        self.device = device if device else torch.device("cpu")
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Q-network and target network
        self.q_network = QNetwork(obs_dim, num_actions).to(self.device)
        self.target_network = QNetwork(obs_dim, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Stats
        self.steps = 0
        self.updates = 0
    
    def select_action(self, state, eval_mode=False):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current observation
            eval_mode: If True, always pick best action (no exploration)
        
        Returns:
            action: Integer action index
        """
        # Exploration
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Exploitation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.steps += 1
    
    def update(self):
        """
        Update Q-network using experience replay.
        
        Returns:
            Dictionary with loss info (or None if buffer too small)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        self.updates += 1
        if self.updates % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "q_mean": current_q_values.mean().item()
        }
    
    def save(self, filepath: str):
        """Save Q-network."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'updates': self.updates
        }, filepath)
    
    def load(self, filepath: str):
        """Load Q-network."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.steps = checkpoint.get('steps', 0)
        self.updates = checkpoint.get('updates', 0)
        self.q_network.eval()
        self.target_network.eval()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example Q-Learning training loop.
    """
    import torch
    from minimal_soccer_env import MinimalSoccerEnv
    
    # Create agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = QLearningAgent(
        obs_dim=8,
        num_actions=5,
        device=device,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # Dummy training loop
    for episode in range(100):
        # Dummy state
        state = np.random.randn(8).astype(np.float32)
        episode_reward = 0
        
        for step in range(200):
            # Select action
            action = agent.select_action(state)
            
            # Simulate step
            next_state = np.random.randn(8).astype(np.float32)
            reward = np.random.randn()
            done = (step == 199)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update
            if step % 4 == 0:  # Update every 4 steps
                losses = agent.update()
                if losses:
                    print(f"Update {agent.updates}: Loss={losses['loss']:.4f}, "
                          f"Epsilon={losses['epsilon']:.3f}")
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}")
    
    # Save
    agent.save("models/q_learning_agent.pth")
    print("Q-Learning agent trained and saved!")
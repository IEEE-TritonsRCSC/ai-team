import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

# -------------------------
# Hyperparameters
# -------------------------
GAMMA = 0.99
LR = 1e-4               # halved – 3e-4 was destroying the policy by update 3
EPS_CLIP = 0.1           # tighter clip – smaller policy steps per update
K_EPOCHS = 3             # fewer passes over the same batch
ENTROPY_COEFF = 0.05     # much stronger exploration pressure to resist collapse
MAX_GRAD_NORM = 0.5      # gradient clipping
LOG_STD_MIN = -1.0       # raised floor – keeps actions stochastic (was -2.0)
LOG_STD_MAX = 0.5        # ceiling on log_std
BATCH_SIZE = 2048        # minimum transitions before a PPO update
HIGH_LEVEL_ACTIONS = 4   # GoToBall, Shoot, Reposition, CatchHold
LOW_LEVEL_DIM = 4        # [v_x, v_y, omega, kick_power]

# -------------------------
# Actor-Critic Network
# -------------------------
class HierarchicalActorCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        # Shared Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # High-level discrete policy head
        self.high_policy = nn.Linear(128, HIGH_LEVEL_ACTIONS)
        
        # Low-level continuous policy head (mean + log_std)
        self.low_mean = nn.Linear(128, LOW_LEVEL_DIM)
        self.low_log_std = nn.Parameter(torch.zeros(LOW_LEVEL_DIM))
        
        # Critic head
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.encoder(x)
        # High-level policy logits
        high_logits = self.high_policy(x)
        # Low-level policy (clamp log_std to prevent collapse / explosion)
        low_mean = self.low_mean(x)
        clamped_log_std = torch.clamp(self.low_log_std, LOG_STD_MIN, LOG_STD_MAX)
        low_std = torch.exp(clamped_log_std)
        # Value
        value = self.value_head(x)
        return high_logits, low_mean, low_std, value


# -------------------------
# PPO Agent
# -------------------------
class PPOAgent:
    def __init__(self, obs_dim, device: torch.device = None):
        """PPO agent.

        Args:
            obs_dim: observation dimensionality
            device: torch device to place model and tensors on (defaults to CUDA if available)
        """
        self.device = device
        self.model = HierarchicalActorCritic(obs_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = []          # state/action/logprob/value per step
        self.rewards = []         # per-step rewards (across episodes)
        self.masks = []           # per-step masks   (across episodes)

    @property
    def batch_ready(self) -> bool:
        """True when enough transitions have been collected for a PPO update."""
        return len(self.memory) >= BATCH_SIZE

    def store_reward_mask(self, reward: float, mask: float):
        """Store reward and done-mask for the most recent step."""
        self.rewards.append(reward)
        self.masks.append(mask)

    def sample_action(self, state):
        """Sample an action without mutating rollout buffers.

        Returns:
            (action_dict, transition_dict)
        """
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            high_logits, low_mean, low_std, value = self.model(state)
        
        # Discrete high-level
        high_dist = Categorical(logits=high_logits)
        high_action = high_dist.sample()
        high_logprob = high_dist.log_prob(high_action)
        
        # Continuous low-level
        low_dist = Normal(low_mean, low_std)
        low_action = low_dist.sample()
        low_logprob = low_dist.log_prob(low_action).sum()

        # If the high-level action is Shoot (1), force kick angle to 0.
        # The simulator derives kick angle from arctan2(v_y, v_x), so
        # we ensure v_y == 0 and v_x is positive so the angle = 0.
        if int(high_action.item()) == 1:
            # work on a copy to avoid in-place issues
            low_action = low_action.clone()
            # set lateral component to zero (v_y)
            if low_action.numel() >= 2:
                low_action[1] = 0.0
            # ensure forward component (v_x) is positive
            if low_action.numel() >= 1:
                low_action[0] = torch.abs(low_action[0]) + 1e-6
            # recompute logprob for the modified low_action
            low_logprob = low_dist.log_prob(low_action).sum()

        transition = {
            "state": state.detach(),
            "high_action": high_action.detach(),
            "low_action": low_action.detach(),
            "high_logprob": high_logprob.detach(),
            "low_logprob": low_logprob.detach(),
            "value": value.detach()
        }
        action = {
            "high_level": high_action.item(),
            "low_level": low_action.detach().cpu().numpy()
        }
        return action, transition

    def append_transition(self, transition: dict, reward: float, mask: float):
        """Append a precomputed transition and its reward/mask."""
        self.memory.append(transition)
        self.rewards.append(reward)
        self.masks.append(mask)

    def select_action(self, state):
        action, transition = self.sample_action(state)
        self.memory.append(transition)
        return action

    def compute_advantages(self, rewards, masks, values):
        advantages = []
        gae = 0
        values = values + [0]  # bootstrap
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * values[i+1] * masks[i] - values[i]
            gae = delta + GAMMA * 0.95 * masks[i] * gae
            advantages.insert(0, gae)
        return advantages

    def update(self):
        """Run a PPO update using all transitions accumulated in the buffer."""
        rewards = self.rewards
        masks = self.masks

        # Prepare memory tensors (detach from computation graph)
        states = torch.stack([m['state'] for m in self.memory]).to(self.device)
        high_actions = torch.stack([m['high_action'] for m in self.memory]).to(self.device)
        low_actions = torch.stack([m['low_action'] for m in self.memory]).to(self.device)
        old_high_logprobs = torch.stack([m['high_logprob'] for m in self.memory]).to(self.device)
        old_low_logprobs = torch.stack([m['low_logprob'] for m in self.memory]).to(self.device)
        old_values = torch.stack([m['value'] for m in self.memory]).squeeze().to(self.device)

        # Compute advantages
        advantages = torch.FloatTensor(self.compute_advantages(rewards, masks, old_values.detach().cpu().tolist())).to(self.device)
        returns = advantages + old_values

        # Normalise advantages to stabilise training
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Clip advantages to prevent catastrophic updates from extreme values
        # This prevents a few very negative episodes from destroying the policy
        advantages = torch.clamp(advantages, -5.0, 5.0)

        # PPO update
        for epoch in range(K_EPOCHS):
            # Re-evaluate actions and values (creates fresh computational graph)
            high_logits, low_mean, low_std, values = self.model(states)
            
            # High-level policy
            high_dist = Categorical(logits=high_logits)
            high_logprobs = high_dist.log_prob(high_actions)
            high_ratio = torch.exp(high_logprobs - old_high_logprobs)
            
            # Low-level policy
            low_dist = Normal(low_mean, low_std)
            low_logprobs = low_dist.log_prob(low_actions).sum(dim=1)
            low_ratio = torch.exp(low_logprobs - old_low_logprobs)
            
            # Clipped surrogate objectives
            high_surr1 = high_ratio * advantages
            high_surr2 = torch.clamp(high_ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            high_surrogate = torch.min(high_surr1, high_surr2)
            
            low_surr1 = low_ratio * advantages
            low_surr2 = torch.clamp(low_ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            low_surrogate = torch.min(low_surr1, low_surr2)
            
            # Combined policy loss
            policy_loss = -(high_surrogate + low_surrogate).mean()
            
            # Entropy bonus — prevents the policy from collapsing to a
            # deterministic action (the main cause of the "beeline" bug)
            high_entropy = high_dist.entropy().mean()
            low_entropy = low_dist.entropy().sum(dim=-1).mean()
            entropy_bonus = ENTROPY_COEFF * (high_entropy + low_entropy)
            
            # Value loss
            value_loss = (returns - values.squeeze()).pow(2).mean()
            
            # Total loss  (subtract entropy bonus because we maximise entropy)
            total_loss = policy_loss + 0.5 * value_loss - entropy_bonus
            
            self.optimizer.zero_grad()
            total_loss.backward()
            # Clip gradients to prevent destructive updates to shared encoder
            nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()
        
        # Clear memory + reward/mask buffers
        self.memory = []
        self.rewards = []
        self.masks = []

        # Return final-epoch losses so the trainer can log / plot them
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def save(self, filepath: str):
        """Save model checkpoint to filepath."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Load model checkpoint from filepath."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()

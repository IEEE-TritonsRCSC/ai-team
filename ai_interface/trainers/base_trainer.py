"""
Base trainer class for different training methodologies.

This module provides the abstract base class that all specific trainers should inherit from,
ensuring a consistent interface across different training approaches.
"""
import abc
from typing import Optional, Dict, Any
import gymnasium as gym
from pathlib import Path
import json
import logging
from datetime import datetime


class BaseTrainer(abc.ABC):
    """Abstract base class for all trainers.
    
    This class defines the common interface that all training implementations must follow,
    regardless of the specific algorithm or approach being used.
    """
    
    def __init__(self, config: Dict[str, Any], log_dir: Optional[str] = None):
        """Initialize the base trainer.
        
        Args:
            config: Configuration dictionary containing training parameters
            log_dir: Optional directory for log files. If None, logs will be created in train_logs/
        """
        self.config = config
        self.log_dir = log_dir or "train_logs"
        
        # Setup logging
        self._setup_logging()
        
        # Initialize metrics tracking
        self.training_metrics = {
            "episodes": 0,
            "total_timesteps": 0,
            "total_reward": 0,
            "episode_rewards": [],
            "episode_lengths": []
        }
    
    def _setup_logging(self):
        """Setup logging configuration with datetime-based log files."""
        # Create log directory
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create datetime-based log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(self.log_dir) / f"training_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()  # Also log to console
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Log initial configuration
        self.logger.info(f"Training session started at {datetime.now()}")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    @abc.abstractmethod
    def setup_environment(self) -> gym.Env:
        """Setup and return the training environment.
        
        Returns:
            gym.Env: The configured environment for training
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def setup_model(self, env: gym.Env):
        """Setup the model/agent for training.
        
        Args:
            env: The environment the model will interact with
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def train(self):
        """Execute the main training loop."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def save_model(self, path: str):
        """Save the trained model.
        
        Args:
            path: Path where the model should be saved
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def load_model(self, path: str):
        """Load a pre-trained model.
        
        Args:
            path: Path to the model to load
        """
        raise NotImplementedError()
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log training metrics.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        self.logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    def log_episode(self, episode: int, reward: float, length: int):
        """Log episode completion.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length in steps
        """
        self.training_metrics["episodes"] = episode
        self.training_metrics["total_reward"] += reward
        self.training_metrics["episode_rewards"].append(reward)
        self.training_metrics["episode_lengths"].append(length)
        
        self.logger.info(f"Episode {episode}: Reward={reward:.2f}, Length={length}")
        
        # Log periodic statistics
        if episode % 10 == 0:
            recent_rewards = self.training_metrics["episode_rewards"][-10:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_length = sum(self.training_metrics["episode_lengths"][-10:]) / 10
            self.logger.info(f"Last 10 episodes - Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")
    
    def cleanup(self):
        """Cleanup resources after training completion."""
        self.logger.info("Training session completed")
        self.logger.info(f"Final metrics: {json.dumps(self.training_metrics, indent=2)}")
"""
Base trainer class for different training methodologies.

This module provides the abstract base class that all specific trainers should inherit from,
ensuring a consistent interface across different training approaches.
"""
import abc
from typing import Optional, Dict, Any, List
import gymnasium as gym
from pathlib import Path
import json
import logging
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np


class BaseTrainer(abc.ABC):
    """Abstract base class for all trainers.
    
    This class defines the common interface that all training implementations must follow,
    regardless of the specific algorithm or approach being used.
    """
    
    def __init__(self, config: Dict[str, Any], log_dir: Optional[str] = None,
                 algorithm_name: Optional[str] = None):
        """Initialize the base trainer.
        
        Args:
            config: Configuration dictionary containing training parameters
            log_dir: Optional directory for log files. If None, logs will be created in train_logs/
            algorithm_name: Short identifier for the algorithm (e.g. "discrete_ppo").
                Used to create a datetime + algorithm named model directory.
        """
        self.config = config
        self.log_dir = log_dir or "train_logs"
        self.algorithm_name = algorithm_name or "unknown"
        
        # Setup logging
        self._setup_logging()
        
        # Setup model output directory (mirrors run_dir pattern)
        self._setup_model_dir()
        
        # Initialize metrics tracking
        self.training_metrics = {
            "episodes": 0,
            "total_timesteps": 0,
            "total_reward": 0,
            "episode_rewards": [],
            "episode_lengths": []
        }
    
    def _setup_logging(self):
        """Setup logging configuration with datetime-based log directory.

        Each training run gets its own sub-directory named after the current
        datetime inside the top-level log_dir (e.g. train_logs/20260207_200530/).
        The log file itself is simply called ``train_log.log``.
        """
        # Create a datetime-named sub-directory for this run
        # Include microseconds to avoid collisions when parallel workers start
        # within the same second.
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(self.log_dir) / self.run_timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.run_dir / "train_log.log"
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # Configure logging. Keep DEBUG diagnostics in train_log.log without
        # printing them to the terminal during long training runs.
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                file_handler,
                stream_handler,
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

    # ------------------------------------------------------------------
    # Model directory helpers
    # ------------------------------------------------------------------
    def _setup_model_dir(self):
        """Create a datetime + algorithm named directory under ``models/``.

        The directory mirrors the ``run_dir`` pattern used for logs, e.g.
        ``models/20260212_153000_discrete_ppo/``.
        """
        self.model_dir = Path("models") / f"{self.run_timestamp}_{self.algorithm_name}"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Model output directory: {self.model_dir}")

    def _get_checkpoint_path(self, filename: str, tag) -> Path:
        """Build a checkpoint path inside :attr:`model_dir`.

        Args:
            filename: Base filename for the checkpoint (e.g. ``"policy.pth"``).
            tag: Episode number or timestep count appended to the stem.

        Returns:
            Full path like ``models/20260212_…_discrete_ppo/policy_ep100.pth``.
        """
        p = Path(filename)
        return self.model_dir / f"{p.stem}_ep{tag}{p.suffix}"

    def _sim_endpoint_for_env(self, env_index: int = 0) -> tuple[str, int, int]:
        """Return simulator endpoint for a specific parallel environment index.

        Port layout:
        - player port:  base + idx * stride
        - trainer port: player + 1
        """
        sim_host = str(self.config.get("sim_host", "127.0.0.1"))
        base_port = int(self.config.get("sim_player_port", 6000))
        stride = max(3, int(self.config.get("sim_port_stride", 10)))
        player_port = base_port + env_index * stride
        trainer_port = player_port + 1
        return sim_host, player_port, trainer_port

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_training(self, window: int = 10) -> None:
        """Plot average reward per *window*-episode blocks using already-tracked metrics.

        The data comes from ``self.training_metrics["episode_rewards"]`` which is
        populated automatically by :meth:`log_episode`.

        Args:
            window: Number of episodes to average over (default 10).
        """
        rewards = self.training_metrics["episode_rewards"]
        if not rewards:
            self.logger.warning("No reward data to plot.")
            return

        n_windows = len(rewards) // window
        if n_windows == 0:
            n_windows = 1

        trimmed = np.array(rewards[: n_windows * window]).reshape(n_windows, window)
        avg_rewards = trimmed.mean(axis=1)
        x = np.arange(1, n_windows + 1) * window  # episode at end of each window

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, avg_rewards, label="Avg Reward", marker="o", markersize=3)
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Average Reward (per {window} episodes)")
        ax.set_title("Training — Average Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = Path(self.run_dir) / "reward_plot.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        self.logger.info(f"Reward plot saved to {plot_path}")

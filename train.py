#!/usr/bin/env python3
"""
Unified Training Script

This script provides a unified interface for training different RL algorithms
with various configurations. It supports:
- Hierarchical PPO
- Discrete PPO
- Stable Baselines3 PPO
- Other custom algorithms (extensible)

Usage:
    python train_unified.py --config configs/hier_ppo_config.json
    python train_unified.py --trainer hier_ppo --team_config team_config.json
    python train_unified.py --trainer sb3_ppo --timesteps 20000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import torch

from ai_interface.trainers import BaseTrainer, HierarchicalPPOTrainer, SB3PPOTrainer, DiscretePPOTrainer, MAPPOTrainer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_default_config(trainer_type: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Create default configuration based on trainer type and command line arguments."""
    base_config = {
        "team_config": args.team_config,
        "env_mode": args.env,
        "team_name": args.team,
    }
    
    if trainer_type == "hier_ppo":
        return {
            **base_config,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "obs_dim": args.obs_dim,
            "save_path": args.save_path or "models/hier_ppo_policy.pth",
            "save_interval": args.save_interval,
            "load_model": args.load_model or args.resume_checkpoint
        }
    elif trainer_type == "discrete_ppo":
        return {
            **base_config,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "obs_dim": args.obs_dim,
            "save_path": args.save_path or "models/discrete_ppo_policy.pth",
            "save_interval": args.save_interval,
            "load_model": args.load_model or args.resume_checkpoint,
            "curriculum": args.curriculum,
            "start_phase": args.start_phase
        }
    elif trainer_type == "mappo":
        return {
            **base_config,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "obs_dim": args.obs_dim,
            "num_agents": getattr(args, "num_agents", 3),
            "save_path": args.save_path or "models/mappo_team.pth",
            "save_interval": args.save_interval,
            "load_model": args.load_model or args.resume_checkpoint
        }
    elif trainer_type == "sb3_ppo":
        return {
            **base_config,
            "timesteps": args.timesteps,
            "learn_batch_timesteps": args.learn_batch_timesteps,
            "save_path": args.save_path or "models/sb3_ppo_policy.zip",
            "save_interval": args.save_interval,
            "load_model": args.load_model or args.resume_checkpoint,
            "model_params": {}
        }
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")


def get_trainer(trainer_type: str, config: Dict[str, Any]) -> BaseTrainer:
    """Create and return the appropriate trainer instance."""
    trainers = {
        "hier_ppo": HierarchicalPPOTrainer,
        "discrete_ppo": DiscretePPOTrainer,
        "mappo": MAPPOTrainer,
        "sb3_ppo": SB3PPOTrainer,
    }
    
    if trainer_type not in trainers:
        raise ValueError(f"Unknown trainer type: {trainer_type}. Available: {list(trainers.keys())}")
    
    # Determine device and pass to trainer so models/data can be placed on CUDA when available
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    return trainers[trainer_type](config, device=device)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Unified RL training script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --config configs/hier_ppo_config.json
  python train.py --trainer hier_ppo --episodes 1000 --max_steps 200
  python train.py --trainer mappo --num_agents 3 --obs_dim 25 --episodes 3000
  python train.py --config configs/mappo_config.json
  python train.py --trainer sb3_ppo --timesteps 20000
  python train.py --trainer discrete_ppo --resume_checkpoint models/old_run/policy_ep200.pth --curriculum --start_phase 2
        """
    )
    
    # Configuration options
    parser.add_argument("--config", type=str, 
                        help="Path to JSON configuration file")
    parser.add_argument("--trainer", type=str, choices=["hier_ppo", "discrete_ppo", "mappo", "sb3_ppo"],
                        default="hier_ppo", help="Type of trainer to use")
    
    # Environment options
    parser.add_argument("--team_config", type=str, default="team_config.json",
                        help="Path to team configuration JSON file")
    parser.add_argument("--env", choices=[
        "sim-only", "sim-mixed", "field-practice", "field-tournament"
    ], default="sim-only", help="Environment mode for Networker")
    parser.add_argument("--team", type=str, default=None,
                        help="Team name to control")
    
    # Hierarchical PPO specific options
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes to train (hier_ppo)")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum steps per episode (hier_ppo)")
    parser.add_argument("--obs_dim", type=int, default=10,
                        help="Observation dimension (hier_ppo, discrete_ppo, mappo)")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="Number of agents per team (mappo)")
    
    # SB3 PPO specific options
    parser.add_argument("--timesteps", type=int, default=10000,
                        help="Total timesteps to train (sb3_ppo)")
    parser.add_argument("--learn_batch_timesteps", type=int, default=2048,
                        help="Timesteps per learning batch (sb3_ppo)")
    
    # Common options
    parser.add_argument("--save_path", type=str, default=None,
                        help="Where to save the trained model")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Save model every N episodes/timesteps")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to load a pre-trained model (optional)")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from. "
                             "The original checkpoint is never modified; new "
                             "checkpoints are saved to a fresh timestamped directory.")
    
    # Curriculum options (discrete_ppo only)
    parser.add_argument("--curriculum", action="store_true", default=False,
                        help="Enable curriculum learning mode (discrete_ppo only)")
    parser.add_argument("--start_phase", type=int, default=0,
                        help="Curriculum phase to start from (0-4). "
                             "Use 2 to start from Phase 3 after Phase 1-2 training.")
    
    parser.add_argument("--log_dir", type=str, default="train_logs",
                        help="Directory for training logs")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            print(f"Loading configuration from {args.config}")
            config = load_config(args.config)
            trainer_type = config.get("trainer_type", args.trainer)
        else:
            print(f"Using command line configuration for {args.trainer} trainer")
            trainer_type = args.trainer
            config = create_default_config(trainer_type, args)
        
        # Validate checkpoint safety: if resuming, warn that original is read-only
        resume_path = config.get("load_model")
        if resume_path:
            import os
            if not os.path.isfile(resume_path):
                print(f"ERROR: Checkpoint not found: {resume_path}")
                sys.exit(1)
            print(f"\nResuming from checkpoint: {resume_path}")
            print(f"  Original checkpoint will NOT be modified.")
            print(f"  New checkpoints will be saved to a fresh timestamped directory.\n")
        
        print(f"Starting {trainer_type} training with configuration:")
        print(json.dumps(config, indent=2))
        
        # Create trainer
        trainer = get_trainer(trainer_type, config)
        
        # Execute training
        try:
            trainer.train()
        finally:
            trainer.cleanup()
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Unified Training Script

This script provides a unified interface for training different RL algorithms
with various configurations. It supports:
- Hierarchical PPO
- Discrete PPO
- MAPPO (Multi-Agent PPO)
- HSM-MARL (Hierarchical State Machine + MAPPO)
- Stable Baselines3 PPO
- TD3 JAL (Twin Delayed DDPG for Joint-Action Learning)

Usage:
    python train.py --config configs/hier_ppo_config.json
    python train.py --trainer hier_ppo --team_config team_config.json
    python train.py --trainer td3_jal --timesteps 400000
    python train.py --config configs/td3_jal_config.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING
import importlib

import torch

if TYPE_CHECKING:
    from ai_interface.trainers import BaseTrainer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def _cli_flag_provided(flag: str) -> bool:
    """Return True when a CLI flag was explicitly present in argv."""
    argv = sys.argv[1:]
    return flag in argv or any(arg.startswith(f"{flag}=") for arg in argv)


def create_default_config(trainer_type: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Create default configuration based on trainer type and command line arguments."""
    base_config = {
        "team_config": args.team_config,
        "env_mode": args.env,
        "team_name": args.team,
        "num_envs": args.num_envs,
        "sim_host": args.sim_host,
        "sim_player_port": args.sim_player_port,
        "sim_port_stride": args.sim_port_stride,
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
    elif trainer_type == "hsm_marl":
        hsm_num_agents = args.num_agents if args.num_agents != 3 else 4
        return {
            **base_config,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "obs_dim": args.obs_dim,
            "num_agents": hsm_num_agents,
            "save_path": args.save_path or "models/hsm_marl_final.pth",
            "save_interval": args.save_interval,
            "load_model": args.load_model or args.resume_checkpoint,
            "algorithm": {
                "learning_rate": 3e-4,
                "critic_learning_rate": 1e-3,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "entropy_coef": 0.03,
                "value_coef": 0.5,
                "max_grad_norm": 0.5,
                "ppo_epochs": 4,
                "rollout_length": 2048,
                "hidden_dims": [128, 128],
            },
            "hsm_thresholds": {
                "possession_distance": 1.35,
                "defensive_x_boundary": -10.0,
                "attacking_x_boundary": 10.0,
                "role_switch_cooldown": 8,
            },
            "curriculum": {
                "stage1": {
                    "episodes": 300,
                    "num_agents": 1,
                    "roles_to_train": ["STRIKER", "SUPPORT", "DEFENDER", "GOALIE"],
                },
                "stage2": {
                    "episodes": 500,
                    "num_agents": 2,
                    "forced_roles": {"0": "STRIKER", "1": "SUPPORT"},
                },
                "stage3": {
                    "episodes": 1200,
                    "num_agents": hsm_num_agents,
                },
            },
        }
    elif trainer_type == "qlearning":
        return {
            **base_config,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "obs_dim": args.obs_dim,
            "save_path": args.save_path or "models/q_learning.pth",
            "save_interval": args.save_interval,
            "load_model": args.load_model or args.resume_checkpoint,
            "lr": args.lr,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "update_frequency": args.update_frequency,
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
    elif trainer_type == "robot_attention":
        return {
            **base_config,
            "num_robots": getattr(args, "num_robots", 6),
            "max_steps": args.max_steps,
            "timesteps": args.timesteps,
            "learn_batch_timesteps": args.learn_batch_timesteps,
            "save_path": args.save_path or "models/robot_attention_policy.zip",
            "save_interval": args.save_interval,
            "load_model": args.load_model or args.resume_checkpoint,
            "feature_extractor_params": {
                "hidden_dim": 64,
                "num_heads": 4,
                "num_attention_layers": 1,
                "time_window": 20,
            },
            "model_params": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "policy_kwargs": {
                    "net_arch": {
                        "pi": [64, 32],
                        "vf": [64, 32]
                    }
                }
            }
        }
    elif trainer_type == "td3_jal":
        return {
            **base_config,
            "num_robots": getattr(args, "num_robots", 2),
            "obs_dim": args.obs_dim,
            "timesteps": args.timesteps,
            "learn_batch_timesteps": args.learn_batch_timesteps,
            "save_path": args.save_path or "models/td3_jal_policy.zip",
            "save_interval": args.save_interval,
            "load_model": args.load_model,
            "model_params": {
                "learning_rate": 0.001,
                "buffer_size": 100000,
                "batch_size": 64,
                "gamma": 0.9,
                "tau": 0.01,
                "policy_delay": 2,
                "target_policy_noise": 0.2,
                "action_noise_std": 0.05,
                "policy_kwargs": {
                    "net_arch": [64, 48, 32]
                }
            }
        }
    elif trainer_type == "hsm_sb3_ppo":
        return {
            **base_config,
            "unum": 1,
            "obs_dim": args.obs_dim if args.obs_dim != 18 else 28,
            "max_steps": args.max_steps,
            "timesteps": args.timesteps,
            "learn_batch_timesteps": args.learn_batch_timesteps,
            "save_path": args.save_path or "models/hsm_sb3_ppo_policy.zip",
            "save_interval": args.save_interval,
            "load_model": args.load_model or args.resume_checkpoint,
            "model_params": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
            },
        }
    elif trainer_type == "hsm_sb3_ppo_curriculum":
        return {
            **base_config,
            "obs_dim": args.obs_dim if args.obs_dim != 18 else 28,
            "max_steps": args.max_steps,
            "learn_batch_timesteps": args.learn_batch_timesteps,
            "save_path": args.save_path or "models/hsm_sb3_ppo_curriculum",
            "save_interval": args.save_interval,
            "load_model": args.load_model or args.resume_checkpoint,
            "model_params": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
            },
        }
    elif trainer_type == "td3_jal_curriculum":
        return {
            **base_config,
            "obs_dim_per_robot": 8,
            "non_robot_obs_dim": 4,
            "max_steps": args.max_steps,
            "learn_batch_timesteps": args.learn_batch_timesteps,
            "save_path": args.save_path or "models/td3_jal_curriculum",
            "save_interval": args.save_interval,
            "load_model": args.load_model,
            "model_params": {
                "learning_rate": 0.001,
                "buffer_size": 100000,
                "batch_size": 64,
                "gamma": 0.9,
                "tau": 0.01,
                "policy_delay": 2,
                "target_policy_noise": 0.2,
                "action_noise_std": 0.05,
                "policy_kwargs": {
                    "net_arch": [64, 48, 32]
                }
            },
            "network_params": {
                "network_type": "expandable_attention",
                "feature_dim": 64,
                "num_heads": 4,
                "max_robots": 3
            },
        }
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")


def get_trainer(trainer_type: str, config: Dict[str, Any]) -> "BaseTrainer":
    """Create and return the appropriate trainer instance."""
    trainers = {
        "hier_ppo": ("ai_interface.trainers.hier_ppo_trainer", "HierarchicalPPOTrainer"),
        "discrete_ppo": ("ai_interface.trainers.discrete_ppo_trainer", "DiscretePPOTrainer"),
        "mappo": ("ai_interface.trainers.mappo_trainer", "MAPPOTrainer"),
        "hsm_marl": ("ai_interface.trainers.hsm_marl_trainer", "HSMMARLTrainer"),
        "hsm_sb3_ppo": ("ai_interface.trainers.hsm_sb3_ppo_trainer", "HSMSB3PPOTrainer"),
        "hsm_sb3_ppo_curriculum": ("ai_interface.trainers.hsm_sb3_ppo_curriculum_trainer", "HSMSB3PPOCurriculumTrainer"),
        "sb3_ppo": ("ai_interface.trainers.sb3_ppo_trainer", "SB3PPOTrainer"),
        "robot_attention": ("ai_interface.trainers.robot_attention_trainer", "RobotAttentionTrainer"),
        "qlearning": ("ai_interface.trainers.qlearning_trainer", "QLearningTrainer"),
        "td3_jal": ("ai_interface.trainers.td3_jal_trainer", "TD3JALTrainer"),
        "td3_jal_curriculum": ("ai_interface.trainers.td3_jal_curriculum_trainer", "TD3JALCurriculumTrainer"),
    }
    
    if trainer_type not in trainers:
        raise ValueError(f"Unknown trainer type: {trainer_type}. Available: {list(trainers.keys())}")

    module_name, class_name = trainers[trainer_type]
    try:
        trainer_module = importlib.import_module(module_name)
        trainer_cls = getattr(trainer_module, class_name)
    except Exception as e:
        raise ImportError(
            f"Failed to load trainer '{trainer_type}' from {module_name}.{class_name}: {e}"
        ) from e
    
    # Determine device and pass to trainer so models/data can be placed on CUDA when available
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    return trainer_cls(config, device=device)


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
    python train.py --trainer hsm_marl --num_agents 4 --obs_dim 33 --episodes 3000
    python train.py --trainer hsm_sb3_ppo --timesteps 300000 --obs_dim 28
  python train.py --trainer td3_jal --timesteps 400000 --num_robots 2
  python train.py --trainer robot_attention --timesteps 400000 --num_robots 6
  python train.py --config configs/td3_jal_config.json
  python train.py --trainer discrete_ppo --resume_checkpoint models/old_run/policy_ep200.pth --curriculum --start_phase 2
  python train.py --trainer qlearning --episodes 2000 --lr 1e-3 --epsilon_decay 0.995
  python train.py --trainer discrete_ppo --episodes 2000 --num_envs 4 --sim_player_port 6000 --sim_port_stride 10
        """
    )
    
    # Configuration options
    parser.add_argument("--config", type=str, 
                        help="Path to JSON configuration file")
    parser.add_argument("--trainer", type=str, 
                        choices=["hier_ppo", "discrete_ppo", "mappo", "hsm_marl", "hsm_sb3_ppo", "hsm_sb3_ppo_curriculum", "sb3_ppo", "robot_attention", "td3_jal", "td3_jal_curriculum", "qlearning"],
                        default="hier_ppo", help="Type of trainer to use")
    
    # Environment options
    parser.add_argument("--team_config", type=str, default="team_config.json",
                        help="Path to team configuration JSON file")
    parser.add_argument("--env", choices=[
        "sim-only", "sim-embedded", "sim-mixed", "field-practice", "field-tournament"
    ], default="sim-only", help="Environment mode for Networker")
    parser.add_argument("--team", type=str, default=None,
                        help="Team name to control")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel simulator environments to use")
    parser.add_argument("--sim_host", type=str, default="127.0.0.1",
                        help="Simulator host for sim-only/sim-mixed")
    parser.add_argument("--sim_player_port", type=int, default=6000,
                        help="Base simulator player port (env0). Trainer port uses player_port+1")
    parser.add_argument("--sim_port_stride", type=int, default=10,
                        help="Port stride between parallel envs (must avoid overlap, recommended >= 3)")
    
    # Episode-based trainers (hier_ppo, discrete_ppo, mappo)
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes to train (hier_ppo, discrete_ppo, mappo)")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum steps per episode (hier_ppo, discrete_ppo, mappo)")
    
    # Observation/action space
    parser.add_argument("--obs_dim", type=int, default=18,
                        help="Observation dimension (default: 18 for TD3 JAL, others may vary)")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="Number of agents per team (mappo)")
    parser.add_argument("--num_robots", type=int, default=2,
                        help="Number of robots in JAL (td3_jal)")
    
    # Timestep-based trainers (sb3_ppo, robot_attention, td3_jal)
    parser.add_argument("--timesteps", type=int, default=400000,
                        help="Total timesteps to train (sb3_ppo, robot_attention, td3_jal)")
    parser.add_argument("--learn_batch_timesteps", type=int, default=2048,
                        help="Timesteps per learning batch (sb3_ppo, robot_attention, td3_jal)")
    
    # Common options
    parser.add_argument("--save_path", type=str, default=None,
                        help="Where to save the trained model")
    parser.add_argument("--save_interval", type=int, default=10000,
                        help="Save model every N episodes/timesteps")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to load a pre-trained model (optional)")
    parser.add_argument("--predict", action="store_true", default=False,
                        help="Run infinite inference instead of training")
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
    
    # Q-Learning specific options
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (qlearning)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (qlearning)")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Initial exploration rate (qlearning)")
    parser.add_argument("--epsilon_end", type=float, default=0.01,
                        help="Final exploration rate (qlearning)")
    parser.add_argument("--epsilon_decay", type=float, default=0.995,
                        help="Epsilon decay rate per update (qlearning)")
    parser.add_argument("--buffer_size", type=int, default=10000,
                        help="Replay buffer capacity (qlearning)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for Q-network updates (qlearning)")
    parser.add_argument("--update_frequency", type=int, default=4,
                        help="Update Q-network every N steps (qlearning)")
    
    parser.add_argument("--log_dir", type=str, default="train_logs",
                        help="Directory for training logs")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            print(f"Loading configuration from {args.config}")
            config = load_config(args.config)
            trainer_type = config.get("trainer_type", args.trainer)
            # Backfill runtime defaults for older config files.
            config.setdefault("env_mode", args.env)
            config.setdefault("team_name", args.team)
            config.setdefault("num_envs", args.num_envs)
            config.setdefault("sim_host", args.sim_host)
            config.setdefault("sim_player_port", args.sim_player_port)
            config.setdefault("sim_port_stride", args.sim_port_stride)
            config.setdefault("predict_only", args.predict)

            # Allow launchers and direct CLI usage to override key runtime
            # environment parameters even when a config file is supplied.
            if _cli_flag_provided("--env"):
                config["env_mode"] = args.env
            if _cli_flag_provided("--team"):
                config["team_name"] = args.team
            if _cli_flag_provided("--num_envs"):
                config["num_envs"] = args.num_envs
            if _cli_flag_provided("--sim_host"):
                config["sim_host"] = args.sim_host
            if _cli_flag_provided("--sim_player_port"):
                config["sim_player_port"] = args.sim_player_port
            if _cli_flag_provided("--sim_port_stride"):
                config["sim_port_stride"] = args.sim_port_stride

            if args.predict:
                config["predict_only"] = True
        else:
            print(f"Using command line configuration for {args.trainer} trainer")
            trainer_type = args.trainer
            config = create_default_config(trainer_type, args)
            config["predict_only"] = bool(args.predict)

        num_envs = int(config.get("num_envs", 1))
        sim_port_stride = int(config.get("sim_port_stride", 10))
        env_mode = config.get("env_mode", "sim-only")
        if num_envs > 1 and env_mode not in ["sim-only", "sim-embedded", "sim-mixed"]:
            raise ValueError("Parallel simulator training requires env_mode to be sim-only, sim-embedded, or sim-mixed")
        if sim_port_stride < 3:
            raise ValueError("sim_port_stride must be >= 3 to avoid simulator port overlap")
        
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
        
        print(f"\nStarting {trainer_type} training with configuration:")
        print(json.dumps(config, indent=2))
        
        # Create trainer
        trainer = get_trainer(trainer_type, config)
        
        # Execute training or inference
        try:
            if config.get("predict_only", False):
                trainer.predict()
            else:
                trainer.train()
        finally:
            trainer.cleanup()

        print("\n" + "="*60)
        print("Run completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

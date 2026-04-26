"""
AI Interface Trainers

This module contains all trainer implementations for different training methodologies.
"""

from .base_trainer import BaseTrainer

_TRAINER_EXPORTS = {
    "HierarchicalPPOTrainer": (".hier_ppo_trainer", "HierarchicalPPOTrainer"),
    "SB3PPOTrainer": (".sb3_ppo_trainer", "SB3PPOTrainer"),
    "DiscretePPOTrainer": (".discrete_ppo_trainer", "DiscretePPOTrainer"),
    "MAPPOTrainer": (".mappo_trainer", "MAPPOTrainer"),
    "TD3JALTrainer": (".td3_jal_trainer", "TD3JALTrainer"),
    "HSMMARLTrainer": (".hsm_marl_trainer", "HSMMARLTrainer"),
    "HSMSB3PPOTrainer": (".hsm_sb3_ppo_trainer", "HSMSB3PPOTrainer"),
    "QLearningTrainer": (".qlearning_trainer", "QLearningTrainer"),
}


def __getattr__(name: str):
    """Load optional trainer modules only when that trainer is requested."""
    if name not in _TRAINER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_name, class_name = _TRAINER_EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    trainer_cls = getattr(module, class_name)
    globals()[name] = trainer_cls
    return trainer_cls

__all__ = [
    "BaseTrainer", 
    "HierarchicalPPOTrainer", 
    "SB3PPOTrainer", 
    "DiscretePPOTrainer", 
    "MAPPOTrainer",
    "TD3JALTrainer",
    "HSMSB3PPOTrainer",
    "QLearningTrainer"
]

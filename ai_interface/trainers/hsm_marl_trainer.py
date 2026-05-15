"""HSM-MARL training entry point.

The dedicated HSM multi-agent environment is not present in this codebase yet,
so this trainer keeps the advertised entry point runnable by using the MAPPO
rollout/training backend with the HSM-MARL config values.
"""

from __future__ import annotations

from typing import Any, Dict

from .mappo_trainer import MAPPOTrainer


class HSMMARLTrainer(MAPPOTrainer):
    """Compatibility trainer for HSM-MARL configurations."""

    def __init__(self, config: Dict[str, Any], log_dir: str = None, device=None):
        super().__init__(
            config,
            log_dir=log_dir,
            device=device,
            algorithm_name="hsm_marl",
        )
        self.logger.info(
            "HSM-MARL entry point is using the MAPPO parallel rollout backend."
        )

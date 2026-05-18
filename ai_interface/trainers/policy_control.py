"""Reusable policy-controller helpers for staged training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

import json

import torch
from stable_baselines3 import TD3

from ai_interface.envs.JAL_env import JALTeamEnv
from ai_interface.envs.JAL_her_env import JALHEREnv
from networking.data_utils import GameState
from networking.networker import Networker


def load_json_config(value: Any) -> Dict[str, Any]:
    """Load a JSON config from a path or return an existing mapping."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, Path)):
        with open(value, "r") as f:
            return json.load(f)
    raise TypeError(f"Unsupported policy config type: {type(value)!r}")


@runtime_checkable
class TeamCommandProvider(Protocol):
    """Protocol for anything that can emit a command list for one team."""

    team_name: str
    num_robots: int

    def predict_commands(self, game_state: GameState) -> list[str]:
        raise NotImplementedError


@dataclass(slots=True)
class FrozenTD3JALPolicySpec:
    """Configuration for loading a frozen TD3 JAL policy for inference."""

    name: str
    model_path: str
    team_name: str
    robot_ids: list[int]
    obs_mode: str = "her"
    obs_dim_per_robot: int = 8
    non_robot_obs_dim: int = 4
    her_distance_threshold: float = 0.1
    max_steps: int = 200
    debug: bool = False
    deterministic: bool = True


class FrozenTD3JALPolicyController:
    """Loads a frozen TD3 policy and emits commands for a single team."""

    def __init__(
        self,
        spec: FrozenTD3JALPolicySpec,
        networker: Networker,
        device: torch.device | str,
    ):
        self.spec = spec
        self.team_name = spec.team_name
        self.num_robots = len(spec.robot_ids)

        helper_env_cls = JALHEREnv if str(spec.obs_mode).lower() == "her" else JALTeamEnv
        helper_env_kwargs: Dict[str, Any] = {
            "networker": networker,
            "team_name": spec.team_name,
            "robot_ids": list(spec.robot_ids),
            "obs_dim_per_robot": int(spec.obs_dim_per_robot),
            "non_robot_obs_dim": int(spec.non_robot_obs_dim),
            "max_steps": int(spec.max_steps),
            "debug": bool(spec.debug),
        }
        if helper_env_cls is JALHEREnv:
            helper_env_kwargs["her_distance_threshold"] = float(spec.her_distance_threshold)

        self.helper_env = helper_env_cls(**helper_env_kwargs)
        self.model = TD3.load(spec.model_path, env=self.helper_env, device=str(device))

    def predict_commands(self, game_state: GameState) -> list[str]:
        obs = self.helper_env._game_state_to_obs(game_state)
        action, _ = self.model.predict(obs, deterministic=self.spec.deterministic)
        commands, _ = self.helper_env._action_to_commands(action, game_state)
        return commands


def build_aux_team_command_providers(
    policy_config: Any,
    networker: Networker,
    device: torch.device | str,
    stage_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, TeamCommandProvider]:
    """Build named auxiliary team controllers from JSON config.

    Config format:
    {
      "policy_catalog": {
        "stage2_policy": {
          "model_path": "models/stage2.zip",
          "team_name": "OpponentTeam",
          "robot_ids": [1],
          "obs_mode": "her"
        }
      }
    }

    A stage can then refer to entries by name via "aux_team_policies": ["stage2_policy"].
    It can also inline full policy specs directly in the stage config.
    """
    config = load_json_config(policy_config)
    catalog = dict(config.get("policy_catalog", {}))

    if stage_config is None:
        stage_config = {}

    stage_specs = stage_config.get("aux_team_policies", [])
    controllers: Dict[str, TeamCommandProvider] = {}

    for raw_spec in stage_specs:
        if isinstance(raw_spec, str):
            if raw_spec not in catalog:
                raise KeyError(f"Unknown policy reference '{raw_spec}'")
            spec_data = dict(catalog[raw_spec])
            spec_data.setdefault("name", raw_spec)
        elif isinstance(raw_spec, Mapping):
            spec_data = dict(raw_spec)
            spec_data.setdefault("name", spec_data.get("team_name", "aux_policy"))
        else:
            raise TypeError(f"Unsupported aux_team_policies entry: {type(raw_spec)!r}")

        required_keys = ["model_path", "team_name", "robot_ids"]
        missing = [key for key in required_keys if key not in spec_data]
        if missing:
            raise ValueError(f"Policy spec '{spec_data.get('name', '<unnamed>')}' is missing keys: {missing}")

        spec = FrozenTD3JALPolicySpec(
            name=str(spec_data["name"]),
            model_path=str(spec_data["model_path"]),
            team_name=str(spec_data["team_name"]),
            robot_ids=list(spec_data["robot_ids"]),
            obs_mode=str(spec_data.get("obs_mode", "her")),
            obs_dim_per_robot=int(spec_data.get("obs_dim_per_robot", 8)),
            non_robot_obs_dim=int(spec_data.get("non_robot_obs_dim", 4)),
            her_distance_threshold=float(spec_data.get("her_distance_threshold", 0.1)),
            max_steps=int(spec_data.get("max_steps", 200)),
            debug=bool(spec_data.get("debug", False)),
            deterministic=bool(spec_data.get("deterministic", True)),
        )
        controllers[spec.team_name] = FrozenTD3JALPolicyController(spec, networker, device=device)

    return controllers
"""Reusable policy-controller helpers for staged training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

import json

import torch
from stable_baselines3 import TD3

import numpy as np

from ai_interface.naive import SoccerAI as HeuristicSoccerAI
from ai_interface.envs.JAL_env import JALTeamEnv
from ai_interface.envs.JAL_her_env import JALHEREnv
from ai_interface.algorithms.ppo_jal import PPOJALAgent
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


@dataclass(slots=True)
class FrozenPPOJALPolicySpec:
    """Configuration for loading a frozen PPO JAL policy for inference."""

    name: str
    model_path: str
    team_name: str
    robot_ids: list[int]
    a_max: int = 5
    c_max: int = 7
    global_dim: int = 6
    per_agent_dim: int = 10
    d_ctx: int = 7
    num_primitives: int = 8
    param_dim: int = 5
    max_steps: int = 200
    deterministic: bool = True


class FrozenPPOJALPolicyController:
    """Loads a frozen PPO JAL policy and emits commands for a single team."""

    def __init__(
        self,
        spec: FrozenPPOJALPolicySpec,
        networker: Networker,
        device: torch.device | str,
    ):
        self.spec = spec
        self.team_name = spec.team_name
        self.num_robots = len(spec.robot_ids)

        obs_dim = spec.global_dim + spec.per_agent_dim * spec.a_max + spec.d_ctx * spec.c_max
        self.agent = PPOJALAgent(
            obs_dim=obs_dim,
            num_robots=self.num_robots,
            a_max=spec.a_max,
            c_max=spec.c_max,
            global_dim=spec.global_dim,
            per_agent_dim=spec.per_agent_dim,
            d_ctx=spec.d_ctx,
            num_primitives=spec.num_primitives,
            param_dim=spec.param_dim,
            device=device,
        )
        self.agent.load(spec.model_path)
        self.agent.model.eval()

        self.helper_env = JALTeamEnv(
            networker=networker,
            team_name=spec.team_name,
            robot_ids=list(spec.robot_ids),
            a_max=spec.a_max,
            c_max=spec.c_max,
            global_dim=spec.global_dim,
            per_agent_dim=spec.per_agent_dim,
            d_ctx=spec.d_ctx,
            max_steps=spec.max_steps,
        )

        self._agent_active_mask = np.zeros(spec.a_max, dtype=np.float32)
        self._agent_active_mask[: self.num_robots] = 1.0
        self._context_active_mask = self.helper_env.context_active_mask.copy()

    def predict_commands(self, game_state: GameState) -> list[str]:
        obs = self.helper_env._game_state_to_obs(game_state)
        action, _ = self.agent.sample_action(
            obs,
            agent_active_mask=self._agent_active_mask,
            context_active_mask=self._context_active_mask,
            deterministic=self.spec.deterministic,
        )
        commands, _ = self.helper_env._action_to_commands(action, game_state)
        return commands


class GoalieCommandProvider:
    """Drive a single scripted Goalie robot via the Goalie state machine."""

    def __init__(self, team_name: str, robot_id: int, side: str = "right"):
        from ai_interface.goalie import Goalie
        self.team_name = team_name
        self.robot_id = int(robot_id)
        self.num_robots = 1
        self._goalie = Goalie(teamname=team_name, unum=self.robot_id, side=side)

    def predict_commands(self, game_state: GameState) -> list[str]:
        if game_state is None:
            return ["turn 0"]
        ball_pos = getattr(game_state, "ball_pos", None)
        if ball_pos is None:
            return ["turn 0"]
        goalie_pose = None
        for entry in getattr(game_state, "robot_poses", {}).get(self.team_name, []):
            if not isinstance(entry, dict):
                continue
            pose = entry.get(self.robot_id)
            if pose is not None and len(pose) >= 3:
                goalie_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
                break
        if goalie_pose is None:
            return ["turn 0"]
        cmd = self._goalie.action(
            ball_pos=ball_pos,
            goalie_pose=goalie_pose,
            game_state=game_state,
        )
        return [cmd]


class DefenderCommandProvider:
    """Drive a single scripted goal-side container defender (Stage 3).

    Unlike the goalie (robot 1), the defender may sit at any robot_id, so the
    returned command list is padded with leading `None`s to place the command at
    the defender's 1-indexed unum slot (the simulator routes list position →
    unum and skips `None` entries), leaving the goalie's own slot untouched.
    """

    def __init__(self, team_name: str, robot_id: int, side: str = "right"):
        from ai_interface.defender import Defender
        self.team_name = team_name
        self.robot_id = int(robot_id)
        self.num_robots = 1
        self._defender = Defender(teamname=team_name, unum=self.robot_id, side=side)

    def _slot_list(self, cmd: str) -> list:
        return [None] * (self.robot_id - 1) + [cmd]

    def predict_commands(self, game_state: GameState) -> list:
        if game_state is None:
            return self._slot_list("turn 0")
        ball_pos = getattr(game_state, "ball_pos", None)
        if ball_pos is None:
            return self._slot_list("turn 0")
        defender_pose = None
        for entry in getattr(game_state, "robot_poses", {}).get(self.team_name, []):
            if not isinstance(entry, dict):
                continue
            pose = entry.get(self.robot_id)
            if pose is not None and len(pose) >= 3:
                defender_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
                break
        if defender_pose is None:
            return self._slot_list("turn 0")
        cmd = self._defender.action(
            ball_pos=ball_pos,
            defender_pose=defender_pose,
            game_state=game_state,
        )
        return self._slot_list(cmd)


class MarkerDefenderCommandProvider:
    """Drive a scripted marker/cover defender (Stage 4+).

    Complements the ball-challenging DefenderCommandProvider by marking
    receivers and covering pass lanes.
    """

    def __init__(
        self,
        team_name: str,
        robot_id: int,
        side: str = "right",
        ball_defender_robot_id: int = 2,
    ):
        from ai_interface.marker_defender import MarkerDefender
        self.team_name = team_name
        self.robot_id = int(robot_id)
        self.num_robots = 1
        self._defender = MarkerDefender(
            teamname=team_name,
            unum=self.robot_id,
            side=side,
            ball_defender_robot_id=ball_defender_robot_id,
        )

    def _slot_list(self, cmd: str) -> list:
        return [None] * (self.robot_id - 1) + [cmd]

    def predict_commands(self, game_state: GameState) -> list:
        if game_state is None:
            return self._slot_list("turn 0")
        ball_pos = getattr(game_state, "ball_pos", None)
        if ball_pos is None:
            return self._slot_list("turn 0")
        defender_pose = None
        for entry in getattr(game_state, "robot_poses", {}).get(self.team_name, []):
            if not isinstance(entry, dict):
                continue
            pose = entry.get(self.robot_id)
            if pose is not None and len(pose) >= 3:
                defender_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
                break
        if defender_pose is None:
            return self._slot_list("turn 0")
        cmd = self._defender.action(
            ball_pos=ball_pos,
            defender_pose=defender_pose,
            game_state=game_state,
        )
        return self._slot_list(cmd)


class ScriptedTeamCommandProvider:
    """Wrap a hard-coded AI implementation behind the TeamCommandProvider protocol."""

    def __init__(self, team_infos: list[Any], team_name: str, num_robots: int, controller_type: str):
        controller_type = str(controller_type).lower()
        if controller_type not in {"naive", "scripted_naive"}:
            raise ValueError(f"Unsupported scripted controller_type: {controller_type!r}")

        self.team_name = team_name
        self.num_robots = int(num_robots)
        self.controller_type = controller_type
        self._ai = HeuristicSoccerAI(team_infos)

    def predict_commands(self, game_state: GameState) -> list[str]:
        actions = self._ai.decide_action(game_state, self.team_name)
        return self._ai.translate_ai_output(actions)


class MixedTeamCommandProvider:
    """Compose a team controller from a frozen policy for some robots and a
    scripted controller for others.

    Config expects a full `robot_ids` ordering for the team and two sub-specs:
      - `frozen_spec`: dict acceptable to FrozenTD3JALPolicySpec (model_path, robot_ids, ...)
      - `scripted_spec`: dict acceptable to ScriptedTeamCommandProvider (team_name, robot_ids, controller_type)

    predict_commands() returns a list of commands ordered according to the
    provided `robot_ids` so it can be passed directly to Networker.execute_ai_output().
    """

    def __init__(
        self,
        team_name: str,
        robot_ids: list[int],
        frozen_spec: Optional[Dict[str, Any]],
        scripted_spec: Optional[Dict[str, Any]],
        networker: Networker,
        device: torch.device | str,
    ):
        self.team_name = team_name
        self.robot_ids = list(robot_ids)

        self.frozen_ctrl: Optional[FrozenTD3JALPolicyController] = None
        if frozen_spec:
            fs = dict(frozen_spec)
            # Ensure robot_ids list exists on frozen_spec
            if "robot_ids" not in fs:
                raise ValueError("frozen_spec must include 'robot_ids' list")
            frozen_dt = FrozenTD3JALPolicySpec(
                name=fs.get("name", "frozen"),
                model_path=str(fs["model_path"]),
                team_name=str(fs.get("team_name", team_name)),
                robot_ids=list(fs["robot_ids"]),
                obs_mode=str(fs.get("obs_mode", "her")),
                obs_dim_per_robot=int(fs.get("obs_dim_per_robot", 8)),
                non_robot_obs_dim=int(fs.get("non_robot_obs_dim", 4)),
                her_distance_threshold=float(fs.get("her_distance_threshold", 0.1)),
                max_steps=int(fs.get("max_steps", 200)),
                debug=bool(fs.get("debug", False)),
                deterministic=bool(fs.get("deterministic", True)),
            )
            self.frozen_ctrl = FrozenTD3JALPolicyController(frozen_dt, networker, device=device)

        self.scripted_ctrl: Optional[ScriptedTeamCommandProvider] = None
        if scripted_spec:
            ss = dict(scripted_spec)
            if "robot_ids" not in ss:
                raise ValueError("scripted_spec must include 'robot_ids' list")
            controller_type = ss.get("controller_type", "scripted_naive")
            # build ScriptedTeamCommandProvider using networker team infos
            team_infos = _networker_team_infos(networker)
            self.scripted_ctrl = ScriptedTeamCommandProvider(
                team_infos=team_infos,
                team_name=str(ss.get("team_name", team_name)),
                num_robots=len(ss["robot_ids"]),
                controller_type=controller_type,
            )
        self._frozen_ids = list(frozen_spec["robot_ids"]) if frozen_spec else []
        self._scripted_ids = list(scripted_spec["robot_ids"]) if scripted_spec else []

    def predict_commands(self, game_state: GameState) -> list[str]:
        # Collect commands from each sub-controller and map them to robot ids.
        cmd_map: dict[int, str] = {}

        if self.frozen_ctrl is not None:
            frozen_cmds = self.frozen_ctrl.predict_commands(game_state)
            # frozen_ctrl was constructed with its own robot_ids; zip accordingly
            for rid, cmd in zip(self._frozen_ids, frozen_cmds):
                cmd_map[int(rid)] = cmd

        if self.scripted_ctrl is not None:
            scripted_cmds = self.scripted_ctrl.predict_commands(game_state)
            for rid, cmd in zip(self._scripted_ids, scripted_cmds):
                cmd_map[int(rid)] = cmd

        # Fill missing robot commands with a safe no-op (turn 0)
        merged: list[str] = []
        for rid in self.robot_ids:
            merged.append(cmd_map.get(int(rid), "turn 0"))

        return merged


def _networker_team_infos(networker: Networker) -> list[Any]:
    commander = getattr(networker, "commander", None)
    if commander is not None and hasattr(commander, "team_infos"):
        return list(commander.team_infos)
    if hasattr(networker, "team_infos"):
        return list(networker.team_infos)
    raise AttributeError("Networker does not expose team information for scripted controllers")


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
    team_infos = _networker_team_infos(networker)

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

        controller_type = str(spec_data.get("controller_type", "frozen_td3")).lower()
        if controller_type == "mixed":
            required_keys = ["team_name", "robot_ids"]
            missing = [key for key in required_keys if key not in spec_data]
            if missing:
                raise ValueError(f"Mixed policy spec '{spec_data.get('name', '<unnamed>')}' is missing keys: {missing}")

            team_name = str(spec_data["team_name"])
            robot_ids = list(spec_data["robot_ids"])
            frozen_spec = spec_data.get("frozen_spec")
            scripted_spec = spec_data.get("scripted_spec")
            controllers[team_name] = MixedTeamCommandProvider(
                team_name=team_name,
                robot_ids=robot_ids,
                frozen_spec=frozen_spec,
                scripted_spec=scripted_spec,
                networker=networker,
                device=device,
            )
            continue
        if controller_type in {"naive", "scripted_naive"}:
            required_keys = ["team_name", "robot_ids"]
            missing = [key for key in required_keys if key not in spec_data]
            if missing:
                raise ValueError(f"Scripted policy spec '{spec_data.get('name', '<unnamed>')}' is missing keys: {missing}")

            team_name = str(spec_data["team_name"])
            robot_ids = list(spec_data["robot_ids"])
            controllers[team_name] = ScriptedTeamCommandProvider(
                team_infos=team_infos,
                team_name=team_name,
                num_robots=len(robot_ids),
                controller_type=controller_type,
            )
            continue

        if controller_type == "frozen_ppo":
            required_keys = ["model_path", "team_name", "robot_ids"]
            missing = [key for key in required_keys if key not in spec_data]
            if missing:
                raise ValueError(f"Frozen PPO spec '{spec_data.get('name', '<unnamed>')}' is missing keys: {missing}")

            ppo_spec = FrozenPPOJALPolicySpec(
                name=str(spec_data.get("name", "frozen_ppo")),
                model_path=str(spec_data["model_path"]),
                team_name=str(spec_data["team_name"]),
                robot_ids=list(spec_data["robot_ids"]),
                a_max=int(spec_data.get("a_max", 5)),
                c_max=int(spec_data.get("c_max", 7)),
                global_dim=int(spec_data.get("global_dim", 6)),
                per_agent_dim=int(spec_data.get("per_agent_dim", 10)),
                d_ctx=int(spec_data.get("d_ctx", 7)),
                num_primitives=int(spec_data.get("num_primitives", 8)),
                param_dim=int(spec_data.get("param_dim", 5)),
                max_steps=int(spec_data.get("max_steps", 200)),
                deterministic=bool(spec_data.get("deterministic", True)),
            )
            controllers[ppo_spec.team_name] = FrozenPPOJALPolicyController(ppo_spec, networker, device=device)
            continue

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
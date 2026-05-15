"""Helpers for creating simulator-backed parallel training environments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym

from networking.networker import Networker, TeamInfo


def make_networked_env_factory(
    env_cls: type[gym.Env],
    team_infos: Sequence[TeamInfo],
    env_mode: str,
    team_name: str,
    sim_host: str,
    sim_player_port: int,
    sim_trainer_port: int,
    env_kwargs: dict[str, Any] | None = None,
    monitor: bool = False,
) -> Callable[[], gym.Env]:
    """Return a picklable factory for one simulator-backed Gym environment."""
    team_infos = list(team_infos)
    env_kwargs = dict(env_kwargs or {})

    def _init() -> gym.Env:
        networker = Networker(
            team_infos,
            env_mode,
            sim_host=sim_host,
            sim_player_port=sim_player_port,
            sim_trainer_port=sim_trainer_port,
        )
        env = env_cls(networker=networker, team_name=team_name, **env_kwargs)
        if monitor:
            from stable_baselines3.common.monitor import Monitor

            env = Monitor(env)
        return env

    return _init


def make_vec_env(
    env_fns: Sequence[Callable[[], gym.Env]],
    backend: str = "subproc",
    start_method: str | None = None,
):
    """Create an SB3 VecEnv, using subprocesses when more than one env is requested."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if not env_fns:
        raise ValueError("At least one environment factory is required")

    backend = (backend or "subproc").lower()
    if len(env_fns) == 1 or backend == "dummy":
        return DummyVecEnv(list(env_fns))
    if backend != "subproc":
        raise ValueError("parallel_backend must be 'subproc' or 'dummy'")
    return SubprocVecEnv(list(env_fns), start_method=start_method)

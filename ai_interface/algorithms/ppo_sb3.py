import typing as _typing
import gym
from stable_baselines3 import PPO

from .base import AlgorithmBase


class PPO_SB3(AlgorithmBase):
    """Thin wrapper around stable-baselines3.PPO exposing a consistent API.

    This wrapper simply forwards calls to SB3's PPO so it can be used
    interchangeably with custom algorithm implementations.
    """

    def __init__(self, env: gym.Env, model_params: _typing.Optional[dict] = None):
        params = model_params or {}
        self._model = PPO("MlpPolicy", env, **params)

    def predict(self, observation, deterministic: bool = True):
        return self._model.predict(observation, deterministic=deterministic)

    def learn(self, total_timesteps: int):
        return self._model.learn(total_timesteps=total_timesteps)

    def save(self, path: str):
        return self._model.save(path)

    @classmethod
    def load(cls, path: str, env=None):
        m = PPO.load(path, env=env)
        inst = object.__new__(cls)
        inst._model = m
        return inst

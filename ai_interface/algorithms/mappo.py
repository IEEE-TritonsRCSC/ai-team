import typing as _typing
import abc
import gym

from .base import AlgorithmBase


class MAPPO(AlgorithmBase):
    """Placeholder for a MAPPO implementation.

    This class provides the same API as `AlgorithmBase` but does not implement
    a concrete algorithm. Add your MAPPO implementation here or subclass this
    to integrate a third-party MAPPO.
    """

    def __init__(self, env: gym.Env, model_params: _typing.Optional[dict] = None):
        # store env/params for a future implementation
        self.env = env
        self.model_params = model_params or {}
        raise NotImplementedError("MAPPO is a placeholder. Implement or plug your MAPPO class here.")

    def predict(self, observation, deterministic: bool = True):
        raise NotImplementedError()

    def learn(self, total_timesteps: int):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str, env=None):
        raise NotImplementedError()

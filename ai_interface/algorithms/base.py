import abc
import typing as _typing

class AlgorithmBase(abc.ABC):
    """Abstract base class for RL algorithm implementations.

    Implementations must provide the minimal API used by `AI_Trainer`:
    - predict(observation, deterministic=True)
    - learn(total_timesteps: int)
    - save(path)
    - load(path, env=None)
    """

    @abc.abstractmethod
    def predict(self, observation, deterministic: bool = True):
        raise NotImplementedError()

    @abc.abstractmethod
    def learn(self, total_timesteps: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, path: str):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str, env=None):
        raise NotImplementedError()

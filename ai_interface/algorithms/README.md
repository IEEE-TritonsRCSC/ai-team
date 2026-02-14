# Adding a new algorithm

This directory contains algorithm implementations that the training code uses.

Location
- Place algorithm implementations in this directory: ai_interface/algorithms/

Required interface
- Subclass `AlgorithmBase` (see `base.py`) and implement:
  - `predict(observation, deterministic: bool = True)` → action or action distribution
  - `learn(total_timesteps: int)` → run learning/updating for the specified timesteps
  - `save(path: str)` → persist model artifacts
  - `@classmethod load(cls, path: str, env=None)` → restore model; `env` optional for some libraries

Naming and layout
- File names should be lower_snake_case (e.g. `my_algo.py`).
- Class names should be `CamelCase` and clearly describe the algorithm (e.g. `MyAlgo`).

Minimal skeleton
```python
from ai_interface.algorithms.base import AlgorithmBase

class MyAlgo(AlgorithmBase):
    def __init__(self, *args, **kwargs):
        # initialize internal model/params
        pass

    def predict(self, observation, deterministic: bool = True):
        # return action (or action, state)
        return 0

    def learn(self, total_timesteps: int):
        # training loop / call into underlying library
        pass

    def save(self, path: str):
        # persist weights/config
        pass

    @classmethod
    def load(cls, path: str, env=None):
        # restore and return instance
        return cls()
```

Testing locally
- Import and instantiate the class from a Python shell or unit test; ensure `predict` and `save`/`load` work as expected.

References
- See `base.py`, `discrete_ppo.py`, and `hier_ppo.py` for examples.

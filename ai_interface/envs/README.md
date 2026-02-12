# Adding a new environment

This directory contains environment implementations used by trainers. Environments are standard Gymnasium environments and should follow Gym's API.

Location
- Place environment implementations in: ai_interface/envs/

Required interface
- Subclass `gym.Env` and implement the core API:
  - `__init__(...)` — set `self.observation_space` and `self.action_space`
  - `reset()` → returns the initial observation (numpy array)
  - `step(action)` → returns `(obs, reward, done, info)`

Notes for this project
- Many environments in this repo interact with a `Networker` (see `networking/networker.py`). If your env talks to the simulator, accept a `networker` instance in `__init__`.
- Keep observation/action shapes and dtypes consistent with the spaces defined.

Minimal skeleton
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyEnv(gym.Env):
    def __init__(self, some_arg=None):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def reset(self):
        # initialize state
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        # apply action, advance sim, compute reward
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info
```

Testing locally
- Import `MyEnv`, call `env = MyEnv()`; run `obs = env.reset()` and `env.step(env.action_space.sample())` to smoke-test.

References
- See `ppo_env.py`, `mappo_env.py`, and `sim_env.py` for concrete examples.

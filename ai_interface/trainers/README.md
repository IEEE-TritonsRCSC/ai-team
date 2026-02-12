# Adding a new trainer

Trainers orchestrate environment setup, model creation, training loops, logging and saving. Use the `BaseTrainer` class as the canonical interface.

Location
- Place trainer implementations in: ai_interface/trainers/

Required interface
- Subclass `BaseTrainer` and implement these abstract methods:
  - `setup_environment(self) -> gym.Env` — return a configured environment
  - `setup_model(self, env: gym.Env)` — create the model/agent to train
  - `train(self)` — run the training loop
  - `save_model(self, path: str)` — persist model artifacts
  - `load_model(self, path: str)` — restore model

Useful helpers
- `BaseTrainer` provides logging setup (`self.logger`) and a per-run `run_dir`.
- Use `self.log_episode(episode, reward, length)` and `self.log_metrics(metrics)` to keep metrics consistent with other trainers.
- `plot_training()` can be used to save a reward plot into `run_dir`.

Minimal skeleton
```python
from ai_interface.trainers.base_trainer import BaseTrainer

class MyTrainer(BaseTrainer):
    def setup_environment(self):
        # return env instance
        return None

    def setup_model(self, env):
        # create and return model
        return None

    def train(self):
        # main training loop — call log_episode() regularly
        pass

    def save_model(self, path: str):
        pass

    def load_model(self, path: str):
        pass
```

Testing locally
- Instantiate your trainer with a config dict and a temporary `log_dir`, call `trainer.setup_environment()` and `trainer.setup_model(env)` to smoke-test initialization; run a short `trainer.train()` for an end-to-end check.

References
- See `base_trainer.py`, `discrete_ppo_trainer.py`, and `mappo_trainer.py` for implementations.

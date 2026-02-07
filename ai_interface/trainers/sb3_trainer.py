import typing as _typing
import gymnasium as gym
from stable_baselines3 import PPO
import torch

from ai_interface.algorithms.base import AlgorithmBase


class AI_Trainer:
    """Trainer wrapper focused on PPO (stable-baselines3) with a small
    extensibility point to plug a custom model class (e.g. MAPPO implementation).

    The trainer accepts either:
    - a class implementing `AlgorithmBase` (recommended for custom algos), or
    - a stable-baselines3-like class (such as `stable_baselines3.PPO`) which
      accepts signature `(policy, env, **params)` and exposes `learn`, `predict`,
      `save`, and `load`.
    """

    def __init__(self, env: _typing.Optional[gym.Env], model_params: _typing.Optional[dict] = None,
                 model_class: _typing.Optional[type] = None,
                            train_batch_size: int = 2048,
                            learn_timesteps_per_batch: int = 2048):
        """Initialize trainer.

        Args:
            env: gym.Env the environment the model will interact with.
            model_params: optional dict passed to the model constructor.
            model_class: optional class implementing AlgorithmBase or SB3-style.
            train_batch_size: number of collected transitions before calling learn.
            learn_timesteps_per_batch: timesteps to pass to model.learn when buffer full.
        """
        self.env = env
        self.train_batch_size = int(train_batch_size)
        self.learn_timesteps_per_batch = int(learn_timesteps_per_batch)
        self._buffer = []  # list of (state, action, reward, next_state, done)

        params = model_params or {}
        # Determine device and add to params if not explicitly set
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if "device" not in params:
            params = dict(params)
            params["device"] = device_str
        # default to SB3 PPO if no class provided
        if model_class is None:
            model_class = PPO

        # Save model_class and params; instantiate the concrete model only if
        # an environment was provided. This allows the caller (main) to create
        # a trainer placeholder and load a pre-trained model later without a
        # gym.Env present at construction time.
        self.model_class = model_class
        self._model_params = params

        if self.env is not None:
            # If it's a subclass of AlgorithmBase, instantiate with (env, params)
            try:
                is_algo_base = issubclass(model_class, AlgorithmBase)
            except Exception:
                is_algo_base = False

            if is_algo_base:
                # AlgorithmBase implementations may expect (env, model_params)
                self.model = model_class(env, model_params)
            else:
                # Assume SB3-style: (policy_str, env, **params)
                self.model = model_class("MlpPolicy", env, **params)
        else:
            self.model = None

    def train_step(self, state, action, reward, next_state, done):
        """Collect a single transition. When buffered transitions reach
        `train_batch_size`, call the model's `learn` for `learn_timesteps_per_batch`.

        For SB3 PPO the internal `learn` call will collect rollouts from the
        environment itself. The buffer exists so user code integrating with a
        step-wise simulator can call `train_step` each time a transition occurs
        and let `AI_Trainer` decide when to invoke learning.
        """
        self._buffer.append((state, action, reward, next_state, done))

        if len(self._buffer) >= self.train_batch_size:
            # If the model implements learn, call it.
            if self.model is None:
                raise RuntimeError("Model not instantiated: provide an env at construction or call load_model before training")

            if hasattr(self.model, "learn"):
                self.model.learn(total_timesteps=self.learn_timesteps_per_batch)
            else:
                # Custom models might implement a different learning entrypoint.
                learn_fn = getattr(self.model, "update_from_buffer", None)
                if callable(learn_fn):
                    learn_fn(self._buffer)
            # Clear buffer after attempting learn/update
            self._buffer.clear()

    def eval_step(self, state):
        """Return an action for `state` using the policy (deterministic).

        State should be in the format expected by the model (e.g., numpy array).
        """
        if self.model is None:
            raise RuntimeError("Model not instantiated: provide an env at construction or call load_model before inference")

        if hasattr(self.model, "predict"):
            action, _ = self.model.predict(state, deterministic=True)
            return action
        # fallback for custom models that implement `act`
        act_fn = getattr(self.model, "act", None)
        if callable(act_fn):
            return act_fn(state)
        raise RuntimeError("Model does not implement predict or act")

    def save_model(self, filepath: str):
        """Save the underlying model to `filepath`."""
        if hasattr(self.model, "save"):
            self.model.save(filepath)
        else:
            save_fn = getattr(self.model, "save_state", None)
            if callable(save_fn):
                save_fn(filepath)
            else:
                raise RuntimeError("Model does not implement a save method")

    def load_model(self, filepath: str, model_class: _typing.Optional[type] = None,
                        model_params: _typing.Optional[dict] = None):
        """Load a model from `filepath` and attach the current environment.

        If `model_class` is provided, it will be used to load; otherwise the
        `model_class` used at construction will be used.
        """
        mclass = model_class or self.model_class
        params = model_params or {}

        try:
            is_algo_base = issubclass(mclass, AlgorithmBase)
        except Exception:
            is_algo_base = False

        if is_algo_base:
            # AlgorithmBase implementations should provide a `load` classmethod
            self.model = mclass.load(filepath, env=self.env)
            self.model_class = mclass
        else:
            # Assume SB3-style
            self.model = mclass.load(filepath, env=self.env)
            self.model_class = mclass


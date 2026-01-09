"""Minimal gym.Env adapter for the project simulator.

This is a thin, test-friendly adapter that shows where to plug the real
`networker` interaction. It intentionally provides placeholder observation
and action spaces so you can run a smoke training session with SB3.

Replace `_game_state_to_obs` and `_action_to_commands` with real mappings for
your simulator when available.
"""
from typing import Optional
import gym
import numpy as np

from networking.networker import Networker


class SimulatorEnv(gym.Env):
    """Gym environment that wraps the project's `Networker`.

    Notes:
    - This implementation uses placeholder spaces and conversions for testing.
    - Replace the conversion functions with domain-specific logic later.
    """

    metadata = {"render.modes": []}

    def __init__(self, networker: Networker, team_name: str):
        super().__init__()
        self.networker = networker
        self.team_name = team_name

        # Placeholder shapes — change to match your true observation/action shapes
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        # Example discrete action space for testing; replace with actual action structure
        self.action_space = gym.spaces.Discrete(5)

    def _game_state_to_obs(self, game_state) -> np.ndarray:
        # TODO: map `game_state` to a numeric observation vector
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _action_to_commands(self, action) -> list:
        # Map discrete actions to valid simulator command strings understood
        # by `Serializer.sim_serialize`. The serializer will convert radians
        # to degrees and handle the 'skick' -> 'kick' rename.
        #
        # Command formats expected before serialization:
        # - "turn <rad_per_sec>" (will be converted using SIM_TIMESTEP)
        # - "dash <power> <direction_rad>"
        # - "skick <power> <direction_rad>" (serializer will strip 's')
        if self.action_space.contains(action):
            if action == 0:
                return ["turn 0.5"]  # gentle clockwise turn
            if action == 1:
                return ["turn -0.5"]  # gentle counter-clockwise turn
            if action == 2:
                return ["dash 1.0 0.0"]  # move forward with full power
            if action == 3:
                return ["dash 0.5 1.5708"]  # move right (approx +90°)
            if action == 4:
                return ["skick 0.8 0.0"]  # kick forward
        # Fallback: no-op
        return [None]

    def step(self, action):
        # Convert action to simulator commands and send them
        commands = self._action_to_commands(action)
        try:
            self.networker.execute_ai_output(commands, self.team_name)
        except Exception:
            # If networking isn't ready in tests, ignore
            pass

        # Observe next game state
        game_state = self.networker.get_game_state()
        obs = self._game_state_to_obs(game_state)

        # Placeholder reward/done values — implement meaningful signals
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def reset(self):
        # If your commander supports an explicit simulator reset, call it here.
        # Example: getattr(self.networker.commander, "reset_sim", lambda: None)()
        try:
            reset_fn = getattr(self.networker.commander, "reset_sim", None)
            if callable(reset_fn):
                reset_fn()
        except Exception:
            pass

        game_state = self.networker.get_game_state()
        return self._game_state_to_obs(game_state)

    def render(self, mode="human"):
        # Optional: implement visualization hooks
        pass

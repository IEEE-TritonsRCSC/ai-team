from typing import Dict, Tuple, Optional
import gym
import numpy as np

from networking.networker import Networker
from constants.field_constants import GOAL_L, GOAL_R


# -----------------------------
# Action definitions (LOCK THIS)
# -----------------------------
ACTION_DASH = 0
ACTION_TURN = 1
ACTION_KICK = 2

N_ACTIONS = 3


class SimulatorEnv(gym.Env):
    """
    Production v1 Gym environment for soccer simulator.
    Designed for hybrid (2-head) policies:
      - discrete: high-level intent
      - continuous: action parameter (power, angle, etc.)
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        networker: Networker,
        team_name: str,
        agent_id: int,
        attacking_goal: Tuple[float, float] = GOAL_R,
        max_steps: int = 300,
    ):
        super().__init__()

        self.networker = networker
        self.team_name = team_name
        self.agent_id = agent_id
        self.goal = np.array(attacking_goal, dtype=np.float32)
        self.max_steps = max_steps

        self.step_count = 0
        self._last_obs: Optional[np.ndarray] = None

        # -----------------
        # Observation space
        # -----------------
        # Relative geometry + heading
        # [ball_dx, ball_dy, goal_dx, goal_dy, cos(theta), sin(theta)]
        obs_high = np.array([105.0, 68.0, 105.0, 68.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # -----------------
        # Action space (2-head)
        # -----------------
        self.action_space = gym.spaces.Dict(
            {
                "discrete": gym.spaces.Discrete(N_ACTIONS),
                # Single scalar parameter ∈ [0, 1]
                "continuous": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )

    # =========================
    # Core Gym API
    # =========================

    def reset(self):
        self.step_count = 0

        # Optional simulator reset hook
        reset_fn = getattr(self.networker.commander, "reset_sim", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception:
                pass

        game_state = self.networker.get_game_state()
        obs = self._build_observation(game_state)
        self._last_obs = obs
        return obs

    def step(self, action: Dict):
        commands = self._action_to_commands(action)

        try:
            self.networker.execute_ai_output(commands, self.team_name)
        except Exception:
            # Networking failures should not crash training
            pass

        game_state = self.networker.get_game_state()
        obs = self._build_observation(game_state)

        reward, done = self._compute_reward_done(game_state)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        self._last_obs = obs
        info = {}

        return obs, reward, done, info

    # =========================
    # Observation
    # =========================

    def _build_observation(self, game_state) -> np.ndarray:
        """
        Convert GameState → relative observation vector.
        """
        try:
            ball_x, ball_y = game_state.ball_pos

            pose = game_state.robot_poses[self.team_name][self.agent_id]
            agent_x, agent_y, agent_theta_deg = pose

            # Relative geometry
            ball_dx = ball_x - agent_x
            ball_dy = ball_y - agent_y
            goal_dx = self.goal[0] - agent_x
            goal_dy = self.goal[1] - agent_y

            theta_rad = np.deg2rad(agent_theta_deg)
            obs = np.array(
                [
                    ball_dx,
                    ball_dy,
                    goal_dx,
                    goal_dy,
                    np.cos(theta_rad),
                    np.sin(theta_rad),
                ],
                dtype=np.float32,
            )
            return obs

        except Exception:
            # Fallback for robustness
            if self._last_obs is not None:
                return self._last_obs
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    # =========================
    # Action decoding
    # =========================

    def _action_to_commands(self, action: Dict) -> list[str]:
        """
        Translate high-level intent → simulator command strings.
        """
        a = int(action["discrete"])
        p = float(np.clip(action["continuous"][0], 0.0, 1.0))

        if a == ACTION_DASH:
            # Dash forward with power-scaled magnitude
            return [f"dash {100.0 * p} 0"]

        if a == ACTION_TURN:
            # Turn in place: map p ∈ [0,1] → [-180, 180]
            angle = (p * 360.0) - 180.0
            return [f"turn {angle}"]

        if a == ACTION_KICK:
            # Kick straight ahead with power
            return [f"kick {100.0 * p} 0"]

        return []

    # =========================
    # Reward & termination
    # =========================

    def _compute_reward_done(self, game_state) -> Tuple[float, bool]:
        """
        Minimal, semantic reward.
        """
        reward = -0.001  # time penalty
        done = False

        # Example terminal condition (replace with real check)
        if self._ball_in_goal(game_state):
            reward = 1.0
            done = True

        return reward, done

    def _ball_in_goal(self, game_state) -> bool:
        # Placeholder: implement real goal detection
        return False

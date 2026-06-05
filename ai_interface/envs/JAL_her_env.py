"""HER-compatible wrapper around JALTeamEnv.

Converts the flat Box observation space to the GoalEnv-style Dict required
by SB3's HerReplayBuffer:
    {
        "observation":   Box(obs_dim)   — same flat obs as JALTeamEnv
        "achieved_goal": Box(2)         — normalized ball position in [-1, 1]
        "desired_goal":  Box(2)         — opponent goal centre [1.0, 0.0] by default
    }

Three methods are overridden relative to JALTeamEnv:

    This wrapper is backend-agnostic: it works with the regular socket-based
    simulator path and with the embedded simulator backend selected by
    ``env_mode = "sim-embedded"``.

  _game_state_to_obs  — returns a dict instead of a flat array; parent's
                        reset() and step() both call this, so returning a
                        dict here is enough to make both hand back HER-style
                        obs without double-fetching state or corrupting
                        ball_pos_history / prev_robot_pose_by_id.

  _calculate_reward   — extends the parent's dense reward with a
                        goal-conditioned progress term (ball toward
                        _desired_goal in normalized coords) plus a bonus
                        when the ball enters the her_distance_threshold.
                        Uses the same normalized space as compute_reward,
                        keeping live and offline rewards coherent.

  reset               — calls super().reset() then clears
                        _prev_ball_to_desired_dist so the progress term
                        starts fresh each episode.

  compute_reward      — stateless sparse reward called by HerReplayBuffer
                        offline to relabel virtual transitions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from ai_interface.envs.JAL_env import JALTeamEnv
from networking.data_utils import GameState


# Weights for the HER-specific reward terms added on top of the parent reward.
_HER_GOAL_PROGRESS_WEIGHT: float = 2.0   # scales ball-toward-desired-goal progress
_HER_GOAL_PROGRESS_CLIP:   float = 0.3   # max/min clamp on that progress per step
_HER_GOAL_REACHED_BONUS:   float = 0.0   # disabled — see note below
# Disabled 2026-05-19: the +10 bonus was firing whenever the ball came within
# 0.1 (normalized) of the desired goal (x=1.0, y=0.0). In raw field units that
# is a ~4.5×3.0 ellipse around (45, 0) — covering the entire penalty area in
# front of the goal mouth, not just the goal itself. On the 200k stage1_180
# run the off-target-but-close kicks (ball ending at e.g. (44, 4.5)) were
# being reinforced as "near-goal achievement" worth +10, on top of the
# parent's +150 goal reward only firing for the true 5-unit-half-height
# mouth. Net effect: a strong gradient toward "kick into the corner of the
# penalty area" — exactly the failure mode we observed (75% off-target).
# Killing this bonus removes the false attractor; goal reward stays sharp.


class JALHEREnv(JALTeamEnv):
    """JALTeamEnv with a GoalEnv-style Dict observation space and HER reward."""

    def __init__(
        self,
        her_distance_threshold: float = 0.1,
        aux_team_command_providers: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Replace the flat Box observation_space with the HER-required Dict.
        # `obs_dim` is already set by the parent __init__.
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.obs_dim,),
                dtype=np.float32,
            ),
            "achieved_goal": spaces.Box(
                low=-1.0, high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
            "desired_goal": spaces.Box(
                low=-1.0, high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
        })

        # Default desired goal: opponent goal centre in normalized field coords.
        # Field x: [-45, 45] → opponent goal at x=45 → normalized = 1.0
        # Goal centred at y=0 → normalized = 0.0
        self._desired_goal = np.array([1.0, 0.0], dtype=np.float32)

        # Distance below which a relabeled "virtual goal" counts as achieved
        # (L2, in normalized coords; field spans [-1, 1] on both axes).
        self.her_distance_threshold = float(her_distance_threshold)

        # Tracks previous ball distance to _desired_goal (normalized) for the
        # progress shaping term in _calculate_reward. Reset each episode.
        self._prev_ball_to_desired_dist: Optional[float] = None
        self.aux_team_command_providers = dict(aux_team_command_providers or {})

    # ------------------------------------------------------------------
    # reset — clear the extra HER episode state
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_ball_to_desired_dist = None
        return obs, info

    # ------------------------------------------------------------------
    # step — send frozen-policy teammates (if any), then delegate to parent
    #
    # The ONLY thing this override adds over JALTeamEnv.step() is sending
    # commands for auxiliary frozen/scripted teams in the same simulator
    # cycle as the learner's own commands. Everything else — goal detection
    # via _check_terminal, the kick-aim bonus, ball-dead / max_steps
    # penalties, "Episode N ended" logging, single-fetch cycle reuse via
    # _cached_game_state — is handled by the parent. A previous version of
    # this method re-implemented step() and hardcoded `terminated = False`,
    # so goals never ended an episode (every episode ran the full max_steps,
    # reward pinned at step_bonus*max_steps) and the policy collapsed to
    # inaction. Delegating to super() keeps those bugs from coming back.
    # ------------------------------------------------------------------

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Send frozen-policy teammate commands, then run the parent step."""
        if self.aux_team_command_providers:
            # Use the state the parent will reuse this cycle so the aux
            # commands are built from the same frame and we avoid a second
            # fetch (which would consume an extra simulator cycle).
            current_game_state = self._cached_game_state
            if current_game_state is None:
                current_game_state = self._get_game_state(
                    retries=self.state_retry_count,
                    sleep_s=self.state_retry_sleep_s,
                )
                self._cached_game_state = current_game_state

            aux_team_commands = self._build_aux_team_commands(current_game_state)
            for team_name, team_commands in aux_team_commands.items():
                self._send_team_commands(team_name, team_commands)

        return super().step(action)

    # ------------------------------------------------------------------
    # _game_state_to_obs override
    # Parent's reset() and step() build their returned obs by calling this.
    # Returning a dict here is what makes the env hand back HER-style obs.
    # ------------------------------------------------------------------

    def _game_state_to_obs(self, game_state):
        flat_obs = super()._game_state_to_obs(game_state)
        # `flat_obs` is always a 1D float32 array of length self.obs_dim,
        # even when game_state was None (parent returns zeros in that case).

        if game_state is not None and getattr(game_state, "ball_pos", None) is not None:
            ball_x = float(game_state.ball_pos[0])
            ball_y = float(game_state.ball_pos[1])
            achieved_goal = np.array([
                np.clip(ball_x / self.field_half_width,  -1.0, 1.0),
                np.clip(ball_y / self.field_half_height, -1.0, 1.0),
            ], dtype=np.float32)
        else:
            achieved_goal = np.zeros(2, dtype=np.float32)

        return {
            "observation":   flat_obs,
            "achieved_goal": achieved_goal,
            "desired_goal":  self._desired_goal.copy(),
        }

    def _send_team_commands(self, team_name: str, commands: List[str]):
        """Send commands for a specific team controller."""
        try:
            if not commands:
                return

            serialized_commands = self._preprocess_commands_for_send(commands)
            self.networker.execute_ai_output(serialized_commands, team_name)
        except Exception as e:
            self.logger.error("Error sending commands for team %s: %s", team_name, e)

    def _build_aux_team_commands(self, game_state: Optional[GameState]) -> Dict[str, List[str]]:
        """Build commands for any auxiliary frozen/scripted team controllers."""
        if game_state is None or not self.aux_team_command_providers:
            return {}

        aux_team_commands: Dict[str, List[str]] = {}
        for team_name, provider in self.aux_team_command_providers.items():
            if team_name == self.team_name:
                continue

            try:
                team_commands = provider.predict_commands(game_state)
                if team_commands:
                    aux_team_commands[team_name] = team_commands
            except Exception as e:
                self.logger.error("Aux team controller failed for %s: %s", team_name, e)
                fallback_count = int(getattr(provider, "num_robots", 0))
                if fallback_count > 0:
                    aux_team_commands[team_name] = ["turn 0"] * fallback_count

        return aux_team_commands

    # ------------------------------------------------------------------
    # _calculate_reward override — HER-extended live reward
    # ------------------------------------------------------------------

    def _calculate_reward(self, current_game_state) -> float:
        """Dense reward used during real (non-relabeled) environment steps.

        Calls the parent's reward for approach shaping, has-ball bonus,
        state bonuses, out-of-bounds penalties, and goal reward, then adds
        two HER-specific terms:

          goal_progress  — clipped change in ball distance to _desired_goal
                           in normalized field coordinates, weighted by
                           _HER_GOAL_PROGRESS_WEIGHT.  Positive when the
                           ball moves closer; negative when it moves away.
                           Using normalized coords keeps this term in the
                           same space as compute_reward, so live shaping
                           and offline HER relabeling are consistent.

          goal_reached   — a one-shot bonus (_HER_GOAL_REACHED_BONUS) when
                           the ball is within her_distance_threshold of
                           _desired_goal in normalized coords.  Distinct
                           from the parent's `goal_reward` (which fires on
                           the actual opponent goal in real coords) so the
                           agent receives a signal for any virtual goal the
                           curriculum might later set.
        """
        if current_game_state is None:
            return 0.0

        reward = super()._calculate_reward(current_game_state)

        if getattr(current_game_state, "ball_pos", None) is None:
            return reward

        ball_x = float(current_game_state.ball_pos[0])
        ball_y = float(current_game_state.ball_pos[1])
        ball_norm = np.array([
            ball_x / self.field_half_width,
            ball_y / self.field_half_height,
        ], dtype=np.float32)

        dist_to_desired = float(np.linalg.norm(ball_norm - self._desired_goal))

        # Goal-conditioned progress shaping
        if self._prev_ball_to_desired_dist is not None:
            progress = self._prev_ball_to_desired_dist - dist_to_desired
            reward += float(np.clip(progress, -_HER_GOAL_PROGRESS_CLIP, _HER_GOAL_PROGRESS_CLIP)) * _HER_GOAL_PROGRESS_WEIGHT

        self._prev_ball_to_desired_dist = dist_to_desired

        # Sparse bonus for reaching the desired goal region
        if dist_to_desired <= self.her_distance_threshold:
            reward += _HER_GOAL_REACHED_BONUS

        return reward

    # ------------------------------------------------------------------
    # compute_reward — called by HerReplayBuffer offline to relabel transitions
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info,
    ) -> np.ndarray:
        """Sparse reward used by HER for relabeled (virtual) transitions.

        Returns 0.0 if `achieved_goal` is within `self.her_distance_threshold`
        (L2 distance) of `desired_goal`, -1.0 otherwise.

        Stateless: this method runs offline, in batch, after episodes end —
        it cannot reference any per-episode state (current_step, prev_*, etc.).

        Critically, the reward depends on `desired_goal`. HER relabels stored
        transitions by replacing `desired_goal` with a *future* ball position
        from the same episode; the agent then "succeeds" whenever the ball
        actually reached close to that position. A `compute_reward` that
        ignores `desired_goal` (e.g. only checks "ball in opponent goal")
        carries no relabeling signal — virtual transitions all collapse to
        -1, and HER stops being useful.

        Real transitions still use the env's dense `_calculate_reward` from
        `step()` — SB3 only invokes this method for HER-relabeled samples.

        Args:
            achieved_goal: shape (..., 2) — normalized ball positions
            desired_goal:  shape (..., 2) — normalized goal targets
            info:          unused; required by the SB3 interface

        Returns:
            rewards: shape (...) — values in {0.0, -1.0}
        """
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired  = np.asarray(desired_goal,  dtype=np.float32)
        distance = np.linalg.norm(achieved - desired, axis=-1)
        return -(distance > self.her_distance_threshold).astype(np.float32)

"""PPO JAL Curriculum Trainer.

Mirrors `td3_jal_curriculum_trainer.py` but drives the PPO hybrid policy in
`ai_interface/algorithms/ppo_jal.py`. Key differences from the TD3 trainer:

  - On-policy: no replay buffer; PPO updates from a rollout buffer that is
    flushed after each update.
  - Custom training loop instead of `model.learn()` — collects transitions
    via env.step() until the rollout buffer is full, then calls update().
  - Action passed to env.step() is the PPO dict format
    {"primitive_idx": np.ndarray, "params": np.ndarray}.
  - No HER (PPO is on-policy; HER doesn't apply).

Per-stage `disabled_actions` is passed both to the env (for honest masking
sanity check + display) and to the policy's `sample_action()` (which applies
-inf masking to disabled primitive logits before sampling). The policy
literally cannot pick a disabled primitive.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ai_interface.algorithms.ppo_jal import PPOJALAgent, PRIMITIVE_NAMES, NUM_PRIMITIVES
from ai_interface.envs.JAL_env import JALTeamEnv
from ai_interface.trainers.base_trainer import BaseTrainer
from networking.networker import Networker, TeamInfo


class _RewardNormalizer:
    """SB3-VecNormalize-style reward scaling.

    Maintains a running estimate of the variance of the discounted return
    R_t = gamma · R_{t-1} + r_t, then divides each raw reward by sqrt(var).
    The discounted return computed from normalized rewards has std ≈ 1.0 by
    construction, so the value function's target is in unit-std scale and the
    value MSE loss stays well-conditioned even with rewards spanning -10 to
    +150. Crucially: does *not* subtract a running mean (preserves reward
    signs).

    Welford's online algorithm is used for the running variance so updates
    are O(1) per step and numerically stable.

    Reference: Engstrom et al. "Implementation Matters in Deep Policy
    Gradients" (2020), §3 — reward scaling via running discounted-return std.
    """

    def __init__(self, gamma: float, clip: float = 10.0, epsilon: float = 1e-8):
        self.gamma = float(gamma)
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        # Welford running stats over discounted-return values R_t
        self._mean = 0.0
        self._m2 = 0.0
        self._count = 0
        # Per-trajectory discounted return tracker (reset on done)
        self._return = 0.0

    @property
    def variance(self) -> float:
        if self._count < 2:
            return 1.0
        return self._m2 / (self._count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance) + self.epsilon

    def normalize(self, reward: float, done: bool) -> float:
        # Update discounted return tracker.
        self._return = self._return * self.gamma + float(reward)
        # Welford update on R_t.
        self._count += 1
        delta = self._return - self._mean
        self._mean += delta / self._count
        delta2 = self._return - self._mean
        self._m2 += delta * delta2
        # Reset tracker on episode end (terminated OR truncated — for stats
        # purposes both end the discounted-return chain).
        if done:
            self._return = 0.0
        # Normalize: divide raw reward by current std estimate, clip.
        normalized = reward / self.std
        if self.clip > 0:
            normalized = max(-self.clip, min(self.clip, normalized))
        return float(normalized)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "gamma": self.gamma,
            "clip": self.clip,
            "epsilon": self.epsilon,
            "mean": self._mean,
            "m2": self._m2,
            "count": self._count,
            "return": self._return,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.gamma = float(state.get("gamma", self.gamma))
        self.clip = float(state.get("clip", self.clip))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self._mean = float(state.get("mean", 0.0))
        self._m2 = float(state.get("m2", 0.0))
        self._count = int(state.get("count", 0))
        self._return = float(state.get("return", 0.0))


class PPOJALCurriculumTrainer(BaseTrainer):
    """Curriculum PPO trainer for JAL hybrid action space."""

    def __init__(self, config: Dict[str, Any], log_dir: str = None, device=None):
        super().__init__(config, log_dir, algorithm_name="ppo_jal_curriculum")
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.networker: Optional[Networker] = None
        self.env: Optional[JALTeamEnv] = None
        self.agent: Optional[PPOJALAgent] = None

        self.curriculum = config.get("curriculum", {})
        self._current_num_robots: Optional[int] = None
        self._current_obs_dim: Optional[int] = None

        # Episode counter spans all stages so log_episode() rolling averages
        # remain coherent across stage transitions.
        self._global_episode_count: int = 0

        # Reward normalizer (SB3-VecNormalize-style). Persists across stages
        # so the variance estimate stabilizes once and stays calibrated.
        # gamma here matches the agent's GAE gamma so the discounted-return
        # tracker is consistent with how the value function will bootstrap.
        normalize_rewards = bool(config.get("normalize_rewards", True))
        model_params = config.get("model_params", {})
        norm_gamma = float(model_params.get("gamma", 0.99))
        norm_clip = float(config.get("reward_norm_clip", 10.0))
        self._reward_normalizer: Optional[_RewardNormalizer] = (
            _RewardNormalizer(gamma=norm_gamma, clip=norm_clip)
            if normalize_rewards else None
        )

        self.logger.info(
            "PPO JAL Curriculum Trainer initialized on device: %s "
            "(reward_normalization=%s, gamma=%.3f, clip=%.1f)",
            self.device,
            "ON" if normalize_rewards else "OFF",
            norm_gamma,
            norm_clip,
        )

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def setup_environment(
        self,
        num_robots: int,
        robot_ids: List[int],
        stage_config: Optional[Dict[str, Any]] = None,
    ) -> JALTeamEnv:
        team_infos = self._load_team_config(self.config["team_config"])
        team_name = self.config.get("team_name") or team_infos[0].name

        if self.networker is None:
            sim_host, sim_player_port, sim_trainer_port = self._sim_endpoint_for_env(0)
            self.networker = Networker(
                team_infos,
                self.config.get("env_mode", "sim-only"),
                sim_host=sim_host,
                sim_player_port=sim_player_port,
                sim_trainer_port=sim_trainer_port,
            )

        stage_config = stage_config or {}

        def _stage_or_top(key, default):
            if key in stage_config:
                return stage_config[key]
            return self.config.get(key, default)

        disabled_actions = _stage_or_top("disabled_actions", [])
        spawn_robot_at_ball = _stage_or_top("spawn_robot_at_ball", False)
        spawn_offset_behind_ball = _stage_or_top("spawn_offset_behind_ball", 1.0)
        random_ball_x = _stage_or_top("random_ball_x", False)
        random_ball_x_range = _stage_or_top("random_ball_x_range", [5.0, 30.0])
        random_ball_y = _stage_or_top("random_ball_y", False)
        random_ball_y_range = _stage_or_top("random_ball_y_range", [-3.0, 3.0])
        random_spawn_theta = _stage_or_top("random_spawn_theta", False)
        random_spawn_theta_range_deg = _stage_or_top("random_spawn_theta_range_deg", [-45.0, 45.0])
        reward_config_overrides = _stage_or_top("reward_config_overrides", None)
        invalid_action_penalty = _stage_or_top("invalid_action_penalty", 0.2)

        self.env = JALTeamEnv(
            networker=self.networker,
            team_name=team_name,
            robot_ids=robot_ids,
            obs_dim_per_robot=int(self.config.get("obs_dim_per_robot", 8)),
            non_robot_obs_dim=int(self.config.get("non_robot_obs_dim", 4)),
            a_max=int(self.config.get("a_max", 5)),
            c_max=int(self.config.get("c_max", 7)),
            global_dim=int(self.config.get("global_dim", 6)),
            per_agent_dim=int(self.config.get("per_agent_dim", 10)),
            d_ctx=int(self.config.get("d_ctx", 7)),
            max_steps=int(self.config.get("max_steps", 200)),
            debug=bool(self.config.get("debug", False)),
            invalid_action_penalty=float(invalid_action_penalty),
            disabled_actions=list(disabled_actions) if disabled_actions else [],
            spawn_robot_at_ball=bool(spawn_robot_at_ball),
            spawn_offset_behind_ball=float(spawn_offset_behind_ball),
            random_ball_x=bool(random_ball_x),
            random_ball_x_range=tuple(random_ball_x_range),
            random_ball_y=bool(random_ball_y),
            random_ball_y_range=tuple(random_ball_y_range),
            random_spawn_theta=bool(random_spawn_theta),
            random_spawn_theta_range_deg=tuple(random_spawn_theta_range_deg),
            reward_config_overrides=dict(reward_config_overrides) if reward_config_overrides else None,
        )

        self.logger.info(
            "JAL environment setup — Team: %s, robots=%s, obs_dim=%d, disabled_actions=%s",
            team_name,
            robot_ids,
            int(self.env.observation_space.shape[0]),
            list(disabled_actions) if disabled_actions else [],
        )
        return self.env

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def setup_model(self, env: JALTeamEnv, num_robots: int):
        self.logger.info("Setting up PPO JAL agent on device: %s", self.device)

        # Merge model_params from config into PPOJALAgent hparams.
        model_params = self.config.get("model_params", {})
        hparams: Dict[str, Any] = {}
        for k in [
            "gamma", "gae_lambda", "clip_range", "target_kl", "n_epochs",
            "minibatch_size", "rollout_size",
            "learning_rate_initial", "learning_rate_final",
            "ent_coef_initial", "ent_coef_final",
            "vf_coef", "max_grad_norm",
            "value_clip_range", "advantage_clip", "entropy_tripwire",
            "kl_lr_halve_factor", "feature_dim", "num_heads",
        ]:
            if k in model_params:
                hparams[k] = model_params[k]

        obs_dim = int(env.observation_space.shape[0])
        self._current_obs_dim = obs_dim
        self.agent = PPOJALAgent(
            obs_dim=obs_dim,
            num_robots=num_robots,
            a_max=int(self.config.get("a_max", 5)),
            c_max=int(self.config.get("c_max", 7)),
            global_dim=int(self.config.get("global_dim", 6)),
            per_agent_dim=int(self.config.get("per_agent_dim", 10)),
            d_ctx=int(self.config.get("d_ctx", 7)),
            num_primitives=int(self.config.get("num_primitives", 8)),
            param_dim=int(self.config.get("param_dim", 5)),
            device=self.device,
            hparams=hparams,
        )

        load_path = self.config.get("load_model")
        if load_path:
            self.logger.info("Loading PPO JAL model from %s", load_path)
            self.agent.load(load_path)
            self.logger.info("PPO JAL model loaded")
        self.logger.info(
            "PPO JAL agent setup — obs_dim=%d num_robots=%d a_max=%d feature_dim=%d",
            obs_dim, num_robots, self.agent.a_max, int(self.agent.hparams["feature_dim"]),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        if not self.curriculum:
            self.logger.error("No curriculum defined in config")
            return

        for stage_name, stage_config in sorted(self.curriculum.items()):
            self._run_stage(stage_name, stage_config)

        self.logger.info("Curriculum training complete!")

    def _run_stage(self, stage_name: str, stage_config: Dict[str, Any]):
        num_robots = int(stage_config.get("num_robots", 1))
        robot_ids = stage_config.get("robot_ids", list(range(1, num_robots + 1)))
        timesteps = int(stage_config.get("timesteps", 200_000))

        if timesteps <= 0:
            self.logger.info("Stage %s disabled (timesteps=0); skipping.", stage_name)
            return

        self.logger.info(
            "Starting %s — %d robot(s) %s for %s timesteps",
            stage_name, num_robots, robot_ids, f"{timesteps:,}",
        )

        try:
            self._run_stage_inner(stage_name, num_robots, robot_ids, timesteps, stage_config=stage_config)
        except Exception:
            self.logger.exception("Stage %s failed — see traceback above", stage_name)
            raise

    def _run_stage_inner(
        self,
        stage_name: str,
        num_robots: int,
        robot_ids: list,
        timesteps: int,
        stage_config: Optional[Dict[str, Any]] = None,
    ):
        # Build the env for this stage.
        self.env = self.setup_environment(num_robots, robot_ids, stage_config=stage_config)

        obs_dim_now = int(self.env.observation_space.shape[0])

        # Rebuild the agent ONLY when the constant obs_dim changes — which, on
        # the expandable backbone, it never should across the curriculum. The
        # network is count-agnostic, so a num_robots change (adding a teammate)
        # is NOT a rebuild: the same shared weights run one more agent slot and
        # the env's active mask handles the rest. This is the whole point of the
        # design — warm-start between stages is a clean weight reuse, not a
        # from-scratch reinit. The rollout buffer is always flushed so the prior
        # stage's reward/spawn distribution doesn't bias the next update.
        needs_new_agent = (
            self.agent is None
            or self._current_obs_dim != obs_dim_now
        )
        if needs_new_agent:
            if self.agent is not None:
                self.logger.warning(
                    "Rebuilding agent: obs_dim %s→%d (structural change).",
                    self._current_obs_dim, obs_dim_now,
                )
            self.setup_model(self.env, num_robots)
            self._current_num_robots = num_robots
        else:
            self.logger.info(
                "Reusing existing agent for %s (count-agnostic warm-start, "
                "num_robots %s→%d); flushing rollout buffer.",
                stage_name, self._current_num_robots, num_robots,
            )
            self.agent.num_robots = num_robots  # for default-mask fallback only
            self._current_num_robots = num_robots
            self.agent.buffer.clear()

        stage_config = stage_config or {}
        # Disabled actions are pulled from stage_config (mirrors setup_environment).
        disabled_actions = list(stage_config.get(
            "disabled_actions", self.config.get("disabled_actions", [])
        ))
        self.logger.info("Stage %s disabled primitives: %s", stage_name, disabled_actions)

        save_interval = int(self.config.get("save_interval", 10_000))
        checkpoint_dir = Path(self.config.get("save_path", "models/ppo_jal_curriculum"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._training_loop(
            stage_name=stage_name,
            total_timesteps=timesteps,
            disabled_actions=disabled_actions,
            save_interval=save_interval,
            checkpoint_dir=checkpoint_dir,
        )

        stage_ckpt = checkpoint_dir / f"{stage_name}_complete.pt"
        stage_ckpt.parent.mkdir(parents=True, exist_ok=True)
        self.save_model(str(stage_ckpt))
        self.logger.info("%s complete — model saved: %s", stage_name, stage_ckpt)
        self.plot_training()

    # ------------------------------------------------------------------
    # Training loop (custom on-policy)
    # ------------------------------------------------------------------

    def _training_loop(
        self,
        stage_name: str,
        total_timesteps: int,
        disabled_actions: Sequence[str],
        save_interval: int,
        checkpoint_dir: Path,
    ):
        assert self.agent is not None and self.env is not None

        # Param-active mask (per stage): which continuous param dims any ENABLED
        # primitive actually uses. Dx,Dy ← goto; Dtheta ← turn; reserved dims off.
        # Gating idle params keeps their head rows from drifting (zero gradient)
        # and preserves them for later activation. Mirrors the disabled-primitive
        # mask on the policy side. See PPO_EXPANDABLE_PLAN.md.
        param_active_mask = np.zeros(self.agent.param_dim, dtype=np.float32)
        if "goto" not in disabled_actions:
            param_active_mask[0] = 1.0  # Dx
            param_active_mask[1] = 1.0  # Dy
        if "turn" not in disabled_actions:
            param_active_mask[2] = 1.0  # Dtheta
        self.logger.info("Stage %s param_active_mask: %s", stage_name, param_active_mask.tolist())

        obs, info = self.env.reset()
        agent_mask = info.get("agent_active_mask")
        context_mask = info.get("context_active_mask")
        ep_return: float = 0.0
        ep_length: int = 0
        ep_entropy_log: List[float] = []  # not used directly but kept for monitoring hooks

        steps_done = 0
        next_save = save_interval
        wall_start = time.time()
        last_update_step = 0

        # Rolling diagnostic summary (goal rate / outcome mix / action mix) —
        # the live signals the curriculum stop-rule watches. log_episode only
        # surfaces avg reward; this adds the goal/aim picture every N episodes
        # so you don't have to run the post-hoc parser to see if bad_aim
        # (≈ off_target rate) is dropping. Window resets after each summary.
        summary_interval = int(self.config.get("episode_summary_interval", 200))
        stage_ep = 0
        win_outcomes: Counter = Counter()
        win_actions: Counter = Counter()
        win_rewards: List[float] = []
        win_lengths: List[int] = []
        cum_outcomes: Counter = Counter()

        while steps_done < total_timesteps:
            # Sample action.
            action, transition = self.agent.sample_action(
                obs=obs,
                disabled_actions=disabled_actions,
                agent_active_mask=agent_mask,
                context_active_mask=context_mask,
                param_active_mask=param_active_mask,
                deterministic=False,
            )

            # Accumulate the chosen primitive(s) for the rolling action mix.
            for p in np.asarray(action["primitive_idx"]).reshape(-1):
                win_actions[PRIMITIVE_NAMES[int(p)]] += 1

            # Step env.
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            # For value bootstrapping: don't bootstrap past a terminal state.
            # truncated (time-limit) should still bootstrap from the next value.
            mask = 0.0 if terminated else 1.0

            # Reward normalization (SB3-VecNormalize-style): divide raw reward
            # by running std of discounted return. The value function will then
            # learn unit-std targets, keeping value MSE well-conditioned even
            # with the +150 (goal) vs ±2 (per-step shaping) spread we have.
            # Episode return tracking uses the raw reward so log_episode() and
            # downstream plots stay in the original reward scale.
            raw_reward = float(reward)
            if self._reward_normalizer is not None:
                norm_reward = self._reward_normalizer.normalize(raw_reward, done)
            else:
                norm_reward = raw_reward

            self.agent.store_transition(transition)
            self.agent.store_reward(norm_reward, mask)

            ep_return += raw_reward
            ep_length += 1
            steps_done += 1

            # Progress schedule (1.0 at start → 0.0 at end of stage).
            progress_remaining = max(0.0, 1.0 - steps_done / total_timesteps)
            self.agent.set_progress_remaining(progress_remaining)

            # PPO update when rollout buffer is full.
            if self.agent.rollout_full:
                # If we landed on a terminal step, last_obs is irrelevant (the
                # bootstrapped value will be ignored via the mask). Otherwise
                # we pass next_obs so the value estimate continues from there.
                last_obs_for_bootstrap = None if terminated else next_obs
                metrics = self.agent.update(last_obs=last_obs_for_bootstrap)
                rollout_steps = steps_done - last_update_step
                last_update_step = steps_done
                cat_ent = metrics.get("categorical_entropy", float("nan"))
                reward_std = (
                    self._reward_normalizer.std
                    if self._reward_normalizer is not None
                    else float("nan")
                )
                self.logger.info(
                    "[%s] PPO update @ step %d/%d  rollout=%d  "
                    "loss(pol=%.4f val=%.4f)  ent=%.3f cat_ent=%.3f kl=%.4f "
                    "lr=%.2e ent_coef=%.4f  reward_std=%.3f",
                    stage_name, steps_done, total_timesteps, rollout_steps,
                    metrics.get("policy_loss", 0.0),
                    metrics.get("value_loss", 0.0),
                    metrics.get("entropy", 0.0),
                    cat_ent,
                    metrics.get("approx_kl", 0.0),
                    metrics.get("lr", 0.0),
                    metrics.get("ent_coef", 0.0),
                    reward_std,
                )
                # Anti-freeze tripwire: warn if Categorical entropy is suspiciously low.
                tripwire = float(self.agent.hparams.get("entropy_tripwire", 0.5))
                if not math.isnan(cat_ent) and cat_ent < tripwire and steps_done < total_timesteps * 0.5:
                    self.logger.warning(
                        "[%s] Categorical entropy %.3f below tripwire %.3f at %d/%d steps — "
                        "possible premature collapse. Raise ent_coef_initial if this persists.",
                        stage_name, cat_ent, tripwire, steps_done, total_timesteps,
                    )

            # Episode end bookkeeping.
            if done:
                self._global_episode_count += 1
                stage_ep += 1
                self.log_episode(self._global_episode_count, ep_return, ep_length)
                self.logger.info(
                    "[%s] Episode %d — reward=%.2f  length=%d  total_timesteps=%d",
                    stage_name, self._global_episode_count, ep_return, ep_length, steps_done,
                )

                # Feed the rolling-summary window.
                outcome = info.get("termination_reason") or "unknown"
                win_outcomes[outcome] += 1
                cum_outcomes[outcome] += 1
                win_rewards.append(ep_return)
                win_lengths.append(ep_length)
                if stage_ep % summary_interval == 0:
                    self._log_episode_summary(
                        stage_name, stage_ep, summary_interval,
                        win_outcomes, win_actions, win_rewards, win_lengths,
                        cum_outcomes,
                    )
                    win_outcomes = Counter()
                    win_actions = Counter()
                    win_rewards = []
                    win_lengths = []

                ep_return = 0.0
                ep_length = 0
                obs, info = self.env.reset()
                agent_mask = info.get("agent_active_mask")
                context_mask = info.get("context_active_mask")
            else:
                obs = next_obs
                agent_mask = info.get("agent_active_mask", agent_mask)
                context_mask = info.get("context_active_mask", context_mask)

            # Periodic checkpoint.
            if steps_done >= next_save:
                ckpt = checkpoint_dir / f"{stage_name}_steps{next_save}.pt"
                self.save_model(str(ckpt))
                self.logger.info(
                    "[%s] Checkpoint saved: %s (wall=%.1fs)",
                    stage_name, ckpt, time.time() - wall_start,
                )
                next_save += save_interval

            # Periodic progress log (every 1k steps).
            if steps_done % 1000 == 0:
                self.logger.info(
                    "[%s] progress: %s/%s timesteps (%.1f%%)  wall=%.1fs",
                    stage_name,
                    f"{steps_done:,}",
                    f"{total_timesteps:,}",
                    100.0 * steps_done / total_timesteps,
                    time.time() - wall_start,
                )

        # Final flush: if buffer has any leftover transitions, run one more update.
        if len(self.agent.buffer) >= int(self.agent.hparams.get("minibatch_size", 64)):
            last_obs_for_bootstrap = obs if ep_length > 0 else None
            self.agent.update(last_obs=last_obs_for_bootstrap)

        # Emit a final summary for the leftover (< summary_interval) episodes.
        if win_outcomes:
            self._log_episode_summary(
                stage_name, stage_ep, sum(win_outcomes.values()),
                win_outcomes, win_actions, win_rewards, win_lengths, cum_outcomes,
            )

    def _log_episode_summary(
        self,
        stage_name: str,
        stage_ep: int,
        window: int,
        win_outcomes: "Counter",
        win_actions: "Counter",
        win_rewards: List[float],
        win_lengths: List[int],
        cum_outcomes: "Counter",
    ) -> None:
        """Log one rolling-window diagnostic line: goal rate, outcome mix, action mix.

        `off_target` (`ball_in_penalty_off_target`) is the live proxy for the
        bad_aim rate the stop-rule tracks — when aim improves, goal rate rises
        and off_target falls. All percentages are over the window just closed;
        the cumulative goal rate spans the whole stage so far.
        """
        n = max(1, sum(win_outcomes.values()))
        cum_n = max(1, sum(cum_outcomes.values()))

        def pct(count: int) -> float:
            return 100.0 * count / n

        goal = win_outcomes.get("goal_scored", 0)
        off = win_outcomes.get("ball_in_penalty_off_target", 0)
        oob = win_outcomes.get("ball_out_of_bounds", 0)
        maxs = win_outcomes.get("max_steps", 0)
        other = n - goal - off - oob - maxs
        cum_goal_rate = 100.0 * cum_outcomes.get("goal_scored", 0) / cum_n
        mean_r = sum(win_rewards) / max(1, len(win_rewards))
        mean_l = sum(win_lengths) / max(1, len(win_lengths))
        a_total = max(1, sum(win_actions.values()))
        amix = "  ".join(f"{k}={100 * v // a_total}%" for k, v in sorted(win_actions.items()))

        self.logger.info(
            "[%s] ── SUMMARY ep %d (last %d) ── goal=%.1f%% (cum %.1f%%)  "
            "off_target=%.0f%%  oob=%.0f%%  max_steps=%.0f%%  other=%.0f%%  "
            "reward μ=%.1f  len μ=%.1f  | actions: %s",
            stage_name, stage_ep, window,
            pct(goal), cum_goal_rate,
            pct(off), pct(oob), pct(maxs), pct(other),
            mean_r, mean_l, amix,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str):
        if self.agent is None:
            raise RuntimeError("Agent not initialized — cannot save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(path)
        self.logger.info("Model saved to %s", path)

    def load_model(self, path: str):
        if self.agent is None:
            raise RuntimeError("Agent must be initialized before loading model")
        self.agent.load(path)
        self.logger.info("Model loaded from %s", path)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        super().cleanup()
        if self.networker:
            try:
                if hasattr(self.networker, "shutdown") and callable(self.networker.shutdown):
                    self.networker.shutdown()
                elif hasattr(self.networker, "disconnect_from_sim") and callable(self.networker.disconnect_from_sim):
                    self.networker.disconnect_from_sim()
            except Exception as e:
                self.logger.error("Error during networker shutdown: %s", e)

    def _load_team_config(self, file_path: str) -> List[TeamInfo]:
        with open(file_path, "r") as f:
            config = json.load(f)["teams"]

        if len(config) != 2:
            raise ValueError("Team configuration must contain exactly two teams.")

        team1_info, team2_info = config
        if len(team1_info) != 3 or len(team2_info) != 3:
            raise ValueError("Each team configuration must have name, n_players, and goalie_id.")

        return [TeamInfo(*team1_info), TeamInfo(*team2_info)]

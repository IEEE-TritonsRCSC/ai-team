"""TD3 JAL HER Trainer — curriculum learning with Hindsight Experience Replay.

Mirrors TD3JALCurriculumTrainer but uses JALHEREnv (GoalEnv-style Dict obs)
and wires SB3's HerReplayBuffer + MultiInputPolicy into the TD3 model.

Can be run in parallel alongside td3_jal_curriculum_trainer.py — the two
trainers share no state and write to separate log/model directories.
"""

import json
import gymnasium as gym
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from ai_interface.envs.JAL_her_env import JALHEREnv
from ai_interface.trainers.base_trainer import BaseTrainer
from ai_interface.trainers.policy_control import build_aux_team_command_providers, load_json_config
from networking.networker import Networker, TeamInfo


class _EpisodeStatsCallback(BaseCallback):
    """SB3 callback that feeds completed-episode stats into BaseTrainer.log_episode().

    SB3 wraps the env with a Monitor automatically when verbose>=1; Monitor
    injects an "episode" key into `info` on the terminal step containing
    {"r": total_reward, "l": episode_length, "t": wall_time}.  We read that
    here so the base trainer's rolling-average logging and plot_training() have
    real data to work with.
    """

    def __init__(self, trainer: "TD3JALHERTrainer", stage_name: str):
        super().__init__(verbose=0)
        self.trainer = trainer
        self.stage_name = stage_name
        self._episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue
            self._episode_count += 1
            reward = float(ep["r"])
            length = int(ep["l"])
            self.trainer.log_episode(self._episode_count, reward, length)
            self.trainer.logger.info(
                "[%s] Episode %d — reward=%.2f  length=%d  "
                "total_timesteps=%d",
                self.stage_name,
                self._episode_count,
                reward,
                length,
                self.num_timesteps,
            )
        return True


class TD3JALHERTrainer(BaseTrainer):
    """Curriculum TD3 trainer that uses HER for sample-efficient goal learning."""

    def __init__(self, config: Dict[str, Any], log_dir: str = None, device=None):
        super().__init__(config, log_dir, algorithm_name="td3_jal_her")
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.networker: Optional[Networker] = None
        self.env: Optional[JALHEREnv] = None
        self.model: Optional[TD3] = None

        self.curriculum = config.get("curriculum", {})
        self.policy_config = load_json_config(config.get("policy_config"))
        # Track which obs/action dim the live model was built for, so we can
        # detect curriculum stage transitions and rebuild the model when
        # num_robots changes (the network shape would otherwise be wrong).
        self._current_num_robots: Optional[int] = None
        self.logger.info("TD3 JAL HER Trainer initialized on device: %s", self.device)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def setup_environment(
        self,
        num_robots: int,
        robot_ids: List[int],
        team_config_path: Optional[Any] = None,
        stage_config: Optional[Dict[str, Any]] = None,
    ) -> JALHEREnv:
        team_config_spec = team_config_path or self.config["team_config"]
        # Reuse the existing networker across stages. Tearing it down and
        # re-initializing UDP connections to rcssserver between stages is
        # unreliable: Client.connect_to_sim has no socket timeout, and the
        # (bye) → new (init) sequence races against the server, causing
        # the new init's recvfrom to hang forever when the server is slow
        # to release the old player slot. The networker holds a stable
        # connection that is identical across stages anyway — only the
        # env's behavioral knobs change.
        team_infos = self._load_team_config(team_config_spec)
        team_name = (
            (stage_config or {}).get("team_name")
            or self.config.get("team_name")
            or team_infos[0].name
        )

        env_mode = str((stage_config or {}).get("env_mode", self.config.get("env_mode", "sim-embedded")))


        if self.networker is None:
            sim_host, sim_player_port, sim_trainer_port = self._sim_endpoint_for_env(0)
            self.networker = Networker(
                team_infos,
                env_mode,
                sim_host=sim_host,
                sim_player_port=sim_player_port,
                sim_trainer_port=sim_trainer_port,
            )

        # Per-stage environment knobs. Stage-level values override top-level
        # config values (so e.g. Stage 1 can disable approach_ball while
        # Stage 2 re-enables it).
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

        aux_team_command_providers = build_aux_team_command_providers(
            self.policy_config,
            self.networker,
            device=self.device,
            stage_config=stage_config,
        )

        self.env = JALHEREnv(
            her_distance_threshold=float(self.config.get("her_distance_threshold", 0.1)),
            networker=self.networker,
            team_name=team_name,
            robot_ids=robot_ids,
            obs_dim_per_robot=int(self.config.get("obs_dim_per_robot", 8)),
            non_robot_obs_dim=int(self.config.get("non_robot_obs_dim", 4)),
            max_steps=int(self.config.get("max_steps", 200)),
            debug=bool(self.config.get("debug", False)),
            aux_team_command_providers=aux_team_command_providers,
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

        # obs_dim is stored on the env; observation_space is now a Dict
        obs_dim = self.env.obs_dim
        self.logger.info(
            "JAL HER environment setup — env_mode=%s, Team: %s, robots=%s, obs_dim=%d, action_dim=%d",
            env_mode,
            team_name,
            robot_ids,
            obs_dim,
            int(self.env.action_space.shape[0]),
        )
        return self.env

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def setup_model(self, env: JALHEREnv, num_robots: int):
        self.logger.info("Setting up TD3+HER model on device: %s", self.device)
        model_params = self.config.get("model_params", {})

        learning_rate     = float(model_params.get("learning_rate", 1e-3))
        buffer_size       = int(model_params.get("buffer_size", 100_000))
        batch_size        = int(model_params.get("batch_size", 64))
        gamma             = float(model_params.get("gamma", 0.9))
        tau               = float(model_params.get("tau", 0.01))
        policy_delay      = int(model_params.get("policy_delay", 2))
        target_policy_noise = float(model_params.get("target_policy_noise", 0.2))
        target_noise_clip = float(model_params.get("target_noise_clip", 0.5))
        action_noise_std  = float(model_params.get("action_noise_std", 0.05))

        # HER-specific params
        n_sampled_goal           = int(model_params.get("her_n_sampled_goal", 4))
        goal_selection_strategy  = str(model_params.get("her_goal_selection_strategy", "future"))

        self.logger.info(
            "TD3+HER hyperparams: lr=%s, buffer=%s, batch=%s, gamma=%s, "
            "n_sampled_goal=%d, strategy=%s",
            learning_rate, buffer_size, batch_size, gamma,
            n_sampled_goal, goal_selection_strategy,
        )

        if self.config.get("load_model"):
            load_path = self.config["load_model"]
            self.logger.info("Loading model from %s", load_path)
            self.model = TD3.load(load_path, env=env, device=str(self.device))
            # Reset step counters so learning_starts applies from scratch.
            # The replay buffer is not saved in checkpoints, so without this
            # SB3 immediately tries to sample an empty HER buffer and crashes.
            self.model.num_timesteps = 0
            self.model._episode_num = 0
            self.logger.info("TD3+HER model loaded (step counters reset for fresh buffer collection)")
            return

        n_actions = int(env.action_space.shape[0])
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=action_noise_std * np.ones(n_actions),
        )

        # net_arch applies to the policy and Q-network heads.
        # With MultiInputPolicy + Dict obs SB3 uses CombinedExtractor by default,
        # which runs each obs key through a small MLP then concatenates.
        user_policy_kwargs = dict(model_params.get("policy_kwargs", {}))
        if "net_arch" not in user_policy_kwargs:
            user_policy_kwargs["net_arch"] = [256, 256]

        self.model = TD3(
            policy="MultiInputPolicy",       # required for Dict observation spaces
            env=env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": n_sampled_goal,
                "goal_selection_strategy": goal_selection_strategy,
            },
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=int(model_params.get("learning_starts", 1000)),
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=int(model_params.get("train_freq", 1)),
            gradient_steps=int(model_params.get("gradient_steps", 1)),
            action_noise=action_noise,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            policy_kwargs=user_policy_kwargs,
            verbose=int(model_params.get("verbose", 1)),
            device=str(self.device),
            tensorboard_log=str(self.run_dir / "tensorboard"),
        )
        self.logger.info("TD3+HER model setup complete")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        if not self.curriculum:
            self.logger.error("No curriculum defined in config")
            return

        for stage_name, stage_config in sorted(self.curriculum.items()):
            self._run_stage(stage_name, stage_config)

        self.logger.info("HER curriculum training complete!")

    def _run_stage(self, stage_name: str, stage_config: Dict[str, Any]):
        num_robots = int(stage_config.get("num_robots", 1))
        robot_ids  = stage_config.get("robot_ids", list(range(1, num_robots + 1)))
        timesteps  = int(stage_config.get("timesteps", 200_000))
        team_config_path = stage_config.get("team_config", self.config["team_config"])
        stage_signature = (
            stage_name,
            self._stage_signature_value(team_config_path),
            tuple(robot_ids),
            tuple(
                sorted(
                    self._stage_signature_value(item)
                    for item in stage_config.get("aux_team_policies", [])
                )
            ),
        )

        self.logger.info("Starting %s — %d robot(s) %s", stage_name, num_robots, robot_ids)

        try:
            self._run_stage_inner(
                stage_name, num_robots, robot_ids, timesteps,
                team_config_path=team_config_path,
                stage_config=stage_config,
            )
        except Exception:
            self.logger.exception("Stage %s failed — see traceback above", stage_name)
            raise

    def _run_stage_inner(self, stage_name: str, num_robots: int, robot_ids: list, timesteps: int,
                         team_config_path: Optional[str] = None,
                         stage_config: Optional[Dict[str, Any]] = None):
        # Always rebuild the env per stage so stage_config (spawn flags,
        # disabled_actions, reward_config_overrides, etc.) actually takes
        # effect. The previous "only rebuild when num_robots changes" check
        # silently dropped stage_config changes between same-sized stages.
        self.env = self.setup_environment(
            num_robots, robot_ids,
            team_config_path=team_config_path,
            stage_config=stage_config,
        )

        # Rebuild the model from scratch only when num_robots changes (obs/action
        # dims differ). Otherwise rebind the existing model to the new env and
        # reset the HER replay buffer + step counters, since transitions stored
        # under the previous stage's reward and spawn distribution would bias
        # updates under the new stage.
        needs_new_model = self.model is None or self._current_num_robots != num_robots
        if needs_new_model:
            if self.model is not None:
                self.logger.warning(
                    "Reinitializing model from scratch for %d robots (was %s).",
                    num_robots, self._current_num_robots,
                )
                self.model = None
            self.setup_model(self.env, num_robots)
            self._current_num_robots = num_robots
        else:
            self.logger.info(
                "Rebinding existing model to new %s env (num_robots=%d unchanged) "
                "and resetting HER replay buffer + step counters.",
                stage_name, num_robots,
            )
            self.model.set_env(self.env)
            self.model.replay_buffer.reset()
            self.model.num_timesteps = 0
            self.model._episode_num = 0

        learn_batch    = int(self.config.get("learn_batch_timesteps", 2048))
        save_interval  = int(self.config.get("save_interval", 10_000))
        checkpoint_dir = Path(self.config.get("save_path", "models/td3_jal_her"))

        remaining         = timesteps
        stage_timesteps   = 0
        next_save         = save_interval

        callback = _EpisodeStatsCallback(trainer=self, stage_name=stage_name)

        self.logger.info("Training %s for %s timesteps", stage_name, f"{timesteps:,}")

        while remaining > 0:
            chunk = min(learn_batch, remaining)
            self.model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=callback)

            remaining       -= chunk
            stage_timesteps += chunk
            self.training_metrics["total_timesteps"] = stage_timesteps

            self.logger.info(
                "%s progress: %s/%s timesteps (%.1f%%)",
                stage_name,
                f"{stage_timesteps:,}",
                f"{timesteps:,}",
                100 * stage_timesteps / timesteps,
            )

            while stage_timesteps >= next_save and remaining > 0:
                ckpt = checkpoint_dir / f"{stage_name}_steps{next_save}.zip"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                self.save_model(str(ckpt))
                self.logger.info("Checkpoint saved: %s", ckpt)
                next_save += save_interval

        stage_ckpt = checkpoint_dir / f"{stage_name}_complete.zip"
        stage_ckpt.parent.mkdir(parents=True, exist_ok=True)
        self.save_model(str(stage_ckpt))
        self.logger.info("%s complete — model saved: %s", stage_name, stage_ckpt)
        self.plot_training()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str):
        if self.model is None:
            raise RuntimeError("Model not initialized — cannot save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        self.logger.info("Model saved to %s", path)

    def load_model(self, path: str):
        if self.env is None:
            raise RuntimeError("Environment must be initialized before loading model")
        self.model = TD3.load(path, env=self.env, device=str(self.device))
        self.logger.info("Model loaded from %s", path)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        super().cleanup()
        if self.networker:
            try:
                if hasattr(self.networker, "shutdown"):
                    self.networker.shutdown()
                elif hasattr(self.networker, "disconnect_from_sim"):
                    self.networker.disconnect_from_sim()
            except Exception as e:
                self.logger.error("Error during networker shutdown: %s", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stage_signature_value(value: Any) -> str:
        if isinstance(value, Mapping):
            return json.dumps(value, sort_keys=True)
        if isinstance(value, (list, tuple)):
            return json.dumps(value)
        return str(value)

    def _load_team_config(self, team_config: Any) -> List[TeamInfo]:
        if isinstance(team_config, Mapping):
            config = dict(team_config)
        else:
            with open(team_config, "r") as f:
                config = json.load(f)

        if "teams" not in config:
            raise ValueError("Team configuration must contain a 'teams' key.")

        teams = config["teams"]

        if len(teams) != 2:
            raise ValueError("Team configuration must contain exactly two teams.")

        team1_info, team2_info = teams
        if len(team1_info) != 3 or len(team2_info) != 3:
            raise ValueError("Each team configuration must have name, n_players, and goalie_id.")

        return [TeamInfo(*team1_info), TeamInfo(*team2_info)]

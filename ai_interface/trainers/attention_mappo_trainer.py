"""Attention-MAPPO trainer with 5-stage curriculum and self-play.

Training flow:
    train()
      for each stage in sorted(config["curriculum"]):
        _run_stage(stage_name, stage_config)
          ├─ Build FullTeamMARLEnv (with stage-specific num_agents + opponent)
          ├─ Build / warm-start AttentionMAPPOAgent
          └─ Episode loop:
               collect rollout → store_transition × max_steps
               if batch_ready: update() → log losses
               every save_interval: checkpoint
               if self_play and episode % swap_interval == 0: update frozen opponent

Warm-starting: encoder + attention + actor weights transfer across stages because
the mean-pooled critic is N-agnostic. Only optimizer state resets between stages.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .base_trainer import BaseTrainer
from ai_interface.algorithms.attention_mappo import AttentionMAPPOAgent, AttentionMAPPOConfig
from ai_interface.envs.full_team_marl_env import FullTeamMARLEnv
from ai_interface.hsm.state_machine import Role
from networking.networker import Networker, TeamInfo


class AttentionMAPPOTrainer(BaseTrainer):
    """Trainer for AttentionMAPPO with multi-stage curriculum and self-play."""

    def __init__(
        self,
        config: Dict[str, Any],
        log_dir: str = None,
        device=None,
    ):
        super().__init__(config, log_dir, algorithm_name="attention_mappo")
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.networker: Optional[Networker] = None
        self.env: Optional[FullTeamMARLEnv] = None
        self.agent: Optional[AttentionMAPPOAgent] = None
        self.frozen_opponent: Optional[AttentionMAPPOAgent] = None
        self._team_infos: Optional[List[TeamInfo]] = None
        self._team_name: Optional[str] = None

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def setup_environment(self, **kwargs) -> FullTeamMARLEnv:
        self._team_infos = self._load_team_config(self.config["team_config"])
        self._team_name = self.config.get("team_name") or self._team_infos[0].name
        sim_host, sim_player_port, sim_trainer_port = self._sim_endpoint_for_env(0)
        self.networker = Networker(
            self._team_infos,
            self.config.get("env_mode", "sim-embedded"),
            sim_host=sim_host,
            sim_player_port=sim_player_port,
            sim_trainer_port=sim_trainer_port,
        )
        self.env = self._build_env(num_agents=6, stage_config={})
        return self.env

    def setup_model(self, env=None) -> None:
        obs_dim = self.config.get("obs_dim", 57)
        algo_cfg = self._build_algo_config()
        self.agent = AttentionMAPPOAgent(
            num_agents=6,
            obs_dim=obs_dim,
            config=algo_cfg,
            device=self.device,
        )
        load_path = self.config.get("load_model")
        if load_path:
            self.logger.info("Loading pre-trained model from %s", load_path)
            self.agent.load(load_path)

    def train(self) -> None:
        """Run all curriculum stages in alphabetical/sorted order."""
        if self.networker is None:
            self.setup_environment()
        if self.agent is None:
            self.setup_model()

        curriculum = self.config.get("curriculum", {})
        stages = sorted(curriculum.items())  # alphabetical sort preserves stage1/2/3...

        prev_checkpoint: Optional[str] = None
        for stage_name, stage_config in stages:
            self.logger.info("=== Starting stage: %s ===", stage_name)
            prev_checkpoint = self._run_stage(stage_name, stage_config, prev_checkpoint)

        self.logger.info("Curriculum training complete.")
        self.plot_training()

    def save_model(self, path: str) -> None:
        if self.agent is None:
            raise RuntimeError("Agent not initialized")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(path)

    def load_model(self, path: str) -> None:
        if self.agent is None:
            self.setup_model()
        self.agent.load(path)

    def cleanup(self) -> None:
        super().cleanup()
        if self.networker and hasattr(self.networker, "shutdown"):
            try:
                self.networker.shutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal: stage runner
    # ------------------------------------------------------------------

    def _run_stage(
        self,
        stage_name: str,
        stage_config: Dict[str, Any],
        prev_checkpoint: Optional[str] = None,
    ) -> Optional[str]:
        """Run one curriculum stage. Returns path of the final checkpoint."""
        num_agents = int(stage_config.get("num_agents", 6))
        episodes = int(stage_config.get("episodes", 500))
        max_steps = int(stage_config.get("max_steps", self.config.get("max_steps", 400)))
        save_interval = int(stage_config.get("save_interval", self.config.get("save_interval", 100)))
        opponent_type = stage_config.get("opponent_type", "none")
        warm_start = bool(stage_config.get("warm_start_from_previous", True))
        selfplay_swap = int(stage_config.get("selfplay_swap_interval",
                                              self.config.get("algorithm", {}).get("selfplay_swap_interval", 200)))

        forced_roles = self._parse_forced_roles(stage_config.get("forced_roles", {}))
        reward_weights = stage_config.get("reward_weights", {})

        # Rebuild env for this stage
        self.env = self._build_env(num_agents, stage_config, forced_roles, reward_weights)
        self.env.set_forced_roles(forced_roles)

        # Rebuild / warm-start agent
        self._rebuild_agent(num_agents, warm_start, prev_checkpoint)

        # Setup frozen opponent for self-play
        if opponent_type == "self_play":
            self.frozen_opponent = copy.deepcopy(self.agent)
            self.frozen_opponent.memory = []
            self.env.set_opponent_controller(self.frozen_opponent)
        elif opponent_type == "scripted":
            self.env.set_opponent_controller("scripted")
        else:
            self.env.set_opponent_controller(None)

        latest_ckpt: Optional[str] = None
        n_updates = 0

        for episode in range(episodes):
            observations = self.env.reset()
            episode_reward = 0.0
            current_roles_list = [
                self.env.current_roles.get(i, Role.SUPPORT)
                for i in range(num_agents)
            ]

            for step in range(max_steps):
                actions, logprobs, value = self.agent.select_actions(
                    observations, current_roles_list, deterministic=False
                )
                next_obs, reward, done, info = self.env.step(actions)

                self.agent.store_transition(
                    observations, current_roles_list, actions, logprobs, value,
                    reward, done,
                )

                episode_reward += reward
                observations = next_obs
                current_roles_list = [
                    Role(info["roles"].get(i, Role.SUPPORT.value))
                    for i in range(num_agents)
                ]

                if done:
                    break

            # PPO update when rollout buffer is full
            if self.agent.batch_ready:
                losses = self.agent.update()
                n_updates += 1
                if losses:
                    self.logger.info(
                        "[%s] Update #%d — actor=%.4f  critic=%.4f  entropy=%.4f",
                        stage_name, n_updates,
                        losses["actor_loss"], losses["critic_loss"], losses["entropy"],
                    )
                    self.log_metrics({"stage": stage_name, "update": n_updates, **losses})

            goal_str = " GOAL!" if info.get("goal_scored") else ""
            self.logger.info(
                "[%s] Ep %d/%d  R=%.2f  steps=%d%s",
                stage_name, episode + 1, episodes, episode_reward, step + 1, goal_str,
            )
            self.log_episode(episode + 1, episode_reward, step + 1)

            # Checkpoint
            if (episode + 1) % save_interval == 0:
                ckpt_name = f"attention_mappo_{stage_name}_ep{episode+1}.pth"
                ckpt_path = self._get_checkpoint_path(ckpt_name, episode + 1)
                self.agent.save(str(ckpt_path))
                latest_ckpt = str(ckpt_path)
                self.logger.info("Checkpoint: %s", ckpt_path)

            # Self-play opponent snapshot update
            if opponent_type == "self_play" and (episode + 1) % selfplay_swap == 0:
                self._update_frozen_opponent()

        # Final stage checkpoint
        final_name = f"attention_mappo_{stage_name}_final.pth"
        final_path = self.model_dir / final_name
        self.agent.save(str(final_path))
        self.logger.info("[%s] Stage complete. Saved: %s", stage_name, final_path)
        return str(final_path)

    # ------------------------------------------------------------------
    # Internal: env / agent construction helpers
    # ------------------------------------------------------------------

    def _build_env(
        self,
        num_agents: int,
        stage_config: Dict[str, Any],
        forced_roles: Optional[Dict[int, Role]] = None,
        reward_weights: Optional[Dict[str, Any]] = None,
    ) -> FullTeamMARLEnv:
        assert self.networker is not None
        opponent_team_name = self._get_opponent_team_name()
        hsm_thresholds = self.config.get("hsm_thresholds", {})
        spawn_config = stage_config.get("spawn_config", {})

        return FullTeamMARLEnv(
            networker=self.networker,
            team_name=self._team_name,
            num_agents=num_agents,
            max_steps=int(stage_config.get("max_steps", self.config.get("max_steps", 400))),
            hsm_thresholds=hsm_thresholds,
            forced_roles=forced_roles or {},
            opponent_team_name=opponent_team_name,
            reward_weights=reward_weights or {},
            spawn_config=spawn_config,
        )

    def _rebuild_agent(
        self,
        num_agents: int,
        warm_start: bool,
        prev_checkpoint: Optional[str],
    ) -> None:
        obs_dim = self.config.get("obs_dim", 57)
        algo_cfg = self._build_algo_config()

        new_agent = AttentionMAPPOAgent(
            num_agents=num_agents,
            obs_dim=obs_dim,
            config=algo_cfg,
            device=self.device,
        )

        if warm_start and prev_checkpoint and Path(prev_checkpoint).exists():
            self.logger.info("Warm-starting from %s", prev_checkpoint)
            new_agent.load(prev_checkpoint, weights_only=True)
        elif self.agent is not None and warm_start:
            # Transfer weights from in-memory agent (no checkpoint written yet)
            new_agent.agent_encoder.load_state_dict(
                self.agent.agent_encoder.state_dict()
            )
            new_agent.team_attention.load_state_dict(
                self.agent.team_attention.state_dict()
            )
            for key in new_agent.actors:
                if key in self.agent.actors:
                    new_agent.actors[key].load_state_dict(
                        self.agent.actors[key].state_dict()
                    )

        self.agent = new_agent

    def _update_frozen_opponent(self) -> None:
        """Snapshot current agent as the frozen self-play opponent."""
        assert self.agent is not None
        self.frozen_opponent = copy.deepcopy(self.agent)
        self.frozen_opponent.memory = []
        if self.env is not None:
            self.env.set_opponent_controller(self.frozen_opponent)
        self.logger.info("Self-play opponent updated to current policy snapshot.")

    def _build_algo_config(self) -> AttentionMAPPOConfig:
        a = self.config.get("algorithm", {})
        return AttentionMAPPOConfig(
            gamma=float(a.get("gamma", 0.99)),
            gae_lambda=float(a.get("gae_lambda", 0.95)),
            lr_actor=float(a.get("lr_actor", 3e-4)),
            lr_critic=float(a.get("lr_critic", 1e-3)),
            lr_encoder=float(a.get("lr_encoder", 5e-4)),
            clip_eps=float(a.get("clip_eps", 0.2)),
            entropy_coef=float(a.get("entropy_coef", 0.01)),
            value_coef=float(a.get("value_coef", 0.5)),
            max_grad_norm=float(a.get("max_grad_norm", 0.5)),
            ppo_epochs=int(a.get("ppo_epochs", 4)),
            rollout_length=int(a.get("rollout_length", 2048)),
            minibatch_size=int(a.get("minibatch_size", 256)),
            min_std=float(a.get("min_std", 0.1)),
            agent_embed_dim=int(a.get("agent_embed_dim", 128)),
            num_attn_heads=int(a.get("num_attn_heads", 4)),
            num_attn_layers=int(a.get("num_attn_layers", 2)),
            actor_hidden=(
                int(a.get("actor_hidden", [256, 256])[0]),
                int(a.get("actor_hidden", [256, 256])[1]),
            ),
        )

    def _get_opponent_team_name(self) -> Optional[str]:
        if self._team_infos and len(self._team_infos) > 1:
            for ti in self._team_infos:
                if ti.name != self._team_name:
                    return ti.name
        return None

    @staticmethod
    def _parse_forced_roles(raw: Dict) -> Dict[int, Role]:
        out: Dict[int, Role] = {}
        role_map = {
            "STRIKER": Role.STRIKER,
            "SUPPORT": Role.SUPPORT,
            "DEFENDER": Role.DEFENDER,
            "GOALIE": Role.GOALIE,
        }
        for k, v in raw.items():
            out[int(k)] = role_map.get(str(v).upper(), Role.SUPPORT)
        return out

    def _load_team_config(self, file_path: str) -> List[TeamInfo]:
        with open(file_path) as f:
            cfg = json.load(f)["teams"]
        if len(cfg) != 2:
            raise ValueError("Team config must have exactly 2 teams.")
        t1, t2 = cfg
        return [TeamInfo(*t1), TeamInfo(*t2)]

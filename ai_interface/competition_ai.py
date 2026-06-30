"""
Competition AI for TritonBots — PPO JAL attacker (robot 1) + 5 scripted controllers.

Mirrors the inference pattern from infer.py:_run_ppo_jal but integrated into the
__main__.py lifecycle (no step-count limit, formation handling for dead-ball states).
"""

import json
import math
import numpy as np
from typing import Optional

from networking.data_utils import GameState, TeamInfo

# Dead-ball states that require formation positioning.
FORMATION_PLAYMODES = {
    "before_kick_off", "kick_off_l", "kick_off_r",
    "kick_in_l", "kick_in_r",
    "corner_kick_l", "corner_kick_r",
    "goal_kick_l", "goal_kick_r",
    "free_kick_l", "free_kick_r",
    "indirect_free_kick_l", "indirect_free_kick_r",
    "offside_l", "offside_r",
}

STOP_PLAYMODES = {"halt", "time_over"}

# Formation positions for our_side="left" (we defend left, attack right).
# Indexed as: {scenario: {robot_id: (x, y)}}
_FORMATION_LEFT = {
    "our_kickoff": {
        1: (0.0,   0.0),
        2: (-8.0,  18.0),
        3: (-8.0, -18.0),
        4: (-22.0,  12.0),
        5: (-22.0, -12.0),
        6: (-42.0,  0.0),
    },
    "their_kickoff": {
        1: (-14.0,  0.0),
        2: (-14.0,  18.0),
        3: (-14.0, -18.0),
        4: (-28.0,  12.0),
        5: (-28.0, -12.0),
        6: (-42.0,  0.0),
    },
    "our_setpiece": {
        # robot 1 navigates to ball; others hold defensive/support shape
        1: None,   # replaced with ball_pos at runtime
        2: (-8.0,  18.0),
        3: (-8.0, -18.0),
        4: (-22.0,  12.0),
        5: (-22.0, -12.0),
        6: (-42.0,  0.0),
    },
    "their_setpiece": {
        1: (-12.0,  0.0),
        2: (-14.0,  18.0),
        3: (-14.0, -18.0),
        4: (-28.0,  12.0),
        5: (-28.0, -12.0),
        6: (-42.0,  0.0),
    },
}


def _flip_formation(formation: dict) -> dict:
    """Flip x-coordinates for our_side='right'."""
    result = {}
    for robot_id, pos in formation.items():
        if pos is None:
            result[robot_id] = None
        else:
            result[robot_id] = (-pos[0], pos[1])
    return result


def _get_formation(playmode: str, our_side: str, ball_pos) -> dict:
    """Return {robot_id: (x, y)} target positions for this playmode."""
    side_suffix = "_l" if our_side == "left" else "_r"
    opp_suffix = "_r" if our_side == "left" else "_l"

    if playmode in ("before_kick_off", f"kick_off{side_suffix}"):
        key = "our_kickoff"
    elif f"kick_off{opp_suffix}" in playmode:
        key = "their_kickoff"
    elif any(playmode.endswith(side_suffix) for s in (
        "kick_in", "corner_kick", "goal_kick", "free_kick", "indirect_free_kick", "offside"
    ) if playmode.startswith(s)):
        key = "our_setpiece"
    else:
        key = "their_setpiece"

    base = dict(_FORMATION_LEFT[key])
    if our_side == "right":
        base = _flip_formation(base)

    # Substitute ball position for robot 1 on our set pieces.
    if base.get(1) is None and ball_pos is not None:
        base[1] = (float(ball_pos[0]), float(ball_pos[1]))
    elif base.get(1) is None:
        base[1] = (0.0, 0.0)

    return base


def _resolve_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class CompetitionAI:
    """
    Full competition stack for TritonBots.

    Robot 1  — PPO JAL (RL attacker) via JALTeamEnv
    Robot 2  — HardcodedSupporterCommandProvider (wing supporter)
    Robot 3  — HardcodedSupporterCommandProvider (wing supporter)
    Robot 4  — DefenderCommandProvider (ball-side defender)
    Robot 5  — MarkerDefenderCommandProvider (marker/cover defender)
    Robot 6  — GoalieCommandProvider
    """

    def __init__(self, team_infos: list, args):
        from ai_interface.algorithms.ppo_jal import PPOJALAgent
        from ai_interface.envs.JAL_env import JALTeamEnv
        from ai_interface.trainers.policy_control import (
            GoalieCommandProvider,
            DefenderCommandProvider,
            MarkerDefenderCommandProvider,
            HardcodedSupporterCommandProvider,
        )
        from networking.networker import Networker

        self.our_side = args.our_side
        self.team_name = "TritonBots"
        opponent_team = team_infos[1].name if len(team_infos) > 1 else "TeamB"

        with open(args.config, "r") as f:
            config = json.load(f)
        stage_config = config["curriculum"][args.stage]

        robot_ids = list(stage_config.get("robot_ids", [1]))
        reward_overrides = stage_config.get("reward_config_overrides")
        num_robots = len(robot_ids)

        self.networker = None  # injected at first step call via setup()
        self._config = config
        self._stage_config = stage_config
        self._args = args
        self._robot_ids = robot_ids
        self._reward_overrides = reward_overrides
        self._opponent_team = opponent_team
        self._num_robots = num_robots

        # Aux controllers for TritonBots robots 2-6
        self.aux_controllers = [
            HardcodedSupporterCommandProvider(
                team_name=self.team_name, robot_id=2, side=self.our_side,
                main_attacker_robot_id=1,
                opponent_team_name=opponent_team,
                opponent_goalie_robot_ids=[team_infos[1].goalie_id] if len(team_infos) > 1 else [1],
                robot_pool_ids=[1, 2],
            ),
            HardcodedSupporterCommandProvider(
                team_name=self.team_name, robot_id=3, side=self.our_side,
                main_attacker_robot_id=1,
                opponent_team_name=opponent_team,
                opponent_goalie_robot_ids=[team_infos[1].goalie_id] if len(team_infos) > 1 else [1],
                robot_pool_ids=[1, 3],
            ),
            DefenderCommandProvider(
                team_name=self.team_name, robot_id=4, side=self.our_side,
            ),
            MarkerDefenderCommandProvider(
                team_name=self.team_name, robot_id=5, side=self.our_side,
                ball_defender_robot_id=4,
            ),
            GoalieCommandProvider(
                team_name=self.team_name, robot_id=6, side=self.our_side,
            ),
        ]

        self.env: Optional[JALTeamEnv] = None
        self.agent: Optional[PPOJALAgent] = None
        self._obs = None
        self._agent_mask = None
        self._context_mask = None
        self._disabled_actions: list = list(stage_config.get("disabled_actions", []))
        self._param_active_mask = None

    def setup(self, networker):
        """Call once after networker is ready to wire up the env and load the model."""
        from ai_interface.algorithms.ppo_jal import PPOJALAgent
        from ai_interface.envs.JAL_env import JALTeamEnv

        self.networker = networker
        config = self._config
        stage_config = self._stage_config
        args = self._args

        opponent_goalie_robot_ids = []
        for spec in stage_config.get("aux_team_policies", []) or []:
            if not isinstance(spec, dict):
                continue
            if str(spec.get("controller_type", "")).lower() != "goalie":
                continue
            if str(spec.get("team_name", "")) == self._opponent_team:
                opponent_goalie_robot_ids.extend(int(r) for r in spec.get("robot_ids", []))
        if not opponent_goalie_robot_ids:
            opp_goalie = self._args.__dict__.get("opp_goalie_id", None)
            opponent_goalie_robot_ids = [int(opp_goalie)] if opp_goalie else [1]

        # Competition deployment owns robots 2-6 via dedicated aux controllers
        # (HardcodedSupporterCommandProvider, DefenderCommandProvider, etc).
        # The training stage's claimant_follow_robot_ids=[1,2] makes the env
        # internally co-opt robot 2 into the PPO pool (deciding per-step which
        # of 1/2 is the "active" ball carrier and driving both) — that directly
        # conflicts with our own robot-2 supporter controller, so force it off.
        env = JALTeamEnv(
            networker=networker,
            team_name=self.team_name,
            robot_ids=self._robot_ids,
            claimant_follow_robot_ids=None,
            ball_claimant_robot_ids=None,
            obs_dim_per_robot=int(config.get("obs_dim_per_robot", 8)),
            non_robot_obs_dim=int(config.get("non_robot_obs_dim", 4)),
            a_max=int(config.get("a_max", 5)),
            c_max=int(config.get("c_max", 7)),
            global_dim=int(config.get("global_dim", 6)),
            per_agent_dim=int(config.get("per_agent_dim", 10)),
            d_ctx=int(config.get("d_ctx", 7)),
            max_steps=int(config.get("max_steps", 300)),
            debug=False,
            invalid_action_penalty=float(stage_config.get("invalid_action_penalty", 0.2)),
            disabled_actions=self._disabled_actions,
            spawn_robot_at_ball=bool(stage_config.get("spawn_robot_at_ball", False)),
            spawn_offset_behind_ball=float(stage_config.get("spawn_offset_behind_ball", 1.0)),
            random_ball_x=bool(stage_config.get("random_ball_x", False)),
            random_ball_x_range=tuple(stage_config.get("random_ball_x_range", [5.0, 30.0])),
            random_ball_y=bool(stage_config.get("random_ball_y", False)),
            random_ball_y_range=tuple(stage_config.get("random_ball_y_range", [-3.0, 3.0])),
            random_spawn_theta=bool(stage_config.get("random_spawn_theta", False)),
            random_spawn_theta_range_deg=tuple(stage_config.get("random_spawn_theta_range_deg", [-45.0, 45.0])),
            ball_action_recovery=bool(stage_config.get("ball_action_recovery", False)),
            claimant_follow_solo_lane_blocked_max=float(stage_config.get("claimant_follow_solo_lane_blocked_max", 0.55)),
            claimant_follow_pass_lane_min=float(stage_config.get("claimant_follow_pass_lane_min", 0.65)),
            claimant_follow_teammate_margin=float(stage_config.get("claimant_follow_teammate_margin", 0.15)),
            claimant_follow_min_pass_distance=float(stage_config.get("claimant_follow_min_pass_distance", 8.0)),
            claimant_follow_post_receive_cooldown_steps=int(stage_config.get("claimant_follow_post_receive_cooldown_steps", 30)),
            hybrid_pass_max_align_steps=(
                int(stage_config["hybrid_pass_max_align_steps"])
                if stage_config.get("hybrid_pass_max_align_steps") is not None else None
            ),
            ball_claimant_switch_margin=float(stage_config.get("ball_claimant_switch_margin", 0.75)),
            turn_stall_limit=int(stage_config.get("turn_stall_limit", 12)),
            turn_stall_displacement=float(stage_config.get("turn_stall_displacement", 0.05)),
            kick_requires_aim=bool(stage_config.get("kick_requires_aim", False)),
            kick_min_aim_quality=float(stage_config.get("kick_min_aim_quality", 0.0)),
            reward_config_overrides=dict(self._reward_overrides) if self._reward_overrides else None,
            opponent_team_name=self._opponent_team,
            num_opponents=int(getattr(args, "num_opponents", 1)),
            opponent_goalie_robot_ids=opponent_goalie_robot_ids,
            our_side=self.our_side,
        )
        self.env = env

        model_params = config.get("model_params", {})
        hparams = {k: model_params[k] for k in [
            "gamma", "gae_lambda", "clip_range", "target_kl", "n_epochs",
            "minibatch_size", "rollout_size",
            "learning_rate_initial", "learning_rate_final",
            "ent_coef_initial", "ent_coef_final",
            "vf_coef", "max_grad_norm",
            "value_clip_range", "advantage_clip", "entropy_tripwire",
            "kl_lr_halve_factor", "feature_dim", "num_heads",
        ] if k in model_params}

        obs_dim = int(env.observation_space.shape[0])
        device = _resolve_device()
        agent = PPOJALAgent(
            obs_dim=obs_dim,
            num_robots=self._num_robots,
            a_max=int(config.get("a_max", 5)),
            c_max=int(config.get("c_max", 7)),
            global_dim=int(config.get("global_dim", 6)),
            per_agent_dim=int(config.get("per_agent_dim", 10)),
            d_ctx=int(config.get("d_ctx", 7)),
            num_primitives=int(config.get("num_primitives", 8)),
            param_dim=int(config.get("param_dim", 5)),
            device=device,
            hparams=hparams,
        )
        print(f"Loading PPO JAL model from {args.model_path} (stage={args.stage})")
        agent.load(args.model_path)
        agent.model.eval()
        self.agent = agent

        param_active_mask = np.zeros(agent.param_dim, dtype=np.float32)
        if "goto" not in self._disabled_actions:
            param_active_mask[0] = 1.0
            param_active_mask[1] = 1.0
        if "turn" not in self._disabled_actions:
            param_active_mask[2] = 1.0
        self._param_active_mask = param_active_mask

        obs, info = env.reset()
        self._obs = obs
        self._agent_mask = info.get("agent_active_mask")
        self._context_mask = info.get("context_active_mask")

    def step(self, game_state: GameState):
        """Drive one cycle of the competition stack."""
        if self.env is None or self.agent is None:
            return

        playmode = game_state.playmode or ""

        if playmode in STOP_PLAYMODES:
            self._send_stop()
            return

        if playmode in FORMATION_PLAYMODES:
            self._send_formation(game_state)
            return

        # Normal play: step aux controllers then RL robot
        cached_gs = getattr(self.env, "_cached_game_state", None) or game_state
        for aux_ctrl in self.aux_controllers:
            try:
                if hasattr(aux_ctrl, "set_active_robot_id"):
                    aux_ctrl.set_active_robot_id(getattr(self.env, "_active_robot_id", None))
                if hasattr(aux_ctrl, "set_claimant_follow_info"):
                    aux_ctrl.set_claimant_follow_info(
                        getattr(self.env, "_claimant_follow_gate_info", {})
                    )
                cmds = aux_ctrl.predict_commands(cached_gs)
                self.networker.execute_ai_output(cmds, self.team_name)
            except Exception as e:
                print(f"[CompetitionAI] aux controller {aux_ctrl.__class__.__name__} error: {e}")

        action, _ = self.agent.sample_action(
            obs=self._obs,
            disabled_actions=self._disabled_actions,
            agent_active_mask=self._agent_mask,
            context_active_mask=self._context_mask,
            param_active_mask=self._param_active_mask,
            primitive_valid_mask=self.env.get_primitive_valid_mask(
                num_primitives=self.agent.num_primitives
            ),
            deterministic=True,
        )

        # Inject param noise to avoid bang-bang turn stall (same as infer.py)
        param_noise_std = float(getattr(self._args, "ppo_param_noise_std", 0.3))
        if param_noise_std > 0.0:
            params = np.asarray(action["params"], dtype=np.float32)
            params = params + np.random.normal(0.0, param_noise_std, size=params.shape).astype(np.float32)
            action["params"] = np.clip(params, -1.0, 1.0)

        obs, _, terminated, truncated, info = self.env.step(action)
        self._obs = obs
        if isinstance(info, dict):
            self._agent_mask = info.get("agent_active_mask", self._agent_mask)
            self._context_mask = info.get("context_active_mask", self._context_mask)
        if terminated or truncated:
            obs, info = self.env.reset()
            self._obs = obs
            self._agent_mask = info.get("agent_active_mask")
            self._context_mask = info.get("context_active_mask")

    def _send_formation(self, game_state: GameState):
        """Send goto commands for all 6 robots to their formation positions."""
        from ai_interface.utils.basic_commands import goto

        playmode = game_state.playmode or ""
        ball_pos = game_state.ball_pos
        targets = _get_formation(playmode, self.our_side, ball_pos)

        # Build pose lookup: {robot_id: (x, y, theta_rad)}
        pose_by_id = {}
        for entry in game_state.robot_poses.get(self.team_name, []):
            if not isinstance(entry, dict):
                continue
            for rid, pose in entry.items():
                if pose is not None and len(pose) >= 3:
                    pose_by_id[int(rid)] = (
                        float(pose[0]), float(pose[1]), math.radians(float(pose[2]))
                    )

        # Build command list indexed 0..5 for robots 1..6
        max_robot_id = max(targets.keys()) if targets else 6
        cmds = [None] * max_robot_id
        for robot_id, target in targets.items():
            if target is None:
                continue
            pose = pose_by_id.get(robot_id)
            if pose is None:
                continue
            cmd = goto(
                self_pose=pose,
                x=target[0],
                y=target[1],
                game_state=game_state,
                is_goalie=(robot_id == 6),
            )
            cmds[robot_id - 1] = cmd

        self.networker.execute_ai_output(cmds, self.team_name)

    def _send_stop(self):
        """Halt all robots."""
        cmds = [None] * 6
        self.networker.execute_ai_output(cmds, self.team_name)

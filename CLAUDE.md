# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Codex adapter & sync contract

This project is also set up for **OpenAI Codex CLI**. Codex reads `AGENTS.md` (root) instead
of this file, with `CLAUDE.md` + `MEMORY.md` as fallbacks (`.codex/config.toml`
`project_doc_fallback_filenames`). `CLAUDE.md` remains the single source of project knowledge;
`AGENTS.md` is a thin adapter that points back here.

**Keep the two in sync. Whenever you change `CLAUDE.md` or add/edit a skill or agent, mirror it
for Codex in the same change:**

- **`CLAUDE.md` changed** → update `AGENTS.md` if the change affects anything it summarizes
  (rules, entry points, project map, commands).
- **Skill added/edited** under `.claude/skills/<name>/` → copy it (`SKILL.md` + supporting
  files, kept together) into `.agents/skills/<name>/`. Codex reads skills from
  `.agents/skills/`, **not** `.codex/skills/`.
- **Agent added/edited** under `.claude/agents/<name>.md` → mirror it to
  `.codex/agents/<name>.toml`, moving the markdown body into `developer_instructions`
  (convention documented in `.codex/agents/README.md`).
- **Durable project fact established** (a diagnosed failure, a settled decision, a "don't
  re-try this") → append a one-line bullet to `MEMORY.md`. That file is the checked-in,
  Codex-readable memory layer; Codex's own `~/.codex/memories/` is global, not project-aware.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- Don't run for any questions about the embedded or external simulator source code.

## Python environment

Run **all** training/inference/tests with the `rcai` conda env — bare `python3`/base conda lack `gymnasium` + `torch`:

```bash
/opt/anaconda3/envs/rcai/bin/python <script.py> ...
```

## Entry points

There are four scripts. The `launch_*` wrappers spawn external `rcssserver` instances and then run the underlying script as a subprocess (inheriting the launcher's `--python`); `train.py`/`infer.py` can also be run directly, and only they support the in-process `sim-embedded` backend.

### `train.py` — single training process
Drives one trainer against one simulator. Key args: `--trainer` (e.g. `ppo_jal_curriculum`, `td3_jal_her`, `discrete_ppo`), `--config <configs/*.json>`, `--env` (`sim-only` | `sim-embedded` | `sim-mixed` | `field-*`), `--timesteps`, `--load_model`, `--save_path`, `--log_dir` (default `train_logs`), `--num_envs`, `--sim_player_port`.

```bash
/opt/anaconda3/envs/rcai/bin/python train.py \
  --trainer ppo_jal_curriculum --config configs/ppo_jal_curriculum_config.json \
  --env sim-embedded --timesteps 200000
```

### `launch_train.py` — multi-sim training launcher
Spawns N external `rcssserver` (+ optional `rcssmonitor`) instances and a `train.py` per env. Args use **dashes**: `--num-envs`, `--base-port` (default 6000), `--port-stride` (10), `--env` (default `sim-embedded` — which skips external sim startup), `--trainer`, `--monitor`, `--sim-cmd`/`--sim-port-flag`.

```bash
/opt/anaconda3/envs/rcai/bin/python launch_train.py \
  --num-envs 2 --sim-cmd rcssserver --sim-port-flag "server::port=" --trainer ppo_jal_curriculum
```

### `infer.py` — single inference/eval process
Loads a checkpoint and drives `JALTeamEnv.step()` (so all env-resident behavior applies at inference automatically). Requires `--model_path` (a **flag**, not positional); key args: `--trainer` (default **`ppo_jal`**), `--config` (default `configs/td3_jal_her_config.json` — pass `configs/ppo_jal_curriculum_config.json` for PPO JAL stages or `--stage` lookups raise `KeyError`), `--stage` (selects env settings from the curriculum config), `--env` (supports `sim-embedded`), `--steps`, `--ppo_param_noise_std` (default **0.3**), `--ppo_stochastic`, `--debug_infer` (writes `step_trace.jsonl`). Writes `infer_logs/<run>/` with `infer_log.log` + `summary.json`.

```bash
/opt/anaconda3/envs/rcai/bin/python infer.py \
  --model_path models/ppo_jal_expandable_wide/stage2g_dribble_param_complete.pt \
  --trainer ppo_jal --config configs/ppo_jal_curriculum_config.json \
  --env sim-embedded --stage stage2g_dribble_param --steps 3000
```

### `launch_infer.py` — multi-sim inference launcher
Spawns external `rcssserver` + monitor (monitor on by default) and an `infer.py` per env. Defaults: `--trainer ppo_jal`, `--config configs/ppo_jal_curriculum_config.json`, `--stage stage2_0_baseline`, `--env sim-only`, `--steps 3000`. Does **not** expose `sim-embedded`. `--ppo_param_noise_std` defaults to `None` and is only forwarded if set, so infer.py's `0.3` applies.

```bash
/opt/anaconda3/envs/rcai/bin/python launch_infer.py \
  models/ppo_jal_expandable_wide/<ckpt>.pt --stage <stage> --env sim-only --steps 3000
```

> PPO JAL inference must **not** run deterministic-mean (causes a bang-bang turn stall). Keep `--ppo_param_noise_std` > 0 (0.3 default). The checkpoint path picks the model; `--stage` picks the env config — they must be consistent.

### `launch_sims.py` — bare simulators only
`python launch_sims.py --env <N> --monitor` launches N `rcssserver` (+ N `rcssmonitor`) with no AI attached.

## Two simulator backends (critical distinction)

- **embedded** (`--env sim-embedded`): `rcssserver_embedded` via `EmbeddedSimulatorBackend` in `networking/socket_utils.py`, run synchronously **in-process**. This is the default for training. It has **local engine patches** (e.g. field-player `catch`/dribble glue, native `drop`) that the stock server does not. `server.conf` overrides do **not** reach it.
- **external** (`--env sim-only`/`sim-mixed`/`field-*`): the stock `rcssserver` UDP binary (the `launch_*` path). `server.conf` overrides apply here. The embedded-only engine patches do **not** exist here, so behavior that depends on them (notably the `catch`-based dribble) will not reproduce. Watch for train(embedded)/infer(external) mismatches.

## Architecture

The control flow at training time is: **launcher → `train.py` → Trainer → Algorithm (policy) + Env → Networker → Simulator**.

- **Trainers** (`ai_interface/trainers/`, subclass `base_trainer.py:BaseTrainer`) own the rollout/update loop and curriculum. The active line of work is `ppo_jal_curriculum_trainer.py:PPOJALCurriculumTrainer`. Each `--trainer` choice maps to one file here.
- **Algorithms** (`ai_interface/algorithms/`, subclass `base.py:AlgorithmBase`) are the networks/update rules. `ppo_jal.py:PPOJALAgent` is the centralized **Joint Action Learner**: one team-level policy emits, per robot, a **discrete primitive** (goto / approach_ball / turn / kick / dribble_to) + **continuous params** (Dx, Dy, Dθ). The expandable backbone uses **one weight-shared per-robot encoder** (robots interchangeable; the goalie is hardcoded, not RL). This is intentionally centralized — do not refactor toward MAPPO/decentralized.
- **Environments** (`ai_interface/envs/`) are Gymnasium envs. `JAL_env.py:JALTeamEnv` is the main one; it decodes the hybrid action into sim commands, runs reward, and exposes `dribble_session_active` / carry telemetry. `infer.py` reuses this exact env, so env-resident changes affect both training and inference.
- **Reward** lives in `reward.py` (`RewardConfig` + `evaluate_reward`); per-stage `reward_config_overrides` in the curriculum config tune it without code changes.
- **Action primitives** are in `ai_interface/utils/basic_commands.py` (`goto`, `approach_ball`, `kick`, `shoot`, `dribble_to`). These emit raw rcssserver command strings (`dash`/`turn`/`kick`/`catch`).
- **Opponent/scripted control**: `ai_interface/goalie.py` + `trainers/policy_control.py` (`GoalieCommandProvider`, `ScriptedTeamCommandProvider`) drive the scripted keeper, stepped each cycle before `env.step`.
- **Networking** (`networking/`): `networker.py` coordinates teams; `socket_utils.py` holds both sim backends; `data_utils.py` defines `GameState` (the parsed world model passed everywhere).

### Curriculum config mechanics (important gotcha)
The curriculum trainer runs **every stage that has `timesteps > 0`**, in sorted order — not just one. The `"ACTIVE"` tag in a stage description only labels the `log-training` parser, **not** the trainer. To run a single stage, set `timesteps: 0` on all others. New `model_params` keys must also be added to the trainer's hparam whitelist or they're silently dropped. Top-level `load_model` sets the warm-start checkpoint.

## Tests

```bash
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/            # all
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/test_stage2_env.py -k <name>   # single
```

## Training log workflow

After a run finishes or is stopped, summarize and append to `docs/TRAINING.md`:

```bash
/opt/anaconda3/envs/rcai/bin/python scripts/parse_training_log.py [train_logs/<run>]
```

It auto-detects the latest run (or takes a path), extracts goal rate / action distribution / aim / outcomes / config diff, and emits a Markdown block to fill in (use the `log-training` skill for the full workflow). `docs/` (CHANGES.md, PARALLEL_TRAINING.md, PPO_EXPANDABLE_*.md, DRIBBLE_TO.md) holds design notes.

## Repo conventions

- Do not commit/push training logs, game logs, model checkpoints, or other non-code artifacts unless explicitly told to.
- Do not push changes under `robocup_downloads/` or `tests/`.
- Commit only when explicitly asked.

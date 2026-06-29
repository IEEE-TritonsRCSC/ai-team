# AGENTS.md

This is the **Codex-facing adapter** for this repository. Codex (OpenAI Codex CLI)
loads `AGENTS.md` automatically; Claude Code loads `CLAUDE.md`. To avoid drift, the
**single source of project knowledge is [CLAUDE.md](CLAUDE.md)** — read it first. This
file only adds the Codex-specific glue and a compact project map; it does **not**
duplicate the long simulator / environment / curriculum sections.

> **Sync contract:** whenever `CLAUDE.md` changes, or a skill is added/edited under
> `.claude/skills/`, mirror it for Codex (this file, `.agents/skills/`) and append any
> durable fact to [MEMORY.md](MEMORY.md). See the "Codex adapter & sync contract"
> section in `CLAUDE.md`.

## Read these first (in order)

1. **[CLAUDE.md](CLAUDE.md)** — full project knowledge: Python env, the 4 entry points,
   the two simulator backends, architecture, curriculum mechanics, tests, conventions.
2. **[MEMORY.md](MEMORY.md)** — accumulated, hard-won project facts (failure modes already
   diagnosed, what *not* to re-try). This is the checked-in, project-scoped memory layer
   that Codex reads (Codex's own `~/.codex/memories/` is global, not project-aware).
3. **`graphify-out/`** — knowledge graph. See the graphify section below.

## graphify (use it for codebase questions)

This project has a knowledge graph at `graphify-out/`. The `graphify` CLI is installed at
`~/.local/bin/graphify` (on PATH). Use it instead of blind grepping:

- For codebase questions, first run `graphify query "<question>"` when
  `graphify-out/graph.json` exists. It returns a scoped subgraph, usually much smaller than
  raw grep output or `graphify-out/GRAPH_REPORT.md`.
- Use `graphify path "<A>" "<B>"` for relationships between two things, and
  `graphify explain "<concept>"` for a focused concept.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source
  browsing. Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when
  query/path/explain don't surface enough.
- **After modifying code, run `graphify update .`** to keep the graph current (AST-only, no
  API cost).
- **Don't** run graphify for questions about the embedded or external simulator source code.

> Sandbox note: the `graphify` binary lives outside the workspace (`~/.local/bin/`) and writes
> its index into `graphify-out/` inside the workspace. Under `sandbox_mode = "workspace-write"`
> Codex may prompt before running it — approve it; it only reads the repo and writes
> `graphify-out/`.

## Critical rules (do not violate)

- **Python:** run **all** training/inference/tests with the `rcai` conda env —
  `/opt/anaconda3/envs/rcai/bin/python`. Bare `python3`/base conda lack `gymnasium`+`torch`.
- **Simulator backends differ:** `sim-embedded` (in-process, has local engine patches) vs
  external `rcssserver` (`sim-only`/`sim-mixed`/`field-*`). `server.conf` overrides reach
  only the external one; the catch-glue dribble exists only in embedded. Watch
  train(embedded)/infer(external) mismatches. (Full detail in CLAUDE.md.)
- **Curriculum gotcha:** the curriculum trainer runs **every** stage with `timesteps > 0`,
  in sorted order — not just the one tagged `ACTIVE`. To run a single stage, set
  `timesteps: 0` on all others. New `model_params` keys must be added to the trainer's
  hparam whitelist or they're silently dropped.
- **Centralized JAL only** — do not refactor toward MAPPO/decentralized.
- **Do not commit** training logs, game logs, model checkpoints, or other non-code
  artifacts unless explicitly told. Commit only when asked. Add files by name (never
  `git add .`). Never `git push --force`.

## Project map

```
ai-team/
├── train.py / launch_train.py        # single / multi-sim TRAINING entry points
├── infer.py / launch_infer.py        # single / multi-sim INFERENCE entry points
├── launch_sims.py                    # bare simulators, no AI attached
├── configs/                          # *.json — ppo_jal_curriculum_config.json is active
├── ai_interface/
│   ├── trainers/                     # rollout/update loop + curriculum (BaseTrainer)
│   │   └── ppo_jal_curriculum_trainer.py   # active line of work
│   ├── algorithms/                   # networks/update rules (AlgorithmBase)
│   │   └── ppo_jal.py                # centralized Joint Action Learner (PPOJALAgent)
│   ├── envs/                         # Gymnasium envs
│   │   ├── JAL_env.py                # main env (JALTeamEnv) — reused by infer.py
│   │   ├── reward.py                 # RewardConfig + evaluate_reward
│   │   └── stage4_support.py         # Stage-4 support-target classification
│   ├── utils/basic_commands.py       # action primitives (goto/approach_ball/kick/...)
│   ├── goalie.py                     # scripted keeper
│   └── trainers/policy_control.py    # scripted opponent/keeper providers
├── networking/                       # networker.py, socket_utils.py (both sim backends),
│                                     #   data_utils.py (GameState world model)
├── scripts/parse_training_log.py     # training-log summarizer (see log-training skill)
├── docs/                             # TRAINING.md, CHANGES.md, design notes
├── tests/                            # pytest suite
├── graphify-out/                     # knowledge graph (query/path/explain/update)
├── CLAUDE.md / AGENTS.md / MEMORY.md  # Claude / Codex adapters + project memory
├── .claude/  (skills, agents, settings)   # Claude Code config
├── .agents/skills/                   # skills mirrored for Codex (see below)
└── .codex/   (config.toml, agents/)  # Codex config + agents
```

Control flow at training time:
**launcher → `train.py` → Trainer → Algorithm (policy) + Env → Networker → Simulator.**

## Common commands

```bash
# Train one stage (embedded) — set timesteps:0 on all other curriculum stages first
/opt/anaconda3/envs/rcai/bin/python train.py \
  --trainer ppo_jal_curriculum --config configs/ppo_jal_curriculum_config.json \
  --env sim-embedded --timesteps 200000

# Inference (env-resident behavior applies automatically). Keep param noise > 0.
/opt/anaconda3/envs/rcai/bin/python infer.py \
  --model_path <ckpt>.pt --trainer ppo_jal \
  --config configs/ppo_jal_curriculum_config.json \
  --env sim-embedded --stage <stage> --steps 3000

# Tests
/opt/anaconda3/envs/rcai/bin/python -m pytest tests/

# Summarize the latest training run (then append to docs/TRAINING.md)
/opt/anaconda3/envs/rcai/bin/python scripts/parse_training_log.py
```

> PPO JAL inference must **not** run deterministic-mean (bang-bang turn stall): keep
> `--ppo_param_noise_std > 0` (0.3 default) and prefer `--ppo_stochastic`.

## Skills (Codex)

Project skills are mirrored from `.claude/skills/` into **`.agents/skills/`** (Codex reads
skills from `.agents/skills/`, **not** `.codex/skills/`). Each keeps its `SKILL.md` and
supporting files together:

- **`log-training`** — after a run finishes/stops, run `scripts/parse_training_log.py`,
  fill in the three narrative sections, append to `docs/TRAINING.md`.
- **`save-plan`** — archive the most recent executed plan into `plans/` with a dated,
  stage-labelled filename (`save_plan.sh`).

## Agents (Codex)

Claude agents (`.claude/agents/*.md`) convert to Codex agents (`.codex/agents/*.toml`)
with the agent body becoming `developer_instructions`. There are currently no project
agents to convert; `.codex/agents/README.md` documents the convention for when one is added.

## Memory in Codex (how it works here)

- Codex's built-in memories are **global** (`~/.codex/memories/`), system-generated, and
  **cannot read a project file**. They're enabled in `.codex/config.toml`
  (`[features] memories = true`) as a cross-session recall layer.
- The **project-scoped** memory that Codex actually reads is the checked-in
  **[MEMORY.md](MEMORY.md)**, wired in via `project_doc_fallback_filenames` in
  `.codex/config.toml`. Treat `MEMORY.md` (+ `CLAUDE.md`) as the authoritative,
  always-applied rules; treat global memories as helpful local recall only.

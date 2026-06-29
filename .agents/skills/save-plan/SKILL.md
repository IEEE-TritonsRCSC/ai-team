---
name: save-plan
description: Archive the most recently executed plan into the repo's plans/ directory. Use after a plan is approved/executed to keep a dated, stage-labelled record of training plans. Copies the latest ~/.claude/plans file into plans/<date>_<time>_<stage>_<title>.md.
---

# save-plan

Save the most recently executed plan into the repo's `plans/` directory as a dated,
stage-labelled markdown file. This keeps a running record of every training plan we've
executed — what we tried, when, and for which stage — so progress is easy to trace later.

Plans live transiently in `~/.claude/plans/` (Claude Code's plan-mode store). This skill
copies the latest one into the repo so it survives and is searchable alongside `TRAINING.md`.

All paths below are relative to the repo root (`/Users/Adnan/Desktop/RoboCup/ai-team/`).

## Naming scheme

Files are written to `plans/` as:

```
<YYYYMMDD>_<HHMMSS>_<stage-token>_<title-slug>.md
```

- **date/time** — taken from the plan file's last-modified time (when the plan was
  finalised/executed), handled automatically by the driver.
- **stage-token** — short stage + sub-stage id, kebab-case. Examples: `stage2a`,
  `stage2c-goalie-balance`, `stage1`. You supply this.
- **title-slug** — concise kebab-case description of the plan. Examples:
  `selective-dribbling-reward-curriculum`, `gk-positioning-fix`. You supply this.

Example: `plans/20260619_115900_stage2a_selective-dribbling-reward-curriculum.md`

## Step 1 — Inspect the latest plan

```bash
.claude/skills/save-plan/save_plan.sh --show
```

This prints the latest plan's path, its formatted mtime, the destination `plans/` dir, and
the full plan content. **Read the content** to decide the two labels:

- **stage-token:** find which stage the plan targets. Look for "STAGE 2A", "stage2c", etc.
  in the plan body. If the plan doesn't name a stage, cross-check the `ACTIVE` stage in
  `configs/ppo_jal_curriculum_config.json` (exactly one stage description starts with
  `ACTIVE`). Map e.g. "STAGE 2A" → `stage2a`, the config stage `stage2c_goalie_balance` →
  `stage2c-goalie-balance`.
- **title-slug:** derive from the plan's top `#` heading — a short kebab-case phrase, no
  stage number duplication (the stage is already its own token).

## Step 2 — Archive it

```bash
.claude/skills/save-plan/save_plan.sh <stage-token> <title-slug>
```

Example:

```bash
.claude/skills/save-plan/save_plan.sh stage2a selective-dribbling-reward-curriculum
```

The driver copies the latest plan to
`plans/<date>_<time>_<stage-token>_<title-slug>.md` and prints the source and destination.
Report the destination path back to the user.

## Notes

- "Most recent plan" = newest file by modification time in `~/.claude/plans/`. If the user
  means a different plan, list `~/.claude/plans/` and pick by name, then pass that file via
  the `PLANS_SRC` env var or copy it manually using the same naming scheme.
- The driver does **not** overwrite-protect: if you run it twice with the same labels and
  the plan's mtime hasn't changed, it re-copies to the same filename (idempotent). Different
  labels produce a new file.
- This skill only records plans. Use `/log-training` to record the *results* of running them.

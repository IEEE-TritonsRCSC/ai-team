---
name: log-training
description: Log a training run to docs/TRAINING.md. Use after any training finishes or is stopped. Extracts goal rate, action distribution, aim quality, config, and episode outcomes. Agent fills in what went wrong and the fix for next run.
---

# log-training

After a training run finishes (or is stopped early), run the parser to extract the key numbers, then append a structured entry to `docs/TRAINING.md`. The goal is to track what each run tried, what it achieved, what failed, and what the fix is — so we never repeat the same mistake twice.

All paths are relative to the repo root (`/Users/Adnan/Desktop/RoboCup/ai-team/`).

## Step 1 — Extract the run summary

```bash
python scripts/parse_training_log.py
```

This reads the **latest** run in `train_logs/`. To target a specific run:

```bash
python scripts/parse_training_log.py train_logs/20260521_020942_188642
```

The script outputs a Markdown block containing:
- Start time, stage name, steps completed vs planned, warm-start model
- Aim of the stage (from the stage description in the config)
- Overall goal rate + last-100-episode goal rate
- Goal rate bucketed by 200-episode windows (shows trend over training)
- Episode outcome breakdown (goal_scored / off-target / OOB / max_steps / etc.)
- Aim quality for first 200 kicks vs last 200 kicks (tracks whether aim improved)
- Action distribution samples at regular intervals (shows what actions the policy favoured)
- Full reward overrides and key model params active during the run

## Step 2 — Fill in the three narrative sections

The output has three placeholder sections at the bottom:

```
### Code changes (non-config)
<!-- Fill in -->

### What went wrong
<!-- Fill in -->

### Fix for next run
<!-- Fill in -->
```

Fill these in based on what you observed:

**Code changes (non-config)** — any change made to source code that is NOT captured by the reward overrides / model params diff. Examples:
- "Removed kicker cone gate in JAL_env.py:891 — kicks now fire whenever ball is in kickable range."
- "Added `ball_dead_*` termination check in JAL_env.py:_check_terminal with -3 penalty."
- "Patched SB3 HerReplayBuffer._sample_goals to clamp upper bound."

Leave this section blank (or delete it) if no source files outside the config were modified.

**What went wrong** — describe the failure mode with specific evidence from the numbers above. Examples:
- "Model idles at ball: action distribution shows approach_ball 90%+ in final episodes; deterministic inference shows no kicks."
- "Aim not improving: bad_aim rate stuck at 51% first 200 kicks vs 43% last 200 — only 8pp improvement over 200k steps."
- "Policy collapsed to kick=100%: action dist shows kick 100% from episode 200 onward, goal rate flat at 9%."

If the run **succeeded**, rename this heading to **"What worked"** and note the key signal (e.g. "Goal rate climbed from 9% → 52% over 200k steps; aim quality improved from bad_aim=51% → 38%").

**Fix for next run** — describe the config change(s) with the reasoning:
- What parameter changed and in which direction
- Why that change addresses the root cause (not just "to improve aim")
- Reference the reward EV math if the change affects reward shaping

If the run succeeded, rename this heading to **"Next stage"** and note what curriculum step comes next.

## Step 3 — Append to docs/TRAINING.md

Add the completed block as a new numbered section at the end of `docs/TRAINING.md`. Sections are numbered sequentially (§7, §8, etc.). Use a `---` separator before the new section.

Keep the tone simple and factual. The audience is a future agent (or teammate) who needs to understand:
1. What was tried
2. What the numbers showed
3. Why it failed (or succeeded)
4. What to do differently next time

## Notes

- The parser auto-detects the active stage by looking for `"ACTIVE"` in the stage description field. **Exactly one** stage should be marked ACTIVE. If zero or multiple are marked, the parser prints a `WARNING:` to stderr and also injects a `⚠️ Config warning` banner at the top of the rendered block — fix the config before logging the run so the wrong stage's aim/overrides don't get recorded.
- Short runs (< 200 episodes) will have only one bucket in the goal-rate chart — that's fine.
- The `step_bonus` in reward overrides shows `0.0` vs `-0.2` — useful for cross-run comparison.
- If training was stopped early, `steps completed / planned` will show the shortfall. Note why it was stopped in the "What went wrong" section.
- The "Code changes (non-config)" section exists so source-level edits (e.g. removing the cone gate in `JAL_env.py`) are tracked alongside the config diff — config diffs alone won't show these.

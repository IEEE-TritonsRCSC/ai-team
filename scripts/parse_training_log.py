"""
Parse a TD3-JAL training log and print a structured summary for use in TRAINING.md.

Usage:
    python scripts/parse_training_log.py                      # latest run
    python scripts/parse_training_log.py train_logs/<dir>     # specific run
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def find_latest_log_dir(base="train_logs") -> Path:
    dirs = sorted(Path(base).iterdir(), key=lambda p: p.name)
    for d in reversed(dirs):
        if (d / "train_log.log").exists():
            return d
    raise FileNotFoundError("No train_log.log found under train_logs/")


def parse(log_path: Path) -> dict:
    lines = log_path.read_text().splitlines()

    result = {
        "log_dir": str(log_path.parent.name),
        "start_time": None,
        "config": {},
        "active_stage": None,
        "active_stage_warning": None,
        "stage_description": None,
        "total_episodes": 0,
        "total_steps": 0,
        "planned_steps": None,
        "end_reasons": Counter(),
        "end_reasons_ordered": [],        # ordered list for last-N slicing
        "goal_rate_buckets": [],          # list of (range_str, goal_pct)
        "action_dist_samples": [],        # list of (episode, dist_str)
        "aim_quality_first": None,        # (avg, bad_pct) from first 200 kicks
        "aim_quality_last": None,         # (avg, bad_pct) from last 200 kicks
        "avg_reward_last": None,
        "avg_length_last": None,
        "load_model": None,
        "reward_overrides": {},
        "model_params": {},
    }

    # ── config block (multi-line JSON after "Configuration: {") ──────────
    config_lines = []
    in_config = False
    brace_depth = 0
    for line in lines:
        if "Configuration: {" in line:
            in_config = True
            config_lines.append("{")
            brace_depth = 1
            continue
        if in_config:
            config_lines.append(line)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                in_config = False
                break

    if config_lines:
        try:
            cfg = json.loads("\n".join(config_lines))
            result["config"] = cfg
            result["load_model"] = cfg.get("load_model")
            result["model_params"] = cfg.get("model_params", {})
            curriculum = cfg.get("curriculum", {})

            # Find all stages marked ACTIVE
            active_stages = [
                (name, scfg) for name, scfg in curriculum.items()
                if "ACTIVE" in scfg.get("description", "")
            ]

            if len(active_stages) == 1:
                stage_name, stage_cfg = active_stages[0]
                result["active_stage"] = stage_name
                result["stage_description"] = stage_cfg.get("description", "")
                result["planned_steps"] = stage_cfg.get("timesteps")
                result["reward_overrides"] = stage_cfg.get("reward_config_overrides", {})
            elif len(active_stages) == 0 and curriculum:
                stage_name = next(iter(curriculum))
                stage_cfg = curriculum[stage_name]
                result["active_stage"] = stage_name
                result["stage_description"] = stage_cfg.get("description", "")
                result["planned_steps"] = stage_cfg.get("timesteps")
                result["reward_overrides"] = stage_cfg.get("reward_config_overrides", {})
                result["active_stage_warning"] = (
                    f"No stage marked 'ACTIVE' in config — fell back to first stage "
                    f"'{stage_name}'. Aim/overrides shown may be wrong."
                )
            elif len(active_stages) > 1:
                names = [n for n, _ in active_stages]
                stage_name, stage_cfg = active_stages[0]
                result["active_stage"] = stage_name
                result["stage_description"] = stage_cfg.get("description", "")
                result["planned_steps"] = stage_cfg.get("timesteps")
                result["reward_overrides"] = stage_cfg.get("reward_config_overrides", {})
                result["active_stage_warning"] = (
                    f"Multiple stages marked 'ACTIVE' ({names}) — used first ('{stage_name}'). "
                    f"Fix the config so exactly one stage is ACTIVE."
                )
        except json.JSONDecodeError:
            pass

    # ── line-by-line ─────────────────────────────────────────────────────
    all_end_reasons = []
    kick_aim_qualities = []
    kick_bad_aim = []
    episode_100_counter = 0
    action_dist_every = 200   # sample every N episodes

    for line in lines:
        # start time
        if result["start_time"] is None:
            m = re.search(r"Training session started at (.+)", line)
            if m:
                result["start_time"] = m.group(1).strip()

        # episode ended
        m = re.search(r"Episode \d+ ended — reason=(\S+)\s+steps=(\d+)", line)
        if m:
            result["total_episodes"] += 1
            all_end_reasons.append(m.group(1))
            result["end_reasons_ordered"].append(m.group(1))
            episode_100_counter += 1

        # total_timesteps
        m = re.search(r"total_timesteps=(\d+)", line)
        if m:
            result["total_steps"] = int(m.group(1))

        # action distribution (sample every action_dist_every episodes)
        m = re.search(r"Episode (\d+) action distribution \(\d+ steps\): (.+)", line)
        if m:
            ep = int(m.group(1))
            dist = m.group(2).strip()
            if ep % action_dist_every == 0 or ep == 1:
                result["action_dist_samples"].append((ep, dist))

        # kick aim quality
        m = re.search(r"aim_quality=([\d.]+)\s+bad_aim=(True|False)", line)
        if m:
            kick_aim_qualities.append(float(m.group(1)))
            kick_bad_aim.append(m.group(2) == "True")

        # last 10 ep average
        m = re.search(r"Last 10 episodes - Avg Reward: ([\d.]+), Avg Length: ([\d.]+)", line)
        if m:
            result["avg_reward_last"] = float(m.group(1))
            result["avg_length_last"] = float(m.group(2))

    result["end_reasons"] = Counter(all_end_reasons)

    # ── aim quality first / last 200 kicks ───────────────────────────────
    def _aim_stats(quals, bads):
        if not quals:
            return None
        avg = sum(quals) / len(quals)
        bad_pct = 100.0 * sum(bads) / len(bads)
        return round(avg, 3), round(bad_pct, 1)

    result["aim_quality_first"] = _aim_stats(kick_aim_qualities[:200], kick_bad_aim[:200])
    result["aim_quality_last"] = _aim_stats(kick_aim_qualities[-200:], kick_bad_aim[-200:])

    # ── goal-rate buckets (200-episode windows) ───────────────────────────
    bucket_size = 200
    for i in range(0, len(all_end_reasons), bucket_size):
        bucket = all_end_reasons[i:i + bucket_size]
        goals = bucket.count("goal_scored")
        pct = round(100.0 * goals / len(bucket), 1)
        result["goal_rate_buckets"].append(
            (f"eps {i+1:4d}-{i+len(bucket):4d}", pct)
        )

    return result


def render(r: dict) -> str:
    lines = []
    a = lines.append

    total_eps = r["total_episodes"]
    total_goals = r["end_reasons"].get("goal_scored", 0)
    overall_rate = round(100.0 * total_goals / total_eps, 1) if total_eps else 0

    last_100_reasons = list(r["end_reasons"].elements())[-100:] if total_eps >= 100 else []
    last_100_goals = last_100_reasons.count("goal_scored") if last_100_reasons else 0

    a(f"## Training run: {r['log_dir']}")
    a(f"")
    if r.get("active_stage_warning"):
        a(f"> ⚠️ **Config warning:** {r['active_stage_warning']}")
        a(f"")
    a(f"**Start time:** {r['start_time']}")
    a(f"**Stage:** `{r['active_stage']}`")
    a(f"**Steps:** {r['total_steps']:,} / {r['planned_steps']:,}" if r['planned_steps'] else f"**Steps:** {r['total_steps']:,}")
    a(f"**Load model:** `{r['load_model']}`")
    a(f"")

    # aim of training (from stage description, shortened)
    if r["stage_description"]:
        # Strip the "ACTIVE (timesteps=...). " prefix
        desc = re.sub(r"^ACTIVE \([^)]+\)\.\s*", "", r["stage_description"])
        a(f"**Aim:** {desc}")
        a(f"")

    a(f"### Results")
    a(f"")
    a(f"| Metric | Value |")
    a(f"|---|---|")
    a(f"| Total episodes | {total_eps} |")
    a(f"| Overall goal rate | {total_goals}/{total_eps} = **{overall_rate}%** |")
    if total_eps >= 100:
        last_100_reasons = r["end_reasons_ordered"][-100:]
        last_100_goal_pct = round(100.0 * last_100_reasons.count("goal_scored") / 100, 1)
        a(f"| Last 100 episodes goal rate | **{last_100_goal_pct}%** |")
    a(f"| Avg reward (last 10 eps) | {r['avg_reward_last']} |")
    a(f"| Avg episode length (last 10 eps) | {r['avg_length_last']} |")
    a(f"")

    a(f"**Episode outcomes:**")
    a(f"")
    for reason, count in sorted(r["end_reasons"].items(), key=lambda x: -x[1]):
        pct = round(100.0 * count / total_eps, 1) if total_eps else 0
        a(f"- `{reason}`: {count} ({pct}%)")
    a(f"")

    a(f"**Goal rate over training (200-episode buckets):**")
    a(f"")
    a(f"```")
    for label, pct in r["goal_rate_buckets"]:
        bar = "█" * int(pct / 5)
        a(f"  {label}: {pct:5.1f}%  {bar}")
    a(f"```")
    a(f"")

    if r["aim_quality_first"] or r["aim_quality_last"]:
        a(f"**Aim quality (kicks):**")
        a(f"")
        a(f"| | avg aim_quality | bad_aim rate |")
        a(f"|---|---|---|")
        if r["aim_quality_first"]:
            aq, bp = r["aim_quality_first"]
            a(f"| First 200 kicks | {aq} | {bp}% |")
        if r["aim_quality_last"]:
            aq, bp = r["aim_quality_last"]
            a(f"| Last 200 kicks | {aq} | {bp}% |")
        a(f"")

    if r["action_dist_samples"]:
        a(f"**Action distribution samples:**")
        a(f"")
        for ep, dist in r["action_dist_samples"][-6:]:
            a(f"- ep {ep}: {dist}")
        a(f"")

    a(f"### Key config")
    a(f"")
    overrides = r["reward_overrides"]
    params = r["model_params"]
    if overrides:
        a(f"**Reward overrides:**")
        for k, v in overrides.items():
            if not k.startswith("_"):
                a(f"- `{k}`: {v}")
        a(f"")
    if params:
        interesting = ["learning_rate", "action_noise_std", "gamma", "batch_size",
                       "her_goal_selection_strategy", "her_n_sampled_goal"]
        a(f"**Model params:**")
        for k in interesting:
            if k in params:
                a(f"- `{k}`: {params[k]}")
        a(f"")

    a(f"### Code changes (non-config)")
    a(f"")
    a(f"<!-- Fill in: any changes to env (JAL_env.py), reward (reward.py), trainer, or algorithm code")
    a(f"     that are NOT captured in the config diff above. Examples: cone gate removed, termination")
    a(f"     check added, action-decoding logic changed. Leave blank if none. -->")
    a(f"")
    a(f"### What went wrong")
    a(f"")
    a(f"<!-- Fill in: describe the failure mode observed, with evidence from the logs above.")
    a(f"     If the run succeeded, rename this to 'What worked' and note the key signal. -->")
    a(f"")
    a(f"### Fix for next run")
    a(f"")
    a(f"<!-- Fill in: what config/reward/algorithm change addresses the root cause, and why.")
    a(f"     If the run succeeded, rename this to 'Next stage' and note what comes next. -->")
    a(f"")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_dir = Path(sys.argv[1])
        if not (log_dir / "train_log.log").exists():
            log_dir = log_dir / "train_log.log"
            if not log_dir.exists():
                print(f"ERROR: no train_log.log at {sys.argv[1]}", file=sys.stderr)
                sys.exit(1)
            log_dir = log_dir.parent
    else:
        log_dir = find_latest_log_dir()

    log_path = log_dir / "train_log.log"
    result = parse(log_path)
    if result.get("active_stage_warning"):
        print(f"WARNING: {result['active_stage_warning']}", file=sys.stderr)
    print(render(result))

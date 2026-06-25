#!/usr/bin/env python3
"""Orchestrate parallel simulator instances and N inference processes.

For ``sim-only``, launches one external rcssserver instance per model on
consecutive port slots, waits for them to initialise, then starts one
infer.py process per model with matching ports. For ``sim-embedded``, starts
one infer.py subprocess per model; each subprocess owns an independent
in-process embedded simulator. All child processes are stopped cleanly when
this script exits.

USAGE
-----
    python launch_infer.py <model1> [<model2> <model3> ...] [options]

One sim + one infer process is launched per model path. With an external
server, N models use consecutive port pairs (6000/6001, 6010/6011, ...).
With the embedded backend, N independent simulator engines run in N infer.py
subprocesses and do not use network ports.

COMMON OPTIONS
--------------
    --steps N            inference steps per model (default 3000)
    --base-port PORT     base player port for env 0 (default 6000)
    --port-stride N      gap between consecutive envs (default 10, min 3)
    --config <path>      training config (default configs/ppo_jal_curriculum_config.json)
    --stage <name>       curriculum stage to mirror (default stage2_0_baseline)
    --env MODE           simulator backend (sim-only or sim-embedded)
    --log-root PATH      parent directory for per-instance logs
    --no-monitor         skip rcssmonitor windows (headless; external only)
    --debug_infer        forward debug flag to each infer.py

Port layout (matches launch_train.py / base_trainer._sim_endpoint_for_env):
    env 0 : player=BASE_PORT,          trainer=BASE_PORT+1
    env 1 : player=BASE_PORT+STRIDE,   trainer=BASE_PORT+STRIDE+1
    ...

Each infer.py writes to its own infer_logs/<timestamp>_env<i>_<model>/
directory (log file + summary.json). When all inferers finish, this
script reads each summary.json and prints a comparison table.

EXAMPLES
--------
    # Single run with auto-launched sim + monitor window
    python launch_infer.py models/<datetime>_td3_jal_her/stage1_9_approach_turn_kick_complete.zip --steps 3000

    # Headless single run, 7000 steps
    python launch_infer.py models/<datetime>_td3_jal_her/stage1_9_approach_turn_kick_complete.zip \\
        --steps 7000 --no-monitor

    # Compare two checkpoints in parallel using independent embedded sims
    python launch_infer.py \\
        models/<datetime>_td3_jal_her/stage1_9_approach_turn_kick_complete.zip \\
        models/<datetime>_td3_jal_her/<other_checkpoint>.zip \\
        --env sim-embedded --steps 7000

    # Run alongside another sim already on 6000 - use a different base port
    python launch_infer.py models/<datetime>_td3_jal_her/stage1_9_approach_turn_kick_complete.zip \\
        --steps 7000 --base-port 7000
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Reuse the battle-tested sim helpers from launch_train.py
from launch_train import (
    build_sim_command,
    command_missing_message,
    player_port,
    stop_processes,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch N simulators + N infer.py processes (one per model).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "model_paths", nargs="+",
        help="One or more model checkpoint paths. One sim + one infer process "
             "is launched per model.",
    )

    # Inference args forwarded to each infer.py
    parser.add_argument("--trainer", default="ppo_jal",
                        help="Trainer type for infer.py (default: ppo_jal, expandable JAL path)")
    parser.add_argument("--config", default="configs/ppo_jal_curriculum_config.json",
                        help="Training config for stage settings (default: "
                             "configs/ppo_jal_curriculum_config.json)")
    parser.add_argument("--stage", default="stage2_0_baseline",
                        help="Curriculum stage to mirror (default: "
                             "stage2_0_baseline)")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Steps per infer process (default: 3000)")
    parser.add_argument("--team_config", default="team_config.json")
    parser.add_argument("--team", default="TritonBots")
    parser.add_argument(
        "--log-root",
        default="infer_logs",
        help="Parent directory for per-instance inference logs "
             "(default: infer_logs).",
    )
    parser.add_argument(
        "--env",
        default="sim-only",
        choices=["sim-only", "sim-embedded", "sim-mixed",
                 "field-practice", "field-tournament"],
        help="Environment mode (default: sim-only). In sim-embedded mode, "
             "every model runs in a separate infer.py process with its own "
             "embedded simulator.",
    )
    parser.add_argument("--debug_infer", action="store_true")
    parser.add_argument("--ppo_stochastic", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="PPO JAL: forward --ppo_stochastic (sample the discrete "
                             "primitive + params), matching how the policy trained. ON by "
                             "default — argmax-primitive inference makes PPO-JAL lock onto "
                             "dribble_to, hold the catch-glued ball and dead-ball-freeze. "
                             "Use --no-ppo_stochastic to force argmax (debugging only).")
    parser.add_argument("--ppo_param_noise_std", type=float, default=None,
                        help="PPO JAL: override the turn param-noise std forwarded to "
                             "infer.py. If unset, infer.py's default (0.3) applies.")

    # Port layout (mirrors launch_train.py)
    parser.add_argument("--base-port", type=int, default=6000)
    parser.add_argument("--port-stride", type=int, default=10,
                        help="Port gap between consecutive envs (min 3)")

    # Sim binary
    parser.add_argument("--sim-cmd", default="rcssserver")
    parser.add_argument("--sim-port-flag", default="server::port=")
    parser.add_argument("--sim-wait", type=float, default=2.0,
                        help="Seconds to wait after launching sims (default: 2.0)")
    parser.add_argument("--sim-delay", type=float, default=0.2,
                        help="Seconds between sequential sim launches (default: 0.2)")

    # Monitor (visual sim window)
    parser.add_argument(
        "--monitor", action=argparse.BooleanOptionalAction, default=True,
        help="Launch rcssmonitor for each env (default: on). Use --no-monitor "
             "to run headless.",
    )
    parser.add_argument("--monitor-cmd", default="rcssmonitor",
                        help='Monitor command (default: "rcssmonitor")')

    # Python interpreter / infer script
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--infer-script", default=str(Path(__file__).parent / "infer.py"))

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-env log dir
# ---------------------------------------------------------------------------

def make_env_log_dir(idx: int, model_path: str, ts: str,
                     root: str = "infer_logs") -> Path:
    stem = Path(model_path).stem
    out = Path(root) / f"{ts}_env{idx}_{stem}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Wait helpers
# ---------------------------------------------------------------------------

def wait_for_all(procs: list[subprocess.Popen]) -> int:
    """Block until every proc exits. Return first non-zero exit code seen."""
    exit_code = 0
    reported: set[int] = set()
    while True:
        for proc in procs:
            rc = proc.poll()
            if rc is None or proc.pid in reported:
                continue
            reported.add(proc.pid)
            if rc != 0 and exit_code == 0:
                exit_code = rc
                print(f"[launcher] PID {proc.pid} exited with code {rc}.",
                      file=sys.stderr)
        if all(p.poll() is not None for p in procs):
            break
        time.sleep(1.0)
    return exit_code


# ---------------------------------------------------------------------------
# Final summary comparison
# ---------------------------------------------------------------------------

def print_comparison(log_dirs: list[Path]) -> None:
    """Read summary.json from each log_dir and print a side-by-side table."""
    rows = []
    for d in log_dirs:
        sj = d / "summary.json"
        if not sj.exists():
            rows.append((d.name, None))
            continue
        try:
            with sj.open() as f:
                rows.append((d.name, json.load(f)))
        except Exception as exc:  # noqa: BLE001
            print(f"[launcher] Could not read {sj}: {exc}", file=sys.stderr)
            rows.append((d.name, None))

    print()
    print("=" * 92)
    print("Inference comparison")
    print("=" * 92)
    header = f"{'Env':<48} {'Eps':>5} {'Goal%':>7} {'AvgR':>8} {'Aim':>6} {'BadAim%':>8}"
    print(header)
    print("-" * 92)
    for name, payload in rows:
        if not payload or not payload.get("all"):
            print(f"{name:<48} (no summary.json — likely crashed early)")
            continue
        a = payload["all"]
        goal_pct = a.get("goal_rate", 0.0) * 100
        bad_pct = (a.get("bad_aim_rate") or 0.0) * 100
        print(f"{name:<48} {a['n']:>5} {goal_pct:>6.1f}% "
              f"{a['avg_reward']:>8.1f} {a['aim_avg']:>6.3f} {bad_pct:>7.1f}%")
    print("=" * 92)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.env != "sim-embedded" and args.port_stride < 3:
        print("--port-stride must be >= 3 to avoid sim port conflicts",
              file=sys.stderr)
        return 1

    num_envs = len(args.model_paths)
    embedded = args.env == "sim-embedded"
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_procs: list[subprocess.Popen] = []
    sim_procs: list[subprocess.Popen] = []
    monitor_procs: list[subprocess.Popen] = []
    infer_procs: list[subprocess.Popen] = []
    log_dirs: list[Path] = []

    try:
        monitor_parts = (
            shlex.split(args.monitor_cmd)
            if args.monitor and not embedded
            else []
        )
    except ValueError as exc:
        print(f"Invalid --monitor-cmd: {exc}", file=sys.stderr)
        return 1

    if embedded:
        print(
            f"[launcher] Starting {num_envs} independent embedded sim + infer "
            "process(es); external ports and monitors are not used.",
            flush=True,
        )
        if args.monitor:
            print("[launcher] --monitor is ignored for sim-embedded.",
                  file=sys.stderr)
    else:
        print(
            f"[launcher] Starting {num_envs} sim(s) + {num_envs} infer "
            f"process(es) (base-port={args.base_port}, "
            f"stride={args.port_stride})",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 1. Prepare every env and launch external simulators when required.
    # Embedded simulators are constructed inside the infer subprocesses so
    # each C++ engine and its process-global state remain isolated.
    # ------------------------------------------------------------------
    try:
        for idx, model_path in enumerate(args.model_paths):
            port = player_port(args.base_port, args.port_stride, idx)
            log_dir = make_env_log_dir(idx, model_path, ts, root=args.log_root)
            log_dirs.append(log_dir)

            if embedded:
                print(
                    f"[launcher] env {idx}: embedded simulator will run inside "
                    f"infer process  log_dir={log_dir}"
                )
                continue

            try:
                sim_parts = build_sim_command(args.sim_cmd, args.sim_port_flag, port)
            except ValueError as exc:
                print(f"[launcher] {exc}", file=sys.stderr)
                stop_processes(all_procs)
                return 1

            sim_log = (log_dir / "rcssserver.log").open("w")
            try:
                sim_proc = subprocess.Popen(sim_parts, stdout=sim_log, stderr=sim_log)
            except FileNotFoundError:
                print(command_missing_message(args.sim_cmd, "--sim-cmd"),
                      file=sys.stderr)
                stop_processes(all_procs)
                return 1
            sim_procs.append(sim_proc)
            all_procs.append(sim_proc)
            print(f"[launcher] env {idx}: sim pid={sim_proc.pid}  "
                  f"player_port={port}  trainer_port={port + 1}  "
                  f"log_dir={log_dir}")

            # Optional rcssmonitor window for this env
            if args.monitor and monitor_parts:
                monitor_cmd = monitor_parts + [
                    "--server-host", "127.0.0.1",
                    "--server-port", str(port),
                ]
                mon_log = (log_dir / "rcssmonitor.log").open("w")
                try:
                    mon_proc = subprocess.Popen(monitor_cmd, stdout=mon_log,
                                                stderr=mon_log)
                except FileNotFoundError:
                    print(command_missing_message(args.monitor_cmd,
                                                  "--monitor-cmd"),
                          file=sys.stderr)
                    print("[launcher] Continuing without monitor (use "
                          "--no-monitor to silence this).", file=sys.stderr)
                else:
                    monitor_procs.append(mon_proc)
                    all_procs.append(mon_proc)
                    print(f"[launcher] env {idx}: monitor pid={mon_proc.pid}")

            if idx < num_envs - 1:
                time.sleep(args.sim_delay)

    except KeyboardInterrupt:
        print("\n[launcher] Interrupted during sim launch. Stopping...",
              file=sys.stderr)
        stop_processes(all_procs)
        return 130

    # ------------------------------------------------------------------
    # 2. Wait for sims to initialise
    # ------------------------------------------------------------------
    if embedded:
        print("[launcher] Embedded simulators need no external startup wait.")
    else:
        print(f"[launcher] Waiting {args.sim_wait}s for simulators to initialise...")
        try:
            time.sleep(args.sim_wait)
        except KeyboardInterrupt:
            print("\n[launcher] Interrupted during wait. Stopping...",
                  file=sys.stderr)
            stop_processes(all_procs)
            return 130

    # ------------------------------------------------------------------
    # 3. Launch one infer.py per model, each targeting its own sim
    # ------------------------------------------------------------------
    try:
        for idx, model_path in enumerate(args.model_paths):
            port = player_port(args.base_port, args.port_stride, idx)
            log_dir = log_dirs[idx]

            infer_cmd = [
                args.python, args.infer_script,
                "--trainer", args.trainer,
                "--team_config", args.team_config,
                "--team", args.team,
                "--env", args.env,
                "--model_path", model_path,
                "--config", args.config,
                "--stage", args.stage,
                "--steps", str(args.steps),
                "--log_dir", str(log_dir),
            ]
            if not embedded:
                infer_cmd += [
                    "--sim_host", "127.0.0.1",
                    "--sim_player_port", str(port),
                    "--sim_trainer_port", str(port + 1),
                ]
            if args.debug_infer:
                infer_cmd.append("--debug_infer")
            if args.ppo_stochastic:
                infer_cmd.append("--ppo_stochastic")
            if args.ppo_param_noise_std is not None:
                infer_cmd += ["--ppo_param_noise_std", str(args.ppo_param_noise_std)]

            # Let infer output go straight to the launcher's terminal so the
            # user sees live progress. infer.py's FileHandler still writes the
            # full log to {log_dir}/infer_log.log for later inspection.
            try:
                p = subprocess.Popen(infer_cmd)
            except FileNotFoundError as exc:
                print(f"[launcher] Cannot start infer.py: {exc}", file=sys.stderr)
                stop_processes(all_procs)
                return 1
            infer_procs.append(p)
            all_procs.append(p)
            backend = "embedded" if embedded else f"ports={port}/{port + 1}"
            print(f"[launcher] env {idx}: infer pid={p.pid}  "
                  f"model={Path(model_path).name}  backend={backend}")
    except KeyboardInterrupt:
        print("\n[launcher] Interrupted during infer launch. Stopping...",
              file=sys.stderr)
        stop_processes(all_procs)
        return 130

    print(f"[launcher] All processes running. Tail any "
          f"{args.log_root}/.../infer_log.log to watch progress. "
          f"Ctrl+C to stop everything.")

    # ------------------------------------------------------------------
    # 4. Wait for infer processes to finish, then kill sims
    # ------------------------------------------------------------------
    try:
        # Wait specifically for the infer processes (sims run forever)
        exit_code = wait_for_all(infer_procs)
    except KeyboardInterrupt:
        print("\n[launcher] Interrupted. Stopping all processes...")
        exit_code = 130
    finally:
        # Kill infer procs first (in case of Ctrl-C), then monitors, then sims.
        stop_processes(infer_procs)
        stop_processes(monitor_procs)
        stop_processes(sim_procs)
        print("[launcher] All subprocesses stopped.")

    # ------------------------------------------------------------------
    # 5. Print comparison table from each summary.json
    # ------------------------------------------------------------------
    print_comparison(log_dirs)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

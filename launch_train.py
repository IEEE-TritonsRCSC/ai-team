#!/usr/bin/env python3
"""Orchestrate parallel simulator instances and a single training process.

Launches N simulator instances on consecutive port slots, waits for them to
initialise, then starts train.py with matching --num_envs / port arguments.
All child processes are stopped cleanly when this script exits.

Port layout (mirrors base_trainer._sim_endpoint_for_env):
  env 0 : player=BASE_PORT,          trainer=BASE_PORT+1
  env 1 : player=BASE_PORT+STRIDE,   trainer=BASE_PORT+STRIDE+1
  ...

Examples:
  python launch_train.py --num-envs 4 --trainer discrete_ppo
  python launch_train.py --num-envs 2 --base-port 7000 --monitor
  python launch_train.py --num-envs 3 --trainer qlearning -- --episodes 5000 --lr 1e-3
  python launch_train.py --num-envs 2 --sim-cmd "rcssserver" --sim-port-flag "server::port="
  python launch_train.py --num-envs 2 --env sim-embedded --trainer discrete_ppo
"""

from __future__ import annotations

import argparse
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch N simulator instances then start train.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Parallel environment settings
    parser.add_argument(
        "--num-envs", type=int, default=1,
        help="Number of parallel simulator environments (default: 1)",
    )
    parser.add_argument(
        "--base-port", type=int, default=6000,
        help="Base player port for env 0 (default: 6000). "
             "Each env uses base_port + idx * port_stride.",
    )
    parser.add_argument(
        "--port-stride", type=int, default=10,
        help="Port gap between consecutive envs (default: 10, minimum: 3). "
             "Must be large enough so player and trainer ports do not overlap.",
    )
    parser.add_argument(
        "--env",
        choices=["sim-only", "sim-embedded", "sim-mixed", "field-practice", "field-tournament"],
        default="sim-only",
        help="Environment mode forwarded to train.py (default: sim-only). "
             "sim-embedded runs the synchronous embedded simulator in-process "
             "and does not launch external rcssserver instances.",
    )

    # Simulator / monitor commands
    parser.add_argument(
        "--sim-cmd", default="rcssserver",
        help='Base simulator command (default: "rcssserver")',
    )
    parser.add_argument(
        "--sim-port-flag", default="server::port=",
        help='CLI flag/prefix used to pass the player port to the simulator '
             '(default: "server::port="). Set to "" to skip passing ports. '
             'Values ending in "=" are emitted as FLAGPORT; other values are '
             'emitted as FLAG PORT.',
    )
    parser.add_argument(
        "--monitor", action="store_true", default=False,
        help="Also launch a monitor process for each simulator instance",
    )
    parser.add_argument(
        "--monitor-cmd", default="rcssmonitor",
        help='Monitor command (default: "rcssmonitor")',
    )

    # Trainer forwarding
    parser.add_argument(
        "--trainer",
        choices=[
            "hier_ppo",
            "discrete_ppo",
            "mappo",
            "hsm_marl",
            "hsm_sb3_ppo",
            "hsm_sb3_ppo_curriculum",
            "sb3_ppo",
            "robot_attention",
            "td3_jal",
            "td3_jal_curriculum",
            "td3_jal_her",
            "qlearning",
        ],
        default="discrete_ppo",
        help="Trainer type forwarded to train.py (default: discrete_ppo)",
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter used to run train.py (default: current interpreter)",
    )
    parser.add_argument(
        "--train-script", default=str(Path(__file__).parent / "train.py"),
        help="Path to train.py (default: train.py next to this script)",
    )

    # Startup timing
    parser.add_argument(
        "--sim-wait", type=float, default=2.0,
        help="Seconds to wait after launching all simulators before starting "
             "the trainer (default: 2.0)",
    )
    parser.add_argument(
        "--sim-delay", type=float, default=0.2,
        help="Seconds to wait between launching individual simulator instances "
             "(default: 0.2)",
    )

    # Unknown args are forwarded verbatim to train.py
    return parser.parse_known_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def player_port(base: int, stride: int, idx: int) -> int:
    """Return the player-side port for environment *idx*."""
    return base + idx * stride


def build_sim_command(base_cmd: str, port_flag: str, port: int) -> list[str]:
    """Construct the full command list for one simulator instance."""
    parts = shlex.split(base_cmd)
    if not parts:
        raise ValueError("--sim-cmd must not be empty")
    if port_flag:
        if "{port}" in port_flag:
            parts.append(port_flag.format(port=port))
        elif port_flag.endswith("="):
            parts.append(f"{port_flag}{port}")
        else:
            parts += [port_flag, str(port)]

        # rcssserver v19 uses config-style args and needs all server-side
        # ports shifted per instance to avoid collisions during parallel runs.
        if port_flag.startswith("server::port"):
            parts += [
                f"server::coach_port={port + 1}",
                f"server::olcoach_port={port + 2}",
            ]
    return parts


def command_missing_message(command: str, arg_name: str) -> str:
    """Return a helpful message when an external executable cannot be found."""
    binary = shlex.split(command)[0] if command else command
    hints = [
        f"[launcher] Cannot run '{binary}'.",
        f"[launcher] Install RoboCup Soccer Simulator or pass {arg_name} /full/path/to/{binary}.",
    ]
    if binary == "rcssserver" and shutil.which("rcsserver"):
        hints.append("[launcher] Found 'rcsserver' on PATH; try --sim-cmd rcsserver.")
    elif binary == "rcsserver":
        hints.append("[launcher] The standard RoboCup 2D server binary is usually named 'rcssserver'.")
    if sys.platform == "darwin":
        hints.append(
            "[launcher] On macOS, install/build rcsoccersim, then pass the built "
            "binary path if it is not on PATH."
        )
    return "\n".join(hints)


def stop_processes(procs: list[subprocess.Popen]) -> None:
    """Gracefully terminate all processes, force-kill after 3 s if needed."""
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()

    deadline = time.monotonic() + 3.0
    for proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        if proc.poll() is None:
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()


def wait_for_exit(procs: list[subprocess.Popen]) -> int:
    """Block until all processes finish. Return the first non-zero exit code."""
    exit_code = 0
    while True:
        still_running = [p for p in procs if p.poll() is None]
        if not still_running:
            break
        # Surface a crashed child process immediately
        for proc in procs:
            rc = proc.poll()
            if rc is not None and rc != 0 and exit_code == 0:
                exit_code = rc
                print(
                    f"[launcher] Process PID {proc.pid} exited with code {rc}.",
                    file=sys.stderr,
                )
        time.sleep(1.0)
    return exit_code


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args, extra_train_args = parse_args()

    # Validate inputs
    if args.num_envs < 1:
        print("--num-envs must be >= 1", file=sys.stderr)
        return 1
    if args.port_stride < 3:
        print("--port-stride must be >= 3 to avoid sim port conflicts", file=sys.stderr)
        return 1

    try:
        monitor_parts = shlex.split(args.monitor_cmd) if args.monitor else []
    except ValueError as exc:
        print(f"Invalid --monitor-cmd: {exc}", file=sys.stderr)
        return 1

    all_procs: list[subprocess.Popen] = []

    print(
        f"[launcher] Starting launcher for env={args.env} "
        f"(num-envs={args.num_envs}, base-port={args.base_port}, stride={args.port_stride})",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 1. Launch simulator (and optional monitor) for each environment
    # ------------------------------------------------------------------
    if args.env == "sim-embedded":
        if args.monitor:
            print("[launcher] --monitor is ignored for sim-embedded.", file=sys.stderr)
    else:
        try:
            for idx in range(args.num_envs):
                port = player_port(args.base_port, args.port_stride, idx)

                # Build per-instance sim command with the correct player port
                try:
                    sim_parts = build_sim_command(args.sim_cmd, args.sim_port_flag, port)
                except ValueError as exc:
                    print(f"[launcher] {exc}", file=sys.stderr)
                    stop_processes(all_procs)
                    return 1

                try:
                    sim_proc = subprocess.Popen(sim_parts)
                except FileNotFoundError:
                    print(command_missing_message(args.sim_cmd, "--sim-cmd"), file=sys.stderr)
                    stop_processes(all_procs)
                    return 1

                all_procs.append(sim_proc)
                print(f"[launcher] env {idx}: sim pid={sim_proc.pid}  port={port}")

                # Launch optional monitor for this env (monitor reads trainer port = port+1)
                if args.monitor:
                    if not monitor_parts:
                        print("[launcher] --monitor-cmd is empty, skipping monitor", file=sys.stderr)
                    else:
                        monitor_cmd = monitor_parts + [
                            "--server-host",
                            "127.0.0.1",
                            "--server-port",
                            str(port),
                        ]
                        try:
                            mon_proc = subprocess.Popen(monitor_cmd)
                        except FileNotFoundError:
                            print(command_missing_message(args.monitor_cmd, "--monitor-cmd"), file=sys.stderr)
                            stop_processes(all_procs)
                            return 1
                        all_procs.append(mon_proc)
                        print(f"[launcher] env {idx}: monitor pid={mon_proc.pid}")

                # Small delay between sequential sim launches to avoid port race
                if idx < args.num_envs - 1:
                    time.sleep(args.sim_delay)

        except KeyboardInterrupt:
            print("\n[launcher] Interrupted during sim launch. Stopping...", file=sys.stderr)
            stop_processes(all_procs)
            return 130

    # ------------------------------------------------------------------
    # 2. Wait for simulators to initialise before connecting the trainer
    # ------------------------------------------------------------------
    if args.env == "sim-embedded":
        print("[launcher] sim-embedded selected; skipping external simulator startup wait.")
    else:
        print(f"[launcher] Waiting {args.sim_wait}s for simulators to initialise...")
        try:
            time.sleep(args.sim_wait)
        except KeyboardInterrupt:
            print("\n[launcher] Interrupted during wait. Stopping...", file=sys.stderr)
            stop_processes(all_procs)
            return 130

    # ------------------------------------------------------------------
    # 3. Build and launch the training command
    # ------------------------------------------------------------------
    train_cmd = [
        args.python,
        args.train_script,
        "--trainer", args.trainer,
        "--env", args.env,
        "--num_envs", str(args.num_envs),
        "--sim_player_port", str(args.base_port),
        "--sim_port_stride", str(args.port_stride),
    ] + extra_train_args  # pass any remaining args straight through

    print(f"[launcher] Starting trainer: {' '.join(train_cmd)}")

    try:
        train_proc = subprocess.Popen(train_cmd)
    except FileNotFoundError as exc:
        print(f"[launcher] Cannot start trainer: {exc}", file=sys.stderr)
        stop_processes(all_procs)
        return 1

    all_procs.append(train_proc)
    print(f"[launcher] Trainer pid={train_proc.pid}")

    # ------------------------------------------------------------------
    # 4. Monitor all processes until they finish or the user interrupts
    # ------------------------------------------------------------------
    print("[launcher] All processes running. Press Ctrl+C to stop everything.")
    try:
        exit_code = wait_for_exit(all_procs)
    except KeyboardInterrupt:
        print("\n[launcher] Interrupted. Stopping all processes...")
        exit_code = 130
    finally:
        stop_processes(all_procs)
        print("[launcher] Done.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

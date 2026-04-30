#!/usr/bin/env python3
"""Launch simulator instances with a minimal CLI.

Examples:
  python launch_sims.py --env 2
  python launch_sims.py --env 2 --monitor
  python launch_sims.py --env 2 --monitor --sim-cmd "rcssserver" --monitor-cmd "rcssmonitor"
"""

from __future__ import annotations

import argparse
import shutil
import shlex
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch simulator and monitor processes")
    parser.add_argument(
        "--env",
        type=int,
        default=1,
        help="How many simulator instances to launch",
    )
    parser.add_argument(
        "--monitor",
        nargs="?",
        const="true",
        default="false",
        help="Whether to launch monitors for each env (true/false). Using --monitor alone means true.",
    )
    parser.add_argument(
        "--sim-cmd",
        default="rcssserver",
        help='Command used to launch one simulator instance (default: "rcssserver")',
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=6000,
        help="Player port for env 0 (default: 6000)",
    )
    parser.add_argument(
        "--port-stride",
        type=int,
        default=10,
        help="Port gap between consecutive envs (default: 10)",
    )
    parser.add_argument(
        "--monitor-cmd",
        default="rcssmonitor",
        help='Command used to launch one monitor instance (default: "rcssmonitor")',
    )
    parser.add_argument(
        "monitor_value",
        nargs="?",
        default=None,
        help="Optional positional boolean for monitor (true/false), e.g.: --env 2 true",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def split_command(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("Command must not be empty")
    return parts


def build_sim_command(base_cmd: list[str], port: int) -> list[str]:
    return base_cmd + [
        f"server::port={port}",
        f"server::coach_port={port + 1}",
        f"server::olcoach_port={port + 2}",
    ]


def command_missing_message(command: str, arg_name: str) -> str:
    binary = shlex.split(command)[0] if command else command
    hints = [
        f"Cannot run '{binary}'.",
        f"Install RoboCup Soccer Simulator or pass {arg_name} /full/path/to/{binary}.",
    ]
    if binary == "rcssserver" and shutil.which("rcsserver"):
        hints.append("Found 'rcsserver' on PATH; try --sim-cmd rcsserver.")
    elif binary == "rcsserver":
        hints.append("The standard RoboCup 2D server binary is usually named 'rcssserver'.")
    if sys.platform == "darwin":
        hints.append(
            "On macOS, install/build rcsoccersim, then pass the built binary path "
            "if it is not on PATH."
        )
    return "\n".join(hints)


def stop_processes(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> int:
    args = parse_args()
    if args.env < 1:
        print("--env must be >= 1", file=sys.stderr)
        return 1
    if args.port_stride < 3:
        print("--port-stride must be >= 3 to avoid sim port conflicts", file=sys.stderr)
        return 1

    try:
        monitor_raw = args.monitor_value if args.monitor_value is not None else args.monitor
        launch_monitor = parse_bool(monitor_raw)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        sim_cmd = split_command(args.sim_cmd)
        monitor_cmd = split_command(args.monitor_cmd)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    procs: list[subprocess.Popen] = []
    print(
        f"Launching {args.env} sim process(es)"
        + (f" and {args.env} monitor process(es)" if launch_monitor else "")
        + "...",
        flush=True,
    )

    try:
        for env_idx in range(args.env):
            port = args.base_port + env_idx * args.port_stride
            instance_sim_cmd = build_sim_command(sim_cmd, port)
            try:
                sim_proc = subprocess.Popen(instance_sim_cmd)
            except FileNotFoundError:
                print(command_missing_message(args.sim_cmd, "--sim-cmd"), file=sys.stderr)
                stop_processes(procs)
                return 1

            procs.append(sim_proc)
            print(f"[env {env_idx}] sim pid={sim_proc.pid} port={port}")
            time.sleep(0.1)

            if launch_monitor:
                instance_monitor_cmd = monitor_cmd + [
                    "--server-host",
                    "127.0.0.1",
                    "--server-port",
                    str(port),
                ]
                try:
                    monitor_proc = subprocess.Popen(instance_monitor_cmd)
                except FileNotFoundError:
                    print(command_missing_message(args.monitor_cmd, "--monitor-cmd"), file=sys.stderr)
                    stop_processes(procs)
                    return 1
                procs.append(monitor_proc)
                print(f"[env {env_idx}] monitor pid={monitor_proc.pid}")
                time.sleep(0.1)

        print("All processes launched. Press Ctrl+C to stop.")
        while True:
            running = sum(1 for proc in procs if proc.poll() is None)
            if running == 0:
                print("All launched processes have exited.")
                return 0
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping processes...")
    finally:
        stop_processes(procs)
        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

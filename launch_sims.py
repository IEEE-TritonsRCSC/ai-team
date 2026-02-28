#!/usr/bin/env python3
"""Launch simulator instances with a minimal CLI.

Examples:
  python launch_sims.py --env 2
  python launch_sims.py --env 2 --monitor
  python launch_sims.py --env 2 --monitor --sim-cmd "rcsserver" --monitor-cmd "monitor"
"""

from __future__ import annotations

import argparse
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
        default="rcsserver",
        help='Command used to launch one simulator instance (default: "rcsserver")',
    )
    parser.add_argument(
        "--monitor-cmd",
        default="monitor",
        help='Command used to launch one monitor instance (default: "monitor")',
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
        + "..."
    )

    try:
        for env_idx in range(args.env):
            try:
                sim_proc = subprocess.Popen(sim_cmd)
            except FileNotFoundError:
                print(
                    f"Cannot run '{sim_cmd[0]}'. Make sure it is in PATH or set --sim-cmd.",
                    file=sys.stderr,
                )
                stop_processes(procs)
                return 1

            procs.append(sim_proc)
            print(f"[env {env_idx}] sim pid={sim_proc.pid}")
            time.sleep(0.1)

            if launch_monitor:
                try:
                    monitor_proc = subprocess.Popen(monitor_cmd)
                except FileNotFoundError:
                    print(
                        f"Cannot run '{monitor_cmd[0]}'. Make sure it is in PATH or set --monitor-cmd.",
                        file=sys.stderr,
                    )
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

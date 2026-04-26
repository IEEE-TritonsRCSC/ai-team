#!/usr/bin/env python3
"""
Launch multiple independent RoboCup simulation instances.

Each instance uses its own server/coach/online-coach port triplet and can run
its own AI arguments so windows/sessions can receive different inputs.
"""

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_team_configs(raw: str | None, instances: int) -> list[str] | None:
    if not raw:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if len(items) != instances:
        raise ValueError(
            f"--team-configs expects exactly {instances} entries, got {len(items)}"
        )
    return items


def parse_instance_args(raw_args: list[str], instances: int) -> list[list[str]]:
    if len(raw_args) > instances:
        raise ValueError(
            f"Too many --ai-args entries: got {len(raw_args)}, expected <= {instances}"
        )
    padded = raw_args + [""] * (instances - len(raw_args))
    return [shlex.split(arg) for arg in padded]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multiple rcssserver/rcssmonitor + ai-team instances."
    )
    parser.add_argument("--instances", type=int, default=2, help="Number of instances")
    parser.add_argument("--sim-host", type=str, default="127.0.0.1", help="Sim host")
    parser.add_argument(
        "--base-port",
        type=int,
        default=6000,
        help="First simulator player port",
    )
    parser.add_argument(
        "--port-step",
        type=int,
        default=10,
        help="Port increment between instances",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=".",
        help="Working directory for spawned processes",
    )

    parser.add_argument(
        "--server-cmd",
        type=str,
        default="rcssserver",
        help="Server executable command",
    )
    parser.add_argument(
        "--server-extra",
        type=str,
        default="",
        help="Extra args appended to each server command",
    )

    parser.add_argument(
        "--monitor-cmd",
        type=str,
        default="rcssmonitor",
        help="Monitor executable command",
    )
    parser.add_argument(
        "--monitor-extra",
        type=str,
        default="",
        help="Extra args appended to each monitor command",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Do not launch rcssmonitor windows",
    )

    parser.add_argument(
        "--ai-cmd",
        type=str,
        default="python .",
        help="AI command base",
    )
    parser.add_argument(
        "--ai-extra",
        type=str,
        default="",
        help="Extra args appended to all AI commands",
    )
    parser.add_argument(
        "--ai-args",
        action="append",
        default=[],
        help=(
            "Per-instance AI args (repeatable). "
            "Order maps to instance index: 0,1,2,..."
        ),
    )
    parser.add_argument(
        "--team-configs",
        type=str,
        default=None,
        help="Comma-separated team_config paths (one per instance)",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Do not launch AI processes",
    )

    parser.add_argument(
        "--stagger",
        type=float,
        default=0.3,
        help="Seconds to wait between process launches",
    )
    return parser


def terminate_processes(processes: list[tuple[str, int, subprocess.Popen]]) -> None:
    for _, _, proc in processes:
        if proc.poll() is None:
            proc.terminate()
    time.sleep(0.5)
    for _, _, proc in processes:
        if proc.poll() is None:
            proc.kill()


def launch(command: list[str], cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(command, cwd=cwd)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.instances <= 0:
        raise ValueError("--instances must be > 0")
    if args.base_port <= 0:
        raise ValueError("--base-port must be > 0")
    if args.port_step <= 0:
        raise ValueError("--port-step must be > 0")

    cwd = Path(args.cwd).resolve()
    team_configs = parse_team_configs(args.team_configs, args.instances)
    per_instance_ai_args = parse_instance_args(args.ai_args, args.instances)

    server_base = shlex.split(args.server_cmd)
    server_extra = shlex.split(args.server_extra)
    monitor_base = shlex.split(args.monitor_cmd)
    monitor_extra = shlex.split(args.monitor_extra)
    ai_base = shlex.split(args.ai_cmd)
    ai_extra = shlex.split(args.ai_extra)

    processes: list[tuple[str, int, subprocess.Popen]] = []

    try:
        for idx in range(args.instances):
            sim_port = args.base_port + (idx * args.port_step)
            coach_port = sim_port + 1
            olcoach_port = sim_port + 2

            server_cmd = server_base + [
                f"server::port={sim_port}",
                f"server::coach_port={coach_port}",
                f"server::olcoach_port={olcoach_port}",
            ] + server_extra
            print(f"[instance {idx}] server: {' '.join(server_cmd)}")
            processes.append(("server", idx, launch(server_cmd, cwd)))
            time.sleep(args.stagger)

            if not args.no_monitor:
                monitor_cmd = monitor_base + [
                    "--server-host",
                    args.sim_host,
                    "--server-port",
                    str(sim_port),
                ] + monitor_extra
                print(f"[instance {idx}] monitor: {' '.join(monitor_cmd)}")
                processes.append(("monitor", idx, launch(monitor_cmd, cwd)))
                time.sleep(args.stagger)

            if not args.no_ai:
                ai_cmd = ai_base + [
                    "--env",
                    "sim-only",
                    "--sim-host",
                    args.sim_host,
                    "--sim-port",
                    str(sim_port),
                ]
                if team_configs:
                    ai_cmd += ["--team_config", team_configs[idx]]
                ai_cmd += ai_extra
                ai_cmd += per_instance_ai_args[idx]
                print(f"[instance {idx}] ai: {' '.join(ai_cmd)}")
                processes.append(("ai", idx, launch(ai_cmd, cwd)))
                time.sleep(args.stagger)

        print("All instances launched. Press Ctrl+C to stop all.")
        while True:
            for role, idx, proc in processes:
                code = proc.poll()
                if code is not None:
                    print(f"[instance {idx}] {role} exited with code {code}")
                    return code
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping all instances...")
    except FileNotFoundError as exc:
        print(f"Failed to launch command: {exc}", file=sys.stderr)
        return 1
    finally:
        terminate_processes(processes)

    return 0


if __name__ == "__main__":
    sys.exit(main())

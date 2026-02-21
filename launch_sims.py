#!/usr/bin/env python3
"""Launch multiple rcssserver instances on non-overlapping ports.

Example:
  python launch_sims.py --num_envs 4 --sim_bin Downloads/rcssserver-19.0.0/src/rcssserver
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch parallel rcssserver instances")
    parser.add_argument(
        "--sim_bin",
        type=str,
        default="Downloads/rcssserver-19.0.0/src/rcssserver",
        help="Path to rcssserver binary",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="How many simulator instances")
    parser.add_argument("--sim_player_port", type=int, default=6000, help="Base player port")
    parser.add_argument(
        "--sim_port_stride",
        type=int,
        default=10,
        help="Port stride between simulator instances (recommended >= 3)",
    )
    parser.add_argument(
        "--disable_logs",
        action="store_true",
        help="Disable text/game logging in server for faster training",
    )
    parser.add_argument(
        "--with_monitor",
        action="store_true",
        help="Launch one rcssmonitor per simulator instance",
    )
    parser.add_argument(
        "--monitor_bin",
        type=str,
        default="Downloads/rcssmonitor-19.0.1/src/rcssmonitor",
        help="Path to rcssmonitor binary",
    )
    parser.add_argument(
        "--monitor_host",
        type=str,
        default="127.0.0.1",
        help="Host passed to rcssmonitor --server-host",
    )
    return parser.parse_args()


def build_server_cmd(sim_bin: str, player_port: int, stride_idx: int, disable_logs: bool) -> list[str]:
    trainer_port = player_port + 1
    online_coach_port = player_port + 2

    text_dir = Path("text_logs") / f"sim_{stride_idx}"
    game_dir = Path("game_logs") / f"sim_{stride_idx}"
    text_dir.mkdir(parents=True, exist_ok=True)
    game_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sim_bin,
        f"server::port={player_port}",
        f"server::coach_port={trainer_port}",
        f"server::olcoach_port={online_coach_port}",
        f"server::text_log_dir={text_dir.as_posix()}/",
        f"server::game_log_dir={game_dir.as_posix()}/",
    ]
    if disable_logs:
        cmd.extend(
            [
                "server::text_logging=false",
                "server::game_logging=false",
            ]
        )
    return cmd


def main() -> int:
    args = parse_args()

    if args.num_envs < 1:
        print("num_envs must be >= 1", file=sys.stderr)
        return 1
    if args.sim_port_stride < 3:
        print("sim_port_stride must be >= 3", file=sys.stderr)
        return 1

    sim_bin = str(Path(args.sim_bin))
    procs: list[subprocess.Popen] = []
    monitor_procs: list[subprocess.Popen] = []
    print(f"Launching {args.num_envs} simulator instances...")

    try:
        for env_idx in range(args.num_envs):
            player_port = args.sim_player_port + env_idx * args.sim_port_stride
            cmd = build_server_cmd(sim_bin, player_port, env_idx, args.disable_logs)
            proc = subprocess.Popen(cmd)
            procs.append(proc)
            print(
                f"[env {env_idx}] player={player_port} trainer={player_port + 1} "
                f"olcoach={player_port + 2} pid={proc.pid}"
            )
            time.sleep(0.15)

            if args.with_monitor:
                monitor_cmd = [
                    str(Path(args.monitor_bin)),
                    "--connect",
                    "--server-host",
                    args.monitor_host,
                    "--server-port",
                    str(player_port),
                ]
                monitor_proc = subprocess.Popen(monitor_cmd)
                monitor_procs.append(monitor_proc)
                print(f"[env {env_idx}] monitor pid={monitor_proc.pid}")
                time.sleep(0.10)

        print("All simulators launched. Press Ctrl+C to stop.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping simulators...")
    finally:
        for proc in monitor_procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in monitor_procs:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()

        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

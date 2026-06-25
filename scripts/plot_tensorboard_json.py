"""Plot TensorBoard JSON exports from the three rollout metrics.

Export the three scalars from TensorBoard as JSON, then run:
    python scripts/plot_tensorboard_json.py \
        --len   "TD3_0.json" \
        --rew   "TD3_0 (1).json" \
        --fps   "TD3_0 (2).json" \
        --out   analysis_tb.png

Each JSON file is a list of [wall_time, timestep, value] triples as
exported by the TensorBoard UI (Download → JSON).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: str):
    with open(path) as f:
        data = json.load(f)
    steps  = np.array([row[1] for row in data])
    values = np.array([row[2] for row in data])
    return steps, values


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) < window:
        return arr.copy()
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", required=True,  dest="len_file",  help="ep_len_mean JSON")
    ap.add_argument("--rew", required=True,  dest="rew_file",  help="ep_rew_mean JSON")
    ap.add_argument("--fps", required=True,  dest="fps_file",  help="time_fps JSON")
    ap.add_argument("--out", default="analysis_tb.png", help="Output PNG path")
    args = ap.parse_args()

    ts_len, ep_len  = load(args.len_file)
    ts_rew, ep_rew  = load(args.rew_file)
    ts_fps, fps_val = load(args.fps_file)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("TD3+HER Training — TensorBoard Metrics", fontsize=13)

    # ── Panel 1: ep_rew_mean ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot(ts_rew / 1000, ep_rew, color="steelblue", linewidth=1.5, label="ep_rew_mean")
    window = max(1, len(ep_rew) // 8)
    if len(ep_rew) >= window:
        rm = rolling_mean(ep_rew, window)
        ax.plot(ts_rew[window - 1:] / 1000, rm, color="orange",
                linewidth=2.5, label=f"Smooth ({window}-pt)")
    ax.axhline(6.0,  color="red",   linestyle="--", linewidth=1, alpha=0.7, label="6.00 ceiling")
    ax.axhline(0.0,  color="grey",  linestyle=":",  linewidth=0.8, alpha=0.5)

    # Annotate the dip and recovery
    dip_idx = int(np.argmin(ep_rew))
    ax.annotate(f"Dip\n{ep_rew[dip_idx]:.1f}",
                xy=(ts_rew[dip_idx] / 1000, ep_rew[dip_idx]),
                xytext=(ts_rew[dip_idx] / 1000 + 3, ep_rew[dip_idx] - 1.5),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)

    final_rew = ep_rew[-1]
    ax.annotate(f"Latest\n{final_rew:.2f}",
                xy=(ts_rew[-1] / 1000, final_rew),
                xytext=(ts_rew[-1] / 1000 - 8, final_rew + 1.5),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)

    ax.set_xlabel("Timesteps (k)")
    ax.set_ylabel("ep_rew_mean")
    ax.set_title("Mean Episode Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: ep_len_mean ─────────────────────────────────────────────
    ax = axes[1]
    ax.plot(ts_len / 1000, ep_len, color="green", linewidth=1.5)
    ax.axhline(200, color="red", linestyle="--", linewidth=1, alpha=0.7, label="max_steps=200")
    ax.set_ylim(0, 220)
    ax.set_xlabel("Timesteps (k)")
    ax.set_ylabel("ep_len_mean")
    ax.set_title("Mean Episode Length\n(200 = always hits max_steps — never terminated early)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: FPS ─────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(ts_fps / 1000, fps_val, color="purple", linewidth=1.5)
    ax.set_ylim(0, 6)
    ax.set_xlabel("Timesteps (k)")
    ax.set_ylabel("Steps / second")
    ax.set_title("Simulator Throughput (FPS)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved: {args.out}")

    # ── Text summary ──────────────────────────────────────────────────────
    print(f"\n=== Summary ===")
    print(f"Timestep range  : {ts_rew[0]:,.0f} – {ts_rew[-1]:,.0f}")
    print(f"ep_rew_mean start  : {ep_rew[0]:.3f}")
    print(f"ep_rew_mean lowest : {ep_rew[dip_idx]:.3f}  (at {ts_rew[dip_idx]:,.0f} steps)")
    print(f"ep_rew_mean latest : {ep_rew[-1]:.3f}")
    print(f"ep_len_mean        : always {ep_len[0]:.0f} (robot never terminated early)")
    print(f"FPS                : {fps_val.min():.0f} – {fps_val.max():.0f}")
    still_rising = ep_rew[-1] > ep_rew[-3]
    print(f"Still rising?      : {'YES' if still_rising else 'NO'}")


if __name__ == "__main__":
    main()

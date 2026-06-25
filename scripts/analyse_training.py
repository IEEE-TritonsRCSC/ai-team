"""Parse a train_log.log file and produce diagnostic plots.

Usage (on Mac mini):
    python scripts/analyse_training.py train_logs/<run_dir>/train_log.log

Outputs a multi-panel PNG alongside the log file:
    train_logs/<run_dir>/analysis.png
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

EP_RE   = re.compile(r"Episode\s+(\d+):\s+Reward=([+-]?\d+\.?\d*),\s+Length=(\d+)")
SB3_TS  = re.compile(r"\|\s+total_timesteps\s+\|\s+(\d+)\s+\|")
SB3_REW = re.compile(r"\|\s+ep_rew_mean\s+\|\s+([+-]?\d+\.?\d*)\s+\|")
SB3_ACT = re.compile(r"\|\s+actor_loss\s+\|\s+([+-]?\d+\.?\d*)\s+\|")
SB3_CRI = re.compile(r"\|\s+critic_loss\s+\|\s+([+-]?\d+\.?\d*)\s+\|")


def parse_log(path: Path):
    episodes, rewards, lengths = [], [], []
    ts_list, rew_mean_list, actor_list, critic_list = [], [], [], []

    _ts = _rew = _act = _cri = None

    with open(path) as f:
        for line in f:
            m = EP_RE.search(line)
            if m:
                episodes.append(int(m.group(1)))
                rewards.append(float(m.group(2)))
                lengths.append(int(m.group(3)))
                continue

            m = SB3_TS.search(line)
            if m:
                _ts = int(m.group(1))
                continue

            m = SB3_REW.search(line)
            if m:
                _rew = float(m.group(1))
                continue

            m = SB3_ACT.search(line)
            if m:
                _act = float(m.group(1))
                continue

            m = SB3_CRI.search(line)
            if m:
                _cri = float(m.group(1))
                if all(v is not None for v in [_ts, _rew, _act, _cri]):
                    ts_list.append(_ts)
                    rew_mean_list.append(_rew)
                    actor_list.append(_act)
                    critic_list.append(_cri)
                _ts = _rew = _act = _cri = None

    return (
        np.array(episodes), np.array(rewards), np.array(lengths),
        np.array(ts_list), np.array(rew_mean_list),
        np.array(actor_list), np.array(critic_list),
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) < window:
        return arr.copy()
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot(log_path: Path):
    eps, rews, lens, ts, rew_mean, actor, critic = parse_log(log_path)

    if len(eps) == 0:
        print("No episode data found in log.", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Training analysis — {log_path.parent.name}", fontsize=13)

    # ── Panel 1: raw episode rewards + rolling mean ───────────────────────
    ax = axes[0, 0]
    ax.scatter(eps, rews, s=8, alpha=0.5, color="steelblue", label="Episode reward")
    window = min(20, max(1, len(rews) // 10))
    if len(rews) >= window:
        rm = rolling_mean(rews, window)
        ax.plot(eps[window - 1:], rm, color="orange", linewidth=2,
                label=f"Rolling mean ({window} ep)")
    ax.axhline(6.0, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="6.00 (step-bonus ceiling)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Rewards")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: reward histogram ─────────────────────────────────────────
    ax = axes[0, 1]
    ax.hist(rews, bins=40, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(6.0, color="red", linestyle="--", linewidth=1.5, label="6.00")
    pct_at_6 = 100 * np.mean(np.isclose(rews, 6.0, atol=0.05))
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title(f"Reward Distribution  ({pct_at_6:.1f}% of episodes at 6.00)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: SB3 ep_rew_mean over timesteps ───────────────────────────
    ax = axes[1, 0]
    if len(ts) > 0:
        ax.plot(ts, rew_mean, color="green", marker="o", markersize=4,
                label="ep_rew_mean")
        ax.axhline(6.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("ep_rew_mean")
        ax.set_title("SB3 Rolling Reward Mean vs Timesteps")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No SB3 stats blocks found", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("SB3 ep_rew_mean")

    # ── Panel 4: actor & critic loss over timesteps ───────────────────────
    ax = axes[1, 1]
    if len(ts) > 0:
        ax2 = ax.twinx()
        l1, = ax.plot(ts, actor, color="darkorange", marker="o", markersize=4,
                      label="actor_loss")
        l2, = ax2.plot(ts, critic, color="purple", marker="s", markersize=4,
                       linestyle="--", label="critic_loss")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Actor loss", color="darkorange")
        ax2.set_ylabel("Critic loss", color="purple")
        ax.set_title("Actor & Critic Losses")
        ax.legend(handles=[l1, l2], fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No SB3 stats blocks found", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Actor & Critic Losses")

    fig.tight_layout()
    out = log_path.parent / "analysis.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Text summary ──────────────────────────────────────────────────────
    pct_6   = 100 * np.mean(np.isclose(rews, 6.0, atol=0.05))
    pct_neg = 100 * np.mean(rews < 0)
    print(f"\n=== Summary ===")
    print(f"Episodes logged : {len(eps)}")
    print(f"Reward range    : {rews.min():.2f} to {rews.max():.2f}")
    print(f"Mean reward     : {rews.mean():.2f}")
    print(f"% at 6.00 (step-bonus ceiling) : {pct_6:.1f}%")
    print(f"% negative rewards             : {pct_neg:.1f}%")
    if len(ts) > 0:
        print(f"Timesteps seen  : {ts[0]:,} – {ts[-1]:,}")
        print(f"Final ep_rew_mean              : {rew_mean[-1]:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path/to/train_log.log>", file=sys.stderr)
        sys.exit(1)
    plot(Path(sys.argv[1]))

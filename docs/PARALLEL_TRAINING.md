# Parallel Training Guide

Run multiple simulator instances simultaneously to accelerate RL training.

---

## How It Works

`launch_train.py` is the single entry-point for parallel training. It:

1. Starts **N simulator instances**, each on its own port slot.
2. Waits a configurable delay for the simulators to initialise.
3. Starts **`train.py`** with matching `--num_envs`, `--sim_player_port`, and `--sim_port_stride` arguments so the trainer connects to every simulator.
4. Monitors all child processes and shuts them all down cleanly on exit or error.

### Port Layout

Each environment occupies a pair of consecutive ports:

| Env index | Player port | Trainer port |
|-----------|-------------|--------------|
| 0 | `BASE_PORT` | `BASE_PORT + 1` |
| 1 | `BASE_PORT + STRIDE` | `BASE_PORT + STRIDE + 1` |
| … | … | … |

Default: `BASE_PORT = 6000`, `STRIDE = 10`.

---

## Quick Start

### Single simulator (unchanged behaviour)

```bash
python launch_train.py --trainer discrete_ppo
```

### 4 parallel simulators

```bash
python launch_train.py --num-envs 4 --trainer discrete_ppo
```

The launcher supports every active training entry point:
`hier_ppo`, `discrete_ppo`, `mappo`, `hsm_marl`, `hsm_sb3_ppo`, `sb3_ppo`, `td3_jal`, and `qlearning`.

### 4 simulators + monitors

```bash
python launch_train.py --num-envs 4 --trainer discrete_ppo --monitor
```

### Custom ports and pass extra args to train.py

```bash
python launch_train.py \
  --num-envs 3 \
  --base-port 7000 \
  --port-stride 10 \
  --trainer qlearning \
  -- --episodes 5000 --lr 1e-3 --epsilon_decay 0.998
```

> Arguments after `--` are forwarded verbatim to `train.py`.

---

## All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-envs N` | `1` | Number of parallel simulator environments |
| `--base-port PORT` | `6000` | Player port for env 0 |
| `--port-stride N` | `10` | Port gap between consecutive envs (min 3) |
| `--sim-cmd CMD` | `rcssserver` | Command used to launch a simulator |
| `--sim-port-flag FLAG` | `server::port=` | CLI flag/prefix passed to the simulator for the player port. With the default, launcher also sets `server::coach_port` and `server::olcoach_port`. |
| `--monitor` | off | Launch a monitor process alongside each simulator |
| `--monitor-cmd CMD` | `rcssmonitor` | Command used to launch a monitor |
| `--trainer TYPE` | `discrete_ppo` | Trainer type forwarded to `train.py` |
| `--sim-wait SECS` | `2.0` | Seconds to wait after launching all sims before starting the trainer |
| `--sim-delay SECS` | `0.2` | Seconds between launching consecutive sim instances |
| `--python PATH` | current interpreter | Python used to run `train.py` |
| `--train-script PATH` | `train.py` | Path to `train.py` |

---

## Relationship to Existing Scripts

| Script | Purpose |
|--------|---------|
| `launch_sims.py` | Launch simulator(s) only — no trainer |
| `train.py` | Run the trainer only — assumes sims are already running |
| **`launch_train.py`** | **Full pipeline: launch sims + trainer together** |

Use `launch_sims.py` or `train.py` individually when you need fine-grained control (e.g. attaching a debugger to the trainer while sims run separately).

---

## How `train.py` Connects to Multiple Simulators

`train.py` (and the underlying trainers) support `--num_envs`, `--sim_player_port`, and `--sim_port_stride`. Each trainer creates one simulator-backed environment per index and calculates its ports via `BaseTrainer._sim_endpoint_for_env`:

```
player_port = base_port + env_index * stride
trainer_port = player_port + 1
```

`launch_train.py` passes matching values so both sides agree on every port.

Custom PPO/MAPPO/Q-learning trainers step those environments with a thread pool. SB3-based trainers use vectorized environments, with subprocess workers by default when `num_envs > 1`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Cannot run 'rcssserver'` | Binary not in PATH | Install `rcssserver` or use `--sim-cmd /full/path/to/rcssserver` |
| Trainer can't connect to sim | Sims not ready in time | Increase `--sim-wait` |
| Port conflict error | Stride too small | Increase `--port-stride` (≥ 3) |
| Monitor doesn't open | Wrong monitor command | Use `--monitor-cmd /path/to/monitor` |

---

## Example: Resuming from a Checkpoint with 4 Envs

```bash
python launch_train.py \
  --num-envs 4 \
  --trainer discrete_ppo \
  -- --resume_checkpoint models/run_abc/policy_ep500.pth
```

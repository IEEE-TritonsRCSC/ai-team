from __future__ import annotations

import argparse
from itertools import chain
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ai_interface.algorithms.robot_attention import (
    ActionHead,
    MultiRobotFeatureExtractor,
)
from ai_interface.pretraining.dataset.RobotAttentionDataset import (
    RobotAttentionTorchDataset,
)


DEFAULT_PRETRAIN_CONFIG_PATH = Path(
    "ai_interface/pretraining/config/robot_attention_pretrain_config.json"
)
DEFAULT_ROBOT_ATTENTION_CONFIG_PATH = Path("configs/robot_attention_config.json")
DEFAULT_DATA_DIR = Path("data/helios-05032026")
DEFAULT_OUTPUT_DIR = Path("models/robot_attention_pretrain")


@dataclass
class EpochMetrics:
    loss: float
    action_ce: float
    param_mse: float
    action_acc: float
    samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavior cloning pretraining for RobotAttention."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PRETRAIN_CONFIG_PATH,
        help="Path to the pretraining config JSON.",
    )
    parser.add_argument(
        "--robot-attention-config",
        type=str,
        default=None,
        help="Optional override path to configs/robot_attention_config.json.",
    )
    return parser.parse_args()


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def build_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    pretrain_config_path = Path(args.config)
    pretrain_config = load_json_file(pretrain_config_path)

    robot_attention_config_path = Path(
        args.robot_attention_config
        or pretrain_config.get(
            "robot_attention_config",
            DEFAULT_ROBOT_ATTENTION_CONFIG_PATH,
        )
    )
    robot_attention_config = load_json_file(robot_attention_config_path)

    feature_params = dict(robot_attention_config.get("feature_extractor_params", {}))
    model_params = dict(robot_attention_config.get("model_params", {}))

    num_robots = int(
        pretrain_config.get(
            "num_robots",
            robot_attention_config.get("num_robots", 6),
        )
    )
    robot_ids = pretrain_config.get("robot_ids")
    if robot_ids is None:
        robot_ids = list(range(1, num_robots + 1))
    else:
        robot_ids = [int(robot_id) for robot_id in robot_ids]

    return {
        "pretrain_config_path": str(pretrain_config_path),
        "robot_attention_config_path": str(robot_attention_config_path),
        "data_dir": Path(pretrain_config.get("data_dir", DEFAULT_DATA_DIR)),
        "team_prefix": str(pretrain_config.get("team_prefix", "t1")),
        "robot_ids": robot_ids,
        "time_window": int(pretrain_config.get("time_window", feature_params.get("time_window", 1))),
        "hidden_dim": int(pretrain_config.get("hidden_dim", feature_params.get("hidden_dim", 64))),
        "num_heads": int(pretrain_config.get("num_heads", feature_params.get("num_heads", 4))),
        "num_attention_layers": int(
            pretrain_config.get(
                "num_attention_layers",
                feature_params.get("num_attention_layers", 1),
            )
        ),
        "batch_size": int(pretrain_config.get("batch_size", model_params.get("batch_size", 256))),
        "epochs": int(pretrain_config.get("epochs", 20)),
        "learning_rate": float(pretrain_config.get("learning_rate", model_params.get("learning_rate", 3e-4))),
        "weight_decay": float(pretrain_config.get("weight_decay", 1e-5)),
        "param_loss_weight": float(pretrain_config.get("param_loss_weight", 1.0)),
        "scheduler": dict(
            pretrain_config.get(
                "scheduler",
                {"type": "cosine", "eta_min": 0.0},
            )
        ),
        "train_split": float(pretrain_config.get("train_split", 0.95)),
        "seed": int(pretrain_config.get("seed", 42)),
        "device": str(
            pretrain_config.get(
                "device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        ),
        "num_workers": int(pretrain_config.get("num_workers", 0)),
        "output_dir": Path(pretrain_config.get("output_dir", DEFAULT_OUTPUT_DIR)),
        "max_files": int(pretrain_config.get("max_files", 0)),
        "load_path": (
            Path(pretrain_config["load_path"])
            if pretrain_config.get("load_path")
            else None
        ),
        "wandb": dict(
            pretrain_config.get(
                "wandb",
                robot_attention_config.get("wandb", {}),
            )
        ),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_csv_paths(data_dir: Path, max_files: int = 0) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    csv_paths = sorted(data_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    if max_files > 0:
        csv_paths = csv_paths[:max_files]
    return csv_paths


def build_dataloaders(
    *,
    data_dir: Path,
    csv_paths: list[Path],
    team_prefix: str,
    robot_ids: list[int],
    time_window: int,
    batch_size: int,
    train_split: float,
    seed: int,
    num_workers: int,
) -> tuple[RobotAttentionTorchDataset, RobotAttentionTorchDataset, DataLoader, DataLoader]:
    if len(csv_paths) < 2:
        raise ValueError("Need at least 2 CSV files to create train/validation splits")

    shuffled_paths = list(csv_paths)
    random.Random(seed).shuffle(shuffled_paths)
    train_file_count = max(1, min(len(shuffled_paths) - 1, int(len(shuffled_paths) * train_split)))
    train_paths = shuffled_paths[:train_file_count]
    val_paths = shuffled_paths[train_file_count:]

    train_dataset = RobotAttentionTorchDataset(
        data_dir=data_dir,
        csv_paths=train_paths,
        team_prefix=team_prefix,
        robot_ids=robot_ids,
        time_window=time_window,
        return_metadata=False,
    )
    val_dataset = RobotAttentionTorchDataset(
        data_dir=data_dir,
        csv_paths=val_paths,
        team_prefix=team_prefix,
        robot_ids=robot_ids,
        time_window=time_window,
        return_metadata=False,
    )

    if train_dataset.obs_dim_per_robot != val_dataset.obs_dim_per_robot:
        raise ValueError("Train/validation datasets disagree on obs_dim_per_robot")
    if train_dataset.action_dim != val_dataset.action_dim:
        raise ValueError("Train/validation datasets disagree on action_dim")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def compute_behavior_cloning_loss(
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    *,
    num_robots: int,
    action_dim_per_robot: int,
    param_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = pred_action.shape[0]
    pred = pred_action.reshape(batch_size, num_robots, action_dim_per_robot)
    target = target_action.reshape(batch_size, num_robots, action_dim_per_robot)

    pred_action_logits = pred[..., :5]
    target_action_onehot = target[..., :5]
    target_action_idx = target_action_onehot.argmax(dim=-1)

    action_ce = F.cross_entropy(
        pred_action_logits.reshape(-1, 5),
        target_action_idx.reshape(-1),
    )

    pred_params = pred[..., 5:]
    target_params = target[..., 5:]

    dash_mask = target_action_idx == 0
    turn_mask = target_action_idx == 1
    param_mask = torch.zeros_like(target_params)
    param_mask[..., 0:3] = dash_mask.unsqueeze(-1)
    param_mask[..., 3:5] = turn_mask.unsqueeze(-1)

    masked_sq_error = ((pred_params - target_params) ** 2) * param_mask
    valid_param_count = param_mask.sum()
    if valid_param_count.item() > 0:
        param_mse = masked_sq_error.sum() / valid_param_count
    else:
        param_mse = masked_sq_error.sum() * 0.0

    total_loss = action_ce + float(param_loss_weight) * param_mse
    action_acc = (pred_action_logits.argmax(dim=-1) == target_action_idx).float().mean()

    return total_loss, {
        "action_ce": action_ce.detach(),
        "param_mse": param_mse.detach(),
        "action_acc": action_acc.detach(),
    }


def run_epoch(
    *,
    encoder: MultiRobotFeatureExtractor,
    action_head: ActionHead,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_robots: int,
    action_dim_per_robot: int,
    param_loss_weight: float,
) -> EpochMetrics:
    is_train = optimizer is not None
    encoder.train(is_train)
    action_head.train(is_train)

    total_loss = 0.0
    total_action_ce = 0.0
    total_param_mse = 0.0
    total_action_acc = 0.0
    total_samples = 0

    for batch in dataloader:
        observations, actions = batch
        observations = observations.to(device=device, dtype=torch.float32, non_blocking=True)
        actions = actions.to(device=device, dtype=torch.float32, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            pred_actions = action_head(encoder(observations))
            loss, metrics = compute_behavior_cloning_loss(
                pred_actions,
                actions,
                num_robots=num_robots,
                action_dim_per_robot=action_dim_per_robot,
                param_loss_weight=param_loss_weight,
            )
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = observations.shape[0]
        total_samples += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_action_ce += float(metrics["action_ce"]) * batch_size
        total_param_mse += float(metrics["param_mse"]) * batch_size
        total_action_acc += float(metrics["action_acc"]) * batch_size

    return EpochMetrics(
        loss=total_loss / max(1, total_samples),
        action_ce=total_action_ce / max(1, total_samples),
        param_mse=total_param_mse / max(1, total_samples),
        action_acc=total_action_acc / max(1, total_samples),
        samples=total_samples,
    )


def save_checkpoint(
    *,
    output_dir: Path,
    epoch: int,
    encoder: MultiRobotFeatureExtractor,
    action_head: ActionHead,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_metrics: EpochMetrics,
    val_metrics: EpochMetrics,
    is_best: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "action_head_state_dict": action_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_metrics": train_metrics.__dict__,
        "val_metrics": val_metrics.__dict__,
    }

    latest_path = output_dir / "latest.pt"
    torch.save(checkpoint, latest_path)
    torch.save(encoder.state_dict(), output_dir / "encoder_latest.pt")
    torch.save(action_head.state_dict(), output_dir / "action_head_latest.pt")

    if is_best:
        torch.save(checkpoint, output_dir / "best.pt")
        torch.save(encoder.state_dict(), output_dir / "encoder_best.pt")
        torch.save(action_head.state_dict(), output_dir / "action_head_best.pt")


def init_wandb(config: dict[str, Any], summary: dict[str, Any]):
    """Initialize a W&B run when enabled in config."""

    wandb_config = dict(config.get("wandb", {}))
    if not wandb_config.get("enabled", False):
        return None

    import wandb

    run_config = {
        **summary,
        "pretrain_config_path": config["pretrain_config_path"],
        "robot_attention_config_path": config["robot_attention_config_path"],
    }
    return wandb.init(
        project=wandb_config.get("project", "robot-attention-pretrain"),
        name=wandb_config.get("name"),
        entity=wandb_config.get("entity"),
        tags=wandb_config.get("tags"),
        group=wandb_config.get("group"),
        config=run_config,
    )


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict[str, Any],
    num_epochs: int,
):
    """Build the configured LR scheduler."""

    scheduler_type = str(scheduler_config.get("type", "cosine")).lower()
    if scheduler_type in {"none", "null", ""}:
        return None
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(scheduler_config.get("t_max", num_epochs))),
            eta_min=float(scheduler_config.get("eta_min", 0.0)),
        )
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(scheduler_config.get("step_size", 1))),
            gamma=float(scheduler_config.get("gamma", 0.1)),
        )
    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def resolve_load_checkpoint_path(load_path: Path | None) -> Path | None:
    """Resolve a resume path from either a checkpoint file or a checkpoint directory."""

    if load_path is None:
        return None
    if load_path.is_file():
        return load_path
    if load_path.is_dir():
        latest_path = load_path / "latest.pt"
        if latest_path.exists():
            return latest_path
        best_path = load_path / "best.pt"
        if best_path.exists():
            return best_path
        raise FileNotFoundError(
            f"No checkpoint file found in directory {load_path}. "
            "Expected latest.pt or best.pt."
        )
    raise FileNotFoundError(f"Checkpoint path does not exist: {load_path}")


def load_training_checkpoint(
    *,
    load_path: Path,
    encoder: MultiRobotFeatureExtractor,
    action_head: ActionHead,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, float]:
    """Load encoder/action head/optimizer state and return resume epoch + best val loss."""

    checkpoint = torch.load(load_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format at {load_path}")

    encoder_state = checkpoint.get("encoder_state_dict")
    action_head_state = checkpoint.get("action_head_state_dict")
    if not isinstance(encoder_state, dict) or not isinstance(action_head_state, dict):
        raise ValueError(
            f"Checkpoint {load_path} does not contain encoder_state_dict and action_head_state_dict"
        )

    encoder.load_state_dict(encoder_state)
    action_head.load_state_dict(action_head_state)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and isinstance(scheduler_state, dict):
        scheduler.load_state_dict(scheduler_state)

    epoch = int(checkpoint.get("epoch", 0))
    val_metrics = checkpoint.get("val_metrics", {})
    best_val_loss = float(val_metrics.get("loss", math.inf)) if isinstance(val_metrics, dict) else math.inf
    return epoch + 1, best_val_loss

def main() -> None:
    args = parse_args()
    config = build_runtime_config(args)
    set_seed(config["seed"])

    device = torch.device(config["device"])
    csv_paths = discover_csv_paths(config["data_dir"], max_files=config["max_files"])
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        data_dir=config["data_dir"],
        csv_paths=csv_paths,
        team_prefix=config["team_prefix"],
        robot_ids=list(config["robot_ids"]),
        time_window=config["time_window"],
        batch_size=config["batch_size"],
        train_split=config["train_split"],
        seed=config["seed"],
        num_workers=config["num_workers"],
    )

    action_dim_per_robot = train_dataset.action_dim // train_dataset.num_robots
    encoder = MultiRobotFeatureExtractor(
        obs_dim=train_dataset.obs_dim_per_robot,
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_attention_layers=config["num_attention_layers"],
    ).to(device)
    action_head = ActionHead(
        hidden_dim=config["hidden_dim"],
        action_dim=action_dim_per_robot,
    ).to(device)
    optimizer = torch.optim.AdamW(
        chain(encoder.parameters(), action_head.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_config=config["scheduler"],
        num_epochs=config["epochs"],
    )

    start_epoch = 1
    best_val_loss = math.inf
    resolved_load_path = resolve_load_checkpoint_path(config["load_path"])
    if resolved_load_path is not None:
        start_epoch, best_val_loss = load_training_checkpoint(
            load_path=resolved_load_path,
            encoder=encoder,
            action_head=action_head,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(f"Resuming from checkpoint: {resolved_load_path} at epoch {start_epoch}")

    run_summary = {
        "data_dir": str(config["data_dir"]),
        "csv_count": len(csv_paths),
        "train_csv_count": len(train_loader.dataset.csv_paths) if hasattr(train_loader.dataset, "csv_paths") else None,
        "val_csv_count": len(val_loader.dataset.csv_paths) if hasattr(val_loader.dataset, "csv_paths") else None,
        "train_num_samples": len(train_dataset),
        "val_num_samples": len(val_dataset),
        "num_robots": train_dataset.num_robots,
        "obs_dim_per_robot": train_dataset.obs_dim_per_robot,
        "action_dim": train_dataset.action_dim,
        "action_dim_per_robot": action_dim_per_robot,
        "time_window": config["time_window"],
        "team_prefix": config["team_prefix"],
        "robot_ids": list(config["robot_ids"]),
        "hidden_dim": config["hidden_dim"],
        "num_heads": config["num_heads"],
        "num_attention_layers": config["num_attention_layers"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "scheduler": config["scheduler"],
        "weight_decay": config["weight_decay"],
        "param_loss_weight": config["param_loss_weight"],
        "train_split": config["train_split"],
        "seed": config["seed"],
        "device": str(device),
        "load_path": str(resolved_load_path) if resolved_load_path is not None else None,
    }
    wandb_run = init_wandb(config, run_summary)

    print(
        f"Loaded train={len(train_dataset)} and val={len(val_dataset)} samples "
        f"from {len(csv_paths)} CSV files. "
        f"obs=({train_dataset.num_robots}, {train_dataset.obs_dim_per_robot}, {config['time_window']}), "
        f"action_dim={train_dataset.action_dim}"
    )

    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            train_metrics = run_epoch(
                encoder=encoder,
                action_head=action_head,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                num_robots=train_dataset.num_robots,
                action_dim_per_robot=action_dim_per_robot,
                param_loss_weight=config["param_loss_weight"],
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    encoder=encoder,
                    action_head=action_head,
                    dataloader=val_loader,
                    optimizer=None,
                    device=device,
                    num_robots=train_dataset.num_robots,
                    action_dim_per_robot=action_dim_per_robot,
                    param_loss_weight=config["param_loss_weight"],
                )

            is_best = val_metrics.loss < best_val_loss
            if is_best:
                best_val_loss = val_metrics.loss

            save_checkpoint(
                output_dir=config["output_dir"],
                epoch=epoch,
                encoder=encoder,
                action_head=action_head,
                optimizer=optimizer,
                scheduler=scheduler,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=is_best,
            )

            current_lr = float(optimizer.param_groups[0]["lr"])

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/lr": current_lr,
                        "train/loss": train_metrics.loss,
                        "train/action_ce": train_metrics.action_ce,
                        "train/param_mse": train_metrics.param_mse,
                        "train/action_acc": train_metrics.action_acc,
                        "val/loss": val_metrics.loss,
                        "val/action_ce": val_metrics.action_ce,
                        "val/param_mse": val_metrics.param_mse,
                        "val/action_acc": val_metrics.action_acc,
                        "val/best_loss": best_val_loss,
                    },
                    step=epoch,
                )

            if scheduler is not None:
                scheduler.step()

            print(
                f"epoch {epoch:03d} | "
                f"lr={current_lr:.6g} "
                f"train_loss={train_metrics.loss:.6f} "
                f"train_ce={train_metrics.action_ce:.6f} "
                f"train_mse={train_metrics.param_mse:.6f} "
                f"train_acc={train_metrics.action_acc:.4f} | "
                f"val_loss={val_metrics.loss:.6f} "
                f"val_ce={val_metrics.action_ce:.6f} "
                f"val_mse={val_metrics.param_mse:.6f} "
                f"val_acc={val_metrics.action_acc:.4f}"
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print(f"Saved checkpoints to {config['output_dir']}")


if __name__ == "__main__":
    main()

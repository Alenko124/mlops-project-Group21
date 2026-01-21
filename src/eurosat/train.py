import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.profiler import ProfilerActivity, profile

from eurosat.data_dvc import DataConfig, create_dataloaders
from eurosat.model import ModelConfig, create_model


def mem(msg):
    print(msg, psutil.Process(os.getpid()).memory_info().rss / 1024**3, "GB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training configuration overrides")
    # data
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset root directory",
    )

    parser.add_argument(
        "--data-batch-size",
        type=int,
        default=None,
        help="Batch size for training",
    )

    parser.add_argument(
        "--data-sample-every",
        type=int,
        default=None,
        help="Load every N-th sample (debugging)",
    )

    parser.add_argument(
        "--data-num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers",
    )

    parser.add_argument(
        "--data-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable pin_memory in DataLoader",
    )

    # model
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model architecture name (e.g. resnet18.a1_in1k)",
    )

    parser.add_argument(
        "--model-num-classes",
        type=int,
        default=None,
        help="Number of output classes",
    )

    parser.add_argument(
        "--model-pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use pretrained weights",
    )

    parser.add_argument(
        "--model-freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Freeze backbone parameters",
    )

    # -------------------------
    # Core training params
    # -------------------------
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "sgd", "adamw"],
    )
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)

    # -------------------------
    # Paths / logging
    # -------------------------
    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--log-dir", type=str)

    # -------------------------
    # Feature toggles
    # -------------------------
    parser.add_argument(
        "--enable-profiling",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enable-wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # -------------------------
    # Weights & Biases
    # -------------------------
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-project", type=str)

    return parser.parse_args()


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    epochs: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    momentum: float = 0.9
    seed: int = 42
    device: Optional[str] = None
    checkpoint_path: str = "models/resnet18_eurosat4.pt"
    log_dir: Optional[str] = "outputs/runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    enable_profiling: bool = True
    enable_wandb: bool = True
    wandb_entity: str = "lp6adi-danmarks-tekniske-universitet-dtu"
    wandb_project: str = "eurosat"


def apply_args_to_config(cfg: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    for key, value in vars(args).items():
        if value is None:
            continue

        key = key.replace("-", "_")

        # 🔹 Nested DataConfig: --data-xxx
        if key.startswith("data_"):
            field_name = key[len("data_") :]
            if hasattr(cfg.data, field_name):
                setattr(cfg.data, field_name, value)
            else:
                raise ValueError(f"Unknown DataConfig field: {field_name}")

        # 🔹 Top-level TrainingConfig
        elif hasattr(cfg, key):
            setattr(cfg, key, value)

        else:
            raise ValueError(f"Unknown config field: {key}")

    return cfg


def train(config: Optional[TrainingConfig] = None) -> List[Dict[str, float]]:
    """Train a classification model using the provided configuration.

    Returns:
        Training history with metrics for each epoch.
    """
    args = parse_args()
    cfg = config or TrainingConfig()
    cfg = apply_args_to_config(cfg, args)
    print("===== FINAL TRAINING CONFIG =====")
    print(json.dumps(asdict(cfg), indent=2))
    print("=================================")
    device = _get_device(cfg)
    print(f"Using device: {device}")
    _set_seed(cfg.seed)
    run = _setup_logging(cfg)

    model = create_model(cfg.model, device=device)
    mem("After model")
    train_loader, val_loader, _ = create_dataloaders(cfg.data)
    mem("After dataloaders")
    criterion = nn.CrossEntropyLoss()
    optimizer = _create_optimizer(model, cfg)

    history: List[Dict[str, float]] = []
    prof = None
    if cfg.enable_profiling:
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        prof = profile(
            activities=activities,
            record_shapes=True,
        )
        prof.start()

    for epoch in range(cfg.epochs):
        train_loss, train_acc = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
        )
        val_loss, val_acc = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=False,
        )

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if run is not None:
            run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

    if prof is not None:
        prof.stop()
        del prof
        prof = None
    _save_checkpoint(model, cfg.checkpoint_path)
    _persist_run(cfg, history, prof)

    if run is not None:
        run.finish()

    return history


def _run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    training: bool,
) -> Tuple[float, float]:
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if training and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    mean_loss = total_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return mean_loss, accuracy


def _setup_logging(config: TrainingConfig) -> Optional[wandb.sdk.wandb_run.Run]:
    """Initialize WandB tracking if enabled."""
    if not config.enable_wandb:
        return None

    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        config={
            "epochs": config.epochs,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "optimizer": config.optimizer,
            "momentum": config.momentum,
            "seed": config.seed,
            "batch_size": config.data.batch_size,
            "model_name": config.model.model_name,
            "num_classes": config.model.num_classes,
            "freeze_backbone": config.model.freeze_backbone,
        },
    )

    run.define_metric("train_loss", summary="min")
    run.define_metric("val_loss", summary="min")
    run.define_metric("train_acc", summary="max")
    run.define_metric("val_acc", summary="max")

    return run


def _create_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer based on configuration."""
    trainable_params = (param for param in model.parameters() if param.requires_grad)

    optimizer_name = config.optimizer.lower()
    if optimizer_name == "adam":
        return optim.Adam(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    elif optimizer_name == "adamw":
        return optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(trainable_params, lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}. Choose from: adam, adamw, sgd")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device(config: TrainingConfig) -> torch.device:
    if config.device:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_checkpoint(model: torch.nn.Module, path: str) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # ---- Upload to GCS ----
    # bucket_name = "mlops-group21"
    # gcs_path = f"models/{checkpoint_path.name}"

    # client = storage.Client()
    # bucket = client.bucket(bucket_name)
    # blob = bucket.blob(gcs_path)

    # blob.upload_from_filename(str(checkpoint_path))
    # print(f"☁️ Uploaded checkpoint to gs://{bucket_name}/{gcs_path}")


def _persist_run(
    config: TrainingConfig,
    history: List[Dict[str, float]],
    prof: Optional[profile] = None,
) -> None:
    if not config.log_dir:
        return

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "history": history,
    }
    output_path = log_dir / "last_run.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved run metadata to {output_path}")

    if prof is not None:
        prof_path = log_dir / "last_run_profiling.json"
        if config.enable_profiling and prof is not None:
            _save_profiling_stats(prof, prof_path)
        print(f"Saved profiling stats to {prof_path}")


def _save_profiling_stats(prof: profile, path: Path) -> None:
    """Save profiling statistics to JSON file."""
    stats = prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1)
    profiling_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": str(stats),
        "total_cpu_time": prof.key_averages().total_average().cpu_time_total,
    }
    if torch.cuda.is_available():
        profiling_data["total_cuda_time"] = prof.key_averages().total_average().cuda_time_total
    with path.open("w", encoding="utf-8") as f:
        json.dump(profiling_data, f, indent=2, default=str)


if __name__ == "__main__":
    train()

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import psutil
import torch
import wandb
from torch.utils.data import DataLoader

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
        "--val-split",
        type=float,
        default=None,
        help="Validation split ratio",
    )

    parser.add_argument(
        "--test-split",
        type=float,
        default=None,
        help="Test split ratio",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader num_workers",
    )

    parser.add_argument(
        "--pin-memory",
        action="store_true",
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
        "--num-classes",
        type=int,
        default=None,
        help="Number of output classes",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )

    # misc
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name",
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity/username",
    )

    return parser.parse_args()


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Data config
    data: DataConfig = field(default_factory=DataConfig)

    # Model config
    model: ModelConfig = field(default_factory=ModelConfig)

    # Training hyperparameters
    lr: float = 0.001
    epochs: int = 10
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "s_kruh_te"


def apply_args_to_config(
    cfg: TrainingConfig, args: argparse.Namespace
) -> TrainingConfig:
    for key, value in vars(args).items():
        if value is None:
            continue

        # Check top-level config
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        # Check data config
        elif hasattr(cfg.data, key):
            setattr(cfg.data, key, value)
        # Check model config
        elif hasattr(cfg.model, key):
            setattr(cfg.model, key, value)

    return cfg


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in enumerate(dataloader):
            # Handle potential unpacking differences if dataloader returns index
            if isinstance(images, int):
                continue
            # Usually dataloader yields (images, labels) directly
            pass

        # Re-iterating correctly for standard PyTorch DataLoader
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main():
    mem("Start")
    args = parse_args()
    cfg = TrainingConfig()
    cfg = apply_args_to_config(cfg, args)

    print(f"Configuration: {asdict(cfg)}")
    set_seed(cfg.seed)

    # Initialize WandB
    if cfg.use_wandb:
        # Check for API key
        if "WANDB_API_KEY" not in os.environ:
            print("Warning: WANDB_API_KEY not found. Disabling WandB.")
            cfg.use_wandb = False
        else:
            wandb.init(
                project=cfg.wandb_project,
                config=asdict(cfg),
                name=f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            )

    mem("Before data")
    # Data
    train_loader, val_loader, _ = create_dataloaders(cfg.data)
    mem("After data")

    # Model
    model = create_model(cfg.model)
    model = model.to(cfg.device)

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/runs/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=4)

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, cfg.device)

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        if cfg.use_wandb:
            wandb.log(metrics)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "config": asdict(cfg),
                },
                output_dir / "best_model.pth",
            )
            # Link to generic name for easy access
            latest_model_path = Path("models/model.pth")
            latest_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), latest_model_path)

    print(f"Training completed. Best Validation Accuracy: {best_val_acc:.4f}")
    mem("End")

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

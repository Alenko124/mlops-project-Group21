import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data import DataConfig, create_dataloaders
from model import ModelConfig, create_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    epochs: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: Optional[str] = None
    checkpoint_path: str = "models/resnet18_eurosat2.pt"
    log_dir: Optional[str] = "outputs/runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def train(config: Optional[TrainingConfig] = None) -> None:
    """Train a classification model using the provided configuration."""

    cfg = config or TrainingConfig()
    _setup_logging()
    device = _get_device(cfg)
    logger.info("Using device %s", device)
    _set_seed(cfg.seed)

    model = create_model(cfg.model, device=device)
    train_loader, val_loader, _ = create_dataloaders(cfg.data)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        (param for param in model.parameters() if param.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    history: List[Dict[str, float]] = []
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

        logger.info(
            "Epoch %d/%d | Train loss %.4f acc %.4f | Val loss %.4f acc %.4f",
            epoch + 1,
            cfg.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
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

    _save_checkpoint(model, cfg.checkpoint_path)
    _persist_run(cfg, history)


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


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


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
    logger.info("Saved checkpoint to %s", checkpoint_path)


def _persist_run(config: TrainingConfig, history: List[Dict[str, float]]) -> None:
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
    logger.info("Saved run metadata to %s", output_path)


if __name__ == "__main__":
    train()

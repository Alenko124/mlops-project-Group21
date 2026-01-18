"""Hyperparameter sweep using WandB for EuroSAT training."""

import wandb

from eurosat.data import DataConfig
from eurosat.model import ModelConfig
from eurosat.train import TrainingConfig, train


def objective() -> None:
    """Training objective function for sweep.
    
    WandB agent automatically sets up run context with sweep config.
    """
    run = wandb.init()
    
    config = TrainingConfig(
        epochs=run.config.epochs,
        lr=run.config.lr,
        weight_decay=run.config.weight_decay,
        optimizer=run.config.optimizer,
        momentum=run.config.momentum,
        seed=run.config.seed,
        data=DataConfig(
            batch_size=run.config.batch_size,
            num_workers=4,
        ),
        model=ModelConfig(
            freeze_backbone=run.config.freeze_backbone,
        ),
        enable_wandb=False,
        enable_profiling=False,
    )

    history = train(config)
    
    best_val_loss = min(h["val_loss"] for h in history)
    best_val_acc = max(h["val_acc"] for h in history)
    
    run.log({
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
    })

def sweep_basic_config() -> str:
    """Define basic hyperparameter sweep for learning rate and optimizer."""
    sweep_config = {
        "method": "random",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "parameters": {
            "epochs": {"value": 5},
            "lr": {"min": 1e-5, "max": 1e-2},
            "weight_decay": {"values": [1e-4, 1e-3, 0]},
            "optimizer": {"values": ["adam", "adamw", "sgd"]},
            "momentum": {"value": 0.9},
            "batch_size": {"values": [16, 32, 64]},
            "seed": {"value": 42},
            "freeze_backbone": {"value": True},
        },
    }
    return wandb.sweep(
        sweep=sweep_config,
        project="eurosat-sweep",
        entity="lp6adi-danmarks-tekniske-universitet-dtu",
    )


def sweep_extended() -> str:
    """Define extended hyperparameter sweep with model config."""
    sweep_config = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "parameters": {
            "epochs": {"value": 10},
            "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
            "optimizer": {"values": ["adam", "adamw", "sgd"]},
            "momentum": {"min": 0.8, "max": 0.99},
            "batch_size": {"values": [16, 32, 64]},
            "seed": {"value": 42},
            "freeze_backbone": {"values": [True, False]},
        },
    }
    return wandb.sweep(
        sweep=sweep_config,
        project="eurosat-sweep",
        entity="lp6adi-danmarks-tekniske-universitet-dtu",
    )


def main(sweep_id: str, count: int = 5) -> None:
    """Run sweep agents.
    
    Args:
        sweep_id: WandB sweep ID from sweep creation.
        count: Number of sweep iterations per agent.
    """
    print(f"Starting sweep with ID: {sweep_id}")
    wandb.agent(sweep_id, function=objective, count=count)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        sweep_type = sys.argv[1]
        sweep_count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    else:
        sweep_type = "basic"
        sweep_count = 5

    if sweep_type == "basic":
        sweep_id = sweep_basic_config()
    elif sweep_type == "extended":
        sweep_id = sweep_extended()
    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}. Choose 'basic' or 'extended'.")

    main(sweep_id, count=sweep_count)

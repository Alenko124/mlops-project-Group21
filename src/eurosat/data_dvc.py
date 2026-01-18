from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


@dataclass
class DataConfig:
    data_dir: str = "data/raw"
    batch_size: int = 32
    sample_every: int = 40
    num_workers: int = 4
    pin_memory: bool = True



def subsample_dataset(dataset, n: int):
    """Return a dataset that contains every n-th sample."""
    if n <= 1:
        return dataset
    indices = list(range(0, len(dataset), n))
    return Subset(dataset, indices)


def create_dataloaders(config: DataConfig,) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    root = Path(config.data_dir)

    train_ds = ImageFolder(root / "train", transform=transform)
    val_ds = ImageFolder(root / "val", transform=transform)
    test_ds = ImageFolder(root / "test", transform=transform)

    # Subsampling
    train_ds = subsample_dataset(train_ds, config.sample_every)
    val_ds = subsample_dataset(val_ds, config.sample_every)
    test_ds = subsample_dataset(test_ds, config.sample_every)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, test_loader

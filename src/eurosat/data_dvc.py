from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
from PIL import Image

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img: Image.Image):
        img = np.array(img)
        augmented = self.transform(image=img)
        return augmented["image"]


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


def get_transforms():
    train_tf = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )

    eval_tf = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )

    return train_tf, eval_tf


def create_dataloaders(
    config: DataConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_tf, eval_tf = get_transforms()

    root = Path(config.data_dir)

    train_ds = ImageFolder(
        root / "train",
        transform=AlbumentationsTransform(train_tf),
    )
    val_ds = ImageFolder(
        root / "val",
        transform=AlbumentationsTransform(eval_tf),
    )
    test_ds = ImageFolder(
        root / "test",
        transform=AlbumentationsTransform(eval_tf),
    )

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

from dataclasses import dataclass
from typing import Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class DataConfig:
    """Configuration for the EuroSAT dataloaders."""

    dataset_name: str = "nielsr/eurosat-demo"
    seed: int = 42
    test_size: float = 0.2
    val_ratio_within_test: float = 0.5
    sample_every: int = 20
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


def load_eurosat_splits(config: DataConfig):
    """Load and split EuroSAT into train/val/test subsets with deterministic shuffling."""

    ds = load_dataset(config.dataset_name)

    split = ds["train"].train_test_split(test_size=config.test_size, seed=config.seed)
    temp = split["test"].train_test_split(test_size=config.val_ratio_within_test, seed=config.seed)

    stride = max(config.sample_every, 1)
    train_ds = split["train"].shuffle(seed=config.seed).select(range(0, len(split["train"]), stride))
    val_ds = temp["train"].shuffle(seed=config.seed).select(range(0, len(temp["train"]), stride))
    test_ds = temp["test"].shuffle(seed=config.seed).select(range(0, len(temp["test"]), stride))

    return train_ds, val_ds, test_ds


class EuroSATDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        image = self.transform(image)
        return image, label


def create_dataloaders(config: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders using the provided configuration."""

    train_ds, val_ds, test_ds = load_eurosat_splits(config)

    train_dataset = EuroSATDataset(train_ds)
    val_dataset = EuroSATDataset(val_ds)
    test_dataset = EuroSATDataset(test_ds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, test_loader

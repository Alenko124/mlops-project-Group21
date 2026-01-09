from datasets import load_dataset
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision import transforms

# --------------------------------------------------
# 1. Load & split dataset
# --------------------------------------------------

def load_eurosat_splits(seed: int = 42):
    ds = load_dataset("nielsr/eurosat-demo")

    split = ds["train"].train_test_split(test_size=0.2, seed=seed)
    temp = split["test"].train_test_split(test_size=0.5, seed=seed)

    train_ds = split["train"].shuffle(seed=seed).select(range(0, len(split["train"]), 10))
    val_ds   = temp["train"].shuffle(seed=seed).select(range(0, len(temp["train"]), 10))
    test_ds  = temp["test"].shuffle(seed=seed).select(range(0, len(temp["test"]), 10))

    return train_ds, val_ds, test_ds


# --------------------------------------------------
# 2. PyTorch Dataset wrapper
# --------------------------------------------------

class EuroSATDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

        # ResNet / ImageNet standard
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # scales to [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        image = item["image"].convert("RGB")  # PIL
        label = item["label"]

        image = self.transform(image)

        return image, label
# --------------------------------------------------
# 3. Dataloaders
# --------------------------------------------------

def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
):
    train_ds, val_ds, test_ds = load_eurosat_splits(seed)

    train_dataset = EuroSATDataset(train_ds)
    val_dataset   = EuroSATDataset(val_ds)
    test_dataset  = EuroSATDataset(test_ds)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


    return train_loader, val_loader, test_loader

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from tqdm import tqdm


# ============================================================
# Data (PREUZETO IZ TVOG KODA)
# ============================================================
@dataclass
class DataConfig:
    data_dir: str = "data/raw"
    batch_size: int = 32
    sample_every: int = 1
    num_workers: int = 16
    pin_memory: bool = True


def subsample_dataset(dataset, n: int):
    if n <= 1:
        return dataset
    indices = list(range(0, len(dataset), n))
    return Subset(dataset, indices)


def create_dataloaders(config: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
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


# ============================================================
# Config
# ============================================================
MODEL_URL = "https://storage.googleapis.com/mlops-group21/models/resnet18_eurosat_latest.pt"
MODEL_PATH = "models/resnet18_eurosat_latest.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10


# ============================================================
# Download model
# ============================================================
Path("models").mkdir(exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
else:
    print("✅ Model already exists")


# ============================================================
# Load data
# ============================================================
data_cfg = DataConfig()
_, _, test_loader = create_dataloaders(data_cfg)

print(f"📦 Test samples: {len(test_loader.dataset)}")


# ============================================================
# Load model (TVOJ SETUP)
# ============================================================
model = timm.create_model(
    "resnet18.a1_in1k",
    pretrained=False,
    num_classes=NUM_CLASSES,
)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()


# ============================================================
# Evaluation
# ============================================================
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total

print("\n==============================")
print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("==============================")

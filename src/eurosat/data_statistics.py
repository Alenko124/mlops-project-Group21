"""Dataset statistics and visualization utilities."""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def get_transforms():
    """Get standard image transforms."""
    return Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def dataset_statistics(data_dir: str = "data/raw", output_dir: str = "reports") -> Dict:
    """Generate comprehensive dataset statistics and visualizations.

    Args:
        data_dir: Path to dataset root directory (should contain train/, val/, test/)
        output_dir: Directory to save report figures and statistics

    Returns:
        Dictionary with statistics summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    transforms = get_transforms()
    data_path = Path(data_dir)

    # Load datasets
    print("Loading datasets...")
    train_dataset = ImageFolder(data_path / "train", transform=transforms)
    val_dataset = ImageFolder(data_path / "val", transform=transforms)
    test_dataset = ImageFolder(data_path / "test", transform=transforms)

    # Get class names
    class_names = train_dataset.classes
    num_classes = len(class_names)

    # Compute statistics
    print("Computing statistics...")
    stats = {
        "dataset_name": "EuroSAT",
        "num_classes": num_classes,
        "class_names": class_names,
        "train": {
            "num_samples": len(train_dataset),
            "image_shape": train_dataset[0][0].shape,
        },
        "val": {
            "num_samples": len(val_dataset),
            "image_shape": val_dataset[0][0].shape,
        },
        "test": {
            "num_samples": len(test_dataset),
            "image_shape": test_dataset[0][0].shape,
        },
    }

    # Print summary in markdown format
    print("# Dataset Statistics Report")
    print(f"\n## Dataset: {stats['dataset_name']}")
    print(f"\n**Number of classes:** {num_classes}")
    print(f"**Class names:** {', '.join(class_names)}")
    print("\n## Dataset Splits")
    print("\n### Train Set")
    print(f"- Samples: **{stats['train']['num_samples']}**")
    print(f"- Image shape: {stats['train']['image_shape']}")
    print("\n### Validation Set")
    print(f"- Samples: **{stats['val']['num_samples']}**")
    print(f"- Image shape: {stats['val']['image_shape']}")
    print("\n### Test Set")
    print(f"- Samples: **{stats['test']['num_samples']}**")
    print(f"- Image shape: {stats['test']['image_shape']}")

    # Get class distributions
    print("Generating class distribution plots...")
    train_targets = torch.tensor([label for _, label in train_dataset.samples])
    val_targets = torch.tensor([label for _, label in val_dataset.samples])
    test_targets = torch.tensor([label for _, label in test_dataset.samples])

    train_dist = torch.bincount(train_targets, minlength=num_classes).numpy()
    val_dist = torch.bincount(val_targets, minlength=num_classes).numpy()
    test_dist = torch.bincount(test_targets, minlength=num_classes).numpy()

    stats["class_distribution"] = {
        "train": train_dist.tolist(),
        "val": val_dist.tolist(),
        "test": test_dist.tolist(),
    }

    # Plot class distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(range(num_classes), train_dist, color="steelblue")
    axes[0].set_title("Train Set Class Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_xticks(range(num_classes))
    axes[0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(range(num_classes), val_dist, color="seagreen")
    axes[1].set_title("Val Set Class Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Number of Samples")
    axes[1].set_xticks(range(num_classes))
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(range(num_classes), test_dist, color="coral")
    axes[2].set_title("Test Set Class Distribution", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Class")
    axes[2].set_ylabel("Number of Samples")
    axes[2].set_xticks(range(num_classes))
    axes[2].set_xticklabels(class_names, rotation=45, ha="right")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "class_distribution.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved class distribution plot to {output_path / 'class_distribution.png'}")
    plt.close()

    # Plot one sample per class for each split
    print("Generating sample image plots...")
    sample_indices = {
        "Train": _get_first_indices_per_class(train_dataset, num_classes),
        "Val": _get_first_indices_per_class(val_dataset, num_classes),
        "Test": _get_first_indices_per_class(test_dataset, num_classes),
    }

    fig, axes = plt.subplots(3, num_classes, figsize=(num_classes * 3, 9))
    axes = np.atleast_2d(axes).reshape(3, num_classes)

    for row_idx, (split_name, dataset, indices) in enumerate(
        (
            ("Train", train_dataset, sample_indices["Train"]),
            ("Val", val_dataset, sample_indices["Val"]),
            ("Test", test_dataset, sample_indices["Test"]),
        )
    ):
        axes[row_idx, 0].set_ylabel(split_name, fontsize=12, fontweight="bold")
        for col_idx, sample_idx in enumerate(indices):
            img, label = dataset[sample_idx]
            img = _denormalize(img)
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].set_title(f"{class_names[label]}", fontsize=10)
            axes[row_idx, col_idx].axis("off")
        for col_idx in range(len(indices), num_classes):
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path / "sample_images.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved sample images plot to {output_path / 'sample_images.png'}")
    plt.close()

    # Save statistics to JSON
    stats_file = output_path / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_file}")

    return stats


def _denormalize(img: torch.Tensor) -> np.ndarray:
    """Denormalize an image tensor to displayable format.

    Args:
        img: Normalized image tensor [C, H, W]

    Returns:
        Denormalized image as numpy array [H, W, C] in [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def _get_first_indices_per_class(dataset: ImageFolder, num_classes: int) -> List[int]:
    """Return the first sample index for each class present in the dataset."""
    seen = set()
    indices: List[int] = []
    for idx, (_, label) in enumerate(dataset.samples):
        if label in seen:
            continue
        indices.append(idx)
        seen.add(label)
        if len(seen) == num_classes:
            break
    return indices


if __name__ == "__main__":
    typer.run(dataset_statistics)

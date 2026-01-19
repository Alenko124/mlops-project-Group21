from pathlib import Path
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import random

SEED = 42
TEST_SIZE = 0.2
VAL_RATIO = 0.5

OUT_DIR = Path("data/raw")
random.seed(SEED)

def save_split(dataset, split_name):
    for idx, item in enumerate(dataset):
        label = item["label"]
        label_dir = OUT_DIR / split_name / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)

        img: Image.Image = item["image"].convert("RGB")
        img.save(label_dir / f"{idx}.jpg")

def main():
    ds = load_dataset("nielsr/eurosat-demo")["train"]

    indices = list(range(len(ds)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=SEED
    )

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=VAL_RATIO, random_state=SEED
    )

    save_split(ds.select(train_idx), "train")
    save_split(ds.select(val_idx), "val")
    save_split(ds.select(test_idx), "test")

    print("✅ EuroSAT slike skinute i spremljene u data/raw")

if __name__ == "__main__":
    main()

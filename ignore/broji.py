from pathlib import Path
from collections import defaultdict


DATA_DIR = Path("data/raw")
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def count_images_per_class(split_dir: Path):
    class_counts = defaultdict(int)

    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        class_counts[class_name] += sum(
            1
            for f in class_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )

    return class_counts


def main():
    grand_total = 0

    for split in SPLITS:
        split_path = DATA_DIR / split
        if not split_path.exists():
            print(f"\n❌ {split}: folder ne postoji")
            continue

        print(f"\n📁 {split.upper()}")

        class_counts = count_images_per_class(split_path)
        split_total = sum(class_counts.values())
        grand_total += split_total

        for cls, count in sorted(class_counts.items()):
            print(f"  ├─ klasa {cls}: {count} slika")

        print(f"  └─ UKUPNO {split}: {split_total} slika")

    print(f"\n📊 UKUPNO SLIKA (svi splitovi): {grand_total}")


if __name__ == "__main__":
    main()

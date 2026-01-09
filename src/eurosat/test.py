from data import create_dataloaders


def main():
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=32,
        num_workers=2,
    )

    print("Dataset sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # Take one batch
    images, labels = next(iter(train_loader))

    print("\nOne batch:")
    print(f"  Images shape: {images.shape}")  # [B, 3, H, W]
    print(f"  Labels shape: {labels.shape}")  # [B]
    print(f"  Labels dtype: {labels.dtype}")


if __name__ == "__main__":
    main()

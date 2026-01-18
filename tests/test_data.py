import pytest
import torch
from datasets import load_dataset

from eurosat.data import DataConfig, EuroSATDataset, load_eurosat, load_eurosat_splits


@pytest.fixture(scope="module")
def dataset():
    cfg = DataConfig(sample_every=20, num_workers=0, pin_memory=False)
    ds = load_eurosat(cfg)
    return ds

@pytest.fixture(scope="module")
def splits():
    cfg = DataConfig(sample_every=20, num_workers=0, pin_memory=False)
    train_ds, val_ds, test_ds = load_eurosat_splits(cfg)
    return cfg, train_ds, val_ds, test_ds

def test_length_splits_matching_total_dataset_length(dataset, splits):
    _, train_ds, val_ds, test_ds = splits
    assert len(dataset) == len(train_ds) + len(val_ds) + len(test_ds)

def test_splits_have_expected_structure(splits):
    _, train_ds, _, _ = splits
    sample = train_ds[0]
    assert "image" in sample
    assert "label" in sample


def test_train_val_test_lengths(splits):
    _, train_ds, val_ds, test_ds = splits

    train_len = len(train_ds)
    val_len = len(val_ds)
    test_len = len(test_ds)

    assert train_len > 0
    assert val_len > 0
    assert test_len > 0

    assert train_len > val_len + test_len


def test_number_of_classes_reasonable(splits):
    _, train_ds, _, _ = splits
    dataset = EuroSATDataset(train_ds)

    labels = set()
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.add(int(y))

    assert len(labels) == 10, "Too few classes observed in the sampled dataset"
    assert min(labels) >= 0
    assert max(labels) <= 9


def test_eurosat_dataset_image_shape(splits):
    _, train_ds, _, _ = splits
    dataset = EuroSATDataset(train_ds)

    x, _ = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 224, 224)


def test_eurosat_dataset_image_normalized_range(splits):
    _, train_ds, _, _ = splits
    dataset = EuroSATDataset(train_ds)

    x, _ = dataset[0]
    assert x.min().item() >= -5.0
    assert x.max().item() <= 5.0


def test_raw_dataset_total_samples():
    raw_dataset = load_dataset("nielsr/eurosat-demo")
    
    assert "train" in raw_dataset, "Dataset should have 'train' split"
    
    total_samples = len(raw_dataset["train"])
    
    assert total_samples == 27000, f"Expected 27000 samples, got {total_samples}"
    print(f"\n✓ Raw dataset has {total_samples} samples")

def test_raw_image_resolution_before_transform():
    """Test that raw images from HF have 64x64 resolution before transformation."""
    raw_dataset = load_dataset("nielsr/eurosat-demo")
    train_split = raw_dataset["train"]
    
    
    resolutions = set()
    for i in range(min(10, len(train_split))):
        sample = train_split[i]
        image = sample["image"]
        
       
        width, height = image.size
        resolutions.add((width, height))
        
        
        assert width == 64, f"Sample {i}: Expected width 64, got {width}"
        assert height == 64, f"Sample {i}: Expected height 64, got {height}"
    
    assert len(resolutions) == 1, f"All images should have same resolution, got {resolutions}"
    assert (64, 64) in resolutions, f"Expected (64, 64), got {resolutions}"

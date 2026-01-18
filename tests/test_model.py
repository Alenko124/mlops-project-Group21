"""Unit tests for the model module.

These tests verify that model creation, configuration, and forward passes work correctly.
"""
import torch
import pytest

from eurosat.model import ModelConfig, create_model

class TestCreateModel:

    def test_model_has_classifier(self):
        """Test that the model has a classifier layer."""
        config = ModelConfig()
        model = create_model(config)

        assert hasattr(model, "get_classifier")
        classifier = model.get_classifier()
        assert classifier is not None


    def test_model_freeze_backbone(self):
        """Test that backbone parameters are frozen when freeze_backbone=True."""
        config = ModelConfig(freeze_backbone=True)
        model = create_model(config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        assert trainable_params < total_params, "Backbone should be frozen"
        assert trainable_params > 0, "Classifier should be trainable"

        print(f"\nFreeze backbone: {trainable_params}/{total_params} params trainable")


class TestModelForwardPass:
    """Test model forward passes and output shapes."""

    def test_forward_pass(self):
        config = ModelConfig()
        model = create_model(config, device=torch.device("cpu"))
        model.eval()

        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {output.shape}"

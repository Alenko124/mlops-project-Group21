from dataclasses import dataclass
from typing import Optional

import timm
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration for creating a classification model."""

    model_name: str = "resnet18.a1_in1k"
    num_classes: int = 10
    pretrained: bool = True
    freeze_backbone: bool = True


def create_model(config: ModelConfig, device: Optional[torch.device] = None) -> nn.Module:
    """Instantiate a timm model configured for EuroSAT classification."""

    model = timm.create_model(
        config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
    )

    if config.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    for param in model.get_classifier().parameters():
        param.requires_grad = True

    if device is not None:
        model = model.to(device)

    return model

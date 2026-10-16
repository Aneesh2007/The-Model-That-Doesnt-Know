"""Classifier construction and checkpoint loading."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Build a ResNet-18 transfer-learning classifier for the known classes."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(checkpoint_path: str | Path, device: torch.device):
    """Load model weights and metadata from a training checkpoint."""
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:  # compatibility with older PyTorch versions
        checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError("Checkpoint has no 'class_names'; retrain using this project version.")
    image_size = int(checkpoint.get("image_size", 128))
    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint

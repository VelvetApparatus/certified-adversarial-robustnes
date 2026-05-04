import os

from torch import nn
from torchvision import models
import torch


# =================== ResNet 18-10 ===================

def _get_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Load a pretrained model
def _load_model(
        model_path,
        device: str = "cpu",
        num_classes=10,
):
    model = _get_resnet18(num_classes=num_classes)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "net" in checkpoint:
        state_dict = checkpoint["net"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


def build_resnet_18_10(
        weights_path: str = None,
        device: str = "cpu",
        pretrained: bool = False,
        num_classes=10,
):
    if not pretrained:
        return _get_resnet18(num_classes=num_classes)

    if weights_path is None:
        raise ValueError("weights_path must be provided when pretrained=True")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            "invalid path for custom weights: {}".format(weights_path)
        )

    return _load_model(
        model_path=weights_path,
        device=device,
        num_classes=num_classes,
    )


def build_wide_resnet_28_10(
        weights_path: str = None,
):
    if not os.path.exists(weights_path):
        raise FileNotFoundError("invalid path for custom weights: {}".format(weights_path))

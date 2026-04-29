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
        device: str = 'cpu',
        num_classes=10,

):
    model = _get_resnet18(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def build_resnet_18_10(
        weights_path: str = None,
        device: str = 'cpu',
        pretrained: bool = False,
):
    # todo: make num_classes configurable
    if not (pretrained):
        return _get_resnet18(num_classes=10)

    if not os.path.exists(weights_path):
        raise FileNotFoundError("invalid path for custom weights: {}".format(weights_path))

    return _load_model(weights_path, device)


def build_wide_resnet_28_10(
        weights_path: str = None,
):
    if not os.path.exists(weights_path):
        raise FileNotFoundError("invalid path for custom weights: {}".format(weights_path))

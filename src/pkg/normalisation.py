# src/normalization.py

import torch
from torch import nn

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]


class InputNormalizer(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            mean,
            std,
    ):
        super().__init__()
        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1),
        )

        self.model = model

    def forward(self, x):
        return self.model((x - self.mean) / self.std)

    def model(self):
        return self.model

    def normalize(self, x):
        return (x - self.mean) / self.std

# src/robustness/gaussian_noise.py

import torch

from src.robustness.input.common import RobustnessRegularization


class GaussianNoiseGenerator(RobustnessRegularization):
    """
    Adds Gaussian noise to input batch.

    Supports two modes:
    1. normalized_space=True:
       noise is added directly to normalized tensors.

    2. normalized_space=False:
       inputs are unnormalized to [0, 1], noise is added there,
       values are clamped to [0, 1], and then normalized back.
    """

    def __init__(
            self,
            sigma: float,
            ratio: float = 1.0,
            normalized_space: bool = True,
    ):
        super().__init__()
        if sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")

        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")

        self.sigma = sigma
        self.ratio = ratio
        self.normalized_space = normalized_space

    def _make_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mean is None or self.std is None:
            raise ValueError(
                "mean and std must be provided when normalized_space=False"
            )

        mean = torch.tensor(
            self.mean,
            dtype=x.dtype,
            device=x.device,
        ).view(1, -1, 1, 1)

        std = torch.tensor(
            self.std,
            dtype=x.dtype,
            device=x.device,
        ).view(1, -1, 1, 1)

        return mean, std

    def _add_noise_normalized_space(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.sigma

    def _add_noise_image_space(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._make_stats(x)

        x_image = x * std + mean
        x_image = x_image + torch.randn_like(x_image) * self.sigma
        x_image = x_image.clamp(0.0, 1.0)

        return (x_image - mean) / std

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalized_space:
            return self._add_noise_normalized_space(x)

        return self._add_noise_image_space(x)

    def augment_on_batch(self, x, y, model=None):
        if self.sigma == 0.0 or self.ratio == 0.0:
            return x, y

        if self.ratio >= 1.0:
            return self._add_noise(x), y

        batch_size = x.size(0)
        mask = torch.rand(batch_size, device=x.device) < self.ratio

        if mask.sum().item() == 0:
            return x, y

        x_aug = x.clone()
        x_aug[mask] = self._add_noise(x_aug[mask])

        return x_aug, y

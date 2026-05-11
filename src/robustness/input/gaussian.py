# src/robustness/gaussian_noise.py

import torch

from src.robustness.input.common import RobustnessRegularization


class GaussianNoiseGenerator(RobustnessRegularization):
    """
    Adds Gaussian noise to raw image-space inputs.

    Expected input:
        x in [0, 1]

    Output:
        x_noisy in [0, 1]

    Normalization, if needed, should happen inside the model
    via InputNormalizer.
    """

    def __init__(
            self,
            sigma: float,
            ratio: float = 1.0,
    ):
        super().__init__()

        if sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")

        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")

        self.sigma = sigma
        self.ratio = ratio

    def _add_noise_image_space(self, x: torch.Tensor) -> torch.Tensor:
        x_noisy = x + torch.randn_like(x) * self.sigma
        return x_noisy.clamp(0.0, 1.0)

    def augment_on_batch(self, x, y, model=None):
        if self.sigma == 0.0 or self.ratio == 0.0:
            return x, y

        if self.ratio >= 1.0:
            return self._add_noise_image_space(x), y

        batch_size = x.size(0)
        mask = torch.rand(batch_size, device=x.device) < self.ratio

        if mask.sum().item() == 0:
            return x, y

        x_aug = x.clone()
        x_aug[mask] = self._add_noise_image_space(x_aug[mask])

        return x_aug, y

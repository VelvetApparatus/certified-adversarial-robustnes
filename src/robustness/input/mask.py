import torch

from src.robustness.input.common import RobustnessRegularization


class MaskGen(RobustnessRegularization):
    def __init__(
            self,
            ratio: float,
            p: float,
    ):
        super().__init__()
        self.ratio = ratio
        self.p = p
        self.last_stats = {
            "masked_channel_fraction": 0.0,
            "masked_sample_fraction": 0.0,
            "num_masked_channels": 0,
            "num_total_channels": 0,
            "num_masked_samples": 0,
            "num_total_samples": 0,
        }

    def _reset_stats(
            self,
            batch_size: int = 0,
            channels: int = 0,
    ):
        self.last_stats = {
            "masked_channel_fraction": 0.0,
            "masked_sample_fraction": 0.0,
            "num_masked_channels": 0,
            "num_total_channels": int(batch_size * channels),
            "num_masked_samples": 0,
            "num_total_samples": int(batch_size),
        }

    def augment_on_batch(self, x, y, model):
        batch_size, channels = x.shape[:2]
        self._reset_stats(
            batch_size=batch_size,
            channels=channels,
        )

        if self.ratio == 0.0 or self.p == 0.0:
            return x, y

        device = x.device

        apply_mask = torch.rand(batch_size, device=device) <= self.ratio
        if apply_mask.sum() == 0:
            return x, y

        x_out = x.clone()

        channel_mask = torch.rand(batch_size, channels, 1, 1, device=device) > self.p

        channel_mask[~apply_mask] = True

        x_out = x_out * channel_mask

        channel_mask_2d = channel_mask.squeeze(-1).squeeze(-1)
        masked_channels = (~channel_mask_2d).sum().item()
        total_channels = batch_size * channels

        masked_per_sample = (~channel_mask_2d).any(dim=1)
        masked_samples = masked_per_sample.sum().item()

        self.last_stats = {
            "masked_channel_fraction": (
                masked_channels / total_channels if total_channels > 0 else 0.0
            ),
            "masked_sample_fraction": (
                masked_samples / batch_size if batch_size > 0 else 0.0
            ),
            "num_masked_channels": int(masked_channels),
            "num_total_channels": int(total_channels),
            "num_masked_samples": int(masked_samples),
            "num_total_samples": int(batch_size),
        }

        return x_out, y

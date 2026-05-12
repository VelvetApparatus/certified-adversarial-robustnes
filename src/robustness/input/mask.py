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

    def augment_on_batch(self, x, y, model):
        if self.ratio == 0.0 or self.p == 0.0:
            return x, y

        batch_size, channels = x.shape[:2]
        device = x.device

        apply_mask = torch.rand(batch_size, device=device) <= self.ratio
        if apply_mask.sum() == 0:
            return x, y

        x_out = x.clone()

        channel_mask = torch.rand(batch_size, channels, 1, 1, device=device) > self.p

        channel_mask[~apply_mask] = True

        x_out = x_out * channel_mask

        return x_out, y

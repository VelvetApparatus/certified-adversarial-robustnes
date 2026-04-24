import torch

from src.robustness.common import RobustnessRegularization
from src.adversaries.common import Adversary


class AdversarialTraining(RobustnessRegularization):
    def __init__(self, adversary: Adversary, ratio: float = 1.0):
        super().__init__()

        if adversary is None:
            raise ValueError("Adversary is None")
        if not (0.0 <= ratio <= 1.0):
            raise ValueError("ratio must be in [0, 1]")

        self.adversary = adversary
        self.ratio = ratio

    def augment_on_batch(self, x, y, model):
        if self.ratio == 0.0:
            return x, y

        batch_size = x.size(0)
        device = x.device

        mask = torch.rand(batch_size, device=device) < self.ratio

        if mask.sum() == 0:
            return x, y

        x_adv_source = x[mask]
        y_adv_source = y[mask]

        x_adv = self.adversary.gen(model, x_adv_source, y_adv_source)

        x_out = x.clone()
        x_out[mask] = x_adv

        return x_out, y

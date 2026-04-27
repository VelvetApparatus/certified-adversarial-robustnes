from src.adversaries.common import Adversary
import torch
import torch.nn as nn


class FGSMAttack(Adversary):
    def __init__(self, eps, loss_fn=None, mean=None, std=None):
        super().__init__(
            name="fgsm",
            params={"eps": eps, "mean": mean, "std": std},
            loss_fn=loss_fn or nn.CrossEntropyLoss(),
        )
        self.eps = eps
        self.mean = mean
        self.std = std

    def _gen(self, model, X, y):
        X_adv = X.detach().clone()
        X_adv.requires_grad_(True)

        logits = model(X_adv)
        loss = self.loss_fn(logits, y)

        grad = torch.autograd.grad(loss, X_adv)[0]

        eps = self.scale_for_normalized_input(self.eps, X_adv)

        X_adv = X_adv + eps * grad.sign()
        X_adv = self.clamp_input(X_adv)

        return X_adv.detach()

    def clamp_input(self, x):
        if self.mean is None or self.std is None:
            return torch.clamp(x, 0.0, 1.0)

        mean = self.as_channel_tensor(self.mean, x)
        std = self.as_channel_tensor(self.std, x)

        lower = (0.0 - mean) / std
        upper = (1.0 - mean) / std

        return torch.max(torch.min(x, upper), lower)

    def scale_for_normalized_input(self, value, x):
        if self.mean is None or self.std is None:
            return value

        std = self.as_channel_tensor(self.std, x)
        return value / std

    @staticmethod
    def as_channel_tensor(value, x):
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=x.device, dtype=x.dtype)
        else:
            tensor = torch.tensor(value, device=x.device, dtype=x.dtype)

        return tensor.view(1, -1, 1, 1)
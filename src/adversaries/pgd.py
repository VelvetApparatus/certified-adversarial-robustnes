from typing import Literal

import torch
from torch import nn

from src.adversaries.common import Adversary


class PGD(Adversary):
    """
    PGD is iterative method of adversarial example generation algorithm.
    It uses FGSM as a step of each iteration, but uses projection to avoid
    perturbation overgrowing.

    Link: https://arxiv.org/abs/1706.06083
    """

    def __init__(
        self,
        epsilon: float,
        alpha: float,
        steps: int,
        loss_fn: nn.Module,
        norm: Literal["Linf", "l2"] = "Linf",
        mean=None,
        std=None,
    ):
        super(PGD, self).__init__(
            name="PGD",
            loss_fn=loss_fn,
            params={
                "epsilon": epsilon,
                "alpha": alpha,
                "steps": steps,
                "norm": norm,
                "mean": mean,
                "std": std,
            },
        )

        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.norm = norm
        self.mean = mean
        self.std = std

    def __repr__(self):
        return (
            f"PGD("
            f"epsilon={self.epsilon}, "
            f"alpha={self.alpha}, "
            f"steps={self.steps}, "
            f"norm={self.norm}"
            f")"
        )

    def __str__(self):
        return self.__repr__()

    def _gen(self, model, X, y):
        X_orig = X.detach().clone()
        X_adv = X_orig.clone()

        eps = self.scale_for_normalized_input(self.epsilon, X)
        alpha = self.scale_for_normalized_input(self.alpha, X)

        for _ in range(self.steps):
            X_adv = X_adv.detach()
            X_adv.requires_grad_(True)

            logits = model(X_adv)
            loss = self.loss_fn(logits, y)

            grad = torch.autograd.grad(loss, X_adv)[0]

            if self.norm == "Linf":
                X_adv = X_adv.detach() + alpha * grad.sign()

            elif self.norm == "l2":
                grad = self.normalize_l2(grad)
                X_adv = X_adv.detach() + self.alpha * grad

            else:
                raise NotImplementedError(f"Unsupported norm: {self.norm}")

            delta = X_adv - X_orig

            if self.norm == "Linf":
                delta = self.l_inf_projection(delta, eps)

            elif self.norm == "l2":
                delta = self.l2_projection(delta, self.epsilon)

            X_adv = X_orig + delta
            X_adv = self.clamp_input(X_adv)

        return X_adv.detach()

    def l_inf_projection(self, delta, eps):
        return torch.clamp(delta, min=-eps, max=eps).detach()

    def l2_projection(self, delta, eps):
        delta_flat = delta.view(delta.size(0), -1)

        norms = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
        factors = torch.clamp(eps / (norms + 1e-12), max=1.0)

        delta_flat = delta_flat * factors

        return delta_flat.view_as(delta).detach()

    def normalize_l2(self, grad):
        grad_flat = grad.view(grad.size(0), -1)

        norms = torch.norm(grad_flat, p=2, dim=1, keepdim=True)
        grad_flat = grad_flat / (norms + 1e-12)

        return grad_flat.view_as(grad)

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
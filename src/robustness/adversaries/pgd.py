from typing import Literal

import torch
from torch import nn

from src.robustness.adversaries.common import Adversary


class PGD(Adversary):
    def __init__(
            self,
            epsilon: float,
            alpha: float,
            steps: int,
            loss_fn: nn.Module,
            norm: Literal["Linf", "l2"] = "Linf",
            random_start: bool = True,
    ):
        super(PGD, self).__init__(
            name="PGD",
            loss_fn=loss_fn,
            params={
                "epsilon": epsilon,
                "alpha": alpha,
                "steps": steps,
                "norm": norm,
                "random_start": random_start,
            },
        )

        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.norm = norm
        self.random_start = random_start

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

        if self.random_start:
            if self.norm == "Linf":
                delta = torch.empty_like(X_orig).uniform_(
                    -self.epsilon,
                    self.epsilon,
                )
                X_adv = X_orig + delta
                X_adv = torch.clamp(X_adv, 0.0, 1.0)

            elif self.norm == "l2":
                delta = torch.randn_like(X_orig)
                delta = self.normalize_l2(delta)

                radius = torch.rand(
                    X_orig.size(0), 1, 1, 1,
                    device=X_orig.device,
                    dtype=X_orig.dtype,
                )

                delta = delta * radius * self.epsilon
                X_adv = X_orig + delta
                X_adv = torch.clamp(X_adv, 0.0, 1.0)

            else:
                raise NotImplementedError(f"Unsupported norm: {self.norm}")
        else:
            X_adv = X_orig.clone()

        for _ in range(self.steps):
            X_adv = X_adv.detach()
            X_adv.requires_grad_(True)

            logits = model(X_adv)
            loss = self.loss_fn(logits, y)

            grad = torch.autograd.grad(loss, X_adv)[0]

            if self.norm == "Linf":
                X_adv = X_adv.detach() + self.alpha * grad.sign()

            elif self.norm == "l2":
                grad = self.normalize_l2(grad)
                X_adv = X_adv.detach() + self.alpha * grad

            else:
                raise NotImplementedError(f"Unsupported norm: {self.norm}")

            delta = X_adv - X_orig

            if self.norm == "Linf":
                delta = self.l_inf_projection(delta, self.epsilon)

            elif self.norm == "l2":
                delta = self.l2_projection(delta, self.epsilon)

            X_adv = X_orig + delta
            X_adv = torch.clamp(X_adv, 0.0, 1.0)

        return X_adv.detach()

    def l_inf_projection(self, delta, eps):
        return torch.clamp(delta, min=-eps, max=eps).detach()

    def l2_projection(self, delta, eps):
        delta_flat = delta.reshape(delta.size(0), -1)

        norms = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
        factors = torch.clamp(eps / (norms + 1e-12), max=1.0)

        delta_flat = delta_flat * factors

        return delta_flat.reshape_as(delta).detach()

    def normalize_l2(self, grad):
        grad_flat = grad.reshape(grad.size(0), -1)

        norms = torch.norm(grad_flat, p=2, dim=1, keepdim=True)
        grad_flat = grad_flat / (norms + 1e-12)

        return grad_flat.reshape_as(grad)

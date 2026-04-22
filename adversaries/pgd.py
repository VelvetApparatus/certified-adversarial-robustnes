from typing import Literal

import torch
from torch import nn

from adversaries.common import Adversary


class PGD(Adversary):
    """
    PGD is iterative method of adversarial example generation algorithm
    it uses FGSM as a step of each iteration, but uses projection to avoid
    perturbation overgrowing

    Link: https://arxiv.org/abs/1706.06083
    """
    def __init__(
        self,
        epsilon: float,
        alpha: float,
        steps: int,
        loss_fn: nn.Module,
        norm: Literal["Linf", "l2"] = "Linf",
    ):
        super(PGD, self).__init__(
            name="PGD",
            loss_fn=loss_fn,
            params={
                "epsilon": epsilon,
                "alpha": alpha,
                "steps": steps,
                "norm": norm,
            }
        )
        self.alpha = alpha
        self.steps = steps
        self.norm = norm
        self.epsilon = epsilon

    def __repr__(self):
        return f"PGD(epsilon={self.epsilon}, alpha={self.alpha}, steps={self.steps}, norm={self.norm})"

    def __str__(self):
        return self.__repr__()

    def gen(self, model, X, y):
        model.eval()

        X_orig = X.detach().clone()
        X_adv = X_orig.clone()

        for _ in range(self.steps):
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
                delta = self.l_inf_projection(delta)
            elif self.norm == "l2":
                delta = self.l2_projection(delta)

            X_adv = X_orig + delta
            X_adv = torch.clamp(X_adv, 0.0, 1.0)

        return X_adv.detach()

    def l_inf_projection(self, delta):
        return torch.clamp(delta, min=-self.epsilon, max=self.epsilon).detach()

    def l2_projection(self, delta):
        delta_flat = delta.view(delta.size(0), -1)
        norms = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
        factors = torch.clamp(self.epsilon / (norms + 1e-12), max=1.0)
        delta_flat = delta_flat * factors
        return delta_flat.view_as(delta).detach()

    def normalize_l2(self, grad):
        grad_flat = grad.view(grad.size(0), -1)
        norms = torch.norm(grad_flat, p=2, dim=1, keepdim=True)
        grad_flat = grad_flat / (norms + 1e-12)
        return grad_flat.view_as(grad)
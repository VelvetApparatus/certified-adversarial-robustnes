from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from src.robustness.adversaries.common import Adversary


class PGD(Adversary):
    def __init__(
            self,
            epsilon: float,
            alpha: float,
            steps: int,
            lossfn: Literal["cross_entropy", "kl_divergence"] = "cross_entropy",
            norm: Literal["Linf", "l2"] = "Linf",
            random_start: bool = True,
    ):

        if lossfn not in ["cross_entropy", "kl_divergence"]:
            raise NotImplementedError(f"Unsupported lossfn: {lossfn}")

        self.kl_divergence = lossfn == "kl_divergence"

        if lossfn == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.KLDivLoss(reduction="batchmean")

        super(PGD, self).__init__(
            name="PGD",
            params={
                "epsilon": epsilon,
                "alpha": alpha,
                "steps": steps,
                "norm": norm,
                "random_start": random_start,
                "loss_fn": lossfn,
            },
        )

        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.norm = norm.lower()
        self.random_start = random_start

    def __repr__(self):
        return (
            f"PGD("
            f"epsilon={self.epsilon}, "
            f"alpha={self.alpha}, "
            f"steps={self.steps}, "
            f"norm={self.norm}"
            f"random_start={self.random_start}"
            f"loss_fn={self.loss_fn}"
            f")"
        )

    def __str__(self):
        return self.__repr__()

    def _gen(self, model, X, y):
        """

        :param model: nn.Module
        :param X: input data
        :param y: target data, omitted in TRADES-style regime (loss_fn=kl_divergence)
        :return:
        X_adv: adversarial examples
        """
        X_orig = X.detach().clone()

        if self.kl_divergence:
            with torch.no_grad():
                clean_probs = F.softmax(model(X_orig), dim=1)

        # Random start to avoid gradient masking
        # https://arxiv.org/abs/1802.00420
        if self.random_start:
            if self.norm == "linf":
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

        # iterative generation
        for _ in range(self.steps):
            X_adv = X_adv.detach()
            X_adv.requires_grad_(True)

            logits_adv = model(X_adv)

            if self.kl_divergence:
                loss = self.loss_fn(
                    F.log_softmax(logits_adv, dim=1),
                    clean_probs,
                )
            else:
                loss = self.loss_fn(logits_adv, y)

            grad = torch.autograd.grad(loss, X_adv)[0]

            if self.norm == "linf":
                X_adv = X_adv.detach() + self.alpha * grad.sign()

            elif self.norm == "l2":
                grad = self.normalize_l2(grad)
                X_adv = X_adv.detach() + self.alpha * grad

            else:
                raise NotImplementedError(f"Unsupported norm: {self.norm}")

            delta = X_adv - X_orig

            if self.norm == "linf":
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


class SmoothPGD(PGD):
    def __init__(self, epsilon, alpha, steps, sigma, num_noise_vec, norm="l2", random_start=True, clamp_noisy=True):
        super().__init__(
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            lossfn="cross_entropy",
            norm=norm,
            random_start=random_start,
        )
        self.sigma = sigma
        self.num_noise_vec = num_noise_vec
        self.clamp_noisy = clamp_noisy

    def _repeat_for_noise_samples(self, x):
        b = x.size(0)
        return (
            x.unsqueeze(1)
            .repeat(1, self.num_noise_vec, 1, 1, 1)
            .view(b * self.num_noise_vec, *x.shape[1:])
        )

    def _gen(self, model, X, y):
        X_orig = X.detach().clone()

        # random start as in PGD
        if self.random_start:
            if self.norm == "linf":
                delta = torch.empty_like(X_orig).uniform_(-self.epsilon, self.epsilon)
            elif self.norm == "l2":
                delta = torch.randn_like(X_orig)
                delta = self.normalize_l2(delta)
                radius = torch.rand(
                    X_orig.size(0), 1, 1, 1,
                    device=X_orig.device,
                    dtype=X_orig.dtype,
                )
                delta = delta * radius * self.epsilon
            else:
                raise NotImplementedError(f"Unsupported norm: {self.norm}")

            X_adv = torch.clamp(X_orig + delta, 0.0, 1.0)
        else:
            X_adv = X_orig.clone()

        y_rep = y.repeat_interleave(self.num_noise_vec)

        for _ in range(self.steps):
            X_adv = X_adv.detach()
            X_adv.requires_grad_(True)

            X_adv_rep = self._repeat_for_noise_samples(X_adv)
            noise = torch.randn_like(X_adv_rep) * self.sigma
            X_noisy_adv = X_adv_rep + noise

            if self.clamp_noisy:
                X_noisy_adv = torch.clamp(X_noisy_adv, 0.0, 1.0)

            logits = model(X_noisy_adv)
            loss = F.cross_entropy(logits, y_rep)

            grad = torch.autograd.grad(loss, X_adv)[0]

            if self.norm == "linf":
                X_adv = X_adv.detach() + self.alpha * grad.sign()
            elif self.norm == "l2":
                X_adv = X_adv.detach() + self.alpha * self.normalize_l2(grad)

            delta = X_adv - X_orig

            if self.norm == "linf":
                delta = self.l_inf_projection(delta, self.epsilon)
            else:
                delta = self.l2_projection(delta, self.epsilon)

            X_adv = torch.clamp(X_orig + delta, 0.0, 1.0)

        return X_adv.detach()

from src.adversaries.common import Adversary
import torch
import torch.nn as nn


class FGSMAttack(Adversary):
    def __init__(
            self,
            eps,
            loss_fn=None,
            alpha=None,
            random_start: bool = False,
    ):
        super().__init__(
            name="fgsm",
            params={
                "eps": eps,
                "alpha": alpha if alpha is not None else eps,
                "random_start": random_start,
            },
            loss_fn=loss_fn or nn.CrossEntropyLoss(),
        )
        self.eps = eps
        self.alpha = alpha if alpha is not None else eps
        self.random_start = random_start

    def __repr__(self):
        return "FGSM attack (eps={eps})".format(eps=self.eps)

    def __str__(self):
        return self.__repr__()

    def _gen(self, model, X, y):
        X_orig = X.detach().clone()

        if self.random_start:
            X_adv = X_orig + torch.empty_like(X_orig).uniform_(
                -self.eps,
                self.eps,
            )
            X_adv = torch.clamp(X_adv, 0.0, 1.0)
        else:
            X_adv = X_orig.clone()

        X_adv.requires_grad_(True)

        logits = model(X_adv)
        loss = self.loss_fn(logits, y)

        grad = torch.autograd.grad(loss, X_adv)[0]

        X_adv = X_adv.detach() + self.alpha * grad.sign()

        delta = torch.clamp(
            X_adv - X_orig,
            min=-self.eps,
            max=self.eps,
        )

        X_adv = X_orig + delta
        X_adv = torch.clamp(X_adv, 0.0, 1.0)

        return X_adv.detach()

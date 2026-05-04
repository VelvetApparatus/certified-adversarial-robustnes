from src.adversaries.common import Adversary
import torch
import torch.nn as nn


class FGSMAttack(Adversary):
    def __init__(
            self,
            eps,
            loss_fn=None,
    ):
        super().__init__(
            name="fgsm",
            params={"eps": eps},
            loss_fn=loss_fn or nn.CrossEntropyLoss(),
        )
        self.eps = eps

    def __repr__(self):
        return "FGSM attack (eps={eps})".format(eps=self.eps)

    def __str__(self):
        return self.__repr__()

    def _gen(self, model, X, y):
        X_adv = X.detach().clone()
        X_adv.requires_grad_(True)

        logits = model(X_adv)
        loss = self.loss_fn(logits, y)

        grad = torch.autograd.grad(loss, X_adv)[0]

        eps = self.eps

        X_adv = X_adv + eps * grad.sign()
        X_adv = torch.clamp(X_adv, 0.0, 1.0)

        return X_adv.detach()

from src.adversaries.common import Adversary
import torch
import torch.nn as nn


class FGSMAttack(Adversary):
    """
        FGSM attack uses gradient of model for adversarial examples generation

        x_adv = x + eps * grad.sign()

        Link: https://arxiv.org/abs/1902.02918

    """

    def __init__(self, eps, loss_fn=None):
        super().__init__(
            name="fgsm",
            params={"eps": eps},
            loss_fn=loss_fn or nn.CrossEntropyLoss()
        )

    def _gen(self, model, X, y):
        X_adv = X.detach().clone()
        X_adv.requires_grad_(True)

        logits = model(X_adv)
        loss = self.loss_fn(logits, y)

        grad = torch.autograd.grad(loss, X_adv)[0]

        eps = self.params["eps"]
        X_adv = X_adv + eps * grad.sign()
        X_adv = torch.clamp(X_adv, 0.0, 1.0)

        return X_adv.detach()

from typing import List

from src.config.conf import AttackConfig, StAdvAttackConfig, FGSMAttackConfig, PGDAttackConfig
from src.adversaries.fgsm import FGSMAttack
from src.adversaries.pgd import PGD
from src.adversaries.stadv import StAdv
from src.pkg.get_loss_fn import get_loss_fn


def get_adversaries(
        cfg: List[AttackConfig],
):
    adversaries = []

    for attack in cfg:
        loss_fn = get_loss_fn(attack.loss_fn)

        if isinstance(attack, FGSMAttackConfig):
            adversaries.append(
                FGSMAttack(
                    attack.epsilon,
                    loss_fn,
                )
            )
        elif isinstance(attack, PGDAttackConfig):
            adversaries.append(
                PGD(
                    attack.epsilon,
                    attack.alpha,
                    attack.steps,
                    loss_fn,
                    attack.norm,
                )
            )
        elif isinstance(attack, StAdvAttackConfig):
            adversaries.append(
                StAdv(
                    attack.steps,
                    attack.alpha,
                    attack.tau,
                    loss_fn,
                    attack.targeted,
                )
            )
        else:
            raise NotImplementedError("undefined attack type")
    return adversaries

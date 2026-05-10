from typing import List

from src.config._parsers import AttackConfig, FGSMAttackConfig, PGDAttackConfig
from src.robustness.adversaries.fgsm import FGSMAttack
from src.robustness.adversaries.pgd import PGD


def get_adversaries(
        cfg: List[AttackConfig],
):
    adversaries = []

    for attack in cfg:

        if isinstance(attack, FGSMAttackConfig):
            adversaries.append(
                FGSMAttack(
                    eps=attack.epsilon,
                    loss_fn=attack.loss_fn,
                )
            )
        elif isinstance(attack, PGDAttackConfig):
            adversaries.append(
                PGD(
                    epsilon=attack.epsilon,
                    alpha=attack.alpha,
                    steps=attack.steps,
                    lossfn=attack.loss_fn,
                    norm=attack.norm,
                )
            )
        else:
            raise NotImplementedError("undefined attack type")
    return adversaries

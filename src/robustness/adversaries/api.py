from typing import List

from src.config.common import (
    AttackConfig,
    FGSMAttackConfig,
    PGDAttackConfig,
    SmoothedAttackConfig,
    get_scheduled,
)
from src.robustness.adversaries.fgsm import FGSMAttack
from src.robustness.adversaries.pgd import PGD, SmoothPGD


def get_adversary(
        cfg: AttackConfig,
        epoch: int | None = None,
        sigma: float | None = None,
        num_noise_vec: int | None = None,
        clamp_noisy: bool | None = None,
):
    if isinstance(cfg, FGSMAttackConfig):
        return FGSMAttack(
            eps=cfg.epsilon,
            loss_fn=cfg.loss_fn,
        )

    if isinstance(cfg, PGDAttackConfig):
        epsilon = get_scheduled(cfg.epsilon, cfg.epsilon_scheduler, epoch) if epoch is not None else cfg.epsilon
        alpha = get_scheduled(cfg.alpha, cfg.alpha_scheduler, epoch) if epoch is not None else cfg.alpha

        return PGD(
            epsilon=epsilon,
            alpha=alpha,
            steps=cfg.steps,
            lossfn=cfg.loss_fn,
            norm=cfg.norm,
            random_start=cfg.random_start,
        )

    if isinstance(cfg, SmoothedAttackConfig):
        epsilon = get_scheduled(cfg.epsilon, cfg.epsilon_scheduler, epoch) if epoch is not None else cfg.epsilon
        alpha = get_scheduled(cfg.alpha, cfg.alpha_scheduler, epoch) if epoch is not None else cfg.alpha

        if sigma is None:
            raise ValueError("sigma is required to build SmoothPGD from SmoothedAttackConfig")
        if num_noise_vec is None:
            raise ValueError("num_noise_vec is required to build SmoothPGD from SmoothedAttackConfig")

        return SmoothPGD(
            epsilon=epsilon,
            alpha=alpha,
            steps=cfg.steps,
            sigma=sigma,
            num_noise_vec=num_noise_vec,
            norm=cfg.norm,
            random_start=cfg.random_start,
            clamp_noisy=cfg.clamp_noisy if clamp_noisy is None else clamp_noisy,
        )

    raise NotImplementedError(f"Unsupported attack config type: {type(cfg)}")


def get_adversaries(
        cfg: List[AttackConfig],
        epoch: int | None = None,
        sigma: float | None = None,
        num_noise_vec: int | None = None,
        clamp_noisy: bool | None = None,
):
    return [
        get_adversary(
            attack,
            epoch=epoch,
            sigma=sigma,
            num_noise_vec=num_noise_vec,
            clamp_noisy=clamp_noisy,
        )
        for attack in cfg
    ]

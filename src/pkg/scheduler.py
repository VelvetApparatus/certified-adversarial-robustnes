import torch
from src.config.conf import ExperimentConfig


def get_scheduler(
        optimizer,
        cfg: ExperimentConfig,
):
    scheduler_cfg = cfg.training.scheduler

    if scheduler_cfg.name == "none":
        return None

    if scheduler_cfg.name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.step_size,
            gamma=scheduler_cfg.gamma,
        )

    if scheduler_cfg.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs,
            eta_min=scheduler_cfg.eta_min,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_cfg.name}")

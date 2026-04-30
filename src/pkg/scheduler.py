import torch
from src.config.common import SchedulerConfig


def get_scheduler(
        optimizer,
        cfg: SchedulerConfig,
):
    scheduler_cfg = cfg

    if scheduler_cfg.name == "none":
        return None

    if scheduler_cfg.name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.step_size,
            gamma=scheduler_cfg.gamma,
        )

    if scheduler_cfg.name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_cfg.milestones,
            gamma=scheduler_cfg.gamma,
        )
    if scheduler_cfg.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs,
            eta_min=scheduler_cfg.eta_min,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_cfg.name}")

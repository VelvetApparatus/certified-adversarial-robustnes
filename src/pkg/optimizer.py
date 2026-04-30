import torch
from src.config._parsers import AdversarialTrainingConfig


def get_optimizer(
        model,
        cfg: AdversarialTrainingConfig,
):
    opt_cfg = cfg.training.optimizer

    if opt_cfg.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg.lr,
            momentum=opt_cfg.momentum,
            weight_decay=opt_cfg.weight_decay,
            nesterov=opt_cfg.nesterov,
        )

    if opt_cfg.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )

    if opt_cfg.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")
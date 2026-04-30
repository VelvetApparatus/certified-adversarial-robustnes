from dataclasses import dataclass, field, replace
from typing import Optional, Literal, List, Union

import torch
import yaml
from torchvision.transforms import Compose

from src.config.certify import CertificationConfig, CertificationParams
from src.config.common import AttackConfig, FGSMAttackConfig, PGDAttackConfig, StAdvAttackConfig, DatasetConfig, \
    OptimizerConfig, WandbConfig, TrainingConfig, SchedulerConfig, ModelConfig
from src.config.macer import MacerParams


def _parse_attack(cfg: dict) -> AttackConfig:
    attack_name = cfg.get("name")
    if attack_name is None:
        raise ValueError("Each attack must contain field 'name'")

    if attack_name == "fgsm":
        return FGSMAttackConfig(
            name="fgsm",
            epsilon=cfg["epsilon"],
            loss_fn=cfg.get("loss_fn", "cross_entropy"),
            mean=cfg.get("mean", None),
            std=cfg.get("std", None),
        )

    if attack_name == "pgd":
        return PGDAttackConfig(
            name="pgd",
            epsilon=cfg["epsilon"],
            alpha=cfg["alpha"],
            steps=cfg["steps"],
            norm=cfg.get("norm", "Linf"),
            loss_fn=cfg.get("loss_fn", "cross_entropy"),
            mean=cfg.get("mean", None),
            std=cfg.get("std", None),
        )

    if attack_name == "stadv":
        return StAdvAttackConfig(
            name="stadv",
            alpha=cfg["alpha"],
            steps=cfg["steps"],
            tau=cfg["tau"],
            targeted=cfg.get("targeted", False),
            loss_fn=cfg.get("loss_fn", "cross_entropy"),
        )

    raise ValueError(f"Unsupported attack name: {attack_name}")


def _parse_dataset(cfg: dict, *, default_train: Optional[bool] = None) -> DatasetConfig:
    if cfg is None:
        raise ValueError("Dataset config must not be empty")

    train = cfg.get("train", default_train if default_train is not None else False)

    return DatasetConfig(
        name=cfg["name"],
        root_dir=cfg.get("root_dir", "./data"),
        train=train,
        download=cfg.get("download", True),
        batch_size=cfg.get("batch_size", 128),
        num_workers=cfg.get("num_workers", 0),
    )


def _parse_optimizer(cfg: Optional[dict]) -> OptimizerConfig:
    cfg = cfg or {}

    return OptimizerConfig(
        name=cfg.get("name", "sgd"),
        lr=cfg.get("lr", 0.1),
        weight_decay=cfg.get("weight_decay", 5e-4),
        momentum=cfg.get("momentum", 0.9),
        nesterov=cfg.get("nesterov", False),
    )


def _parse_scheduler(cfg: Optional[dict]) -> SchedulerConfig:
    cfg = cfg or {}

    return SchedulerConfig(
        name=cfg.get("name", "none"),
        step_size=cfg.get("step_size", 30),
        gamma=cfg.get("gamma", 0.1),
        eta_min=cfg.get("eta_min", 0.0),
    )


def _parse_wandb(cfg: Optional[dict]) -> WandbConfig:
    cfg = cfg or {}

    return WandbConfig(
        enabled=cfg.get("enabled", False),
        project=cfg.get("project", "certified-robustness"),
        entity=cfg.get("entity", None),
        run_name=cfg.get("run_name", None),
        tags=cfg.get("tags", []),
    )


def _parse_training(cfg: Optional[dict]) -> TrainingConfig:
    cfg = cfg or {}

    return TrainingConfig(
        enabled=cfg.get("enabled", True),
        epochs=cfg.get("epochs", 100),
        seed=cfg.get("seed", 42),

        optimizer=_parse_optimizer(cfg.get("optimizer")),
        scheduler=_parse_scheduler(cfg.get("scheduler")),
        wandb=_parse_wandb(cfg.get("wandb")),

        save_dir=cfg.get("save_dir", "./checkpoints"),
        save_best=cfg.get("save_best", True),
        save_last=cfg.get("save_last", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "test_accuracy"),
    )


def _parse_model(cfg: Optional[dict]) -> ModelConfig:
    cfg = cfg or {}
    return ModelConfig(
        name=cfg.get("name", None),
        pretrained=cfg.get("pretrained", True),
        weights_path=cfg.get("weights_path", None),
        loss_fn=cfg.get("loss_fn", "cross_entropy"),
    )


def _parse_macer_params(cfg: Optional[dict]) -> MacerParams:
    cfg = cfg or {}
    return MacerParams(
        output_dir=cfg.get("output_dir", "./checkpoints"),
        seed=cfg.get("seed", 42),
        gauss_samples=cfg.get("gauss_samples", 100),
        sigma=cfg.get("sigma", 0.01),
        beta=cfg.get("beta", 0.01),
        num_classes=cfg.get("num_classes", 100),
        gamma=cfg.get("gamma", 0.1),
        lbd=cfg.get("lbd", 0.01),
        epochs=cfg.get("epochs", 100),
        certificate_every_epoch=cfg.get("certificate_every_epoch", False),
        certificate_epoch_threshold=cfg.get("certificate_epoch_threshold", 200),
        checkpoint=cfg.get("checkpoint", None),
        cert_start=cfg.get("cert_start", 0),
        cert_num=cfg.get("cert_num", 100),
    )


def _parse_certification_params(cfg: dict) -> CertificationParams:
    return CertificationParams(
        sigma=cfg["sigma"],
        output_dir=cfg["output_dir"],
        n0=cfg["n0"],
        n=cfg["n"],
        alpha=cfg["alpha"],
    )

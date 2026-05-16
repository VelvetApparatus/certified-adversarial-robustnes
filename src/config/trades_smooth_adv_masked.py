from dataclasses import dataclass

import yaml

from src.config._parsers import (
    _parse_dataset,
    _parse_dataset_split,
    _parse_input_mask_params,
    _parse_model,
    _parse_normalization,
    _parse_pgd,
    _parse_smoothed_attack,
    _parse_trades_smooth_adv_params,
    _parse_training,
)
from src.config.common import (
    DatasetConfig,
    DatasetSplitConfig,
    InputMaskParams,
    ModelConfig,
    NormalizeConfig,
    PGDAttackConfig,
    SmoothedAttackConfig,
    TradesSmoothAdvParams,
    TrainingConfig,
)


@dataclass
class TradesSmoothAdvMaskedExperimentConfig:
    model: ModelConfig
    dataset: DatasetConfig
    split: DatasetSplitConfig
    training: TrainingConfig
    params: TradesSmoothAdvParams
    trainPGD: PGDAttackConfig
    evalPGD: PGDAttackConfig
    attack: SmoothedAttackConfig
    input_mask: InputMaskParams
    normalization: NormalizeConfig


def load_trades_smooth_adv_masked_config(
        path: str,
) -> TradesSmoothAdvMaskedExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    required = [
        "model",
        "dataset",
        "split",
        "training",
        "params",
        "attack",
        "train_pgd",
        "eval_pgd",
        "input_mask",
        "normalization",
    ]
    for key in required:
        if key not in raw:
            raise ValueError(f"Config must contain '{key}'")

    train_pgd = _parse_pgd(raw["train_pgd"])
    eval_pgd = _parse_pgd(raw["eval_pgd"])
    attack = _parse_smoothed_attack(raw["attack"])
    input_mask = _parse_input_mask_params(raw["input_mask"])

    if train_pgd.loss_fn != "kl_divergence":
        raise ValueError(
            "train_pgd.loss_fn must be 'kl_divergence' for trades_smooth_adv_masked"
        )
    if attack.name != "smooth_pgd":
        raise ValueError("attack.name must be 'smooth_pgd' for trades_smooth_adv_masked")
    if input_mask.p < 0.0 or input_mask.ratio < 0.0:
        raise ValueError("input_mask.p and input_mask.ratio must be non-negative")
    if input_mask.p > 1.0 or input_mask.ratio > 1.0:
        raise ValueError("input_mask.p and input_mask.ratio must be <= 1.0")

    return TradesSmoothAdvMaskedExperimentConfig(
        model=_parse_model(raw["model"]),
        dataset=_parse_dataset(raw["dataset"]),
        split=_parse_dataset_split(raw["split"]),
        training=_parse_training(raw["training"]),
        params=_parse_trades_smooth_adv_params(raw["params"]),
        trainPGD=train_pgd,
        evalPGD=eval_pgd,
        attack=attack,
        input_mask=input_mask,
        normalization=_parse_normalization(raw["normalization"]),
    )

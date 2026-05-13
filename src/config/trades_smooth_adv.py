from dataclasses import dataclass

import yaml

from src.config._parsers import (
    _parse_dataset,
    _parse_dataset_split,
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
    ModelConfig,
    NormalizeConfig,
    PGDAttackConfig,
    SmoothedAttackConfig,
    TradesSmoothAdvParams,
    TrainingConfig,
)


@dataclass
class TradesSmoothAdvConfig:
    model: ModelConfig
    dataset: DatasetConfig
    split: DatasetSplitConfig
    training: TrainingConfig
    params: TradesSmoothAdvParams
    trainPGD: PGDAttackConfig
    evalPGD: PGDAttackConfig
    attack: SmoothedAttackConfig
    normalization: NormalizeConfig


def load_trades_smooth_adv_config(path: str) -> TradesSmoothAdvConfig:
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
        "train_pgd",
        "eval_pgd",
        "attack",
        "normalization",
    ]
    for key in required:
        if key not in raw:
            raise ValueError(f"Config must contain '{key}'")

    train_pgd = _parse_pgd(raw["train_pgd"])
    if train_pgd.loss_fn != "kl_divergence":
        raise ValueError("train_pgd.loss_fn must be 'kl_divergence' for trades_smooth_adv")

    return TradesSmoothAdvConfig(
        model=_parse_model(raw["model"]),
        dataset=_parse_dataset(raw["dataset"]),
        split=_parse_dataset_split(raw["split"]),
        training=_parse_training(raw["training"]),
        params=_parse_trades_smooth_adv_params(raw["params"]),
        trainPGD=train_pgd,
        evalPGD=_parse_pgd(raw["eval_pgd"]),
        attack=_parse_smoothed_attack(raw["attack"]),
        normalization=_parse_normalization(raw["normalization"]),
    )

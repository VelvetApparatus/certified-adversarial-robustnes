from dataclasses import dataclass

import yaml

from src.config._parsers import (
    _parse_awp_params,
    _parse_dataset,
    _parse_dataset_split,
    _parse_input_mask_params,
    _parse_model,
    _parse_normalization,
    _parse_pgd,
    _parse_trades_masked_params,
    _parse_training,
)
from src.config.common import (
    AWPParams,
    DatasetConfig,
    DatasetSplitConfig,
    InputMaskParams,
    ModelConfig,
    NormalizeConfig,
    PGDAttackConfig,
    TradesMaskedParams,
    TrainingConfig,
)


@dataclass
class TradesAWPMaskedExperimentConfig:
    model: ModelConfig
    dataset: DatasetConfig
    split: DatasetSplitConfig
    training: TrainingConfig
    trades: TradesMaskedParams
    trainPGD: PGDAttackConfig
    evalPGD: PGDAttackConfig
    awp: AWPParams
    input_mask: InputMaskParams
    normalization: NormalizeConfig


def load_trades_awp_masked_config(path: str) -> TradesAWPMaskedExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    required = [
        "model",
        "dataset",
        "split",
        "training",
        "trades",
        "train_pgd",
        "eval_pgd",
        "awp",
        "input_mask",
        "normalization",
    ]
    for key in required:
        if key not in raw:
            raise ValueError(f"Config must contain '{key}'")

    train_pgd = _parse_pgd(raw["train_pgd"])
    eval_pgd = _parse_pgd(raw["eval_pgd"])

    if train_pgd.loss_fn != "kl_divergence":
        raise ValueError("train_pgd.loss_fn must be 'kl_divergence' for TRADES-AWP-Masked")

    return TradesAWPMaskedExperimentConfig(
        model=_parse_model(raw["model"]),
        dataset=_parse_dataset(raw["dataset"]),
        split=_parse_dataset_split(raw["split"]),
        training=_parse_training(raw["training"]),
        trades=_parse_trades_masked_params(raw["trades"]),
        trainPGD=train_pgd,
        evalPGD=eval_pgd,
        awp=_parse_awp_params(raw["awp"]),
        input_mask=_parse_input_mask_params(raw["input_mask"]),
        normalization=_parse_normalization(raw["normalization"]),
    )

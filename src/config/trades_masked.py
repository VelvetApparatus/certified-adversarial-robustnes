from dataclasses import dataclass

import yaml

from src.config.common import (
    TrainingConfig, DatasetConfig,
    DatasetSplitConfig, PGDAttackConfig, ModelConfig,
    NormalizeConfig, InputMaskParams, TradesMaskedParams
)
from src.config._parsers import (
    _parse_dataset, _parse_dataset_split,
    _parse_training, _parse_model, _parse_normalization,
    _parse_input_mask_params, _parse_pgd, _parse_trades_masked_params
)


@dataclass
class TradesMaskedTrainingConfig:
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig
    split: DatasetSplitConfig
    trades: TradesMaskedParams
    input_mask: InputMaskParams
    evalPGD: PGDAttackConfig
    trainPGD: PGDAttackConfig
    normalization: NormalizeConfig


def load_trades_masked_config(path: str) -> TradesMaskedTrainingConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    required = [
        "model",
        "training",
        "dataset",
        "split",
        "trades",
        "input_mask",
        "normalization",
        "eval_pgd",
        "train_pgd",

    ]

    for key in required:
        if key not in raw:
            raise ValueError(f"Config must contain '{key}'")

    return TradesMaskedTrainingConfig(
        model=_parse_model(raw["model"]),
        training=_parse_training(raw["training"]),
        dataset=_parse_dataset(raw["dataset"]),
        split=_parse_dataset_split(raw["split"]),
        trades=_parse_trades_masked_params(raw["trades"]),
        input_mask=_parse_input_mask_params(raw["input_mask"]),
        normalization=_parse_normalization(raw["normalization"]),
        evalPGD=_parse_pgd(raw["eval_pgd"]),
        trainPGD=_parse_pgd(raw["train_pgd"]),
    )

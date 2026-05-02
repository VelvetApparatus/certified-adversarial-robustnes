from dataclasses import dataclass

import yaml

from src.config._parsers import (
    _parse_dataset, _parse_training, _parse_model, _parse_trades_params,
    _parse_dataset_split
)
from src.config.common import ModelConfig, DatasetConfig, TradesParams, TrainingConfig, DatasetSplitConfig


@dataclass
class TradesConfig:
    training: TrainingConfig
    params: TradesParams
    model: ModelConfig
    dataset: DatasetConfig
    split: DatasetSplitConfig


def load_trades_config(path: str) -> TradesConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    if "model" not in raw:
        raise ValueError("Config must contain 'model'")
    if "dataset" not in raw:
        raise ValueError("Config must contain 'dataset'")
    if "training" not in raw:
        raise ValueError("Config must contain 'training'")
    if "split" not in raw:
        raise ValueError("Config must contain 'split'")

    return TradesConfig(
        training=_parse_training(raw["training"]),
        params=_parse_trades_params(raw["trades_params"]),
        model=_parse_model(raw["model"]),
        dataset=_parse_dataset(raw["dataset"]),
        split=_parse_dataset_split(raw["split"]),
    )

from dataclasses import dataclass

import yaml

from src.config._parsers import (
    _parse_trades_params, _parse_model, _parse_dataset,
    _parse_optimizer, _parse_scheduler
)
from src.config.common import ModelConfig, DatasetConfig, SchedulerConfig, OptimizerConfig, TradesParams


@dataclass
class TradesConfig:
    params: TradesParams
    model: ModelConfig
    train_dataset: DatasetConfig
    test_dataset: DatasetConfig
    scheduler: SchedulerConfig
    optimizer: OptimizerConfig


def load_trades_config(path: str) -> TradesConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    if "model" not in raw:
        raise ValueError("Config must contain 'model'")
    if "test_dataset" not in raw:
        raise ValueError("Config must contain 'test_dataset'")

    return TradesConfig(
        params=_parse_trades_params(raw["trades_params"]),
        model=_parse_model(raw["model"]),
        train_dataset=_parse_dataset(raw["train_dataset"]),
        test_dataset=_parse_dataset(raw["test_dataset"]),
        scheduler=_parse_scheduler(raw["scheduler"]),
        optimizer=_parse_optimizer(raw["optimizer"]),
    )

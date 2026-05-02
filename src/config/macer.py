from dataclasses import dataclass, field

import yaml

from src.config._parsers import _parse_model, _parse_dataset, _parse_dataset_split, _parse_training, _parse_macer_params
from src.config.common import (
    ModelConfig,
    DatasetConfig,
    DatasetSplitConfig,
    TrainingConfig, MacerTrainingParams,
)


@dataclass
class MacerTrainingConfig:
    model: ModelConfig
    dataset: DatasetConfig
    split: DatasetSplitConfig
    training: TrainingConfig
    params: MacerTrainingParams


def load_macer_training_config(path: str) -> MacerTrainingConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return MacerTrainingConfig(

        model=_parse_model(raw["model"]),
        dataset=_parse_dataset(raw["dataset"], default_train=True),
        split=_parse_dataset_split(raw.get("split")),
        training=_parse_training(raw["training"]),
        params=_parse_macer_params(raw.get("params")),
    )

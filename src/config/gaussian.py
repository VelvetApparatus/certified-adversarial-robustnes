from dataclasses import dataclass

import yaml

from src.config.common import TrainingConfig, ModelConfig, DatasetConfig, DatasetSplitConfig, GaussianTrainingParams
from src.config._parsers import _parse_training, _parse_model, _parse_dataset, _parse_dataset_split, \
    _parse_gaussian_params


@dataclass
class GaussianConfig:
    params: GaussianTrainingParams
    train: TrainingConfig
    model: ModelConfig
    dataset: DatasetConfig
    split: DatasetSplitConfig


def load_gaussian_train_config(path: str) -> GaussianConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    if "train" not in raw:
        raise ValueError("Config must contain 'train'")
    if "model" not in raw:
        raise ValueError("Config must contain 'model'")
    if "dataset" not in raw:
        raise ValueError("Config must contain 'dataset'")
    if "split" not in raw:
        raise ValueError("Config must contain 'split'")

    return GaussianConfig(
        params=_parse_gaussian_params(raw["params"]),
        train=_parse_training(raw["train"]),
        model=_parse_model(raw["model"]),
        dataset=_parse_dataset(raw["dataset"]),
        split=_parse_dataset_split(raw["split"])
    )

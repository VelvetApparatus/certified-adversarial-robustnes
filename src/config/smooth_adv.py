from dataclasses import dataclass

import yaml

from src.config._parsers import (
    _parse_model, _parse_dataset, _parse_dataset_split,
    _parse_training, _parse_normalization, _parse_smooth_adv_params,
    _parse_smoothed_attack,
)

from src.config.common import ModelConfig, TrainingConfig, DatasetConfig, DatasetSplitConfig, SmoothAdvTrainingParams, \
    NormalizeConfig, SmoothedAttackConfig


@dataclass
class SmoothAdvTrainConfig:
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig
    split: DatasetSplitConfig
    params: SmoothAdvTrainingParams
    attack: SmoothedAttackConfig
    normalization: NormalizeConfig


def load_smooth_adv_train_config(path: str) -> SmoothAdvTrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    if "model" not in raw:
        raise ValueError("Config must contain 'model'")
    if "dataset" not in raw and "datasets" not in raw:
        raise ValueError("Config must contain 'dataset' or 'datasets'")
    if "split" not in raw:
        raise ValueError("Config must contain 'split'")
    if "training" not in raw:
        raise ValueError("Config must contain 'training'")
    if "normalization" not in raw:
        raise ValueError("Config must contain 'normalization'")

    return SmoothAdvTrainConfig(
        model=_parse_model(raw["model"]),
        training=_parse_training(raw["training"]),
        dataset=_parse_dataset(raw["dataset"]),
        split=_parse_dataset_split(raw["split"]),
        normalization=_parse_normalization(raw["normalization"]),
        params=_parse_smooth_adv_params(raw["params"]),
        attack=_parse_smoothed_attack(raw.get("attack", {
            "name": "smooth_pgd",
            "epsilon": raw["params"].get("epsilon", 0.25),
            "alpha": raw["params"].get("step_size", 0.025),
            "steps": raw["params"].get("steps", 10),
            "norm": raw["params"].get("norm", "l2"),
            "clamp_noisy": raw["params"].get("clamp_noisy", True),
        })),
    )

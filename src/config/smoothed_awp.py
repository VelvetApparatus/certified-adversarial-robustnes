from dataclasses import dataclass

import yaml

from src.config._parsers import (
    _parse_model,
    _parse_dataset,
    _parse_dataset_split,
    _parse_training,
    _parse_normalization,
    _parse_smooth_adv_params,
    _parse_awp_params,
    _parse_smoothed_attack,
)

from src.config.common import (
    ModelConfig,
    TrainingConfig,
    DatasetConfig,
    DatasetSplitConfig,
    SmoothAdvTrainingParams,
    NormalizeConfig,
    AWPParams,
    SmoothedAttackConfig,
)


@dataclass
class SmoothedAWPTrainConfig:
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig
    split: DatasetSplitConfig
    params: SmoothAdvTrainingParams
    attack: SmoothedAttackConfig
    awp: AWPParams
    normalization: NormalizeConfig
    eval_smoothed: dict


def load_smoothed_awp_config(path: str) -> SmoothedAWPTrainConfig:
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
        "awp",
        "normalization",
    ]

    for key in required:
        if key not in raw:
            raise ValueError(f"Config must contain '{key}'")

    return SmoothedAWPTrainConfig(
        model=_parse_model(raw["model"]),
        training=_parse_training(raw["training"]),
        dataset=_parse_dataset(raw["dataset"]),
        split=_parse_dataset_split(raw["split"]),
        normalization=_parse_normalization(raw["normalization"]),
        params=_parse_smooth_adv_params(raw["params"]),
        attack=_parse_smoothed_attack(raw["attack"]),
        awp=_parse_awp_params(raw["awp"]),
        eval_smoothed=raw.get("eval_smoothed", {}),
    )

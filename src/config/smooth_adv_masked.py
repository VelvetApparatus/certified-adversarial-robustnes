from dataclasses import dataclass

import yaml

from src.config.common import (
    TrainingConfig, SmoothMaskedTrainingParams, DatasetConfig,
    DatasetSplitConfig, ModelConfig, NormalizeConfig,
    SmoothedAttackConfig, InputMaskParams,

)
from src.config._parsers import (
    _parse_dataset, _parse_dataset_split, _parse_input_mask_params,
    _parse_training, _parse_model, _parse_normalization,
    _parse_smooth_adv_masked_params, _parse_smoothed_attack
)


@dataclass
class SmoothAdvMaskedConfig:
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig
    split: DatasetSplitConfig
    params: SmoothMaskedTrainingParams
    attack: SmoothedAttackConfig
    input_mask: InputMaskParams
    normalization: NormalizeConfig
    eval_smoothed: dict


def load_smoothed_adv_masked_config(path: str) -> SmoothAdvMaskedConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    required = [
        "model",
        "training",
        "dataset",
        "split",
        "params",
        "attack",
        "input_mask",
        "normalization",
    ]

    for key in required:
        if key not in raw:
            raise ValueError(f"Config must contain '{key}'")

    return SmoothAdvMaskedConfig(
        model=_parse_model(raw["model"]),
        training=_parse_training(raw["training"]),
        dataset=_parse_dataset(raw["dataset"]),
        split=_parse_dataset_split(raw["split"]),
        params=_parse_smooth_adv_masked_params(raw["params"]),
        attack=_parse_smoothed_attack(raw["attack"]),
        input_mask=_parse_input_mask_params(raw["input_mask"]),
        normalization=_parse_normalization(raw["normalization"]),
        eval_smoothed=raw.get("eval_smoothed", {}),
    )

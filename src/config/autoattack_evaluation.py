import os.path
from dataclasses import dataclass, field

import yaml

from src.config._parsers import (
    _parse_autoattack_config,
    _parse_autoattack_evaluation_params,
    _parse_dataset,
    _parse_model,
    _parse_normalization,
    _parse_wandb,
)
from src.config.common import (
    AutoAttackConfig,
    AutoAttackEvaluationParams,
    DatasetConfig,
    ModelConfig,
    NormalizeConfig,
    WandbConfig,
)


@dataclass
class AutoAttackEvaluationExperimentConfig:
    model: ModelConfig
    test_dataset: DatasetConfig
    params: AutoAttackEvaluationParams
    normalization: NormalizeConfig
    autoattack: AutoAttackConfig
    wandb: WandbConfig = field(default_factory=WandbConfig)


def load_autoattack_evaluate_config(path: str) -> AutoAttackEvaluationExperimentConfig:
    if not os.path.exists(path):
        raise ValueError(f"Config path does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    required_sections = ("model", "test_dataset", "params", "normalization", "autoattack")
    for section in required_sections:
        if section not in raw:
            raise ValueError(f"Config must contain '{section}'")

    return AutoAttackEvaluationExperimentConfig(
        model=_parse_model(raw["model"]),
        test_dataset=_parse_dataset(raw["test_dataset"], default_train=False),
        params=_parse_autoattack_evaluation_params(raw["params"]),
        normalization=_parse_normalization(raw["normalization"]),
        autoattack=_parse_autoattack_config(raw["autoattack"]),
        wandb=_parse_wandb(raw.get("wandb")),
    )

from dataclasses import dataclass
from typing import List

import yaml

from src.config._parsers import _parse_dataset, _parse_attack
from src.config.common import ModelConfig, DatasetConfig, AttackConfig


@dataclass
class EvaluationExperimentConfig:
    model: ModelConfig
    test_dataset: DatasetConfig
    attacks: List[AttackConfig]
    evaluation_root: str


def load_evaluate_config(path: str) -> EvaluationExperimentConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    if "model" not in raw:
        raise ValueError("Config must contain 'model'")
    if "test_dataset" not in raw:
        raise ValueError("Config must contain 'test_dataset'")

    model_cfg = ModelConfig(
        name=raw["model"]["name"],
        weights_path=raw["model"].get("weights_path"),
    )

    dataset_cfg = _parse_dataset(raw["test_dataset"])

    attacks_raw = raw.get("attacks", [])
    attacks_cfg = [_parse_attack(a) for a in attacks_raw]

    evaluation_root = raw.get("evaluation_root")

    return EvaluationExperimentConfig(
        model=model_cfg,
        test_dataset=dataset_cfg,
        attacks=attacks_cfg,
        evaluation_root=evaluation_root
    )

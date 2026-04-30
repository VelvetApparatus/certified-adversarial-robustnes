from dataclasses import dataclass, field, replace
from typing import Optional, List

import yaml

from src.config._parsers import _parse_dataset, _parse_attack, _parse_training
from src.config.common import ModelConfig, DatasetConfig, DatasetSplitsConfig, TrainingConfig, AttackConfig


@dataclass
class AdversarialTrainingConfig:
    model: ModelConfig
    dataset: Optional[DatasetConfig] = None
    datasets: Optional[DatasetSplitsConfig] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation_root: Optional[str] = None
    attacks: List[AttackConfig] = field(default_factory=list)

    @property
    def train_dataset(self) -> DatasetConfig:
        if self.datasets and self.datasets.train is not None:
            return self.datasets.train
        if self.dataset is not None:
            return replace(self.dataset, train=True)
        raise ValueError("Train dataset config is not provided")

    @property
    def test_dataset(self) -> DatasetConfig:
        if self.datasets and self.datasets.test is not None:
            return self.datasets.test
        if self.dataset is not None:
            return replace(self.dataset, train=False)
        raise ValueError("Test dataset config is not provided")


def load_adversarial_training_config(path: str) -> AdversarialTrainingConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    if "model" not in raw:
        raise ValueError("Config must contain 'model'")
    if "dataset" not in raw and "datasets" not in raw:
        raise ValueError("Config must contain 'dataset' or 'datasets'")

    model_cfg = ModelConfig(
        name=raw["model"]["name"],
        weights_path=raw["model"].get("weights_path"),
        pretrained=raw["model"].get("pretrained"),
    )

    dataset_cfg = None
    if "dataset" in raw:
        dataset_cfg = _parse_dataset(raw["dataset"])

    datasets_cfg = None
    if "datasets" in raw:
        datasets_raw = raw["datasets"] or {}
        train_cfg = datasets_raw.get("train")
        test_cfg = datasets_raw.get("test")

        if train_cfg is None and test_cfg is None:
            raise ValueError("Config field 'datasets' must contain 'train' or 'test'")

        datasets_cfg = DatasetSplitsConfig(
            train=_parse_dataset(train_cfg, default_train=True) if train_cfg is not None else None,
            test=_parse_dataset(test_cfg, default_train=False) if test_cfg is not None else None,
        )

    attacks_raw = raw.get("attacks", [])
    attacks_cfg = [_parse_attack(a) for a in attacks_raw]

    training_cfg = _parse_training(raw.get("training"))
    evaluation_root = raw.get("evaluation_root")

    return AdversarialTrainingConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        datasets=datasets_cfg,
        attacks=attacks_cfg,
        training=training_cfg,
        evaluation_root=evaluation_root,
    )

from dataclasses import dataclass, field, replace
from typing import Optional, List

import yaml
from mpmath.libmp import normalize

from src.config._parsers import (_parse_dataset, _parse_dataset_split, _parse_training, _parse_model, _parse_fgsm,
                                 _parse_pgd, _parse_normalization
                                 )
from src.config.common import (
    ModelConfig, DatasetConfig, TrainingConfig, PGDAttackConfig, FGSMAttackConfig,
    DatasetSplitConfig, NormalizeConfig
)


@dataclass
class AdversarialTrainingConfig:
    model: ModelConfig
    dataset: Optional[DatasetConfig] = None
    split: Optional[DatasetSplitConfig] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # PGD attack configuration used to estimate empirical adversarial robustness.
    pgd: PGDAttackConfig = field(default_factory=PGDAttackConfig)

    # FGSM attack configuration used as a fast single-step adversarial baseline.
    fgsm: FGSMAttackConfig = field(default_factory=FGSMAttackConfig)

    normalization: NormalizeConfig = field(default_factory=NormalizeConfig)


def load_adversarial_training_config(path: str) -> AdversarialTrainingConfig:
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
    if "pgd" not in raw:
        raise ValueError("Config must contain 'pgd'")
    if "fgsm" not in raw:
        raise ValueError("Config must contain 'fgsm'")
    if "normalize" not in raw:
        raise ValueError("Config must contain 'normalization'")

    return AdversarialTrainingConfig(
        model=_parse_model(raw["model"]),
        dataset=_parse_dataset(raw["dataset"]),
        split=_parse_dataset_split(raw["split"]),
        training=_parse_training(raw["training"]),
        pgd=_parse_pgd(raw["pgd"]),
        fgsm=_parse_fgsm(raw["fgsm"]),
        normalization=_parse_normalization(raw["normalization"])
    )

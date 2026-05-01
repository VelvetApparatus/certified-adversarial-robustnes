import os.path
from dataclasses import dataclass
import yaml

from src.config._parsers import _parse_dataset, _parse_fgsm, _parse_pgd, _parse_evaluation_table_params, _parse_model
from src.config.common import ModelConfig, DatasetConfig, PGDAttackConfig, FGSMAttackConfig, EvaluationTableParams


@dataclass
class EvaluationExperimentConfig:
    # model for evaluation on clean/noisy/adversarial examples
    model: ModelConfig

    # test dataset
    test_dataset: DatasetConfig

    # PGD attack configuration used to estimate empirical adversarial robustness.
    pgd: PGDAttackConfig

    # FGSM attack configuration used as a fast single-step adversarial baseline.
    fgsm: FGSMAttackConfig

    # parameters
    params: EvaluationTableParams


def load_evaluate_config(path: str) -> EvaluationExperimentConfig:
    if not os.path.exists(path):
        raise ValueError("Config must contain 'path'")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    if "model" not in raw:
        raise ValueError("Config must contain 'model'")

    if "test_dataset" not in raw:
        raise ValueError("Config must contain 'test_dataset'")

    if "pgd" not in raw:
        raise ValueError("Config must contain 'pgd'")

    if "fgsm" not in raw:
        raise ValueError("Config must contain 'fgsm'")

    if "params" not in raw:
        raise ValueError("Config must contain 'params'")

    return EvaluationExperimentConfig(
        model=_parse_model(raw["model"]),
        test_dataset=_parse_dataset(raw["test_dataset"], default_train=False),
        pgd=_parse_pgd(raw["pgd"]),
        fgsm=_parse_fgsm(raw["fgsm"]),
        params=_parse_evaluation_table_params(raw["params"]),
    )

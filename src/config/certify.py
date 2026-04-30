from dataclasses import dataclass

import yaml

from src.config._parsers import _parse_dataset, _parse_certification_params
from src.config.common import ModelConfig, DatasetConfig


@dataclass
class CertificationParams:
    sigma: float
    output_dir: str
    n0: int = 100
    n: int = 100000
    alpha: float = 0.001
    seed: int = 42


@dataclass
class CertificationConfig:
    model: ModelConfig
    dataset: DatasetConfig
    certification: CertificationParams


def load_certification_config(path: str) -> CertificationConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    if "certification" not in raw:
        raise ValueError("Config must contain 'certification'")
    if "model" not in raw:
        raise ValueError("Config must contain 'model'")
    if "dataset" not in raw:
        raise ValueError("Config must contain 'dataset'")

    model_cfg = ModelConfig(
        name=raw["model"]["name"],
        weights_path=raw["model"].get("weights_path"),
        pretrained=raw["model"].get("pretrained"),
    )
    dataset_cfg = _parse_dataset(raw["dataset"])
    certification_params = _parse_certification_params(raw["certification"])
    return CertificationConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        certification=certification_params,
    )

from dataclasses import dataclass

import yaml

from src.config._parsers import _parse_model, _parse_dataset, _parse_optimizer, _parse_scheduler, _parse_macer_params
from src.config.common import DatasetConfig, SchedulerConfig, OptimizerConfig, ModelConfig


@dataclass
class MacerParams:
    output_dir: str
    seed: int
    gauss_samples: int
    sigma: float
    beta: float
    num_classes: int
    gamma: float
    lbd: float
    epochs: int
    certificate_every_epoch: int
    certificate_epoch_threshold: int
    checkpoint: str
    cert_start: int
    cert_num: int



@dataclass
class MacerConfig:
    params: MacerParams
    model: ModelConfig
    train_dataset: DatasetConfig
    test_dataset: DatasetConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig


def load_macer_config(path: str) -> MacerConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    if "model" not in raw:
        raise ValueError("Config must contain 'model'")
    if "test_dataset" not in raw:
        raise ValueError("Config must contain 'test_dataset'")

    model_cfg = _parse_model(raw["model"])
    train_ds_cfg = _parse_dataset(raw["train_dataset"])
    test_ds_cfg = _parse_dataset(raw["test_dataset"])
    optimizer_cfg = _parse_optimizer(raw["optimizer"])
    scheduler_cfg = _parse_scheduler(raw["scheduler"])
    macer_params = _parse_macer_params(raw["macer_params"])
    return MacerConfig(
        model=model_cfg,
        train_dataset=train_ds_cfg,
        test_dataset=test_ds_cfg,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        params=macer_params,
    )

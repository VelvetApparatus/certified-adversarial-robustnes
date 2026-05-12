from dataclasses import dataclass

import yaml

from src.config.common import (
    TrainingConfig, AWPParams,
    TradesParams, DatasetConfig,
    DatasetSplitConfig, PGDAttackConfig, ModelConfig, NormalizeConfig
)
from src.config._parsers import (
    _parse_dataset, _parse_dataset_split,
    _parse_training, _parse_model, _parse_normalization,
    _parse_awp_params, _parse_trades_params, _parse_pgd
)


@dataclass
class AWPTradesConfig:
    training: TrainingConfig
    trades: TradesParams
    model: ModelConfig
    awp: AWPParams
    dataset: DatasetConfig
    split: DatasetSplitConfig
    evalPGD: PGDAttackConfig
    trainPGD: PGDAttackConfig

    normalization: NormalizeConfig


def load_awp_config(path: str) -> AWPTradesConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if "awp" not in raw:
        raise FileNotFoundError("AWP not found in config")
    if "train" not in raw and "training" not in raw:
        raise FileNotFoundError("Train not found in config")
    if "trades" not in raw and "trades_params" not in raw:
        raise FileNotFoundError("Trades not found in config")
    if "dataset" not in raw:
        raise FileNotFoundError("Dataset not found in config")
    if "split" not in raw:
        raise FileNotFoundError("Split not found in config")
    if "eval_pgd" not in raw:
        raise ValueError("Config must contain 'eval_pgd'")
    if "train_pgd" not in raw:
        raise ValueError("Config must contain 'train_pgd'")
    if "normalization" not in raw:
        raise FileNotFoundError("Normalization not found in config")

    return AWPTradesConfig(
        training=_parse_training(raw.get("train", raw.get("training"))),
        trades=_parse_trades_params(raw.get("trades", raw.get("trades_params"))),
        model=_parse_model(raw["model"]),
        awp=_parse_awp_params(raw["awp"]),
        dataset=_parse_dataset(raw["dataset"]),
        split=_parse_dataset_split(raw["split"]),
        normalization=_parse_normalization(raw["normalization"]),
        evalPGD=_parse_pgd(raw["eval_pgd"]),
        trainPGD=_parse_pgd(raw["train_pgd"]),
    )

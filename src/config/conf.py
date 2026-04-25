from dataclasses import dataclass, field, replace
from typing import Optional, Literal, List, Union

import torch
import yaml
from torchvision.transforms import Compose

LossName = Literal["cross_entropy"]
NormName = Literal["Linf", "l2"]
AttackName = Literal["fgsm", "pgd", "stadv"]


@dataclass
class ModelConfig:
    name: str
    weights_path: Optional[str] = None
    loss_fn: LossName = "cross_entropy"


@dataclass
class DatasetConfig:
    name: str
    root_dir: str = "./data"
    train: bool = False
    download: bool = True
    batch_size: int = 128
    num_workers: int = 0
    transform: Optional[Compose] = None


@dataclass
class DatasetSplitsConfig:
    train: Optional[DatasetConfig] = None
    test: Optional[DatasetConfig] = None


@dataclass
class FGSMAttackConfig:
    name: Literal["fgsm"]
    epsilon: float
    loss_fn: LossName = "cross_entropy"


@dataclass
class PGDAttackConfig:
    name: Literal["pgd"]
    epsilon: float
    alpha: float
    steps: int
    norm: NormName = "Linf"
    loss_fn: LossName = "cross_entropy"
    mean: torch.Tensor = None
    std: torch.Tensor = None


@dataclass
class StAdvAttackConfig:
    name: Literal["stadv"]
    alpha: float
    steps: int
    tau: float
    targeted: bool = False
    loss_fn: LossName = "cross_entropy"


AttackConfig = Union[FGSMAttackConfig, PGDAttackConfig, StAdvAttackConfig]


@dataclass
class ExperimentConfig:
    model: ModelConfig
    dataset: Optional[DatasetConfig] = None
    datasets: Optional[DatasetSplitsConfig] = None
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


def _parse_attack(cfg: dict) -> AttackConfig:
    attack_name = cfg.get("name")
    if attack_name is None:
        raise ValueError("Each attack must contain field 'name'")

    if attack_name == "fgsm":
        return FGSMAttackConfig(
            name="fgsm",
            epsilon=cfg["epsilon"],
            loss_fn=cfg.get("loss_fn", "cross_entropy"),
        )

    if attack_name == "pgd":
        return PGDAttackConfig(
            name="pgd",
            epsilon=cfg["epsilon"],
            alpha=cfg["alpha"],
            steps=cfg["steps"],
            norm=cfg.get("norm", "Linf"),
            loss_fn=cfg.get("loss_fn", "cross_entropy"),
            mean=cfg.get("mean", None),
            std=cfg.get("std", None),
        )

    if attack_name == "stadv":
        return StAdvAttackConfig(
            name="stadv",
            alpha=cfg["alpha"],
            steps=cfg["steps"],
            tau=cfg["tau"],
            targeted=cfg.get("targeted", False),
            loss_fn=cfg.get("loss_fn", "cross_entropy"),
        )

    raise ValueError(f"Unsupported attack name: {attack_name}")


def _parse_dataset(cfg: dict, *, default_train: Optional[bool] = None) -> DatasetConfig:
    if cfg is None:
        raise ValueError("Dataset config must not be empty")

    train = cfg.get("train", default_train if default_train is not None else False)

    return DatasetConfig(
        name=cfg["name"],
        root_dir=cfg.get("root_dir", "./data"),
        train=train,
        download=cfg.get("download", True),
        batch_size=cfg.get("batch_size", 128),
        num_workers=cfg.get("num_workers", 0),
    )


def load_config(path: str) -> ExperimentConfig:
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

    return ExperimentConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        datasets=datasets_cfg,
        attacks=attacks_cfg,
    )

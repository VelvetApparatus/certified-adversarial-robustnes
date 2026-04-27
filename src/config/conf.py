from dataclasses import dataclass, field, replace
from typing import Optional, Literal, List, Union

import torch
import yaml
from torchvision.transforms import Compose

LossName = Literal["cross_entropy"]
NormName = Literal["Linf", "l2"]
AttackName = Literal["fgsm", "pgd", "stadv"]
OptimizerName = Literal["sgd", "adam", "adamw"]

SchedulerName = Literal["none", "step_lr", "cosine"]

DeviceName = Literal["cpu", "cuda", "mps", "auto"]


@dataclass
class OptimizerConfig:
    name: OptimizerName = "sgd"
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    nesterov: bool = False


@dataclass
class SchedulerConfig:
    name: SchedulerName = "none"
    step_size: int = 30
    gamma: float = 0.1
    eta_min: float = 0.0


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "certified-robustness"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)


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
    mean: torch.Tensor = None
    std: torch.Tensor = None


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


@dataclass
class TrainingConfig:
    enabled: bool = True
    epochs: int = 100
    seed: int = 42
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    save_dir: str = "./checkpoints"
    save_best: bool = True
    save_last: bool = True
    metric_for_best_model: str = "test_accuracy"


AttackConfig = Union[FGSMAttackConfig, PGDAttackConfig, StAdvAttackConfig]


@dataclass
class ExperimentConfig:
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


def _parse_attack(cfg: dict) -> AttackConfig:
    attack_name = cfg.get("name")
    if attack_name is None:
        raise ValueError("Each attack must contain field 'name'")

    if attack_name == "fgsm":
        return FGSMAttackConfig(
            name="fgsm",
            epsilon=cfg["epsilon"],
            loss_fn=cfg.get("loss_fn", "cross_entropy"),
            mean=cfg.get("mean", None),
            std=cfg.get("std", None),
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


def _parse_optimizer(cfg: Optional[dict]) -> OptimizerConfig:
    cfg = cfg or {}

    return OptimizerConfig(
        name=cfg.get("name", "sgd"),
        lr=cfg.get("lr", 0.1),
        weight_decay=cfg.get("weight_decay", 5e-4),
        momentum=cfg.get("momentum", 0.9),
        nesterov=cfg.get("nesterov", False),
    )


def _parse_scheduler(cfg: Optional[dict]) -> SchedulerConfig:
    cfg = cfg or {}

    return SchedulerConfig(
        name=cfg.get("name", "none"),
        step_size=cfg.get("step_size", 30),
        gamma=cfg.get("gamma", 0.1),
        eta_min=cfg.get("eta_min", 0.0),
    )


def _parse_wandb(cfg: Optional[dict]) -> WandbConfig:
    cfg = cfg or {}

    return WandbConfig(
        enabled=cfg.get("enabled", False),
        project=cfg.get("project", "certified-robustness"),
        entity=cfg.get("entity", None),
        run_name=cfg.get("run_name", None),
        tags=cfg.get("tags", []),
    )


def _parse_training(cfg: Optional[dict]) -> TrainingConfig:
    cfg = cfg or {}

    return TrainingConfig(
        enabled=cfg.get("enabled", True),
        epochs=cfg.get("epochs", 100),
        seed=cfg.get("seed", 42),

        optimizer=_parse_optimizer(cfg.get("optimizer")),
        scheduler=_parse_scheduler(cfg.get("scheduler")),
        wandb=_parse_wandb(cfg.get("wandb")),

        save_dir=cfg.get("save_dir", "./checkpoints"),
        save_best=cfg.get("save_best", True),
        save_last=cfg.get("save_last", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "test_accuracy"),
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

    training_cfg = _parse_training(raw.get("training"))
    evaluation_root = raw.get("evaluation_root")

    return ExperimentConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        datasets=datasets_cfg,
        attacks=attacks_cfg,
        training=training_cfg,
        evaluation_root=evaluation_root,
    )

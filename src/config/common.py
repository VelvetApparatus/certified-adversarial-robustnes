from dataclasses import field, dataclass
from typing import Optional, List, Literal, Union

import torch
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
    pretrained: bool
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


AttackConfig = Union[FGSMAttackConfig, PGDAttackConfig, StAdvAttackConfig]


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

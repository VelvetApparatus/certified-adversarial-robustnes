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
    milestones: List[int] = field(default_factory=list)


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
    num_classes: Optional[int] = None


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
    checkpoint: str = None
    save_best: bool = True
    save_last: bool = True
    metric_for_best_model: str = "test_accuracy"


@dataclass
class CertificationParams:
    sigma: float
    output_dir: str
    n0: int = 100
    n: int = 100000
    alpha: float = 0.001
    seed: int = 42


@dataclass
class EvaluationTableParams:
    method: str
    comment: str

    # Loss function used for clean/noisy/adversarial evaluation.
    loss_fn: LossName

    # Standard deviation of Gaussian noise used for noisy evaluation
    # and randomized smoothing certification.
    sigma: float

    # Randomized smoothing certification mode.
    # "hard" uses standard majority-vote certification.
    # "soft" uses soft prediction probabilities and empirical Bernstein bound.
    # "both" computes both hard and soft certificates.
    cert_mode: str

    # Number of Monte Carlo samples used for class selection.
    # The most frequent / most probable class is selected using these samples
    # before running the main certification procedure.
    N0: int

    # Number of Monte Carlo samples used to estimate the certified radius.
    # Larger values give tighter statistical bounds but increase runtime.
    N: int

    # Failure probability for the statistical confidence bound.
    # Smaller alpha gives stronger confidence but may reduce certified radius.
    alpha: float

    # Temperature / smoothing parameter used by Soft-RS mode.
    # Relevant mainly when cert_mode is "soft" or "both".
    beta: float

    # path to evaluation dir
    evaluation_dir: str


@dataclass
class DatasetSplitConfig:
    enabled: bool = False

    # Fraction of the original train dataset used for evaluation/validation.
    # Example: 0.1 means 90% train / 10% eval.
    eval_ratio: float = 0.1

    # Seed for deterministic train/eval split.
    seed: int = 42

    # Whether to shuffle indices before splitting.
    shuffle: bool = True

    # Optional hard limit for eval subset size.
    # Useful for fast experiments.
    eval_size: Optional[int] = None


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
class TradesParams:
    epochs: int
    lr: float
    momentum: float
    epsilon: float
    num_steps: int
    step_size: int
    sigma: float
    beta: float
    seed: int
    output_dir: str
    certificate_every_epoch: int
    certificate_epoch_threshold: int
    checkpoint: str
    cert_start: int
    cert_num: int

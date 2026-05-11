from dataclasses import field, dataclass
from typing import Optional, List, Literal, Union

import torch
from torchvision.transforms import Compose

LossName = Literal["cross_entropy"]
NormName = Literal["Linf", "l2"]
AttackName = Literal["fgsm", "pgd", "stadv", "smooth_pgd"]
AttackLossName = Literal["cross_entropy", "kl_divergence"]
OptimizerName = Literal["sgd", "adam", "adamw"]
BestMetricMode = Literal["auto", "min", "max"]

SchedulerName = Literal["none", "step_lr", "cosine"]

DeviceName = Literal["cpu", "cuda", "mps", "auto"]


@dataclass
class LinearScheduleConfig:
    enabled: bool = False
    type: str = "linear"
    start: float = 0.0
    end: float = 12.0
    warmup_epochs: int = 0
    ramp_epochs: int = 20


def get_scheduled(base: float, schedule: Optional[LinearScheduleConfig], epoch: int) -> float:
    if schedule is None or not schedule.enabled:
        return float(base)

    if schedule.type != "linear":
        raise ValueError(f"Unsupported schedule type: {schedule.type}")

    if epoch <= schedule.warmup_epochs:
        return schedule.start

    if schedule.ramp_epochs <= 0:
        return schedule.end

    progress = (epoch - schedule.warmup_epochs) / schedule.ramp_epochs
    progress = max(0.0, min(1.0, progress))

    return schedule.start + progress * (schedule.end - schedule.start)


@dataclass
class NormalizeConfig:
    enabled: bool = False
    std: torch.Tensor = None
    mean: torch.Tensor = None


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
    normalize: Optional[NormalizeConfig] = field(default_factory=NormalizeConfig)


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
    loss_fn: AttackLossName = "cross_entropy"
    random_start: bool = True
    epsilon_scheduler: Optional[LinearScheduleConfig] = None
    alpha_scheduler: Optional[LinearScheduleConfig] = None


@dataclass
class SmoothedAttackConfig:
    name: Literal["smooth_pgd"]
    epsilon: float
    alpha: float
    steps: int
    norm: NormName = "l2"
    random_start: bool = True
    clamp_noisy: bool = True
    epsilon_scheduler: Optional[LinearScheduleConfig] = None
    alpha_scheduler: Optional[LinearScheduleConfig] = None


@dataclass
class StAdvAttackConfig:
    name: Literal["stadv"]
    alpha: float
    steps: int
    tau: float
    targeted: bool = False
    loss_fn: LossName = "cross_entropy"


AttackConfig = Union[
    FGSMAttackConfig,
    PGDAttackConfig,
    SmoothedAttackConfig,
    StAdvAttackConfig,
]


@dataclass
class TrainingConfig:
    enabled: bool = True
    epochs: int = 100
    seed: int = 42
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    criterion: LossName = "cross_entropy"
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoint: str = None
    save_best: bool = True
    save_last: bool = True
    metric_for_best_model: str = "eval_acc"
    metric_mode_for_best_model: BestMetricMode = "auto"
    save_dir: str = None


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
class GaussianTrainingParams:
    sigma: float
    clean_loss_weight: float = 0.0
    noisy_loss_weight: float = 1.0
    noise_ratio: float = 1.0
    normalized_space: bool = True


@dataclass
class MacerTrainingParams:
    # Number of Gaussian samples per input during training.
    gauss_samples: int = 16

    # Gaussian noise standard deviation.
    sigma: float = 0.25

    # Number of classes in the dataset.
    num_classes: int = 10

    # Softmax sharpening parameter used in MACER robustness term.
    beta: float = 16.0
    beta_scheduler: Optional[LinearScheduleConfig] = field(default_factory=LinearScheduleConfig)

    # Target margin in inverse Gaussian CDF space.
    gamma: float = 8.0

    # Weight of MACER robustness regularization.
    lbd: float = 12.0
    # optional scheduler for lbd to more accurate training
    lbd_scheduler: Optional[LinearScheduleConfig] = field(default_factory=LinearScheduleConfig)

    # Numerical stability epsilon for probabilities before icdf/log.
    eps: float = 1e-6


@dataclass
class TradesParams:
    epochs: int = 100
    lr: float = 0.1
    momentum: float = 0.9
    epsilon: float = 0.01
    num_steps: int = 10
    step_size: float = 0.01
    sigma: float = 0.0
    beta: float = 6.0
    seed: int = 42
    output_dir: str | None = None
    certificate_every_epoch: int = 0
    certificate_epoch_threshold: int = 200
    checkpoint: str | None = None
    cert_start: int = 0
    cert_num: int = 100
    distance: str = "l_inf"


@dataclass
class SmoothAdvTrainingParams:
    sigma: float = 0.25
    sigma_scheduler: Optional[LinearScheduleConfig] = None

    epsilon: float = 0.25
    epsilon_scheduler: Optional[LinearScheduleConfig] = None

    step_size: float = 0.025
    steps: int = 10

    num_noise_vec: int = 2
    norm: str = "l2"

    beta: float = 16.0
    beta_scheduler: Optional[LinearScheduleConfig] = None

    train_multi_noise: bool = True
    clamp_noisy: bool = True


@dataclass
class AWPParams:
    weights_diff_coef: float = 0.0
    weights_epsilon: float = 0.0
    warmup_steps: int = 10
    proxy_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

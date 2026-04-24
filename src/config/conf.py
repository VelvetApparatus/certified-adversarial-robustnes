from dataclasses import dataclass, field
from typing import Optional, Literal, List, Union
import yaml

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
    dataset: DatasetConfig
    attacks: List[AttackConfig] = field(default_factory=list)


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


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("YAML config is empty")

    if "model" not in raw:
        raise ValueError("Config must contain 'model'")
    if "dataset" not in raw:
        raise ValueError("Config must contain 'dataset'")

    model_cfg = ModelConfig(
        name=raw["model"]["name"],
        weights_path=raw["model"].get("weights_path"),
    )

    dataset_cfg = DatasetConfig(
        root_dir=raw["dataset"].get("root_dir", "./data"),
        train=raw["dataset"].get("train", False),
        download=raw["dataset"].get("download", True),
        batch_size=raw["dataset"].get("batch_size", 128),
    )

    attacks_raw = raw.get("attacks", [])
    attacks_cfg = [_parse_attack(a) for a in attacks_raw]

    return ExperimentConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        attacks=attacks_cfg,
    )

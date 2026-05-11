from typing import Optional

from src.config.common import AttackConfig, FGSMAttackConfig, PGDAttackConfig, StAdvAttackConfig, DatasetConfig, \
    OptimizerConfig, WandbConfig, TrainingConfig, SchedulerConfig, ModelConfig, CertificationParams, TradesParams, \
    EvaluationTableParams, DatasetSplitConfig, GaussianTrainingParams, MacerTrainingParams, NormalizeConfig, \
    LinearScheduleConfig, SmoothAdvTrainingParams, AWPParams


def _parse_linear_schedule(cfg: Optional[dict]) -> LinearScheduleConfig:
    cfg = cfg or {}
    return LinearScheduleConfig(
        enabled=cfg.get("enabled", False),
        type=cfg.get("type", "linear"),
        start=cfg.get("start", 0.0),
        end=cfg.get("end", 1.0),
        warmup_epochs=cfg.get("warmup_epochs", 0),
        ramp_epochs=cfg.get("ramp_epochs", 0),

    )


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


def _parse_normalization(cfg: Optional[dict]) -> Optional[NormalizeConfig]:
    cfg = cfg or {}
    return NormalizeConfig(
        enabled=cfg.get("enabled", False),
        mean=cfg.get("mean", None),
        std=cfg.get("std", None),
    )


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
        milestones=cfg.get("milestones", [10, 20]),
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
        criterion=cfg.get("criterion"),
        wandb=_parse_wandb(cfg.get("wandb")),

        checkpoint=cfg.get("checkpoint", None),
        save_dir=cfg.get("save_dir", "./checkpoints"),
        save_best=cfg.get("save_best", True),
        save_last=cfg.get("save_last", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_acc"),
        metric_mode_for_best_model=cfg.get("metric_mode_for_best_model", "auto"),
    )


def _parse_model(cfg: Optional[dict]) -> ModelConfig:
    cfg = cfg or {}
    return ModelConfig(
        name=cfg.get("name", None),
        pretrained=cfg.get("pretrained", True),
        weights_path=cfg.get("weights_path", None),
        loss_fn=cfg.get("loss_fn", "cross_entropy"),
        num_classes=cfg.get("num_classes", 100),
    )


def _parse_macer_params(cfg: dict) -> MacerTrainingParams:
    cfg = cfg or {}

    return MacerTrainingParams(
        gauss_samples=cfg.get("gauss_samples", 16),
        sigma=cfg.get("sigma", 0.25),
        num_classes=cfg.get("num_classes", 10),
        beta=cfg.get("beta", {}).get("value", 16.0),
        beta_scheduler=_parse_linear_schedule(cfg.get("beta", {}).get("scheduler", None)),
        gamma=cfg.get("gamma", 8.0),
        lbd=cfg.get("lbd", {}).get("value", 12.0),
        lbd_scheduler=_parse_linear_schedule(cfg.get("lbd", {}).get("scheduler", None)),
        eps=cfg.get("eps", 1e-6),
    )


def _parse_trades_params(cfg: Optional[dict]) -> TradesParams:
    cfg = cfg or {}
    return TradesParams(
        epochs=cfg.get("epochs", 100),
        lr=cfg.get("lr", 0.1),
        momentum=cfg.get("momentum", 0.9),
        epsilon=cfg.get("epsilon", 0.01),
        num_steps=cfg.get("num_steps", 100),
        step_size=cfg.get("step_size", 30),
        beta=cfg.get("beta", 0.01),
        sigma=cfg.get("sigma", 0.01),
        seed=cfg.get("seed", 42),
        output_dir=cfg.get("output_dir", None),
        certificate_every_epoch=cfg.get("certificate_every_epoch", False),
        certificate_epoch_threshold=cfg.get("certificate_epoch_threshold", 200),
        checkpoint=cfg.get("checkpoint", None),
        cert_start=cfg.get("cert_start", 0),
        cert_num=cfg.get("cert_num", 100),
    )


def _parse_pgd(cfg: Optional[dict]) -> PGDAttackConfig:
    cfg = cfg or {}
    return PGDAttackConfig(
        name=cfg.get("name", "pgd"),
        epsilon=cfg.get("epsilon", 0.01),
        alpha=cfg.get("alpha", 0.01),
        steps=cfg.get("steps", 100),
        norm=cfg.get("norm", "l2"),
        loss_fn=cfg.get("loss_fn", None),
        random_start=cfg.get("random_start", True)
    )


def _parse_fgsm(cfg: Optional[dict]) -> FGSMAttackConfig:
    cfg = cfg or {}
    return FGSMAttackConfig(
        name=cfg.get("name", "fgsm"),
        epsilon=cfg.get("epsilon", 0.01),
        loss_fn=cfg.get("loss_fn", None),
    )


def _parse_evaluation_table_params(cfg: Optional[dict]) -> EvaluationTableParams:
    cfg = cfg or {}
    return EvaluationTableParams(
        method=cfg.get("method", None),
        comment=cfg.get("comment", None),
        loss_fn=cfg.get("loss_fn", None),
        sigma=cfg.get("sigma", 0.01),
        cert_mode=cfg.get("cert_mode", "hard"),
        N0=cfg.get("N0", 100),
        N=cfg.get("N", 100),
        alpha=cfg.get("alpha", 0.01),
        beta=cfg.get("beta", 0.01),
        evaluation_dir=cfg.get("evaluation_dir", None),
    )


def _parse_certification_params(cfg: dict) -> CertificationParams:
    return CertificationParams(
        sigma=cfg["sigma"],
        output_dir=cfg["output_dir"],
        n0=cfg["n0"],
        n=cfg["n"],
        alpha=cfg["alpha"],
    )


def _parse_dataset_split(cfg: Optional[dict]) -> DatasetSplitConfig:
    cfg = cfg or {}

    return DatasetSplitConfig(
        enabled=cfg.get("enabled", False),
        eval_ratio=cfg.get("eval_ratio", 0.1),
        seed=cfg.get("seed", 42),
        shuffle=cfg.get("shuffle", True),
        eval_size=cfg.get("eval_size", None),

    )


def _parse_gaussian_params(cfg: Optional[dict]) -> GaussianTrainingParams:
    cfg = cfg or {}
    return GaussianTrainingParams(
        sigma=cfg.get("sigma", 0.01),
        clean_loss_weight=cfg.get("clean_loss_weight", 1.0),
        noisy_loss_weight=cfg.get("noisy_loss_weight", 1.0),
        noise_ratio=cfg.get("noise_ratio", 1.0),
        normalized_space=cfg.get("normalized_space", True),

    )


def _parse_value_with_scheduler(
        cfg: dict,
        key: str,
        default_value: float,
):
    raw = cfg.get(key, default_value)

    if isinstance(raw, (int, float)):
        return float(raw), None

    if isinstance(raw, dict):
        value = float(raw.get("value", default_value))
        scheduler = _parse_linear_schedule(raw.get("scheduler", None))
        return value, scheduler

    raise TypeError(
        f"params.{key} must be either a number or a dict with "
        f"'value' and optional 'scheduler'. Got {type(raw)}"
    )


def _parse_smooth_adv_params(cfg: dict) -> SmoothAdvTrainingParams:
    cfg = cfg or {}

    sigma, sigma_scheduler = _parse_value_with_scheduler(
        cfg=cfg,
        key="sigma",
        default_value=0.25,
    )

    epsilon, epsilon_scheduler = _parse_value_with_scheduler(
        cfg=cfg,
        key="epsilon",
        default_value=0.25,
    )

    beta, beta_scheduler = _parse_value_with_scheduler(
        cfg=cfg,
        key="beta",
        default_value=1,
    )

    return SmoothAdvTrainingParams(
        sigma=sigma,
        sigma_scheduler=sigma_scheduler,
        epsilon=epsilon,
        epsilon_scheduler=epsilon_scheduler,
        step_size=float(cfg.get("step_size", 0.025)),
        steps=int(cfg.get("steps", 10)),
        num_noise_vec=int(cfg.get("num_noise_vec", 2)),
        norm=cfg.get("norm", "l2"),
        train_multi_noise=bool(cfg.get("train_multi_noise", True)),
        clamp_noisy=bool(cfg.get("clamp_noisy", True)),
        beta=beta,
        beta_scheduler=beta_scheduler,
    )


def _parse_awp_params(cfg: dict) -> AWPParams:
    cfg = cfg or {}

    beta, beta_scheduler = _parse_value_with_scheduler(
        cfg=cfg,
        key="beta",
        default_value=0.1,
    )

    return AWPParams(
        weights_epsilon=cfg.get("weights_epsilon", 0.01),
        weights_diff_coef=cfg.get("weights_diff_coef", 0.01),
        warmup_steps=cfg.get("warmup_steps", 100),
        beta=beta,
        beta_scheduler=beta_scheduler,
    )

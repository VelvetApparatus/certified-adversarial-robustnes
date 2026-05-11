import datetime
import math
import os
import shutil
from typing import Callable, Literal

import torch
import wandb
from tqdm import tqdm

from src.config.common import (
    ModelConfig,
    DatasetConfig,
    DatasetSplitConfig,
    TrainingConfig,
    NormalizeConfig,
    BestMetricMode,
)
from src.config.secret import WANDB_TOKEN
from src.db.api import build_train_eval_loaders
from src.model.api import get_model
from src.pkg import init_metrics, update_metrics, finalize_metrics, get_optimizer, get_scheduler, InputNormalizer


def prefix_metrics(prefix: str, metrics: dict) -> dict:
    return {
        f"{prefix}/{k}": v
        for k, v in metrics.items()
        if isinstance(v, (int, float))
    }


def select_metric(
        train_metrics: dict,
        eval_metrics: dict | None,
        metric_name: str,
) -> float:
    available = {}

    available.update({f"train_{k}": v for k, v in train_metrics.items()})

    if eval_metrics is not None:
        available.update({f"eval_{k}": v for k, v in eval_metrics.items()})

    if metric_name not in available:
        raise ValueError(
            f"Unknown metric_for_best_model='{metric_name}'. "
            f"Available metrics: {list(available.keys())}"
        )

    return float(available[metric_name])


def resolve_metric_mode(
        metric_name: str,
        metric_mode: BestMetricMode,
) -> Literal["min", "max"]:
    if metric_mode in ("min", "max"):
        return metric_mode

    metric_name = metric_name.lower()
    if "loss" in metric_name:
        return "min"

    return "max"


def initial_best_metric(metric_mode: Literal["min", "max"]) -> float:
    return math.inf if metric_mode == "min" else -math.inf


def is_better_metric(
        current: float,
        best: float,
        metric_mode: Literal["min", "max"],
) -> bool:
    if metric_mode == "min":
        return current < best

    return current > best


def save_checkpoint(
        path: str,
        model,
        optimizer,
        scheduler,
        epoch: int,
        best_metric: float,
        best_metric_name: str,
        best_metric_mode: str,
):
    net = model.model if isinstance(model, InputNormalizer) else model

    state = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "best_metric_mode": best_metric_mode,
    }
    torch.save(state, path)


def resolve_model_name(model: ModelConfig | torch.nn.Module) -> str:
    if isinstance(model, ModelConfig):
        return model.name

    if isinstance(model, InputNormalizer):
        return resolve_model_name(model.model)

    return getattr(model, "name", model.__class__.__name__)


def train(
        name: str,
        cfg: TrainingConfig,
        norm_cfg: NormalizeConfig | None,
        model: ModelConfig | torch.nn.Module,
        device,
        train_dataset_config: DatasetConfig,
        split_config: DatasetSplitConfig,
        loss_fn,
        train_epoch_fn: Callable,
        eval_fn: Callable,
        config_path: str | None = None,
        optimizer=None,
        model_is_prepared: bool = False,
        training_kwargs: dict | None = None,
        eval_kwargs: dict | None = None,
):
    if isinstance(model, ModelConfig):
        model = get_model(model, device).to(device)
    else:
        model = model.to(device)

    training_kwargs = dict(training_kwargs or {})
    eval_kwargs = dict(eval_kwargs or {})

    train_loader, eval_loader, train_dataset, eval_dataset = build_train_eval_loaders(
        dataset_cfg=train_dataset_config,
        split_cfg=split_config,
    )

    if optimizer is None:
        optimizer = get_optimizer(model, cfg.optimizer)

    if norm_cfg is not None and norm_cfg.enabled and not model_is_prepared:
        model = InputNormalizer(
            model=model,
            std=norm_cfg.std,
            mean=norm_cfg.mean,
        )
        model = model.to(device)

    model_name = resolve_model_name(model)

    scheduler = get_scheduler(optimizer, cfg.scheduler, epochs=cfg.epochs)

    run_name = "{}_{}_{}".format(
        model_name,
        train_dataset_config.name,
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    use_wandb = init_wandb_if_needed(
        name=name,
        cfg=cfg,
        dataset_cfg=train_dataset_config,
        model_name=model_name,
        split=split_config,
    )

    output_dir = os.path.join(cfg.save_dir, run_name)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    if config_path is not None:
        shutil.copy(config_path, os.path.join(output_dir, "config.yaml"))

    start_epoch = 1
    best_metric_mode = resolve_metric_mode(
        metric_name=cfg.metric_for_best_model,
        metric_mode=cfg.metric_mode_for_best_model,
    )
    best_metric = initial_best_metric(best_metric_mode)

    if cfg.checkpoint is not None:
        checkpoint = torch.load(cfg.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["net"])

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"] + 1

        checkpoint_metric_name = checkpoint.get("best_metric_name")
        checkpoint_metric_mode = checkpoint.get("best_metric_mode")

        if (
                checkpoint_metric_name == cfg.metric_for_best_model
                and checkpoint_metric_mode == best_metric_mode
                and "best_metric" in checkpoint
        ):
            best_metric = checkpoint["best_metric"]
        elif "best_metric" in checkpoint:
            print(
                "==> Checkpoint best metric metadata does not match current "
                "metric_for_best_model / metric_mode_for_best_model. "
                "Best metric tracking will be reset for this run."
            )

    print(
        "Best checkpoint metric: "
        f"{cfg.metric_for_best_model} ({best_metric_mode})"
    )

    for epoch in tqdm(range(start_epoch, cfg.epochs + 1), desc="Epoch"):
        model.train()

        train_metrics = train_epoch_fn(
            model=model,
            train_loader=train_loader,
            criterion=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            **training_kwargs,
        )

        print("=" * 80)
        print("Train Evaluation")
        print(f"Epoch {epoch}: {train_metrics}")
        print("=" * 80)

        eval_metrics = None
        if eval_loader is not None:
            model.eval()
            eval_call_kwargs = dict(eval_kwargs)
            eval_call_kwargs["epoch"] = epoch
            eval_metrics = eval_fn(
                model=model,
                loader=eval_loader,
                criterion=loss_fn,
                device=device,
                **eval_call_kwargs,
            )

            print("=" * 80)
            print("Eval Clean Evaluation")
            print(f"Epoch {epoch}: {eval_metrics}")
            print("=" * 80)

        current_lr = optimizer.param_groups[0]["lr"]

        if use_wandb:
            log_data = {
                "epoch": epoch,
                "lr": current_lr,
            }
            log_data.update(prefix_metrics("train", train_metrics))

            if eval_metrics is not None:
                log_data.update(prefix_metrics("eval", eval_metrics))

            wandb.log(log_data)

        metric_value = select_metric(
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            metric_name=cfg.metric_for_best_model,
        )

        if not math.isfinite(metric_value):
            print(
                f"==> Metric '{cfg.metric_for_best_model}' is not finite "
                f"({metric_value}); skipping best-checkpoint update."
            )
        elif cfg.save_best and is_better_metric(
                current=metric_value,
                best=best_metric,
                metric_mode=best_metric_mode,
        ):
            best_metric = metric_value

            best_path = os.path.join(checkpoints_dir, "best.pth")
            print(
                f"==> Saving best checkpoint: {best_path} "
                f"({cfg.metric_for_best_model}={best_metric:.4f})"
            )

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                best_metric_name=cfg.metric_for_best_model,
                best_metric_mode=best_metric_mode,
            )

        if scheduler is not None:
            scheduler.step()

    if cfg.save_last:
        last_path = os.path.join(checkpoints_dir, "last.pth")
        print(f"==> Saving last checkpoint: {last_path}")

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=cfg.epochs,
            best_metric=best_metric,
            best_metric_name=cfg.metric_for_best_model,
            best_metric_mode=best_metric_mode,
        )

    if use_wandb:
        wandb.finish()

    print("\nTraining finished")
    print(
        f"Best metric ({cfg.metric_for_best_model}, {best_metric_mode}): "
        f"{best_metric:.4f}"
    )

    return {
        "output_dir": output_dir,
        "best_metric": best_metric,
    }


def init_wandb_if_needed(
        name: str,
        cfg: TrainingConfig,
        dataset_cfg: DatasetConfig,
        model_name: str,
        split: DatasetSplitConfig,
):
    use_wandb = cfg.wandb.enabled

    if not use_wandb:
        return False

    wandb.login(key=WANDB_TOKEN)

    run_name = cfg.wandb.run_name or (
        f"{model_name}|{dataset_cfg.name}|{name}|"
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        tags=cfg.wandb.tags,
        config={
            "dataset": dataset_cfg.name,
            "model": model_name,
            "method": name,
            "epochs": cfg.epochs,
            "seed": cfg.seed,
            "optimizer": {
                "name": cfg.optimizer.name,
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
                "momentum": cfg.optimizer.momentum,
                "nesterov": cfg.optimizer.nesterov,
            },
            "scheduler": {
                "name": cfg.scheduler.name,
                "step_size": cfg.scheduler.step_size,
                "gamma": cfg.scheduler.gamma,
                "eta_min": cfg.scheduler.eta_min,
            },
            "split": {
                "enabled": split.enabled,
                "eval_ratio": split.eval_ratio,
                "eval_size": split.eval_size,
                "seed": split.seed,
                "shuffle": split.shuffle,
            },
            "train_dataset": {
                "batch_size": dataset_cfg.batch_size,
                # "num_workers": dataset_cfg.num_workers,
            },
        },
    )

    return True

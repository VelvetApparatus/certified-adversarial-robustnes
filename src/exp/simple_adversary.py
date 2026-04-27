import argparse
import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb

from src.config.conf import load_config
from src.model.api import get_model
from src.db.api import get_dataset
from src.adversaries.api import get_adversaries
from src.pkg import get_device, get_loss_fn, set_seed
from src.config.secret import WANDB_TOKEN

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", default="config/config.yaml")
args = arg_parser.parse_args()


def init_metrics():
    return {
        "loss": 0.0,
        "correct": 0,
        "total": 0,
    }


def update_metrics(storage, logits, y, loss, batch_size):
    preds = logits.argmax(dim=1)

    storage["loss"] += loss.item() * batch_size
    storage["correct"] += (preds == y).sum().item()
    storage["total"] += batch_size


def finalize_metrics(storage):
    total = max(storage["total"], 1)

    return {
        "loss": storage["loss"] / total,
        "acc": storage["correct"] / total,
        "total": storage["total"],
    }


def get_optimizer(model, cfg):
    opt_cfg = cfg.training.optimizer

    if opt_cfg.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg.lr,
            momentum=opt_cfg.momentum,
            weight_decay=opt_cfg.weight_decay,
            nesterov=opt_cfg.nesterov,
        )

    if opt_cfg.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )

    if opt_cfg.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")


def get_scheduler(optimizer, cfg):
    scheduler_cfg = cfg.training.scheduler

    if scheduler_cfg.name == "none":
        return None

    if scheduler_cfg.name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.step_size,
            gamma=scheduler_cfg.gamma,
        )

    if scheduler_cfg.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs,
            eta_min=scheduler_cfg.eta_min,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_cfg.name}")


def maybe_make_train_subset(train_dataset, cfg):
    subset_fraction = getattr(cfg.training, "train_subset_fraction", None)

    if subset_fraction is None:
        return train_dataset

    if not 0 < subset_fraction <= 1:
        raise ValueError("training.train_subset_fraction must be in range (0, 1]")

    batch_size = cfg.train_dataset.batch_size

    subset_size = int(len(train_dataset) * subset_fraction)
    subset_size = (subset_size // batch_size) * batch_size

    if subset_size <= 0:
        raise ValueError("Subset size became zero. Increase train_subset_fraction.")

    return Subset(train_dataset, range(subset_size))


def evaluate_clean(model, test_loader, criterion, device):
    model.eval()
    metrics = init_metrics()

    with torch.no_grad():
        for x, y in tqdm(test_loader, total=len(test_loader), desc="Clean evaluation"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            batch_size = x.size(0)

            logits = model(x)
            loss = criterion(logits, y)

            update_metrics(
                storage=metrics,
                logits=logits,
                y=y,
                loss=loss,
                batch_size=batch_size,
            )

    return finalize_metrics(metrics)


def adversarial_train_one_epoch(
        model,
        train_loader,
        adversary,
        criterion,
        optimizer,
        device,
        epoch: int,
        cfg,
):
    model.train()
    metrics = init_metrics()

    clean_loss_weight = getattr(cfg.training, "clean_loss_weight", 0.8)
    adv_loss_weight = getattr(cfg.training, "adv_loss_weight", 0.2)

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch + 1} | Adv training [{adversary.name}]",
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = x.size(0)
        x_adv = adversary.gen(model, x, y).detach()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits_clean = model(x)
        loss_clean = criterion(logits_clean, y)

        logits_adv = model(x_adv)
        loss_adv = criterion(logits_adv, y)

        loss = clean_loss_weight * loss_clean + adv_loss_weight * loss_adv

        loss.backward()
        optimizer.step()

        # Для train accuracy логируем accuracy на adversarial examples.
        update_metrics(
            storage=metrics,
            logits=logits_adv,
            y=y,
            loss=loss,
            batch_size=batch_size,
        )

        current = finalize_metrics(metrics)
        progress.set_postfix(
            loss=f"{current['loss']:.4f}",
            acc=f"{current['acc']:.4f}",
        )

    return finalize_metrics(metrics)


def save_checkpoint(
        path: str,
        model,
        optimizer,
        scheduler,
        cfg,
        epoch: int,
        best_metric: Optional[float] = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "best_metric": best_metric,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def init_wandb_if_needed(cfg, train_adversary):
    use_wandb = cfg.training.wandb.enabled

    if not use_wandb:
        return False

    wandb.login(key=WANDB_TOKEN)

    dataset_name = cfg.dataset.name if cfg.dataset is not None else cfg.train_dataset.name

    run_name = cfg.training.wandb.run_name or (
        f"{cfg.model.name}|{dataset_name}|advtrain-{train_adversary.name}|"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )

    wandb.init(
        project=cfg.training.wandb.project,
        entity=cfg.training.wandb.entity,
        name=run_name,
        tags=cfg.training.wandb.tags,
        config={
            "dataset": dataset_name,
            "model": cfg.model.name,
            "method": "adversarial-training",
            "train_attack": train_adversary.name,
            "epochs": cfg.training.epochs,
            "seed": cfg.training.seed,
            "optimizer": {
                "name": cfg.training.optimizer.name,
                "lr": cfg.training.optimizer.lr,
                "weight_decay": cfg.training.optimizer.weight_decay,
                "momentum": cfg.training.optimizer.momentum,
                "nesterov": cfg.training.optimizer.nesterov,
            },
            "scheduler": {
                "name": cfg.training.scheduler.name,
                "step_size": cfg.training.scheduler.step_size,
                "gamma": cfg.training.scheduler.gamma,
                "eta_min": cfg.training.scheduler.eta_min,
            },
            "loss_weights": {
                "clean": getattr(cfg.training, "clean_loss_weight", 0.8),
                "adversarial": getattr(cfg.training, "adv_loss_weight", 0.2),
            },
            "train_dataset": {
                "batch_size": cfg.train_dataset.batch_size,
                "num_workers": cfg.train_dataset.num_workers,
            },
            "test_dataset": {
                "batch_size": cfg.test_dataset.batch_size,
                "num_workers": cfg.test_dataset.num_workers,
            },
        },
    )

    return True


def main():
    cfg = load_config(args.config)

    set_seed(cfg.training.seed)

    device = get_device()

    train_dataset_cfg = cfg.train_dataset
    test_dataset_cfg = cfg.test_dataset

    model = get_model(cfg.model, device).to(device)
    criterion = get_loss_fn(cfg.model.loss_fn)

    train_dataset = get_dataset(train_dataset_cfg)
    test_dataset = get_dataset(test_dataset_cfg)

    train_dataset = maybe_make_train_subset(train_dataset, cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_dataset_cfg.batch_size,
        shuffle=True,
        num_workers=train_dataset_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_dataset_cfg.batch_size,
        shuffle=False,
        num_workers=test_dataset_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    adversaries = get_adversaries(cfg.attacks)

    if len(adversaries) == 0:
        raise ValueError("Adversarial training requires at least one attack in config.attacks")

    # Пока берём первую атаку из конфига как атаку для обучения.
    # Позже лучше вынести это в cfg.training.train_attack.
    train_adversary = adversaries[0]

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    use_wandb = init_wandb_if_needed(cfg, train_adversary)

    best_metric = -1.0

    save_dir = cfg.training.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print("\nTraining started")
    print(f"Device: {device}")
    print(f"Model: {cfg.model.name}")
    print(f"Train attack: {train_adversary.name}")
    print(f"Epochs: {cfg.training.epochs}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")

    for epoch in range(cfg.training.epochs):
        train_metrics = adversarial_train_one_epoch(
            model=model,
            train_loader=train_loader,
            adversary=train_adversary,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            cfg=cfg,
        )

        test_metrics = evaluate_clean(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"lr={current_lr:.8f} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_adv_acc={train_metrics['acc']:.4f} | "
            f"test_clean_loss={test_metrics['loss']:.4f} | "
            f"test_clean_acc={test_metrics['acc']:.4f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "lr": current_lr,
                    "train/loss": train_metrics["loss"],
                    "train/adv_acc": train_metrics["acc"],
                    "test/clean_loss": test_metrics["loss"],
                    "test/clean_acc": test_metrics["acc"],
                }
            )

        metric_value = test_metrics["acc"]

        if cfg.training.save_best and metric_value > best_metric:
            best_metric = metric_value

            best_path = os.path.join(
                save_dir,
                f"{cfg.model.name}_{train_dataset_cfg.name}_advtrain_{train_adversary.name}_best.pt",
            )

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                epoch=epoch + 1,
                best_metric=best_metric,
            )

            print(f"Saved best checkpoint: {best_path}")

    if cfg.training.save_last:
        last_path = os.path.join(
            save_dir,
            f"{cfg.model.name}_{train_dataset_cfg.name}_advtrain_{train_adversary.name}_last.pt",
        )

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            epoch=cfg.training.epochs,
            best_metric=best_metric,
        )

        print(f"Saved last checkpoint: {last_path}")

    if use_wandb:
        wandb.finish()

    print("\nTraining finished")
    print(f"Best clean test accuracy: {best_metric:.4f}")


if __name__ == "__main__":
    main()

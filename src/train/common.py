import datetime
import os

import torch
import wandb
from tqdm import tqdm

from src.config.common import ModelConfig, DatasetConfig, DatasetSplitConfig, TrainingConfig
from src.config.secret import WANDB_TOKEN
from src.db.api import split_train_eval_dataset, get_dataset
from src.model.api import get_model
from src.pkg import init_metrics, update_metrics, finalize_metrics, get_optimizer, get_scheduler


def train(
        name: str,
        cfg: TrainingConfig,
        model_cfg: ModelConfig,
        device,
        train_dataset_config: DatasetConfig,
        split_config: DatasetSplitConfig,
        output_dir: str,
        loss_fn,
        train_epoch_fn,
        **kwargs
):
    model = get_model(model_cfg, device)
    model.to(device)

    train_dataset = get_dataset(train_dataset_config)
    train_loader, eval_loader = split_train_eval_dataset(train_dataset, split_config)

    # optimizer
    optimizer = get_optimizer(model, cfg.optimizer)

    # scheduler
    scheduler = get_scheduler(optimizer, cfg.scheduler)

    run_name = "{}_{}_{}".format(
        model_cfg.name,
        train_dataset_config.name,
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    output_dir = os.path.join(output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    start_epoch = 1
    best_metric = -1.0

    # checkpoint
    if cfg.checkpoint is not None:
        checkpoint = torch.load(cfg.checkpoint)
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        scheduler.step(start_epoch)

    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    use_wandb = init_wandb_if_needed(name, cfg, train_dataset_config, model_cfg, split_config)

    for epoch in tqdm(range(start_epoch, cfg.epochs + 1), desc="Epoch"):
        model.train()
        train_metrics = train_epoch_fn(
            modal=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            *kwargs,
        )
        print(
            f"============================================================================================",
            "Train Evaluation",
            f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f}, total={train_metrics['total']}"
            f"============================================================================================",
            sep="\n")
        scheduler.step()

        model.eval()
        eval_metrics = evaluate_clean(model, eval_loader, loss_fn, device)
        print(
            f"============================================================================================",
            "Clean Evaluation",
            f"Epoch {epoch}: loss={eval_metrics['loss']:.4f}, acc={eval_metrics['acc']:.4f}, total={eval_metrics['total']}"
            f"============================================================================================",
            sep="\n"
        )
        # todo: mb add adversarial

        current_lr = optimizer.param_groups[0]["lr"]

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "lr": current_lr,
                    "train/loss": train_metrics["loss"],
                    "train/adv_acc": train_metrics["acc"],
                    "test/clean_loss": eval_metrics["loss"],
                    "test/clean_acc": eval_metrics["acc"],
                }
            )

        metric_value = eval_metrics["acc"]

        if cfg.save_best and metric_value > best_metric:
            best_metric = metric_value

            # Save checkpoint
            print('==> Saving {}.pth..'.format(epoch))
            try:
                state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                }
                torch.save(state, '{}/{}_best.pth'.format(checkpoints_dir, epoch))
            except OSError:
                print('OSError while saving {}.pth'.format(epoch))
                print('Ignoring...')

    if cfg.save_last:
        # Save checkpoint
        print('==> Saving {}.pth..'.format(cfg.epochs))
        try:
            state = {
                'net': model.state_dict(),
                'epoch': cfg.epochs,
            }
            torch.save(state, '{}/{}_last.pth'.format(checkpoints_dir, cfg.epochs))
        except OSError:
            print('OSError while saving {}.pth'.format(cfg.epochs))
            print('Ignoring...')

    if use_wandb:
        wandb.finish()

    print("\nTraining finished")
    print(f"Best clean test accuracy: {best_metric:.4f}")


def evaluate_clean(model, test_loader, criterion, device):
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


def init_wandb_if_needed(
        name: str,
        cfg: TrainingConfig,
        dataset_cfg: DatasetConfig,
        model_cfg: ModelConfig,
        split: DatasetSplitConfig

):
    use_wandb = cfg.wandb.enabled

    if not use_wandb:
        return False

    wandb.login(key=WANDB_TOKEN)

    run_name = cfg.wandb.run_name or (
        f"{model_cfg.name}|{dataset_cfg.name}|{name}|"
        f"{datetime.time.strftime('%Y%m%d-%H%M%S')}"
    )

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        tags=cfg.wandb.tags,
        config={
            "dataset": dataset_cfg.name,
            "model": model_cfg.name,
            "method": "adversarial-training",
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
            "loss_weights": {
                "clean": getattr(cfg, "clean_loss_weight", 0.8),
                "adversarial": getattr(cfg, "adv_loss_weight", 0.2),
            },
            "train_dataset": {
                "batch_size": dataset_cfg.batch_size,
                "num_workers": dataset_cfg.num_workers,
                "split.eval_size": split.eval_size,
            },
        },
    )

    return True

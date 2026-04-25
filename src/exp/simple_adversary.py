import argparse
import os
import copy

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config.conf import load_config
from src.model.api import get_model
from src.db.api import get_dataset
from src.adversaries.api import get_adversaries
from src.pkg.device import get_device
from src.pkg.get_loss_fn import get_loss_fn


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", default="config/config.yaml")
args = arg_parser.parse_args()


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


def init_metrics():
    return {"loss": 0.0, "correct": 0, "total": 0}


def evaluate_model(model, test_loader, adversaries, criterion, device, results_prefix, results):
    model.eval()

    clean_key = f"{results_prefix}_clean"
    if clean_key not in results:
        results[clean_key] = init_metrics()

    for adversary in adversaries:
        adv_key = f"{results_prefix}_{adversary.name}"
        if adv_key not in results:
            results[adv_key] = init_metrics()

    for x, y in tqdm(test_loader, total=len(test_loader), desc=f"Testing {results_prefix}"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = x.size(0)

        # clean evaluation
        with torch.no_grad():
            clean_logits = model(x)
            clean_loss = criterion(clean_logits, y)

        update_metrics(
            results[clean_key],
            clean_logits,
            y,
            clean_loss,
            batch_size,
        )

        # adversarial evaluation
        for adversary in adversaries:
            adv_key = f"{results_prefix}_{adversary.name}"

            x_adv = adversary.gen(model, x, y)

            with torch.no_grad():
                adv_logits = model(x_adv)
                adv_loss = criterion(adv_logits, y)

            update_metrics(
                results[adv_key],
                adv_logits,
                y,
                adv_loss,
                batch_size,
            )


def adversarial_train(model, train_loader, adversary, criterion, optimizer, device):
    model.train()

    for x, y in tqdm(train_loader, total=len(train_loader), desc=f"Adv Training {adversary.name}"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_adv = adversary.gen(model, x, y)

        model.train()
        optimizer.zero_grad()

        logits_clean = model(x)
        loss_clean = criterion(logits_clean, y)

        logits_adv = model(x_adv)
        loss_adv = criterion(logits_adv, y)

        loss = 0.8 * loss_clean + 0.2 * loss_adv

        loss.backward()
        optimizer.step()

def main():
    cfg = load_config(args.config)
    device = get_device()

    test_dataset_cfg = cfg.test_dataset
    train_dataset_cfg = cfg.train_dataset

    model = get_model(cfg.model, device).to(device)
    criterion = get_loss_fn("cross_entropy")

    test_dataset = get_dataset(test_dataset_cfg)
    train_dataset = get_dataset(train_dataset_cfg)

    # Берём четверть train_dataset с учётом batch_size
    train_batch_size = train_dataset_cfg.batch_size
    quarter_size = len(train_dataset) // 4

    # Округляем вниз до кратного batch_size
    quarter_size = (quarter_size // train_batch_size) * train_batch_size

    train_dataset = Subset(train_dataset, range(quarter_size))

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
    )

    adversaries = get_adversaries(cfg.attacks)

    results = {}

    # 1. Оценка исходной модели
    evaluate_model(
        model=model,
        test_loader=test_loader,
        adversaries=adversaries,
        criterion=criterion,
        device=device,
        results_prefix="base",
        results=results,
    )

    # 2. Для каждой атаки обучаем отдельную копию модели
    for train_adversary in adversaries:
        trained_model = copy.deepcopy(model).to(device)

        optimizer = torch.optim.Adam(
            trained_model.parameters(),
            lr=1e-5,
        )

        adversarial_train(
            model=trained_model,
            train_loader=train_loader,
            adversary=train_adversary,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # 3. Оцениваем обученную копию
        results_prefix = f"trained_with_{train_adversary.name}"

        evaluate_model(
            model=trained_model,
            test_loader=test_loader,
            adversaries=adversaries,
            criterion=criterion,
            device=device,
            results_prefix=results_prefix,
            results=results,
        )

    print("\nResults:")
    for name, metrics in results.items():
        final = finalize_metrics(metrics)
        print(
            f"{name}: "
            f"loss={final['loss']:.4f}, "
            f"acc={final['acc']:.4f}, "
            f"total={final['total']}"
        )


if __name__ == "__main__":
    print(os.getcwd())
    main()
import argparse
import json
import os
import shutil
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.adversaries.api import get_adversaries
from src.config.conf import load_experiment_config, load_evaluate_config, EvaluationExperimentConfig
from src.db.api import get_dataset
from src.model.api import get_model
from src.pkg import (
    set_seed,
    get_device,
    get_loss_fn,
    init_metrics,
    update_metrics,
    finalize_metrics,
)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", default="config/config.yaml")
args = arg_parser.parse_args()


def save_evaluation_results(
        results,
        cfg: EvaluationExperimentConfig,
        config_path: str,
):
    dataset_name = cfg.test_dataset.name

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    output_root = cfg.evaluation_root

    run_name = f"{cfg.model.name}_{dataset_name}_{timestamp}"
    run_dir = os.path.join(output_root, run_name)

    os.makedirs(run_dir, exist_ok=True)

    results_path = os.path.join(run_dir, "results.json")
    config_copy_path = os.path.join(run_dir, "config.yaml")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    shutil.copyfile(config_path, config_copy_path)

    print(f"\nSaved evaluation results to: {results_path}")
    print(f"Saved config copy to: {config_copy_path}")


def evaluate_clean(model, test_loader, criterion, device):
    model.eval()
    clean_metrics = init_metrics()

    for x, y in tqdm(test_loader, total=len(test_loader), desc="Clean evaluation"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = x.size(0)

        with torch.no_grad():
            logits = model(x)
            loss = criterion(logits, y)

        update_metrics(
            storage=clean_metrics,
            logits=logits,
            y=y,
            loss=loss,
            batch_size=batch_size,
        )

    return finalize_metrics(clean_metrics)


def evaluate_adversarial(model, test_loader, adversary, criterion, device):
    model.eval()
    adv_metrics = init_metrics()

    for x, y in tqdm(
            test_loader,
            total=len(test_loader),
            desc=f"Adversarial evaluation [{adversary.name}]",
    ):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = x.size(0)

        # Здесь no_grad использовать нельзя:
        # FGSM / PGD обычно требуют градиенты по входу.
        x_adv = adversary.gen(model, x, y)

        model.eval()

        with torch.no_grad():
            logits = model(x_adv)
            loss = criterion(logits, y)

        update_metrics(
            storage=adv_metrics,
            logits=logits,
            y=y,
            loss=loss,
            batch_size=batch_size,
        )

    return finalize_metrics(adv_metrics)


def main():
    cfg = load_evaluate_config(args.config)

    device = get_device()

    test_dataset_cfg = cfg.test_dataset

    model = get_model(cfg.model, device).to(device)
    criterion = get_loss_fn(cfg.model.loss_fn)

    test_dataset = get_dataset(test_dataset_cfg)

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_dataset_cfg.batch_size,
        shuffle=False,
        num_workers=test_dataset_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    adversaries = get_adversaries(cfg.attacks)

    print("\nEvaluation started")
    print(f"Device: {device}")
    print(f"Model: {cfg.model.name}")
    print(f"Dataset: {test_dataset_cfg.name}")
    print(f"Test samples: {len(test_dataset)}")
    print()

    clean_result = evaluate_clean(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    results = {
        "clean": clean_result,
    }

    for adversary in adversaries:
        adv_result = evaluate_adversarial(
            model=model,
            test_loader=test_loader,
            adversary=adversary,
            criterion=criterion,
            device=device,
        )

        results[adversary.name] = adv_result

    print("\nResults:")
    for name, result in results.items():
        print(
            f"{name}: "
            f"loss={result['loss']:.4f}, "
            f"acc={result['acc']:.4f}, "
            f"total={result['total']}"
        )

    save_evaluation_results(
        results=results,
        cfg=cfg,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()

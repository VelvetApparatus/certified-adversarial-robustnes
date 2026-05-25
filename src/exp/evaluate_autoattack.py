import argparse
import json
import os
from typing import Any

import pandas as pd

from src.config.autoattack_evaluation import load_autoattack_evaluate_config
from src.db.api import get_dataset
from src.eval.autoattack import evaluate_autoattack
from src.exp.evaluate import (
    append_row_to_aggregate_csv,
    build_run_output_dir,
    save_effective_config,
    to_serializable,
)
from src.model.api import get_model
from src.pkg import InputNormalizer, get_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name")

    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument("--wandb", action="store_true", dest="force_wandb")
    wandb_group.add_argument("--no-wandb", action="store_true", dest="disable_wandb")

    return parser.parse_args()


def init_autoattack_wandb_if_needed(cfg, run_eval_dir: str, config_path: str, timestamp: str):
    if not cfg.wandb.enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging is enabled for AutoAttack evaluation, but the 'wandb' package is not installed."
        ) from exc

    from src.config.secret import WANDB_TOKEN

    if not WANDB_TOKEN:
        raise RuntimeError(
            "W&B logging is enabled for AutoAttack evaluation, but WANDB_TOKEN is not configured."
        )

    run_name = cfg.wandb.run_name or f"autoattack|{cfg.params.method}|{timestamp}"
    tags = list(dict.fromkeys([*cfg.wandb.tags, "evaluation", "autoattack", cfg.params.method]))

    try:
        wandb.login(key=WANDB_TOKEN)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            tags=tags,
            config={
                "run_type": "autoattack_evaluation",
                "method": cfg.params.method,
                "comment": cfg.params.comment,
                "model": cfg.model.name,
                "weights_path": cfg.model.weights_path,
                "dataset": cfg.test_dataset.name,
                "batch_size": cfg.test_dataset.batch_size,
                "loss_fn": cfg.params.loss_fn,
                "autoattack": {
                    "norm": cfg.autoattack.norm,
                    "epsilon": cfg.autoattack.epsilon,
                    "version": cfg.autoattack.version,
                    "attacks_to_run": cfg.autoattack.attacks_to_run,
                    "max_examples": cfg.autoattack.max_examples,
                    "seed": cfg.autoattack.seed,
                },
                "output_dir": run_eval_dir,
                "config_path": config_path,
            },
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize W&B for AutoAttack evaluation. Check WANDB_TOKEN, credentials, and connectivity."
        ) from exc

    return wandb


def log_autoattack_metrics_to_wandb(wandb, row: dict, run_eval_dir: str):
    raw_log_data = {
        "eval/clean_acc": row.get("clean_acc"),
        "eval/autoattack_acc": row.get("autoattack_acc"),
        "eval/autoattack_error": row.get("autoattack_error"),
        "eval/num_examples": row.get("num_examples"),
    }

    log_data = {
        key: value
        for key, value in raw_log_data.items()
        if value is not None and not pd.isna(value)
    }
    wandb.log(log_data)

    summary = {
        "final/clean_acc": row.get("clean_acc"),
        "final/autoattack_acc": row.get("autoattack_acc"),
        "final/autoattack_error": row.get("autoattack_error"),
        "final/num_examples": row.get("num_examples"),
    }

    for key, value in summary.items():
        if value is not None and not pd.isna(value):
            wandb.run.summary[key] = value

    for filename in ("config.yaml", "autoattack_eval.csv", "metrics.json", "autoattack.log"):
        wandb.save(os.path.join(run_eval_dir, filename))


def _serialize_attacks(attacks_to_run: Any) -> str:
    return json.dumps(to_serializable(attacks_to_run), ensure_ascii=False)


def main():
    args = parse_args()
    cfg = load_autoattack_evaluate_config(args.config)

    if args.output_dir:
        cfg.params.evaluation_dir = args.output_dir

    if args.run_name:
        cfg.wandb.run_name = args.run_name

    if args.force_wandb:
        cfg.wandb.enabled = True
    elif args.disable_wandb:
        cfg.wandb.enabled = False

    base_eval_dir = cfg.params.evaluation_dir
    run_suffix = args.run_name or cfg.params.method
    timestamp, run_id, run_eval_dir = build_run_output_dir(base_eval_dir, run_suffix)

    os.makedirs(base_eval_dir, exist_ok=True)
    os.makedirs(run_eval_dir, exist_ok=True)

    config_path = os.path.join(run_eval_dir, "config.yaml")
    run_csv_path = os.path.join(run_eval_dir, "autoattack_eval.csv")
    metrics_json_path = os.path.join(run_eval_dir, "metrics.json")
    log_path = os.path.join(run_eval_dir, "autoattack.log")
    aggregate_csv_path = os.path.join(base_eval_dir, "autoattack_eval_all.csv")

    save_effective_config(cfg, config_path)

    wandb = init_autoattack_wandb_if_needed(
        cfg=cfg,
        run_eval_dir=run_eval_dir,
        config_path=config_path,
        timestamp=timestamp,
    )

    try:
        device = get_device()

        model = get_model(cfg.model, device).to(device)
        if cfg.normalization.enabled:
            model = InputNormalizer(
                model=model,
                std=cfg.normalization.std,
                mean=cfg.normalization.mean,
            )
            model = model.to(device)

        test_dataset = get_dataset(cfg.test_dataset)

        print("==========================================")
        print("\nAutoAttack evaluation started")
        print(f"Device: {device}")
        print(f"Model: {cfg.model.name}")
        print(f"Dataset: {cfg.test_dataset.name}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Run output dir: {run_eval_dir}")
        print("==========================================")

        eval_metrics = evaluate_autoattack(
            model=model,
            eval_dataset=test_dataset,
            device=device,
            batch_size=cfg.test_dataset.batch_size,
            epsilon=cfg.autoattack.epsilon,
            norm=cfg.autoattack.norm,
            version=cfg.autoattack.version,
            attacks_to_run=cfg.autoattack.attacks_to_run,
            max_examples=cfg.autoattack.max_examples,
            seed=cfg.autoattack.seed,
            log_path=log_path,
        )

        row = {
            "timestamp": timestamp,
            "run_id": run_id,
            "run_eval_dir": run_eval_dir,
            "method": cfg.params.method,
            "model": cfg.model.name,
            "dataset": cfg.test_dataset.name,
            "weights_path": cfg.model.weights_path,
            "comment": cfg.params.comment,
            "norm": eval_metrics.get("norm"),
            "epsilon": eval_metrics.get("epsilon"),
            "version": eval_metrics.get("version"),
            "attacks_to_run": _serialize_attacks(eval_metrics.get("attacks_to_run")),
            "max_examples": cfg.autoattack.max_examples,
            "num_examples": eval_metrics.get("num_examples"),
            "num_total_dataset": eval_metrics.get("num_total_dataset"),
            "clean_acc": eval_metrics.get("clean_acc"),
            "autoattack_acc": eval_metrics.get("autoattack_acc"),
            "autoattack_error": eval_metrics.get("autoattack_error"),
            "num_clean_correct": eval_metrics.get("num_clean_correct"),
            "num_robust_correct": eval_metrics.get("num_robust_correct"),
        }

        row_df = pd.DataFrame([to_serializable(row)])
        row_df.to_csv(run_csv_path, index=False)
        append_row_to_aggregate_csv(row_df, aggregate_csv_path)

        metrics_payload = {
            "method": cfg.params.method,
            "model": cfg.model.name,
            "dataset": cfg.test_dataset.name,
            "comment": cfg.params.comment,
            "run_eval_dir": run_eval_dir,
            "autoattack_metrics": eval_metrics,
            "row": row,
        }
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(to_serializable(metrics_payload), f, indent=2, ensure_ascii=False)

        if wandb is not None:
            log_autoattack_metrics_to_wandb(
                wandb=wandb,
                row=row,
                run_eval_dir=run_eval_dir,
            )

        print("AutoAttack reports saved in", run_eval_dir)
        print("Aggregate CSV updated at", aggregate_csv_path)
    finally:
        if wandb is not None:
            wandb.finish()


if __name__ == "__main__":
    main()

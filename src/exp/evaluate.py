import argparse
import datetime
import json
import os
import re
from dataclasses import asdict
from typing import Any

import pandas as pd
import yaml
from pandas.errors import EmptyDataError

from src.certify.table import certify
from src.config.evaluation import load_evaluate_config
from src.db.api import get_dataset
from src.eval.table import evaluate
from src.model.api import get_model
from src.pkg import (
    InputNormalizer,
    get_device,
    get_loss_fn,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name")

    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument("--wandb", action="store_true", dest="force_wandb")
    wandb_group.add_argument("--no-wandb", action="store_true", dest="disable_wandb")

    return parser.parse_args()


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    sanitized = sanitized.strip("._")
    return sanitized or "evaluation"


def build_run_output_dir(base_dir: str, suffix: str) -> tuple[str, str, str]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_suffix = sanitize_name(suffix)
    run_id = f"{timestamp}_{safe_suffix}"
    run_dir = os.path.join(base_dir, run_id)

    counter = 1
    while os.path.exists(run_dir):
        counter += 1
        run_id = f"{timestamp}_{safe_suffix}_{counter}"
        run_dir = os.path.join(base_dir, run_id)

    return timestamp, run_id, run_dir


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_serializable(v) for v in value]

    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]

    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except TypeError:
            pass

    return value


def save_effective_config(cfg, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            to_serializable(asdict(cfg)),
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def is_cert_metric_key(key: str) -> bool:
    return key.startswith("cert_acc_") or key in ("avg_radius", "median_radius")


def flatten_cert_metrics(cert_metrics: dict, preferred_mode: str | None = None) -> dict:
    if "hard" in cert_metrics and "soft" in cert_metrics:
        result = {}

        for mode_name in ("hard", "soft"):
            for key, value in cert_metrics[mode_name].items():
                if not is_cert_metric_key(key):
                    continue
                result[f"{mode_name}_{key}"] = value

        selected_mode = preferred_mode
        if selected_mode in (None, "both") or selected_mode not in cert_metrics:
            selected_mode = "hard"

        for key, value in cert_metrics[selected_mode].items():
            if not is_cert_metric_key(key):
                continue
            result[key] = value

        result["selected_cert_mode"] = selected_mode
        return result

    result = {
        key: value
        for key, value in cert_metrics.items()
        if is_cert_metric_key(key)
    }
    result["selected_cert_mode"] = cert_metrics.get("mode", preferred_mode or "hard")
    return result


def safe_metric(row: dict, key: str) -> float:
    value = row.get(key)
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def compute_robust_composite_score(row: dict) -> float:
    avg_radius = safe_metric(row, "avg_radius")
    avg_radius_normalized = min(avg_radius / 2.25, 1.0)

    return (
        0.15 * safe_metric(row, "clean_acc")
        + 0.30 * safe_metric(row, "pgd_acc")
        + 0.20 * safe_metric(row, "noisy_acc")
        + 0.25 * safe_metric(row, "cert_acc_050")
        + 0.10 * avg_radius_normalized
    )


def init_eval_wandb_if_needed(cfg, run_eval_dir: str, config_path: str, timestamp: str):
    if not cfg.wandb.enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging is enabled for evaluation, but the 'wandb' package is not installed."
        ) from exc

    from src.config.secret import WANDB_TOKEN

    if not WANDB_TOKEN:
        raise RuntimeError(
            "W&B logging is enabled for evaluation, but WANDB_TOKEN is not configured."
        )

    run_name = cfg.wandb.run_name or f"eval|{cfg.params.method}|{timestamp}"
    tags = list(dict.fromkeys([*cfg.wandb.tags, "evaluation", cfg.params.method]))

    try:
        wandb.login(key=WANDB_TOKEN)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            tags=tags,
            config={
                "run_type": "evaluation",
                "method": cfg.params.method,
                "comment": cfg.params.comment,
                "model": cfg.model.name,
                "weights_path": cfg.model.weights_path,
                "dataset": cfg.test_dataset.name,
                "batch_size": cfg.test_dataset.batch_size,
                "loss_fn": cfg.params.loss_fn,
                "sigma": cfg.params.sigma,
                "cert_mode": cfg.params.cert_mode,
                "N0": cfg.params.N0,
                "N": cfg.params.N,
                "alpha": cfg.params.alpha,
                "beta": cfg.params.beta,
                "pgd": {
                    "epsilon": cfg.pgd.epsilon,
                    "alpha": cfg.pgd.alpha,
                    "steps": cfg.pgd.steps,
                    "norm": cfg.pgd.norm,
                    "loss_fn": cfg.pgd.loss_fn,
                },
                "fgsm": {
                    "epsilon": cfg.fgsm.epsilon,
                    "loss_fn": cfg.fgsm.loss_fn,
                },
                "output_dir": run_eval_dir,
                "config_path": config_path,
            },
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize W&B for evaluation. Check WANDB_TOKEN, credentials, and connectivity."
        ) from exc

    return wandb


def log_eval_metrics_to_wandb(wandb, row: dict, cert_metrics: dict, run_eval_dir: str):
    raw_log_data = {
        "eval/clean_loss": row.get("clean_loss"),
        "eval/clean_acc": row.get("clean_acc"),
        "eval/pgd_loss": row.get("pgd_loss"),
        "eval/pgd_acc": row.get("pgd_acc"),
        "eval/fgsm_loss": row.get("fgsm_loss"),
        "eval/fgsm_acc": row.get("fgsm_acc"),
        "eval/noisy_loss": row.get("noisy_loss"),
        "eval/noisy_acc": row.get("noisy_acc"),
        "cert/acc_000": row.get("cert_acc_000"),
        "cert/acc_025": row.get("cert_acc_025"),
        "cert/acc_050": row.get("cert_acc_050"),
        "cert/acc_075": row.get("cert_acc_075"),
        "cert/acc_100": row.get("cert_acc_100"),
        "cert/acc_125": row.get("cert_acc_125"),
        "cert/acc_150": row.get("cert_acc_150"),
        "cert/acc_175": row.get("cert_acc_175"),
        "cert/acc_200": row.get("cert_acc_200"),
        "cert/acc_225": row.get("cert_acc_225"),
        "cert/avg_radius": row.get("avg_radius"),
        "cert/median_radius": row.get("median_radius"),
        "score/robust_composite": row.get("robust_composite_score"),
    }

    if "hard" in cert_metrics and "soft" in cert_metrics:
        for mode_name in ("hard", "soft"):
            metrics = cert_metrics[mode_name]
            raw_log_data[f"cert/{mode_name}/acc_050"] = metrics.get("cert_acc_050")
            raw_log_data[f"cert/{mode_name}/avg_radius"] = metrics.get("avg_radius")
            raw_log_data[f"cert/{mode_name}/median_radius"] = metrics.get("median_radius")

    log_data = {
        key: value
        for key, value in raw_log_data.items()
        if value is not None and not pd.isna(value)
    }
    wandb.log(log_data)

    summary = {
        "final/clean_acc": row.get("clean_acc"),
        "final/pgd_acc": row.get("pgd_acc"),
        "final/fgsm_acc": row.get("fgsm_acc"),
        "final/noisy_acc": row.get("noisy_acc"),
        "final/cert_acc_050": row.get("cert_acc_050"),
        "final/avg_radius": row.get("avg_radius"),
        "final/median_radius": row.get("median_radius"),
        "final/robust_composite_score": row.get("robust_composite_score"),
    }

    for key, value in summary.items():
        if value is not None and not pd.isna(value):
            wandb.run.summary[key] = value

    for filename in ("config.yaml", "eval.csv", "metrics.json"):
        wandb.save(os.path.join(run_eval_dir, filename))


def append_row_to_aggregate_csv(row_df: pd.DataFrame, csv_path: str):
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
        except EmptyDataError:
            old_df = pd.DataFrame()
        combined_df = pd.concat([old_df, row_df], ignore_index=True, sort=False)
    else:
        combined_df = row_df

    combined_df.to_csv(csv_path, index=False)


def main():
    args = parse_args()
    cfg = load_evaluate_config(args.config)

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
    run_csv_path = os.path.join(run_eval_dir, "eval.csv")
    metrics_json_path = os.path.join(run_eval_dir, "metrics.json")
    aggregate_csv_path = os.path.join(base_eval_dir, "eval_all.csv")

    save_effective_config(cfg, config_path)

    wandb = init_eval_wandb_if_needed(
        cfg=cfg,
        run_eval_dir=run_eval_dir,
        config_path=config_path,
        timestamp=timestamp,
    )

    try:
        device = get_device()

        test_dataset_cfg = cfg.test_dataset

        model = get_model(cfg.model, device).to(device)
        if cfg.normalization.enabled:
            model = InputNormalizer(
                model=model,
                std=cfg.normalization.std,
                mean=cfg.normalization.mean,
            )
            model = model.to(device)
        criterion = get_loss_fn(cfg.params.loss_fn)

        test_dataset = get_dataset(test_dataset_cfg)

        print("==========================================")
        print("\nEvaluation started")
        print(f"Device: {device}")
        print(f"Model: {cfg.model.name}")
        print(f"Dataset: {test_dataset_cfg.name}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Run output dir: {run_eval_dir}")
        print("==========================================")

        eval_metrics = evaluate(
            model=model,
            eval_dataset=test_dataset,
            device=device,
            batch_size=test_dataset_cfg.batch_size,
            loss_fn=criterion,
            pgd_conf=cfg.pgd,
            fgsm_conf=cfg.fgsm,
            sigma=cfg.params.sigma,
        )

        cert_num_img = 500

        cert_metrics = certify(
            model=model,
            device=device,
            dataset=test_dataset,
            num_classes=cfg.model.num_classes,
            mode=cfg.params.cert_mode,
            start_img=0,
            num_img=cert_num_img,
            skip=1,
            sigma=cfg.params.sigma,
            N0=cfg.params.N0,
            N=cfg.params.N,
            alpha=cfg.params.alpha,
            batch=test_dataset_cfg.batch_size,
            verbose=True,
            grid=(0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25),
            beta=cfg.params.beta,
        )

        flat_cert_metrics = flatten_cert_metrics(
            cert_metrics=cert_metrics,
            preferred_mode=cfg.params.cert_mode,
        )

        row = {
            "timestamp": timestamp,
            "run_id": run_id,
            "run_eval_dir": run_eval_dir,
            "method": cfg.params.method,
            "model": cfg.model.name,
            "dataset": cfg.test_dataset.name,
            "test_samples": len(test_dataset),
            "comment": cfg.params.comment,
            "fgsm_epsilon": cfg.fgsm.epsilon,
            "pgd_epsilon": cfg.pgd.epsilon,
            "pgd_alpha": cfg.pgd.alpha,
            "pgd_norm": cfg.pgd.norm,
            "clean_loss": eval_metrics.get("clean_loss"),
            "clean_acc": eval_metrics.get("clean_acc"),
            "pgd_loss": eval_metrics.get("pgd_loss"),
            "pgd_acc": eval_metrics.get("pgd_acc"),
            "fgsm_loss": eval_metrics.get("fgsm_loss"),
            "fgsm_acc": eval_metrics.get("fgsm_acc"),
            "noisy_loss": eval_metrics.get("noisy_loss"),
            "noisy_acc": eval_metrics.get("noisy_acc"),
            "cert_mode": cfg.params.cert_mode,
            "sigma": cfg.params.sigma,
            "cert_num_img": cert_num_img,
            "cert_N0": cfg.params.N0,
            "cert_N": cfg.params.N,
            "cert_alpha": cfg.params.alpha,
            "cert_batch": test_dataset_cfg.batch_size,
            **flat_cert_metrics,
        }
        row["robust_composite_score"] = compute_robust_composite_score(row)

        row_df = pd.DataFrame([to_serializable(row)])
        row_df.to_csv(run_csv_path, index=False)
        append_row_to_aggregate_csv(row_df, aggregate_csv_path)

        metrics_payload = {
            "method": cfg.params.method,
            "model": cfg.model.name,
            "dataset": cfg.test_dataset.name,
            "comment": cfg.params.comment,
            "run_eval_dir": run_eval_dir,
            "eval_metrics": eval_metrics,
            "cert_metrics": cert_metrics,
            "row": row,
        }
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(to_serializable(metrics_payload), f, indent=2, ensure_ascii=False)

        if wandb is not None:
            log_eval_metrics_to_wandb(
                wandb=wandb,
                row=row,
                cert_metrics=cert_metrics,
                run_eval_dir=run_eval_dir,
            )

        print("Eval reports saved in", run_eval_dir)
        print("Aggregate CSV updated at", aggregate_csv_path)
    finally:
        if wandb is not None:
            wandb.finish()


if __name__ == "__main__":
    main()

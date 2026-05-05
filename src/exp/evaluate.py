import argparse
import os
import shutil

from src.config.evaluation import load_evaluate_config
from src.db.api import get_dataset
from src.model.api import get_model
import pandas as pd
from src.evaluation.table import evaluate
from src.certify.table import certify
from src.pkg import (
    get_device,
    get_loss_fn, InputNormalizer,
)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config")
args = arg_parser.parse_args()


def main():
    cfg = load_evaluate_config(args.config)
    os.makedirs(cfg.params.evaluation_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(cfg.params.evaluation_dir, f"config-eval-{cfg.params.method}.yaml"))

    csv_path = os.path.join(cfg.params.evaluation_dir, "eval.csv")

    df = None

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

    if df is None or len(df) == 0:
        df = pd.DataFrame(
            columns=[
                # =====================
                # Experiment info
                # =====================
                "method",
                "model",
                "dataset",
                "test_samples",
                "comment",

                # =====================
                # Evaluation config
                # =====================
                "fgsm_epsilon",
                "pgd_epsilon",
                "pgd_alpha",
                "pgd_norm",

                # =====================
                # Evaluation metrics
                # =====================
                "clean_loss",
                "clean_acc",

                "pgd_loss",
                "pgd_acc",

                "fgsm_loss",
                "fgsm_acc",

                "noisy_loss",
                "noisy_acc",

                # =====================
                # Certification config
                # =====================
                "cert_mode",
                "sigma",
                "cert_num_img",
                "cert_N0",
                "cert_N",
                "cert_alpha",
                "cert_batch",

                # =====================
                # Certification metrics
                # =====================
                "cert_acc_000",
                "cert_acc_025",
                "cert_acc_050",
                "cert_acc_075",
                "cert_acc_100",
                "cert_acc_125",
                "cert_acc_150",
                "cert_acc_175",
                "cert_acc_200",
                "cert_acc_225",

                "avg_radius",
                "median_radius",
            ]
        )

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

    cert_metrics = certify(
        model=model,
        device=device,
        dataset=test_dataset,
        num_classes=cfg.model.num_classes,
        mode=cfg.params.cert_mode,
        # todo: to config
        start_img=0,
        num_img=500,
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

    row = {
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

        "cert_mode": cert_metrics.get("mode"),
        "sigma": cert_metrics.get("sigma"),
        "cert_num_img": cert_metrics.get("num_img"),
        "cert_N0": cert_metrics.get("N0"),
        "cert_N": cert_metrics.get("N"),
        "cert_alpha": cert_metrics.get("alpha"),
        "cert_batch": cert_metrics.get("batch"),

        "cert_acc_000": cert_metrics.get("cert_acc_000"),
        "cert_acc_025": cert_metrics.get("cert_acc_025"),
        "cert_acc_050": cert_metrics.get("cert_acc_050"),
        "cert_acc_075": cert_metrics.get("cert_acc_075"),
        "cert_acc_100": cert_metrics.get("cert_acc_100"),
        "cert_acc_125": cert_metrics.get("cert_acc_125"),
        "cert_acc_150": cert_metrics.get("cert_acc_150"),
        "cert_acc_175": cert_metrics.get("cert_acc_175"),
        "cert_acc_200": cert_metrics.get("cert_acc_200"),
        "cert_acc_225": cert_metrics.get("cert_acc_225"),

        "avg_radius": cert_metrics.get("avg_radius"),
        "median_radius": cert_metrics.get("median_radius"),
    }
    df.loc[len(df)] = row

    df.to_csv(csv_path, index=False)

    print("Eval reports saved in", csv_path)


if __name__ == "__main__":
    main()

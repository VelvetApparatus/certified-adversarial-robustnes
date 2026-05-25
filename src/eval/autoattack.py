import warnings

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.pkg import set_seed


def _normalize_autoattack_norm(norm: str) -> str:
    norm_key = norm.strip().lower().replace("_", "")
    if norm_key in ("linf", "inf"):
        return "Linf"
    if norm_key in ("l2", "2"):
        return "L2"
    raise ValueError(f"Unsupported AutoAttack norm: {norm}")


def _validate_autoattack_version(version: str):
    supported_versions = {"standard", "plus", "rand", "custom"}
    if version not in supported_versions:
        supported = ", ".join(sorted(supported_versions))
        raise ValueError(
            f"Unsupported AutoAttack version: {version}. Supported versions: {supported}"
        )


def _collect_examples(eval_dataset, batch_size: int, max_examples: int | None):
    num_total_dataset = len(eval_dataset)
    if num_total_dataset <= 0:
        raise ValueError("Evaluation dataset is empty")

    if max_examples is None:
        num_examples = num_total_dataset
        dataset_to_collect = eval_dataset
    else:
        num_examples = min(num_total_dataset, int(max_examples))
        if num_examples <= 0:
            raise ValueError("autoattack.max_examples must be positive when provided")
        dataset_to_collect = Subset(eval_dataset, range(num_examples))

    loader = DataLoader(
        dataset_to_collect,
        shuffle=False,
        batch_size=batch_size,
    )

    x_batches = []
    y_batches = []
    for x, y in tqdm(loader, desc="collect_autoattack_examples"):
        x_batches.append(x.cpu())
        y_batches.append(y.cpu())

    x_all = torch.cat(x_batches, dim=0)
    y_all = torch.cat(y_batches, dim=0)

    x_min = float(x_all.min().item())
    x_max = float(x_all.max().item())
    if x_min < 0.0 or x_max > 1.0:
        warnings.warn(
            f"AutoAttack expects inputs in [0, 1], got range [{x_min:.6f}, {x_max:.6f}]",
            stacklevel=2,
        )

    return x_all, y_all, num_examples, num_total_dataset


def _compute_accuracy(model, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> tuple[int, float]:
    total = y.size(0)
    correct = 0

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = start + batch_size
            logits = model(x[start:end])
            preds = logits.argmax(dim=1)
            correct += preds.eq(y[start:end]).sum().item()

    return correct, correct / total


def evaluate_autoattack(
    model,
    eval_dataset,
    device,
    batch_size: int,
    epsilon: float,
    norm: str,
    version: str,
    attacks_to_run: list[str] | None,
    max_examples: int | None,
    seed: int,
    log_path: str | None = None,
) -> dict:
    try:
        from autoattack import AutoAttack
    except ImportError as exc:
        raise RuntimeError(
            "AutoAttack is not installed. Install it with: pip install autoattack"
        ) from exc

    aa_norm = _normalize_autoattack_norm(norm)
    version = str(version).strip().lower()
    _validate_autoattack_version(version)

    if version == "custom" and not attacks_to_run:
        raise ValueError("autoattack.attacks_to_run must be provided when version='custom'")

    set_seed(seed)
    model.eval()

    x_all, y_all, num_examples, num_total_dataset = _collect_examples(
        eval_dataset=eval_dataset,
        batch_size=batch_size,
        max_examples=max_examples,
    )

    x_all = x_all.to(device)
    y_all = y_all.to(device)

    clean_correct, clean_acc = _compute_accuracy(
        model=model,
        x=x_all,
        y=y_all,
        batch_size=batch_size,
    )

    if log_path is not None:
        with open(log_path, "a", encoding="utf-8"):
            pass

    adversary = AutoAttack(
        model,
        norm=aa_norm,
        eps=float(epsilon),
        version=version,
        seed=seed,
        device=str(device),
        log_path=log_path,
    )

    if attacks_to_run is not None:
        adversary.attacks_to_run = list(attacks_to_run)

    x_adv = adversary.run_standard_evaluation(
        x_all,
        y_all,
        bs=batch_size,
    )

    robust_correct, autoattack_acc = _compute_accuracy(
        model=model,
        x=x_adv,
        y=y_all,
        batch_size=batch_size,
    )

    resolved_attacks = getattr(adversary, "attacks_to_run", attacks_to_run)

    return {
        "clean_acc": clean_acc,
        "autoattack_acc": autoattack_acc,
        "autoattack_error": 1.0 - autoattack_acc,
        "epsilon": float(epsilon),
        "norm": aa_norm,
        "version": version,
        "attacks_to_run": list(resolved_attacks) if resolved_attacks is not None else [],
        "num_examples": num_examples,
        "num_total_dataset": num_total_dataset,
        "num_clean_correct": int(clean_correct),
        "num_robust_correct": int(robust_correct),
    }

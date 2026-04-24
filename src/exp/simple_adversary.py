import argparse

import torch
from torch.utils.data import DataLoader
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


def main():
    cfg = load_config(args.config)
    device = get_device()

    model = get_model(cfg.model).to(device)
    criterion = get_loss_fn("cross_entropy")

    if cfg.dataset.train:
        raise ValueError("In this experiment dataset must be test")

    dataset = get_dataset(cfg.dataset)

    test_loader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=getattr(cfg.dataset, "num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    adversaries = get_adversaries(cfg.attacks)

    results = {
        "clean": {"loss": 0.0, "correct": 0, "total": 0},
    }
    for adversary in adversaries:
        results[adversary.name] = {"loss": 0.0, "correct": 0, "total": 0}

    model.eval()

    for x, y in tqdm(test_loader, total=len(test_loader), desc="Adversarial Testing"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = x.size(0)

        # clean evaluation
        with torch.no_grad():
            clean_logits = model(x)
            clean_loss = criterion(clean_logits, y)
        update_metrics(results["clean"], clean_logits, y, clean_loss, batch_size)

        # adversarial evaluation
        for adversary in adversaries:
            x_adv = adversary.gen(model, x, y)

            with torch.no_grad():
                adv_logits = model(x_adv)
                adv_loss = criterion(adv_logits, y)

            update_metrics(results[adversary.name], adv_logits, y, adv_loss, batch_size)

    print("\nResults:")
    for name, metrics in results.items():
        final = finalize_metrics(metrics)
        print(f"{name}: loss={final['loss']:.4f}, acc={final['acc']:.4f}, total={final['total']}")


if __name__ == "__main__":
    main()

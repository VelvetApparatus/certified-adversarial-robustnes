from src.robustness.adversaries.fgsm import FGSMAttack
from src.robustness.adversaries.pgd import PGD
from src.config.common import PGDAttackConfig, FGSMAttackConfig
from src.pkg import get_loss_fn

import torch
from tqdm import tqdm
from src.robustness.input.adversarial_training import AdversarialGenerator


def evaluate(
        model,
        eval_dataset,
        device,
        batch_size: int,
        loss_fn,
        pgd_conf: PGDAttackConfig,
        fgsm_conf: FGSMAttackConfig,
        sigma: float,
):
    loader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=batch_size,
    )

    # =====================
    # Initialize PGD attack
    # =====================
    pgd_loss_fn = get_loss_fn(pgd_conf.loss_fn)
    pgd = PGD(
        alpha=pgd_conf.alpha,
        epsilon=pgd_conf.epsilon,
        loss_fn=pgd_loss_fn,
        norm=pgd_conf.norm,
        steps=pgd_conf.steps,
    )

    pgd_adversary = AdversarialGenerator(
        adversary=pgd,
        ratio=1.0,
    )

    # =====================
    # Initialize FGSM attack
    # =====================
    fgsm_loss_fn = get_loss_fn(fgsm_conf.loss_fn)
    fgsm_attack = FGSMAttack(
        eps=fgsm_conf.epsilon,
        loss_fn=fgsm_loss_fn,
    )

    fgsm_adversary = AdversarialGenerator(
        adversary=fgsm_attack,
        ratio=1.0,
    )

    clean_loss = 0.0
    clean_correct = 0

    pgd_loss = 0.0
    pgd_correct = 0
    pgd_samples = 0

    fgsm_loss = 0.0
    fgsm_correct = 0
    fgsm_samples = 0

    noisy_loss = 0.0
    noisy_correct = 0

    total_samples = 0

    model.eval()

    for x, y in tqdm(loader, desc="evaluation"):
        x = x.to(device)
        y = y.to(device)

        current_batch_size = y.size(0)
        total_samples += current_batch_size

        # =====================
        # Clean evaluation
        # =====================
        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits, y)

            preds = logits.argmax(dim=1)
            clean_correct += preds.eq(y).sum().item()
            clean_loss += loss.item() * current_batch_size

        # =====================
        # Noisy evaluation
        # =====================
        with torch.no_grad():
            noise = torch.randn_like(x) * sigma
            x_noisy = x + noise

            logits = model(x_noisy)
            loss = loss_fn(logits, y)

            preds = logits.argmax(dim=1)
            noisy_correct += preds.eq(y).sum().item()
            noisy_loss += loss.item() * current_batch_size

        # =====================
        # PGD evaluation
        # =====================
        model.eval()
        x_pgd_adv, y_pgd_adv = pgd_adversary.augment_on_batch(
            x=x,
            y=y,
            model=model,
        )

        x_pgd_adv = x_pgd_adv.to(device)
        y_pgd_adv = y_pgd_adv.to(device)

        pgd_batch_size = y_pgd_adv.size(0)
        pgd_samples += pgd_batch_size

        with torch.no_grad():
            logits = model(x_pgd_adv)
            loss = loss_fn(logits, y_pgd_adv)

            preds = logits.argmax(dim=1)
            pgd_correct += preds.eq(y_pgd_adv).sum().item()
            pgd_loss += loss.item() * pgd_batch_size

        # =====================
        # FGSM evaluation
        # =====================
        model.eval()
        x_fgsm_adv, y_fgsm_adv = fgsm_adversary.augment_on_batch(
            x=x,
            y=y,
            model=model,
        )

        x_fgsm_adv = x_fgsm_adv.to(device)
        y_fgsm_adv = y_fgsm_adv.to(device)

        fgsm_batch_size = y_fgsm_adv.size(0)
        fgsm_samples += fgsm_batch_size

        with torch.no_grad():
            logits = model(x_fgsm_adv)
            loss = loss_fn(logits, y_fgsm_adv)

            preds = logits.argmax(dim=1)
            fgsm_correct += preds.eq(y_fgsm_adv).sum().item()
            fgsm_loss += loss.item() * fgsm_batch_size

    return {
        "clean_loss": clean_loss / total_samples,
        "clean_acc": clean_correct / total_samples,

        "noisy_loss": noisy_loss / total_samples,
        "noisy_acc": noisy_correct / total_samples,

        "pgd_loss": pgd_loss / pgd_samples,
        "pgd_acc": pgd_correct / pgd_samples,

        "fgsm_loss": fgsm_loss / fgsm_samples,
        "fgsm_acc": fgsm_correct / fgsm_samples,
    }

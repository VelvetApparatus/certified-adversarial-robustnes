from tqdm import tqdm

import torch
import torch.nn.functional as F

from src.robustness.adversaries.common import Adversary


def trades_train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        pgd: Adversary,
        epoch: int,
        beta: float,
):
    """
    One training epoch for TRADES.

    Loss:
        CE(model(x), y) + beta * KL(model(x_adv), model(x))
    """

    model.train()

    total_loss = 0.0
    total_clean_loss = 0.0
    total_robust_loss = 0.0

    total_clean_correct = 0
    total_adv_correct = 0
    total_samples = 0

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch} | TRADES",
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = y.size(0)
        total_samples += batch_size

        x_adv = pgd.gen(model=model, x=x, y=None)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits_clean = model(x)
        logits_adv = model(x_adv)

        loss_clean = criterion(logits_clean, y)

        loss_robust = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean.detach(), dim=1),
            reduction="batchmean",
        )

        loss = loss_clean + beta * loss_robust

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            clean_preds = logits_clean.argmax(dim=1)
            adv_preds = logits_adv.argmax(dim=1)

            total_clean_correct += clean_preds.eq(y).sum().item()
            total_adv_correct += adv_preds.eq(y).sum().item()

            total_loss += loss.item() * batch_size
            total_clean_loss += loss_clean.item() * batch_size
            total_robust_loss += loss_robust.item() * batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            clean_loss=f"{total_clean_loss / total_samples:.4f}",
            robust_loss=f"{total_robust_loss / total_samples:.4f}",
            beta_robust=f"{beta * total_robust_loss / total_samples:.4f}",
            clean_acc=f"{total_clean_correct / total_samples:.4f}",
            adv_acc=f"{total_adv_correct / total_samples:.4f}",
        )

    return {
        "loss": total_loss / total_samples,
        "clean_loss": total_clean_loss / total_samples,
        "robust_loss": total_robust_loss / total_samples,
        "clean_acc": total_clean_correct / total_samples,
        "adv_acc": total_adv_correct / total_samples,
    }

# src/train/macer.py

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm


def _repeat_for_gaussian_samples(
        x: torch.Tensor,
        gauss_samples: int,
) -> torch.Tensor:
    """
    Converts:
        [B, C, H, W]
    into:
        [B * gauss_samples, C, H, W]
    """

    batch_size = x.size(0)

    return (
        x.unsqueeze(1)
        .repeat(1, gauss_samples, 1, 1, 1)
        .view(batch_size * gauss_samples, *x.shape[1:])
    )


def macer_loss(
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        normal: Normal,
        gauss_samples: int,
        sigma: float,
        num_classes: int,
        beta: float,
        gamma: float,
        lbd: float,
        eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """
    MACER loss.

    Loss:
        classification_loss + lambda * robustness_loss

    classification_loss:
        NLL over averaged softmax probabilities from noisy samples.

    robustness_loss:
        hinge penalty based on top-2 class probabilities and inverse Gaussian CDF.
    """

    batch_size = x.size(0)

    # [B, C, H, W] -> [B * K, C, H, W]
    x_repeated = _repeat_for_gaussian_samples(
        x=x,
        gauss_samples=gauss_samples,
    )

    noise = torch.randn_like(x_repeated) * sigma
    x_noisy = x_repeated + noise
    x_noisy = torch.clamp(x_noisy, 0.0, 1.0)

    logits = model(x_noisy)

    logits = logits.view(batch_size, gauss_samples, num_classes)

    # =====================
    # Classification loss
    # =====================
    probs = F.softmax(logits, dim=2).mean(dim=1)
    probs = probs.clamp(eps, 1.0)

    log_probs = torch.log(probs)

    classification_loss = F.nll_loss(
        input=log_probs,
        target=y,
        reduction="sum",
    )

    # =====================
    # Robustness loss
    # =====================
    beta_logits = logits * beta
    beta_probs = F.softmax(beta_logits, dim=2).mean(dim=1)
    beta_probs = beta_probs.clamp(eps, 1.0 - eps)

    top2_scores, top2_indices = torch.topk(
        beta_probs,
        k=2,
        dim=1,
    )

    correct_mask = top2_indices[:, 0].eq(y)

    if correct_mask.sum().item() == 0:
        robustness_loss = torch.zeros(
            (),
            dtype=x.dtype,
            device=x.device,
        )
    else:
        p_a = top2_scores[correct_mask, 0].clamp(eps, 1.0 - eps)
        p_b = top2_scores[correct_mask, 1].clamp(eps, 1.0 - eps)

        margin = normal.icdf(p_a) - normal.icdf(p_b)

        valid_mask = torch.isfinite(margin)

        if valid_mask.sum().item() == 0:
            robustness_loss = torch.zeros(
                (),
                dtype=x.dtype,
                device=x.device,
            )
        else:
            margin = margin[valid_mask]

            robustness_loss = (
                    torch.clamp(gamma - margin, min=0.0).sum()
                    * sigma
                    / 2.0
            )

    loss = classification_loss + lbd * robustness_loss
    loss = loss / batch_size

    with torch.no_grad():
        preds = probs.argmax(dim=1)
        clean_correct = preds.eq(y).sum().item()

    metrics = {
        "loss": loss.detach().item(),
        "classification_loss": (classification_loss / batch_size).detach().item(),
        "robustness_loss": (robustness_loss / batch_size).detach().item(),
        "acc": clean_correct / batch_size,
        "macer_acc": clean_correct / batch_size,
    }

    return loss, metrics


def macer_train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch: int,
        gauss_samples: int,
        sigma: float,
        num_classes: int,
        beta: float,
        gamma: float,
        lbd: float,
        eps: float = 1e-6,
):
    """
    One MACER training epoch.

    This function follows the same interface as other train_epoch_fn methods:
        model, train_loader, criterion, optimizer, device, epoch, **kwargs

    Note:
        criterion is accepted for compatibility with the common train loop,
        but MACER uses its own NLL-style classification term.
    """

    model.train()

    normal = Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
    )

    total_loss = 0.0
    total_classification_loss = 0.0
    total_robustness_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch} | MACER",
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = y.size(0)

        optimizer.zero_grad(set_to_none=True)

        loss, batch_metrics = macer_loss(
            model=model,
            x=x,
            y=y,
            normal=normal,
            gauss_samples=gauss_samples,
            sigma=sigma,
            num_classes=num_classes,
            beta=beta,
            gamma=gamma,
            lbd=lbd,
            eps=eps,
        )

        loss.backward()
        optimizer.step()

        total_samples += batch_size

        total_loss += batch_metrics["loss"] * batch_size
        total_classification_loss += batch_metrics["classification_loss"] * batch_size
        total_robustness_loss += batch_metrics["robustness_loss"] * batch_size
        total_correct += int(batch_metrics["acc"] * batch_size)

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            cl=f"{total_classification_loss / total_samples:.4f}",
            rl=f"{total_robustness_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return {
        "loss": total_loss / total_samples,
        "classification_loss": total_classification_loss / total_samples,
        "robustness_loss": total_robustness_loss / total_samples,
        "acc": total_correct / total_samples,
        "macer_acc": total_correct / total_samples,
    }

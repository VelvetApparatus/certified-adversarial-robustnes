import torch
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm

from src.config.common import MacerTrainingParams, get_scheduled


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
        .reshape(batch_size * gauss_samples, *x.shape[1:])
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
        Hinge penalty based on:
            margin = Phi^{-1}(p_y) - Phi^{-1}(max_{j != y} p_j)

        gamma:
            Target margin in inverse Gaussian CDF space, not radius space.

        certified_radius proxy:
            radius = sigma / 2 * margin
    """

    batch_size = x.size(0)

    # [B, C, H, W] -> [B * K, C, H, W]
    x_repeated = _repeat_for_gaussian_samples(
        x=x,
        gauss_samples=gauss_samples,
    )

    # Noise is added in pixel-space [0, 1].
    # InputNormalizer, if used, should live inside model.
    noise = torch.randn_like(x_repeated) * sigma
    x_noisy = x_repeated + noise
    x_noisy = torch.clamp(x_noisy, 0.0, 1.0)

    logits = model(x_noisy)
    logits = logits.reshape(batch_size, gauss_samples, num_classes)

    # =====================
    # Classification loss
    # =====================
    probs = F.softmax(logits, dim=2).mean(dim=1)
    probs = probs.clamp(min=eps)

    log_probs = torch.log(probs)

    classification_loss = F.nll_loss(
        input=log_probs,
        target=y,
        reduction="sum",
    ) / batch_size

    # Smoothed/noisy prediction from averaged probabilities
    smoothed_preds = probs.argmax(dim=1)
    correct_mask = smoothed_preds.eq(y)

    # =====================
    # Robustness loss
    # =====================
    beta_logits = logits * beta
    beta_probs = F.softmax(beta_logits, dim=2).mean(dim=1)
    beta_probs = beta_probs.clamp(min=eps, max=1.0 - eps)

    # p_y: probability of the true class
    p_y = beta_probs.gather(
        dim=1,
        index=y.reshape(-1, 1),
    ).squeeze(1)

    # p_other: max probability among all incorrect classes
    other_probs = beta_probs.clone()
    other_probs.scatter_(
        dim=1,
        index=y.reshape(-1, 1),
        value=-1.0,
    )

    p_other = other_probs.max(dim=1).values
    p_other = p_other.clamp(min=eps, max=1.0 - eps)

    if correct_mask.any():
        p_a = p_y[correct_mask].clamp(min=eps, max=1.0 - eps)
        p_b = p_other[correct_mask].clamp(min=eps, max=1.0 - eps)

        margin = normal.icdf(p_a) - normal.icdf(p_b)

        valid_mask = torch.isfinite(margin)

        if valid_mask.any():
            margin_valid = margin[valid_mask]

            # gamma is in inverse-CDF-margin space.
            robustness_loss = (
                torch.clamp(gamma - margin_valid, min=0.0).sum()
                * sigma
                / 2.0
            ) / batch_size

            radius_valid = sigma * margin_valid / 2.0
        else:
            robustness_loss = torch.zeros(
                (),
                dtype=x.dtype,
                device=x.device,
            )
            margin_valid = torch.empty(0, dtype=x.dtype, device=x.device)
            radius_valid = torch.empty(0, dtype=x.dtype, device=x.device)
    else:
        robustness_loss = torch.zeros(
            (),
            dtype=x.dtype,
            device=x.device,
        )
        margin_valid = torch.empty(0, dtype=x.dtype, device=x.device)
        radius_valid = torch.empty(0, dtype=x.dtype, device=x.device)

    loss = classification_loss + lbd * robustness_loss

    # =====================
    # Metrics / diagnostics
    # =====================
    with torch.no_grad():
        clean_logits = model(x)
        clean_preds = clean_logits.argmax(dim=1)
        clean_correct = clean_preds.eq(y).sum().item()

        smoothed_correct = correct_mask.sum().item()

        target_radius = sigma * gamma / 2.0

        if margin_valid.numel() > 0:
            margin_mean = margin_valid.mean().item()
            margin_min = margin_valid.min().item()
            margin_max = margin_valid.max().item()

            radius_mean = radius_valid.mean().item()
            radius_min = radius_valid.min().item()
            radius_max = radius_valid.max().item()
        else:
            margin_mean = 0.0
            margin_min = 0.0
            margin_max = 0.0

            radius_mean = 0.0
            radius_min = 0.0
            radius_max = 0.0

        p_y_mean = p_y.mean().item()
        p_other_mean = p_other.mean().item()

    metrics = {
        "loss": loss.detach().item(),

        "classification_loss": classification_loss.detach().item(),
        "robustness_loss": robustness_loss.detach().item(),

        # Clean deterministic accuracy
        "clean_acc": clean_correct / batch_size,

        # Accuracy of averaged noisy probabilities
        "macer_acc": smoothed_correct / batch_size,
        "smoothed_acc": smoothed_correct / batch_size,

        # Fraction of samples that actually received robustness penalty
        "robust_active_frac": smoothed_correct / batch_size,

        # Probability diagnostics
        "p_y_mean": p_y_mean,
        "p_other_mean": p_other_mean,

        # Margin diagnostics
        "margin_mean": margin_mean,
        "margin_min": margin_min,
        "margin_max": margin_max,

        # Radius proxy diagnostics
        "radius_mean": radius_mean,
        "radius_min": radius_min,
        "radius_max": radius_max,
        "target_radius": target_radius,
    }

    return loss, metrics

def macer_train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch: int,
        params: MacerTrainingParams,
):
    model.train()

    normal = Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
    )

    total_loss = 0.0
    total_classification_loss = 0.0
    total_robustness_loss = 0.0

    total_clean_correct = 0.0
    total_macer_correct = 0.0
    total_samples = 0

    total_margin_mean = 0.0
    total_radius_mean = 0.0
    total_robust_active = 0.0

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

        lbd_eff = get_scheduled(params.lbd, params.lbd_scheduler, epoch)
        beta_eff = get_scheduled(params.beta, params.beta_scheduler, epoch)

        loss, batch_metrics = macer_loss(
            model=model,
            x=x,
            y=y,
            normal=normal,
            gauss_samples=params.gauss_samples,
            sigma=params.sigma,
            num_classes=params.num_classes,
            beta=beta_eff,
            gamma=params.gamma,
            lbd=lbd_eff,
            eps=params.eps,
        )

        loss.backward()
        optimizer.step()

        total_samples += batch_size

        total_loss += batch_metrics["loss"] * batch_size
        total_classification_loss += batch_metrics["classification_loss"] * batch_size
        total_robustness_loss += batch_metrics["robustness_loss"] * batch_size

        total_clean_correct += batch_metrics["clean_acc"] * batch_size
        total_macer_correct += batch_metrics["macer_acc"] * batch_size

        total_margin_mean += batch_metrics["margin_mean"] * batch_size
        total_radius_mean += batch_metrics["radius_mean"] * batch_size
        total_robust_active += batch_metrics["robust_active_frac"] * batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            cl=f"{total_classification_loss / total_samples:.4f}",
            rl=f"{total_robustness_loss / total_samples:.4f}",
            clean_acc=f"{total_clean_correct / total_samples:.4f}",
            macer_acc=f"{total_macer_correct / total_samples:.4f}",
            radius=f"{total_radius_mean / total_samples:.4f}",
            active=f"{total_robust_active / total_samples:.4f}",
        )

    return {
        "loss": total_loss / total_samples,
        "classification_loss": total_classification_loss / total_samples,
        "robustness_loss": total_robustness_loss / total_samples,

        "clean_acc": total_clean_correct / total_samples,
        "acc": total_clean_correct / total_samples,

        "macer_acc": total_macer_correct / total_samples,
        "smoothed_acc": total_macer_correct / total_samples,

        "margin_mean": total_margin_mean / total_samples,
        "radius_mean": total_radius_mean / total_samples,
        "robust_active_frac": total_robust_active / total_samples,

        "target_radius": params.sigma * params.gamma / 2.0,
    }
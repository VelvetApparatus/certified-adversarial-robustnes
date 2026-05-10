import torch
import torch.nn.functional as F
from torch.distributions import Normal

from src.config.common import get_scheduled


@torch.no_grad()
def evaluate_clean(
        model,
        loader,
        criterion,
        device,
        **kwargs
) -> dict:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_samples += batch_size

        total_loss += loss.item() * batch_size
        total_correct += logits.argmax(dim=1).eq(y).sum().item()

    return {
        "acc": total_correct / total_samples,
        "loss": total_loss / total_samples,
    }


def evaluate_adversarial(
        model,
        loader,
        adversary,
        criterion,
        device,
        metric_prefix: str,
        **kwargs
) -> dict:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Important: adversary.gen needs gradients, so no torch.no_grad() here.
        x_adv = adversary.gen(model, x, y).detach()

        with torch.no_grad():
            logits = model(x_adv)
            loss = criterion(logits, y)

        batch_size = y.size(0)
        total_samples += batch_size

        total_loss += loss.item() * batch_size
        total_correct += logits.argmax(dim=1).eq(y).sum().item()

    return {
        f"{metric_prefix}_acc": total_correct / total_samples,
        f"{metric_prefix}_loss": total_loss / total_samples,
    }


@torch.no_grad()
def evaluate_noisy(
        model,
        loader,
        criterion,
        device,
        sigma: float,
        samples: int = 3,
        clamp: bool = True,
        **kwargs
) -> dict:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = y.size(0)

        x_rep = (
            x.unsqueeze(1)
            .repeat(1, samples, 1, 1, 1)
            .reshape(batch_size * samples, *x.shape[1:])
        )
        y_rep = y.repeat_interleave(samples)

        noise = torch.randn_like(x_rep) * sigma
        x_noisy = x_rep + noise

        if clamp:
            x_noisy = torch.clamp(x_noisy, 0.0, 1.0)

        logits = model(x_noisy)
        loss = criterion(logits, y_rep)

        total_samples += y_rep.size(0)
        total_loss += loss.item() * y_rep.size(0)
        total_correct += logits.argmax(dim=1).eq(y_rep).sum().item()

    return {
        "noisy_acc": total_correct / total_samples,
        "noisy_loss": total_loss / total_samples,
    }


@torch.no_grad()
def evaluate_smoothed(
        model,
        loader,
        device,
        epoch,
        sigma: float,
        num_classes: int,
        samples: int = 32,
        beta: float = 1.0,
        beta_scheduler=None,
        eps: float = 1e-6,
        clamp: bool = True,
        **kwargs
) -> dict:
    """
    Fast proxy evaluation for smoothed / MACER-style models.

    Important:
    This function does NOT perform rigorous randomized smoothing certification.
    It estimates a differentiable/proxy smoothing quality using Monte-Carlo
    averaging of softmax probabilities.

    Use this function for:
      - monitoring smoothed accuracy during training;
      - approximate MACER radius diagnostics;
      - checkpoint selection via smoothed_acc / macer_acc / macer_score.

    Use separate certification code, e.g. Smooth.certify(...), for final
    certified accuracy and certified radius.

    Returned metrics:
      clean_acc:
          Accuracy of the base classifier on clean inputs.

      smoothed_acc / macer_acc:
          Accuracy of the smoothed classifier based on averaged probabilities.

      margin_mean_correct:
          Mean normal-quantile margin only over correctly smoothed-classified
          examples with finite margin.

      radius_mean_correct:
          Mean proxy radius only over correctly smoothed-classified examples.

      margin_mean_all:
          Mean normal-quantile margin over all examples, assigning 0 to
          incorrectly smoothed-classified examples.

      radius_mean_all:
          Mean proxy radius over all examples, assigning 0 to incorrectly
          smoothed-classified examples.

      radius_coverage:
          Fraction of all examples for which a positive finite proxy radius
          was computed.

      macer_score:
          Simple combined metric. Useful for checkpoint selection when one
          wants both smoothed accuracy and non-trivial radius.
    """
    model.eval()

    normal = Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
    )

    total_clean_correct = 0
    total_smoothed_correct = 0
    total_samples = 0

    total_margin_correct = 0.0
    total_radius_correct = 0.0
    total_valid_correct = 0

    total_margin_all = 0.0
    total_radius_all = 0.0


    beta_eff = get_scheduled(beta, schedule=beta_scheduler, epoch=epoch)


    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = y.size(0)

        clean_logits = model(x)
        clean_preds = clean_logits.argmax(dim=1)
        total_clean_correct += clean_preds.eq(y).sum().item()

        x_rep = (
            x.unsqueeze(1)
            .repeat(1, samples, 1, 1, 1)
            .reshape(batch_size * samples, *x.shape[1:])
        )

        noise = torch.randn_like(x_rep) * sigma
        x_noisy = x_rep + noise

        if clamp:
            x_noisy = torch.clamp(x_noisy, 0.0, 1.0)

        logits = model(x_noisy)
        logits = logits.reshape(batch_size, samples, num_classes)

        # Smoothed prediction. This is the classifier used for smoothed_acc.
        probs = F.softmax(logits, dim=2).mean(dim=1)
        smoothed_preds = probs.argmax(dim=1)

        smoothed_correct = smoothed_preds.eq(y)
        total_smoothed_correct += smoothed_correct.sum().item()

        # Temperature-scaled probabilities for MACER-style proxy radius.
        beta_probs = F.softmax(beta_eff * logits, dim=2).mean(dim=1)
        beta_probs = beta_probs.clamp(min=eps, max=1.0 - eps)

        p_y = beta_probs.gather(1, y.reshape(-1, 1)).squeeze(1)

        other_probs = beta_probs.clone()
        other_probs.scatter_(1, y.reshape(-1, 1), -1.0)
        p_other = other_probs.max(dim=1).values
        p_other = p_other.clamp(min=eps, max=1.0 - eps)

        # For correctly smoothed-classified examples, p_y should be the top
        # class probability. The proxy radius is based on the margin between
        # true class and closest competitor.
        margin = normal.icdf(p_y) - normal.icdf(p_other)
        radius = sigma * margin / 2.0

        # Numerical safety.
        valid = torch.isfinite(margin) & torch.isfinite(radius) & smoothed_correct
        radius = torch.clamp(radius, min=0.0)
        margin = torch.clamp(margin, min=0.0)

        # Metrics over correctly classified smoothed examples only.
        if valid.any():
            valid_margin = margin[valid]
            valid_radius = radius[valid]

            total_margin_correct += valid_margin.sum().item()
            total_radius_correct += valid_radius.sum().item()
            total_valid_correct += valid_radius.numel()

        # Metrics over all examples: non-valid / incorrect examples contribute 0.
        radius_all = torch.where(valid, radius, torch.zeros_like(radius))
        margin_all = torch.where(valid, margin, torch.zeros_like(margin))

        total_radius_all += radius_all.sum().item()
        total_margin_all += margin_all.sum().item()

        total_samples += batch_size

    clean_acc = total_clean_correct / total_samples
    smoothed_acc = total_smoothed_correct / total_samples

    if total_valid_correct > 0:
        margin_mean_correct = total_margin_correct / total_valid_correct
        radius_mean_correct = total_radius_correct / total_valid_correct
    else:
        margin_mean_correct = 0.0
        radius_mean_correct = 0.0

    margin_mean_all = total_margin_all / total_samples
    radius_mean_all = total_radius_all / total_samples
    radius_coverage = total_valid_correct / total_samples

    # Simple checkpoint metric: rewards both correctness and radius.
    # Since radius_mean_all already includes zeros for incorrect examples,
    # it is often a better selection metric than radius_mean_correct.
    macer_score = radius_mean_all

    return {
        "clean_acc": clean_acc,

        # Keep both names for compatibility with your current configs/logging.
        "smoothed_acc": smoothed_acc,
        "macer_acc": smoothed_acc,

        # Proxy metrics over correctly smoothed-classified samples only.
        "margin_mean_correct": margin_mean_correct,
        "radius_mean_correct": radius_mean_correct,

        # Safer proxy metrics over all samples.
        "margin_mean_all": margin_mean_all,
        "radius_mean_all": radius_mean_all,

        # Fraction of samples that are both smoothed-correct and have finite radius.
        "radius_coverage": radius_coverage,

        # Recommended single metric if you want radius-aware checkpointing.
        "macer_score": macer_score,
    }
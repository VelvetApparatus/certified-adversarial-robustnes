import torch
import torch.nn.functional as F
from torch.distributions import Normal


@torch.no_grad()
def evaluate_clean(
        model,
        loader,
        criterion,
        device,
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
        sigma: float,
        num_classes: int,
        samples: int = 32,
        beta: float = 1.0,
        eps: float = 1e-6,
) -> dict:
    model.eval()

    normal = Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
    )

    total_clean_correct = 0
    total_smoothed_correct = 0
    total_samples = 0

    total_margin = 0.0
    total_radius = 0.0
    total_valid_radius_samples = 0

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
        x_noisy = torch.clamp(x_rep + noise, 0.0, 1.0)

        logits = model(x_noisy)
        logits = logits.reshape(batch_size, samples, num_classes)

        probs = F.softmax(logits, dim=2).mean(dim=1)
        smoothed_preds = probs.argmax(dim=1)

        total_smoothed_correct += smoothed_preds.eq(y).sum().item()

        beta_probs = F.softmax(beta * logits, dim=2).mean(dim=1)
        beta_probs = beta_probs.clamp(min=eps, max=1.0 - eps)

        p_y = beta_probs.gather(1, y.reshape(-1, 1)).squeeze(1)

        other_probs = beta_probs.clone()
        other_probs.scatter_(1, y.reshape(-1, 1), -1.0)
        p_other = other_probs.max(dim=1).values.clamp(min=eps, max=1.0 - eps)

        correct_mask = smoothed_preds.eq(y)

        if correct_mask.any():
            p_a = p_y[correct_mask]
            p_b = p_other[correct_mask]

            margin = normal.icdf(p_a) - normal.icdf(p_b)
            valid = torch.isfinite(margin)

            if valid.any():
                margin = margin[valid]
                radius = sigma * margin / 2.0

                total_margin += margin.sum().item()
                total_radius += radius.sum().item()
                total_valid_radius_samples += radius.numel()

        total_samples += batch_size

    if total_valid_radius_samples > 0:
        margin_mean = total_margin / total_valid_radius_samples
        radius_mean = total_radius / total_valid_radius_samples
    else:
        margin_mean = 0.0
        radius_mean = 0.0

    return {
        "clean_acc": total_clean_correct / total_samples,
        "smoothed_acc": total_smoothed_correct / total_samples,
        "macer_acc": total_smoothed_correct / total_samples,
        "margin_mean": margin_mean,
        "radius_mean": radius_mean,
    }

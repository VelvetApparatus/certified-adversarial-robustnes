from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config.common import SmoothAdvTrainingParams, get_scheduled


def _repeat_for_noise_samples(
        x: torch.Tensor,
        num_noise_vec: int,
) -> torch.Tensor:
    batch_size = x.size(0)

    return (
        x.unsqueeze(1)
        .repeat(1, num_noise_vec, 1, 1, 1)
        .view(batch_size * num_noise_vec, *x.shape[1:])
    )


def _l2_normalize(delta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    flat = delta.view(delta.size(0), -1)
    norm = flat.norm(p=2, dim=1).view(-1, 1, 1, 1)

    return delta / (norm + eps)


def _project_l2(delta: torch.Tensor, epsilon: float, eps: float = 1e-12) -> torch.Tensor:
    flat = delta.view(delta.size(0), -1)
    norm = flat.norm(p=2, dim=1).view(-1, 1, 1, 1)

    factor = torch.clamp(epsilon / (norm + eps), max=1.0)

    return delta * factor


def _project_linf(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.clamp(delta, min=-epsilon, max=epsilon)


def generate_smooth_adv_examples(
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: float,
        epsilon: float,
        step_size: float,
        steps: int,
        num_noise_vec: int,
        norm: str = "l2",
        clamp_noisy: bool = True,
) -> torch.Tensor:
    """
    Generate SmoothAdv adversarial examples.

    x is expected to be raw image tensor in [0, 1].
    model is expected to handle normalization internally.
    """

    was_training = model.training
    model.eval()

    norm = norm.lower()

    if norm not in ("l2", "linf", "l_inf"):
        raise ValueError(f"Unsupported SmoothAdv norm: {norm}")

    batch_size = x.size(0)

    # calculate delta for a specific norm
    if norm == "l2":
        delta = torch.randn_like(x)
        delta = _l2_normalize(delta) * torch.empty(
            batch_size,
            1,
            1,
            1,
            device=x.device,
            dtype=x.dtype,
        ).uniform_(0.0, epsilon)
    else:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)

    x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    y_rep = y.repeat_interleave(num_noise_vec)


    # iteratively add noise to input
    for _ in range(steps):
        x_adv.requires_grad_(True)

        x_adv_rep = _repeat_for_noise_samples(
            x=x_adv,
            num_noise_vec=num_noise_vec,
        )

        # add gaussian noise
        noise = torch.randn_like(x_adv_rep) * sigma
        x_noisy_adv = x_adv_rep + noise

        # clamp in pixel space
        if clamp_noisy:
            x_noisy_adv = torch.clamp(x_noisy_adv, 0.0, 1.0)

        logits = model(x_noisy_adv)
        loss = F.cross_entropy(logits, y_rep)

        # calculate gradient for adversarial examples
        grad = torch.autograd.grad(
            loss,
            x_adv,
            retain_graph=False,
            create_graph=False,
        )[0]

        # project to epsilon
        if norm == "l2":
            x_adv = x_adv.detach() + step_size * _l2_normalize(grad.detach())
            delta = x_adv - x
            delta = _project_l2(delta, epsilon)
        else:
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            delta = x_adv - x
            delta = _project_linf(delta, epsilon)

        x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    if was_training:
        model.train()

    return x_adv.detach()


def smooth_adv_loss(
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion,
        sigma: float,
        epsilon: float,
        step_size: float,
        steps: int,
        num_noise_vec: int,
        norm: str = "l2",
        train_multi_noise: bool = True,
        clamp_noisy: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    SmoothAdv loss.

    1. Generate x_adv by attacking noisy copies.
    2. Train on x_adv + Gaussian noise.
    """

    batch_size = x.size(0)

    x_adv = generate_smooth_adv_examples(
        model=model,
        x=x,
        y=y,
        sigma=sigma,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        num_noise_vec=num_noise_vec,
        norm=norm,
        clamp_noisy=clamp_noisy,
    )

    if train_multi_noise:
        x_adv_train = _repeat_for_noise_samples(
            x=x_adv,
            num_noise_vec=num_noise_vec,
        )
        y_train = y.repeat_interleave(num_noise_vec)
    else:
        x_adv_train = x_adv
        y_train = y

    noise = torch.randn_like(x_adv_train) * sigma
    x_noisy_adv = x_adv_train + noise

    # clamp in pixel space
    if clamp_noisy:
        x_noisy_adv = torch.clamp(x_noisy_adv, 0.0, 1.0)

    logits = model(x_noisy_adv)
    loss = criterion(logits, y_train)

    with torch.no_grad():
        clean_logits = model(x)
        clean_preds = clean_logits.argmax(dim=1)
        clean_acc = clean_preds.eq(y).float().mean().item()

        noisy_preds = logits.argmax(dim=1)
        noisy_acc = noisy_preds.eq(y_train).float().mean().item()

        delta = x_adv - x

        if norm == "l2":
            delta_norm = delta.view(batch_size, -1).norm(p=2, dim=1).mean().item()
        else:
            delta_norm = delta.abs().view(batch_size, -1).max(dim=1)[0].mean().item()

    metrics = {
        "loss": loss.detach().item(),
        "clean_acc": clean_acc,
        "smooth_adv_acc": noisy_acc,
        "adv_delta_norm": delta_norm,
        "epsilon": float(epsilon),
        "sigma": float(sigma),
    }

    return loss, metrics


def smooth_adv_train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch: int,
        params: SmoothAdvTrainingParams,
):
    model.train()

    total_loss = 0.0
    total_clean_acc = 0.0
    total_smooth_adv_acc = 0.0
    total_delta_norm = 0.0
    total_samples = 0

    sigma_eff = get_scheduled(params.sigma, params.sigma_scheduler, epoch)
    epsilon_eff = get_scheduled(params.epsilon, params.epsilon_scheduler, epoch)

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch} | SmoothAdv",
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = y.size(0)

        optimizer.zero_grad(set_to_none=True)

        loss, batch_metrics = smooth_adv_loss(
            model=model,
            x=x,
            y=y,
            criterion=criterion,
            sigma=sigma_eff,
            epsilon=epsilon_eff,
            step_size=params.step_size,
            steps=params.steps,
            num_noise_vec=params.num_noise_vec,
            norm=params.norm,
            train_multi_noise=params.train_multi_noise,
            clamp_noisy=params.clamp_noisy,
        )

        loss.backward()
        optimizer.step()

        total_samples += batch_size

        total_loss += batch_metrics["loss"] * batch_size
        total_clean_acc += batch_metrics["clean_acc"] * batch_size
        total_smooth_adv_acc += batch_metrics["smooth_adv_acc"] * batch_size
        total_delta_norm += batch_metrics["adv_delta_norm"] * batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            clean_acc=f"{total_clean_acc / total_samples:.4f}",
            adv_acc=f"{total_smooth_adv_acc / total_samples:.4f}",
            delta=f"{total_delta_norm / total_samples:.4f}",
            sigma=f"{sigma_eff:.4f}",
            eps=f"{epsilon_eff:.4f}",
        )

    return {
        "loss": total_loss / total_samples,
        "clean_acc": total_clean_acc / total_samples,
        "smooth_adv_acc": total_smooth_adv_acc / total_samples,
        "adv_delta_norm": total_delta_norm / total_samples,
        "sigma": sigma_eff,
        "epsilon": epsilon_eff,
    }
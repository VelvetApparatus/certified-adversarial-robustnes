from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config.common import SmoothedAttackConfig, TradesSmoothAdvParams, get_scheduled
from src.robustness.adversaries.api import get_adversary
from src.robustness.adversaries.pgd import PGD


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


def consistency_loss(
        logits_clean,
        logits_adv,
        logits_smooth,
        consistency_type: str,
        detach_clean: bool = True,
):
    if consistency_type == "none":
        return torch.zeros((), device=logits_clean.device, dtype=logits_clean.dtype)

    clean_target = logits_clean.detach() if detach_clean else logits_clean

    if logits_smooth.size(0) != logits_clean.size(0):
        repeat_factor = logits_smooth.size(0) // logits_clean.size(0)
        clean_for_smooth = clean_target.repeat_interleave(repeat_factor, dim=0)
    else:
        clean_for_smooth = clean_target

    if consistency_type == "kl_clean_smooth":
        return F.kl_div(
            F.log_softmax(logits_smooth, dim=1),
            F.softmax(clean_for_smooth, dim=1),
            reduction="batchmean",
        )

    if consistency_type == "kl_clean_adv":
        return F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(clean_target, dim=1),
            reduction="batchmean",
        )

    if consistency_type == "kl_clean_adv_smooth":
        return (
            F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(clean_target, dim=1),
                reduction="batchmean",
            )
            + F.kl_div(
                F.log_softmax(logits_smooth, dim=1),
                F.softmax(clean_for_smooth, dim=1),
                reduction="batchmean",
            )
        )

    if consistency_type == "mse_logits_clean_smooth":
        return F.mse_loss(logits_smooth, clean_for_smooth)

    if consistency_type == "mse_logits_clean_adv":
        return F.mse_loss(logits_adv, clean_target)

    if consistency_type == "mse_logits_clean_adv_smooth":
        return (
            F.mse_loss(logits_adv, clean_target)
            + F.mse_loss(logits_smooth, clean_for_smooth)
        )

    raise ValueError(f"Unknown consistency_type: {consistency_type}")


def train_trades_smooth_adv(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch: int,
        params: TradesSmoothAdvParams,
        train_pgd_cfg,
        smooth_attack_cfg: SmoothedAttackConfig,
):
    model.train()

    if train_pgd_cfg.loss_fn != "kl_divergence":
        raise ValueError("trades_smooth_adv requires train_pgd.loss_fn='kl_divergence'")

    sigma_eff = get_scheduled(params.sigma, params.sigma_scheduler, epoch)
    beta_eff = get_scheduled(params.beta, params.beta_scheduler, epoch)
    lambda_smooth_eff = get_scheduled(
        params.lambda_smooth,
        params.lambda_smooth_scheduler,
        epoch,
    )
    consistency_weight_eff = get_scheduled(
        params.consistency_weight,
        params.consistency_scheduler,
        epoch,
    )

    linf_adversary = get_adversary(train_pgd_cfg, epoch=epoch)
    smooth_adversary = get_adversary(
        smooth_attack_cfg,
        epoch=epoch,
        sigma=sigma_eff,
        num_noise_vec=params.num_noise_vec,
        clamp_noisy=params.clamp_noisy,
    )

    total_loss = 0.0
    total_clean_ce = 0.0
    total_trades_kl = 0.0
    total_smooth_ce = 0.0
    total_consistency_loss = 0.0
    total_clean_acc = 0.0
    total_linf_adv_acc = 0.0
    total_smooth_adv_acc = 0.0
    total_samples = 0

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch} | TRADES + SmoothAdv",
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = y.size(0)

        x_linf = linf_adversary.gen(model=model, x=x, y=y)
        x_linf = torch.clamp(x_linf, 0.0, 1.0)

        x_smooth_adv = smooth_adversary.gen(model=model, x=x, y=y)
        x_smooth_adv = torch.clamp(x_smooth_adv, 0.0, 1.0)

        if params.train_multi_noise:
            x_smooth_train = _repeat_for_noise_samples(
                x=x_smooth_adv,
                num_noise_vec=params.num_noise_vec,
            )
            y_smooth = y.repeat_interleave(params.num_noise_vec)
        else:
            x_smooth_train = x_smooth_adv
            y_smooth = y

        noise = torch.randn_like(x_smooth_train) * sigma_eff
        x_smooth_noisy = x_smooth_train + noise
        if params.clamp_noisy:
            x_smooth_noisy = torch.clamp(x_smooth_noisy, 0.0, 1.0)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits_clean = model(x)
        logits_linf = model(x_linf)
        logits_smooth = model(x_smooth_noisy)

        clean_ce = criterion(logits_clean, y)
        trades_kl = F.kl_div(
            F.log_softmax(logits_linf, dim=1),
            F.softmax(logits_clean.detach(), dim=1),
            reduction="batchmean",
        )
        smooth_ce = criterion(logits_smooth, y_smooth)
        cons_loss = consistency_loss(
            logits_clean=logits_clean,
            logits_adv=logits_linf,
            logits_smooth=logits_smooth,
            consistency_type=params.consistency_type,
            detach_clean=params.consistency_detach_clean,
        )

        loss = (
            clean_ce
            + beta_eff * trades_kl
            + lambda_smooth_eff * smooth_ce
            + consistency_weight_eff * cons_loss
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            clean_acc = logits_clean.argmax(dim=1).eq(y).float().mean().item()
            linf_adv_acc = logits_linf.argmax(dim=1).eq(y).float().mean().item()
            smooth_adv_acc = logits_smooth.argmax(dim=1).eq(y_smooth).float().mean().item()

        total_samples += batch_size
        total_loss += loss.detach().item() * batch_size
        total_clean_ce += clean_ce.detach().item() * batch_size
        total_trades_kl += trades_kl.detach().item() * batch_size
        total_smooth_ce += smooth_ce.detach().item() * batch_size
        total_consistency_loss += cons_loss.detach().item() * batch_size
        total_clean_acc += clean_acc * batch_size
        total_linf_adv_acc += linf_adv_acc * batch_size
        total_smooth_adv_acc += smooth_adv_acc * batch_size

        progress.set_postfix(
            loss=total_loss / total_samples,
            clean_acc=total_clean_acc / total_samples,
            linf_adv_acc=total_linf_adv_acc / total_samples,
            smooth_adv_acc=total_smooth_adv_acc / total_samples,
            cons=total_consistency_loss / total_samples,
            gamma_consistency=consistency_weight_eff,
        )

    return {
        "loss": total_loss / total_samples,
        "clean_ce": total_clean_ce / total_samples,
        "trades_kl": total_trades_kl / total_samples,
        "smooth_ce": total_smooth_ce / total_samples,
        "clean_acc": total_clean_acc / total_samples,
        "linf_adv_acc": total_linf_adv_acc / total_samples,
        "smooth_adv_acc": total_smooth_adv_acc / total_samples,
        "consistency_loss": total_consistency_loss / total_samples,
        "consistency_weight": consistency_weight_eff,
        "consistency_type": params.consistency_type,
        "sigma": sigma_eff,
        "beta": beta_eff,
        "lambda_smooth": lambda_smooth_eff,
        "linf_epsilon": linf_adversary.epsilon,
        "linf_alpha": linf_adversary.alpha,
        "smooth_epsilon": smooth_adversary.epsilon,
        "smooth_alpha": smooth_adversary.alpha,
    }

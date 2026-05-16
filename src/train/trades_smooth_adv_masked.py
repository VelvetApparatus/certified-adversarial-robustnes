from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config.common import SmoothedAttackConfig, TradesSmoothAdvParams, get_scheduled
from src.robustness.adversaries.api import get_adversary
from src.robustness.input.mask import MaskGen
from src.train.trades_smooth_adv import _repeat_for_noise_samples, consistency_loss


def train_trades_smooth_adv_masked(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch: int,
        trades_attack_cfg,
        smooth_attack_cfg: SmoothedAttackConfig,
        params: TradesSmoothAdvParams,
        mask_gen: MaskGen,
        mask_gen_warmup: int,
):
    model.train()

    if trades_attack_cfg.loss_fn != "kl_divergence":
        raise ValueError(
            "trades_smooth_adv_masked requires train_pgd.loss_fn='kl_divergence'"
        )

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

    linf_adversary = get_adversary(trades_attack_cfg, epoch=epoch)
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
    total_masked_channels = 0.0
    total_masked_samples = 0.0
    mask_stat_count = 0

    mask_enabled = epoch >= mask_gen_warmup

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch} | TRADES + SmoothAdv + Consistency + Mask",
    )

    for x_clean, y in progress:
        x_clean = x_clean.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = y.size(0)

        if mask_enabled:
            x_masked, _ = mask_gen.augment_on_batch(
                x=x_clean,
                y=y,
                model=model,
            )
            stats = getattr(mask_gen, "last_stats", {})
            total_masked_channels += stats.get("masked_channel_fraction", 0.0)
            total_masked_samples += stats.get("masked_sample_fraction", 0.0)
            mask_stat_count += 1
        else:
            x_masked = x_clean

        x_linf_adv = linf_adversary.gen(model=model, x=x_masked, y=None)
        x_linf_adv = torch.clamp(x_linf_adv, 0.0, 1.0)

        x_smooth_adv = smooth_adversary.gen(model=model, x=x_masked, y=y)
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

        x_smooth_noisy = x_smooth_train + torch.randn_like(x_smooth_train) * sigma_eff
        if params.clamp_noisy:
            x_smooth_noisy = torch.clamp(x_smooth_noisy, 0.0, 1.0)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits_clean = model(x_clean)
        logits_linf = model(x_linf_adv)
        logits_smooth = model(x_smooth_noisy)

        clean_ce = criterion(logits_clean, y)
        trades_kl = F.kl_div(
            F.log_softmax(logits_linf, dim=1),
            F.softmax(logits_clean.detach(), dim=1),
            reduction="batchmean",
        )
        smooth_ce = criterion(logits_smooth, y_smooth)
        consistency_loss_value = consistency_loss(
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
            + consistency_weight_eff * consistency_loss_value
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
        total_consistency_loss += consistency_loss_value.detach().item() * batch_size
        total_clean_acc += clean_acc * batch_size
        total_linf_adv_acc += linf_adv_acc * batch_size
        total_smooth_adv_acc += smooth_adv_acc * batch_size

        masked_channel_fraction = (
            total_masked_channels / mask_stat_count if mask_stat_count > 0 else 0.0
        )
        masked_sample_fraction = (
            total_masked_samples / mask_stat_count if mask_stat_count > 0 else 0.0
        )

        progress.set_postfix(
            loss=total_loss / total_samples,
            clean_acc=total_clean_acc / total_samples,
            linf_adv_acc=total_linf_adv_acc / total_samples,
            smooth_adv_acc=total_smooth_adv_acc / total_samples,
            beta=beta_eff,
            lambda_smooth=lambda_smooth_eff,
            consistency_weight=consistency_weight_eff,
            mask_ch=masked_channel_fraction,
            mask_img=masked_sample_fraction,
        )

    return {
        "loss": total_loss / total_samples,
        "clean_ce": total_clean_ce / total_samples,
        "trades_kl": total_trades_kl / total_samples,
        "smooth_ce": total_smooth_ce / total_samples,
        "consistency_loss": total_consistency_loss / total_samples,
        "clean_acc": total_clean_acc / total_samples,
        "linf_adv_acc": total_linf_adv_acc / total_samples,
        "smooth_adv_acc": total_smooth_adv_acc / total_samples,
        "beta": beta_eff,
        "lambda_smooth": lambda_smooth_eff,
        "sigma": sigma_eff,
        "consistency_weight": consistency_weight_eff,
        "consistency_type": params.consistency_type,
        "smooth_epsilon": smooth_adversary.epsilon,
        "smooth_alpha": smooth_adversary.alpha,
        "linf_epsilon": linf_adversary.epsilon,
        "linf_alpha": linf_adversary.alpha,
        "mask_enabled": mask_enabled,
        "mask_ratio": mask_gen.ratio,
        "mask_p": mask_gen.p,
        "masked_channel_fraction": (
            total_masked_channels / mask_stat_count if mask_stat_count > 0 else 0.0
        ),
        "masked_sample_fraction": (
            total_masked_samples / mask_stat_count if mask_stat_count > 0 else 0.0
        ),
    }

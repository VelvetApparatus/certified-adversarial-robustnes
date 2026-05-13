import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config.common import AWPParams, PGDAttackConfig, TradesMaskedParams, get_scheduled
from src.robustness.adversaries import get_adversary
from src.robustness.input.mask import MaskGen
from src.robustness.model.awp import TradesAWP


def _accumulate_mask_stats(mask_gen: MaskGen, totals: dict) -> None:
    stats = getattr(mask_gen, "last_stats", {})
    totals["masked_channels"] += stats.get("num_masked_channels", 0)
    totals["total_channels"] += stats.get("num_total_channels", 0)
    totals["masked_samples"] += stats.get("num_masked_samples", 0)
    totals["total_samples"] += stats.get("num_total_samples", 0)


def _fraction(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def train_trades_awp_masked(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch: int,
        mask_gen: MaskGen,
        mask_gen_warmup: int,
        attack_cfg: PGDAttackConfig,
        params: TradesMaskedParams,
        awp_adversary: TradesAWP,
        awp_cfg: AWPParams,
):
    model.train()

    if attack_cfg.loss_fn != "kl_divergence":
        raise ValueError("TRADES-AWP-Masked requires PGD(loss_fn='kl_divergence')")

    total_loss = 0.0
    total_clean_loss = 0.0
    total_robust_loss = 0.0
    total_delta_norm = 0.0

    mask_totals = {
        "masked_channels": 0,
        "total_channels": 0,
        "masked_samples": 0,
        "total_samples": 0,
    }

    total_clean_correct = 0
    total_trades_adv_correct = 0
    total_samples = 0

    sigma_eff = get_scheduled(params.sigma, params.sigma_scheduler, epoch)
    beta_eff = get_scheduled(params.beta, params.beta_scheduler, epoch)

    adversary = get_adversary(
        attack_cfg,
        epoch=epoch,
        sigma=sigma_eff,
    )
    epsilon_eff = adversary.epsilon
    alpha_eff = adversary.alpha

    need_mask = epoch >= mask_gen_warmup
    awp_enabled = awp_adversary is not None
    need_awp = awp_enabled and epoch >= awp_cfg.warmup_steps
    attack_source = "clean" if params.pgd_on_clean else "masked"

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch} | TRADES + AWP + Input Masking | PGD on {attack_source}",
    )

    for x, y in progress:
        x_clean = x.to(device, non_blocking=True)
        y_clean = y.to(device, non_blocking=True)

        batch_size = x_clean.size(0)

        if need_mask and not params.pgd_on_clean:
            x_attack, y_attack = mask_gen.augment_on_batch(
                x=x_clean,
                y=y_clean,
                model=model,
            )
            _accumulate_mask_stats(mask_gen, mask_totals)
        else:
            x_attack = x_clean
            y_attack = y_clean

        x_adv = adversary.gen(
            model=model,
            x=x_attack,
            y=y_attack,
        )
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        if need_mask and params.pgd_on_clean:
            x_train, y_train = mask_gen.augment_on_batch(
                x=x_adv,
                y=y_clean,
                model=model,
            )
            _accumulate_mask_stats(mask_gen, mask_totals)
        else:
            x_train = x_adv
            y_train = y_attack

        weights_diff = None
        if need_awp:
            weights_diff = awp_adversary.calc_awp(
                inputs_adv=x_train.detach(),
                inputs_clean=x_clean.detach(),
                targets=y_clean,
                beta=beta_eff,
            )
            awp_adversary.perturb(weights_diff)

        try:
            model.train()
            optimizer.zero_grad(set_to_none=True)

            logits_clean = model(x_clean)
            logits_train = model(x_train)

            clean_loss = criterion(logits_clean, y_clean)
            robust_loss = F.kl_div(
                F.log_softmax(logits_train, dim=1),
                F.softmax(logits_clean.detach(), dim=1),
                reduction="batchmean",
            )
            loss = clean_loss + beta_eff * robust_loss
            loss.backward()
        finally:
            if weights_diff is not None:
                awp_adversary.restore(weights_diff)

        optimizer.step()

        with torch.no_grad():
            clean_pred = logits_clean.argmax(dim=1)
            trades_adv_pred = logits_train.argmax(dim=1)

            clean_correct = clean_pred.eq(y_clean).sum().item()
            trades_adv_correct = trades_adv_pred.eq(y_train).sum().item()

            delta_base = x_clean if params.pgd_on_clean else x_attack
            delta = x_adv - delta_base
            if params.norm == "l2":
                delta_norm = delta.view(batch_size, -1).norm(p=2, dim=1).mean().item()
            else:
                delta_norm = delta.abs().view(batch_size, -1).max(dim=1)[0].mean().item()

        total_samples += batch_size
        total_loss += loss.detach().item() * batch_size
        total_clean_loss += clean_loss.detach().item() * batch_size
        total_robust_loss += robust_loss.detach().item() * batch_size
        total_clean_correct += clean_correct
        total_trades_adv_correct += trades_adv_correct
        total_delta_norm += delta_norm * batch_size

        masked_channel_fraction = _fraction(
            mask_totals["masked_channels"],
            mask_totals["total_channels"],
        )
        masked_sample_fraction = _fraction(
            mask_totals["masked_samples"],
            mask_totals["total_samples"],
        )

        progress.set_postfix(
            loss=total_loss / total_samples,
            clean_acc=total_clean_correct / total_samples,
            trades_adv_acc=total_trades_adv_correct / total_samples,
            beta=beta_eff,
            delta_norm=total_delta_norm / total_samples,
            mask_ch=masked_channel_fraction,
            mask_img=masked_sample_fraction,
            awp_active=int(need_awp),
        )

    return {
        "loss": total_loss / total_samples,
        "clean_loss": total_clean_loss / total_samples,
        "robust_loss": total_robust_loss / total_samples,
        "clean_acc": total_clean_correct / total_samples,
        "trades_adv_acc": total_trades_adv_correct / total_samples,
        "adv_delta_norm": total_delta_norm / total_samples,
        "sigma": sigma_eff,
        "beta": beta_eff,
        "epsilon": epsilon_eff,
        "alpha": alpha_eff,
        "pgd_on_clean": params.pgd_on_clean,
        "masking_enabled": need_mask,
        "mask_ratio": mask_gen.ratio,
        "mask_p": mask_gen.p,
        "masked_channel_fraction": _fraction(
            mask_totals["masked_channels"],
            mask_totals["total_channels"],
        ),
        "masked_sample_fraction": _fraction(
            mask_totals["masked_samples"],
            mask_totals["total_samples"],
        ),
        "awp_enabled": awp_enabled,
        "awp_active": need_awp,
        "awp_gamma": awp_adversary.wcoef if awp_enabled else 0.0,
        "awp_weights_diff_coef": awp_cfg.weights_diff_coef,
        "awp_weights_epsilon": awp_cfg.weights_epsilon,
        "awp_warmup_steps": awp_cfg.warmup_steps,
    }

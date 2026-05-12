import torch
from tqdm import tqdm

from src.robustness.adversaries import get_adversary
import torch.nn.functional as F
from src.robustness.input.mask import MaskGen
from src.config.common import PGDAttackConfig, get_scheduled, TradesMaskedParams


def train_trades_masked(
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
):
    model.train()

    if attack_cfg.loss_fn != "kl_divergence":
        raise ValueError("TRADES-Masked requires PGD(loss_fn='kl_divergence')")

    total_loss = 0.0
    total_clean_loss = 0.0
    total_robust_loss = 0.0
    total_delta_norm = 0.0

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

    attack_source = "clean" if params.pgd_on_clean else "masked"
    need_mask = epoch >= mask_gen_warmup

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch} | TRADES + Input Masking | PGD on {attack_source}",
    )

    for x, y in progress:
        x_clean = x.to(device, non_blocking=True)
        y_clean = y.to(device, non_blocking=True)

        batch_size = x_clean.size(0)

        # Choose the input from which PGD/TRADES adversarial examples are generated.
        # pgd_on_clean=True:
        #   clean x -> PGD -> mask -> TRADES robust loss
        # pgd_on_clean=False:
        #   clean x -> mask -> PGD -> TRADES robust loss
        if need_mask and not params.pgd_on_clean:
            x_attack, y_attack = mask_gen.augment_on_batch(
                x=x_clean,
                y=y_clean,
                model=model,
            )
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
        else:
            x_train = x_adv
            y_train = y_attack

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

        progress.set_postfix(
            loss=total_loss / total_samples,
            clean_acc=total_clean_correct / total_samples,
            trades_adv_acc=total_trades_adv_correct / total_samples,
            delta_norm=total_delta_norm / total_samples,
            beta=beta_eff,
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
    }

import torch
from tqdm import tqdm

from src.config.common import SmoothedAttackConfig, get_scheduled, SmoothAdvTrainingParams, SmoothMaskedTrainingParams
from src.robustness.adversaries import get_adversary
from src.robustness.input.gaussian import GaussianNoiseGenerator
from src.robustness.input.mask import MaskGen


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


def train_smooth_adv_masked(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch: int,
        mask_gen: MaskGen,
        mask_warmup: int,
        attack_cfg: SmoothedAttackConfig,
        params: SmoothMaskedTrainingParams,
        pgd_on_clean: bool = True,

):
    model.train()

    total_loss = 0.0
    total_clean_acc = 0.0
    total_smooth_adv_acc = 0.0
    total_delta_norm = 0.0
    total_samples = 0

    sigma_eff = get_scheduled(params.sigma, params.sigma_scheduler, epoch)
    adversary = get_adversary(
        attack_cfg,
        epoch=epoch,
        sigma=sigma_eff,
        num_noise_vec=params.num_noise_vec,
        clamp_noisy=params.clamp_noisy,
    )
    epsilon_eff = adversary.epsilon
    alpha_eff = adversary.alpha

    noise_gen = GaussianNoiseGenerator(
        sigma=sigma_eff,
        ratio=1.0
    )

    attack_source = "clean" if params.pgd_on_clean else "masked"

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch} | SmoothAdv + Input Masking | PGD on {attack_source}",
    )

    need_masking = epoch >= mask_warmup

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_clean = x
        y_clean = y
        batch_size = y_clean.size(0)

        if need_masking and not pgd_on_clean:
            x_attack, y_attack = mask_gen.augment_on_batch(
                x=x_clean,
                y=y_clean,
                model=model,
            )
        else:
            x_attack = x_clean
            y_attack = y_clean

        x_adv = adversary.gen(model, x_attack, y_attack)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        if need_masking and pgd_on_clean:
            x_train, y_train_base = mask_gen.augment_on_batch(
                x=x_adv,
                y=y_clean,
                model=model,
            )
        else:
            x_train = x_adv
            y_train_base = y_attack

        if params.train_multi_noise:
            x_train = _repeat_for_noise_samples(
                x=x_train,
                num_noise_vec=params.num_noise_vec,
            )
            y_train = y_train_base.repeat_interleave(params.num_noise_vec)
        else:
            y_train = y_train_base

        x_noisy_adv, y_train = noise_gen.augment_on_batch(
            x_train,
            y_train,
            model,
        )

        if params.clamp_noisy:
            x_noisy_adv = torch.clamp(x_noisy_adv, 0.0, 1.0)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x_noisy_adv)
        loss = criterion(logits, y_train)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            clean_logits = model(x_clean)
            clean_acc = clean_logits.argmax(dim=1).eq(y_clean).float().mean().item()

            noisy_acc = logits.argmax(dim=1).eq(y_train).float().mean().item()

            delta_base = x_clean if pgd_on_clean else x_attack
            delta = x_adv - delta_base
            if params.norm == "l2":
                delta_norm = delta.view(batch_size, -1).norm(p=2, dim=1).mean().item()
            else:
                delta_norm = delta.abs().view(batch_size, -1).max(dim=1)[0].mean().item()

        total_samples += batch_size
        total_loss += loss.detach().item() * batch_size
        total_clean_acc += clean_acc * batch_size
        total_smooth_adv_acc += noisy_acc * batch_size
        total_delta_norm += delta_norm * batch_size
    return {
        "loss": total_loss / total_samples,
        "clean_acc": total_clean_acc / total_samples,
        "smooth_adv_acc": total_smooth_adv_acc / total_samples,
        "adv_delta_norm": total_delta_norm / total_samples,
        "sigma": sigma_eff,
        "epsilon": epsilon_eff,
        "alpha": alpha_eff,
        "pgd_on_clean": pgd_on_clean,
    }

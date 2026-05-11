import torch
from tqdm import tqdm

from src.config.common import SmoothAdvTrainingParams, get_scheduled
from src.robustness.input.gaussian import GaussianNoiseGenerator
from src.robustness.adversaries.pgd import SmoothPGD
from src.robustness.model.awp import AWPCrossEntropy


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


def train_smoothed_awp(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch: int,
        awp: AWPCrossEntropy,
        beta: float,
        awp_warmup: int,
        adversary: SmoothPGD,
        params: SmoothAdvTrainingParams,
        **kwargs
):
    model.train()

    total_loss = 0.0
    total_clean_acc = 0.0
    total_smooth_adv_acc = 0.0
    total_delta_norm = 0.0
    total_samples = 0

    sigma_eff = get_scheduled(params.sigma, params.sigma_scheduler, epoch)
    epsilon_eff = get_scheduled(params.epsilon, params.epsilon_scheduler, epoch)
    adversary.alpha = epsilon_eff / 4

    noise_gen = GaussianNoiseGenerator(
        sigma=sigma_eff,
        ratio=1.0
    )

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch} | SmoothAdv + AWP + PGD",
    )

    need_awp = epoch >= awp_warmup
    weights_diff = None

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = y.size(0)

        if hasattr(adversary, "epsilon"):
            adversary.epsilon = epsilon_eff

        x_adv = adversary.gen(model, x, y)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        if params.train_multi_noise:
            x_adv_train = _repeat_for_noise_samples(
                x=x_adv,
                num_noise_vec=params.num_noise_vec,
            )
            y_train = y.repeat_interleave(params.num_noise_vec)
        else:
            x_adv_train = x_adv
            y_train = y

        x_noisy_adv, y_train = noise_gen.augment_on_batch(
            x_adv_train,
            y_train,
            model,
        )

        if params.clamp_noisy:
            x_noisy_adv = torch.clamp(x_noisy_adv, 0.0, 1.0)

        weights_diff = None
        if need_awp:
            weights_diff = awp.calc_awp(
                inputs_adv=x_noisy_adv.detach(),
                targets=y_train,
                beta=beta,
            )
            awp.perturb(weights_diff)

        try:
            optimizer.zero_grad(set_to_none=True)

            logits = model(x_noisy_adv)
            loss = criterion(logits, y_train)

            loss.backward()

        finally:
            if weights_diff is not None:
                awp.restore(weights_diff)

        optimizer.step()

        with torch.no_grad():
            clean_logits = model(x)
            clean_acc = clean_logits.argmax(dim=1).eq(y).float().mean().item()

            noisy_acc = logits.argmax(dim=1).eq(y_train).float().mean().item()

            delta = x_adv - x
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
    }

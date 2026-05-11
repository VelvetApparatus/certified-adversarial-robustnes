import torch
from tqdm import tqdm
from src.config.common import PGDAttackConfig
from src.robustness.adversaries.api import get_adversary
from src.robustness.model.awp import TradesAWP
import torch.nn.functional as F


def trades_awp_train(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        awp: TradesAWP,
        awp_warmup: int,
        pgd_cfg: PGDAttackConfig,
        epoch: int,
        beta: float,
        **kwargs

):
    model.train()
    adversary = get_adversary(pgd_cfg, epoch=epoch)

    total_loss = 0.0
    total_clean_loss = 0.0
    total_robust_loss = 0.0

    total_clean_correct = 0
    total_adv_correct = 0
    total_samples = 0

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc="Train Epoch {}".format(epoch),
    )

    need_awp = epoch > awp_warmup
    weights_diff = None

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = x.size(0)
        total_samples += batch_size

        # not need to provide y with TRADES method
        x_adv = adversary.gen(model=model, x=x, y=None)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        model.train()

        if need_awp:
            weights_diff = awp.calc_awp(
                inputs_adv=x_adv,
                inputs_clean=x,
                targets=y,
                beta=beta,
            )

            awp.perturb(weights_diff)

        optimizer.zero_grad(set_to_none=True)
        logits_clean = model(x)
        logits_adv = model(x_adv)

        # robustness_loss
        loss_robust = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean.detach(), dim=1),
            reduction="batchmean",
        )

        # natural Loss
        cl_loss = criterion(logits_clean, y)

        loss = cl_loss + beta * loss_robust

        loss.backward()

        if need_awp and weights_diff is not None:
            awp.restore(weights_diff)

        optimizer.step()

        with torch.no_grad():
            clean_preds = logits_clean.argmax(dim=1)
            adv_preds = logits_adv.argmax(dim=1)

            total_clean_correct += clean_preds.eq(y).sum().item()
            total_adv_correct += adv_preds.eq(y).sum().item()

            total_loss += loss.item() * batch_size
            total_clean_loss += cl_loss.item() * batch_size
            total_robust_loss += loss_robust.item() * batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            clean_loss=f"{total_clean_loss / total_samples:.4f}",
            robust_loss=f"{total_robust_loss / total_samples:.4f}",
            beta_robust=f"{beta * total_robust_loss / total_samples:.4f}",
            clean_acc=f"{total_clean_correct / total_samples:.4f}",
            adv_acc=f"{total_adv_correct / total_samples:.4f}",
        )

    return {
        "loss": total_loss / total_samples,
        "clean_loss": total_clean_loss / total_samples,
        "robust_loss": total_robust_loss / total_samples,
        "clean_acc": total_clean_correct / total_samples,
        "adv_acc": total_adv_correct / total_samples,
    }

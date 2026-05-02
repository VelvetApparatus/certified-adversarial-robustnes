from tqdm import tqdm

import torch
import torch.nn.functional as F


def _l2_normalize(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    flat = tensor.view(tensor.size(0), -1)
    norm = flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
    return tensor / (norm + eps)


def _project_l2(delta: torch.Tensor, epsilon: float, eps: float = 1e-12) -> torch.Tensor:
    flat = delta.view(delta.size(0), -1)
    norm = flat.norm(p=2, dim=1).view(-1, 1, 1, 1)

    factor = torch.clamp(epsilon / (norm + eps), max=1.0)
    return delta * factor


def generate_trades_adversarial_examples(
        model,
        x,
        step_size: float,
        epsilon: float,
        perturb_steps: int,
        distance: str = "l_inf",
):
    model.eval()

    distance = distance.lower()

    if distance in ("l_inf", "linf", "inf"):
        x_adv = x.detach() + 0.001 * torch.randn_like(x)

    elif distance in ("l2", "l_2"):
        delta = torch.randn_like(x)
        delta = _l2_normalize(delta) * 0.001
        x_adv = x.detach() + delta

    else:
        raise ValueError(f"Unsupported TRADES distance: {distance}")

    for _ in range(perturb_steps):
        x_adv.requires_grad_()

        with torch.enable_grad():
            logits_clean = model(x)
            logits_adv = model(x_adv)

            loss_kl = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean.detach(), dim=1),
                reduction="batchmean",
            )

        grad = torch.autograd.grad(
            loss_kl,
            [x_adv],
            retain_graph=False,
            create_graph=False,
        )[0]

        if distance in ("l_inf", "linf", "inf"):
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)

        elif distance in ("l2", "l_2"):
            grad_normalized = _l2_normalize(grad.detach())
            x_adv = x_adv.detach() + step_size * grad_normalized

            delta = x_adv - x
            delta = _project_l2(delta, epsilon)
            x_adv = x + delta

    return x_adv.detach()


def trades_train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch: int,
        step_size: float,
        epsilon: float,
        perturb_steps: int,
        beta: float,
        distance: str = "l_inf",
):
    """
    One training epoch for TRADES.

    Loss:
        CE(model(x), y) + beta * KL(model(x_adv), model(x))
    """

    model.train()

    total_loss = 0.0
    total_clean_loss = 0.0
    total_robust_loss = 0.0

    total_clean_correct = 0
    total_adv_correct = 0
    total_samples = 0

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch} | TRADES",
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = y.size(0)
        total_samples += batch_size

        x_adv = generate_trades_adversarial_examples(
            model=model,
            x=x,
            step_size=step_size,
            epsilon=epsilon,
            perturb_steps=perturb_steps,
            distance=distance,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits_clean = model(x)
        logits_adv = model(x_adv)

        loss_clean = criterion(logits_clean, y)

        loss_robust = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean.detach(), dim=1),
            reduction="batchmean",
        )

        loss = loss_clean + beta * loss_robust

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            clean_preds = logits_clean.argmax(dim=1)
            adv_preds = logits_adv.argmax(dim=1)

            total_clean_correct += clean_preds.eq(y).sum().item()
            total_adv_correct += adv_preds.eq(y).sum().item()

            total_loss += loss.item() * batch_size
            total_clean_loss += loss_clean.item() * batch_size
            total_robust_loss += loss_robust.item() * batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
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

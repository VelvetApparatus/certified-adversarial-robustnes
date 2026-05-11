import torch
from tqdm import tqdm

from src.robustness.input.gaussian import GaussianNoiseGenerator


def gaussian_train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch: int,
        sigma: float,
        clean_loss_weight: float = 0.0,
        noisy_loss_weight: float = 1.0,
        noise_ratio: float = 1.0,
):
    model.train()

    noise_generator = GaussianNoiseGenerator(
        sigma=sigma,
        ratio=noise_ratio,
    )

    total_loss = 0.0
    total_noisy_loss = 0.0
    total_clean_loss = 0.0

    total_noisy_correct = 0
    total_clean_correct = 0
    total_samples = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1} | Gaussian Training",
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = y.size(0)
        total_samples += batch_size

        x_noisy, y_noisy = noise_generator.augment_on_batch(
            x=x,
            y=y,
            model=model,
        )

        optimizer.zero_grad(set_to_none=True)

        loss = 0.0

        if clean_loss_weight > 0.0:
            logits_clean = model(x)
            loss_clean = criterion(logits_clean, y)
            loss = loss + clean_loss_weight * loss_clean
        else:
            logits_clean = None
            loss_clean = None

        logits_noisy = model(x_noisy)
        loss_noisy = criterion(logits_noisy, y_noisy)
        loss = loss + noisy_loss_weight * loss_noisy

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item() * batch_size
            total_noisy_loss += loss_noisy.item() * batch_size

            noisy_preds = logits_noisy.argmax(dim=1)
            total_noisy_correct += noisy_preds.eq(y_noisy).sum().item()

            if logits_clean is not None:
                total_clean_loss += loss_clean.item() * batch_size
                clean_preds = logits_clean.argmax(dim=1)
                total_clean_correct += clean_preds.eq(y).sum().item()

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            noisy_acc=f"{total_noisy_correct / total_samples:.4f}",
        )

    metrics = {
        "loss": total_loss / total_samples,
        "noisy_loss": total_noisy_loss / total_samples,
        "noisy_acc": total_noisy_correct / total_samples,
    }

    if clean_loss_weight > 0.0:
        metrics["clean_loss"] = total_clean_loss / total_samples
        metrics["clean_acc"] = total_clean_correct / total_samples

    return metrics

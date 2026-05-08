from tqdm import tqdm
import torch

from src.config.adversarial_training import AdversarialTrainingConfig


def adversarial_train_one_epoch(
        model,
        train_loader,
        adversary,
        criterion,
        optimizer,
        device,
        epoch: int,
        adversarial_config: AdversarialTrainingConfig,
        **kwargs,
):
    model.train()

    clean_loss_weight = getattr(adversarial_config.training, "clean_loss_weight", 0.0)
    adv_loss_weight = getattr(adversarial_config.training, "adv_loss_weight", 1.0)

    total_loss = 0.0
    total_clean_loss = 0.0
    total_adv_loss = 0.0

    total_clean_correct = 0
    total_adv_correct = 0
    total_samples = 0

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch + 1} | Adv training [{adversary.name}]",
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = x.size(0)

        # Generate adversarial examples.
        # adversary.gen should not leave model in eval mode.
        x_adv = adversary.gen(model, x, y).detach()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits_clean = model(x)
        loss_clean = criterion(logits_clean, y)

        logits_adv = model(x_adv)
        loss_adv = criterion(logits_adv, y)

        loss = clean_loss_weight * loss_clean + adv_loss_weight * loss_adv

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            clean_preds = logits_clean.argmax(dim=1)
            adv_preds = logits_adv.argmax(dim=1)

            clean_correct = clean_preds.eq(y).sum().item()
            adv_correct = adv_preds.eq(y).sum().item()

        total_samples += batch_size

        total_loss += loss.detach().item() * batch_size
        total_clean_loss += loss_clean.detach().item() * batch_size
        total_adv_loss += loss_adv.detach().item() * batch_size

        total_clean_correct += clean_correct
        total_adv_correct += adv_correct

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            clean_loss=f"{total_clean_loss / total_samples:.4f}",
            adv_loss=f"{total_adv_loss / total_samples:.4f}",
            clean_acc=f"{total_clean_correct / total_samples:.4f}",
            adv_acc=f"{total_adv_correct / total_samples:.4f}",
        )

    clean_acc = total_clean_correct / total_samples
    adv_acc = total_adv_correct / total_samples

    return {
        "loss": total_loss / total_samples,
        "clean_loss": total_clean_loss / total_samples,
        "adv_loss": total_adv_loss / total_samples,

        "clean_acc": clean_acc,
        "adv_acc": adv_acc,

        # compatibility: for adversarial training, acc means adv_acc
        "acc": adv_acc,
    }

from tqdm import tqdm

from src.pkg import update_metrics, finalize_metrics, init_metrics


def adversarial_train_one_epoch(
        model,
        train_loader,
        adversary,
        criterion,
        optimizer,
        device,
        epoch: int,
        cfg,
):
    model.train()
    metrics = init_metrics()

    clean_loss_weight = getattr(cfg.training, "clean_loss_weight", 0.8)
    adv_loss_weight = getattr(cfg.training, "adv_loss_weight", 0.2)

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch {epoch + 1} | Adv training [{adversary.name}]",
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = x.size(0)
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

        # Для train accuracy логируем accuracy на adversarial examples.
        update_metrics(
            storage=metrics,
            logits=logits_adv,
            y=y,
            loss=loss,
            batch_size=batch_size,
        )

        current = finalize_metrics(metrics)
        progress.set_postfix(
            loss=f"{current['loss']:.4f}",
            acc=f"{current['acc']:.4f}",
        )

    return finalize_metrics(metrics)

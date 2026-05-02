from src.config.common import TradesParams
from src.pkg import init_metrics, update_metrics
from src.robustness.trades import trades_loss


def train_trades_one_epoch(
        cfg: TradesParams,
        model,
        train_loader,
        optimizer,
        device,
):
    total_loss = 0.0
    total_acc = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        train_loss = trades_loss(
            model=model,
            x_natural=data,
            y=target,
            step_size=cfg.step_size,
            epsilon=cfg.epsilon,
            perturb_steps=10,
            beta=cfg.beta,
            distance="l_2"
        )

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_loss += train_loss.item()
        total_acc += train_loss.item()

    return {
        "loss": total_loss / len(train_loader),
        "acc": total_acc / len(train_loader),
    }


def init_metrics():
    return {
        "loss": 0.0,
        "correct": 0,
        "total": 0,
    }


def update_metrics(storage, logits, y, loss, batch_size):
    preds = logits.argmax(dim=1)

    storage["loss"] += loss.item() * batch_size
    storage["correct"] += (preds == y).sum().item()
    storage["total"] += batch_size


def finalize_metrics(storage):
    total = max(storage["total"], 1)

    return {
        "loss": storage["loss"] / total,
        "acc": storage["correct"] / total,
        "total": storage["total"],
    }


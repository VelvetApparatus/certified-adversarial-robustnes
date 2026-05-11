import argparse
import copy

from src.config.awp import load_awp_config
from src.eval.validation import evaluate_adversarial
from src.model.api import get_model
from src.pkg import *
from src.robustness.adversaries.pgd import PGD
from src.robustness.model.awp import TradesAWP
from src.train.trades_awp import trades_awp_train
from src.train.common import train

parser = argparse.ArgumentParser(description='PyTorch AWP+TRADES Adversarial Training')
parser.add_argument("--config")
args = parser.parse_args()


def main():
    config = load_awp_config(args.config)

    device = get_device()
    set_seed(config.training.seed)
    model = get_model(config.model, device).to(device)

    if config.normalization.enabled:
        model = InputNormalizer(
            model=model,
            mean=config.normalization.mean,
            std=config.normalization.std,
        ).to(device)

    optimizer = get_optimizer(model, config.training.optimizer)

    proxy = copy.deepcopy(model).to(device)
    proxy_optim = get_optimizer(proxy, config.awp.proxy_optimizer)

    awp = TradesAWP(
        model=model,
        proxy=proxy,
        proxy_optim=proxy_optim,
        wcoef=config.awp.weights_diff_coef,
        weps=config.awp.weights_epsilon,
    )

    train_adversary = PGD(
        epsilon=config.trainPGD.epsilon,
        alpha=config.trainPGD.alpha,
        steps=config.trainPGD.steps,
        lossfn=config.trainPGD.loss_fn,
        norm=config.trainPGD.norm,
        random_start=config.trainPGD.random_start,
    )
    eval_adversary = PGD(
        epsilon=config.evalPGD.epsilon,
        alpha=config.evalPGD.alpha,
        steps=config.evalPGD.steps,
        lossfn=config.evalPGD.loss_fn,
        norm=config.evalPGD.norm,
        random_start=config.evalPGD.random_start,
    )

    print(f"Train PGD: {train_adversary}")
    print(f"train PGD loss_fn = {train_adversary.lossfn}")
    print(f"Eval PGD: {eval_adversary}")
    print(f"eval PGD loss_fn = {eval_adversary.lossfn}")

    train(
        name="AWP+TRADES",
        cfg=config.training,
        model=model,
        optimizer=optimizer,
        norm_cfg=config.normalization,
        model_is_prepared=True,
        device=device,
        train_dataset_config=config.dataset,
        split_config=config.split,
        loss_fn=get_loss_fn(config.training.criterion),
        config_path=args.config,

        # train epoch
        train_epoch_fn=trades_awp_train,
        eval_fn=evaluate_adversarial,
        training_kwargs={
            "awp": awp,
            "awp_warmup": config.awp.warmup_steps,
            "adversary": train_adversary,
            "beta": config.trades.beta,
        },
        eval_kwargs={
            "metric_prefix": "pgd",
            "adversary": eval_adversary,
        },

    )


if __name__ == "__main__":
    main()

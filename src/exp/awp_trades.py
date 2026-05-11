import argparse
import copy
import shutil
import os

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

    os.makedirs(config.training.save_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(config.training.save_dir, "config.yaml"))

    adversary = PGD(
        epsilon=config.pgd.epsilon,
        alpha=config.pgd.alpha,
        steps=config.pgd.steps,
        lossfn=config.pgd.loss_fn,
        norm=config.pgd.norm,
        random_start=config.pgd.random_start
    )

    model = get_model(config.model, device).to(device)
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

    train(
        name="AWP+TRADES",
        cfg=config.training,
        model=model,
        optimizer=optimizer,
        norm_cfg=config.normalization,
        device=device,
        train_dataset_config=config.dataset,
        split_config=config.split,
        loss_fn=get_loss_fn(config.training.criterion),

        # train epoch
        train_epoch_fn=trades_awp_train,
        eval_fn=evaluate_adversarial,
        training_kwargs={
            "awp": awp,
            "awp_warmup": config.awp.warmup_steps,
            "adversary": adversary,
            "beta": config.awp.beta,
        },
        eval_kwargs={
            "metric_prefix": "pgd",
            "adversary": adversary,
        },

    )


if __name__ == "__main__":
    main()

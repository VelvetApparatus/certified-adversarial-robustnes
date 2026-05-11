from __future__ import print_function
import argparse

from src.robustness.adversaries.pgd import PGD
from src.config.trades import load_trades_config
from src.pkg import *
from src.eval.validation import evaluate_adversarial
from src.train.common import train
from src.train.trades import trades_train_one_epoch

parser = argparse.ArgumentParser(description='PyTorch TRADES Adversarial Training')
parser.add_argument("--config")
args = parser.parse_args()


def main():
    config = load_trades_config(args.config)

    device = get_device()
    set_seed(config.params.seed)

    adversary = PGD(
        epsilon=config.evalPGD.epsilon,
        alpha=config.evalPGD.alpha,
        steps=config.evalPGD.steps,
        loss_fn=get_loss_fn(config.training.criterion),
        norm=config.evalPGD.norm,
    )

    train(
        "TRADES-training",
        cfg=config.training,
        model_cfg=config.model,
        norm_cfg=config.normalization,
        device=device,
        train_dataset_config=config.dataset,
        split_config=config.split,
        loss_fn=get_loss_fn(config.training.criterion),
        config_path=args.config,

        # train epoch
        train_epoch_fn=trades_train_one_epoch,

        # eval
        eval_fn=evaluate_adversarial,

        training_kwargs={
            "pgd": adversary,
            "beta": config.params.beta,
        },
        eval_kwargs={
            "metric_prefix": "pgd",
            "adversary": adversary,
        },

    )


if __name__ == '__main__':
    main()

from __future__ import print_function
import argparse

from src.config.trades import load_trades_config
from src.pkg import *
from src.eval.validation import evaluate_adversarial
from src.robustness.adversaries.api import get_adversary
from src.train.common import train
from src.train.trades import trades_train_one_epoch

parser = argparse.ArgumentParser(description='PyTorch TRADES Adversarial Training')
parser.add_argument("--config")
args = parser.parse_args()


def main():
    config = load_trades_config(args.config)

    device = get_device()
    set_seed(config.params.seed)
    eval_adversary = get_adversary(config.evalPGD)

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
            "pgd_cfg": config.trainPGD,
            "beta": config.params.beta,
        },
        eval_kwargs={
            "metric_prefix": "pgd",
            "adversary": eval_adversary,
        },

    )


if __name__ == '__main__':
    main()

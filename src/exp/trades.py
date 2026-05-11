from __future__ import print_function
import os
import argparse
import shutil

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

    os.makedirs(config.training.save_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(config.training.save_dir, "config.yaml"))

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
        model=config.model,
        norm_cfg=config.normalization,
        device=device,
        train_dataset_config=config.dataset,
        split_config=config.split,
        loss_fn=get_loss_fn(config.training.criterion),

        # train epoch
        train_epoch_fn=trades_train_one_epoch,

        # eval
        eval_fn=evaluate_adversarial,

        # kwargs
        metric_prefix="pgd",
        adversary=adversary,
        step_size=config.params.step_size,
        epsilon=config.params.epsilon,
        perturb_steps=config.evalPGD.steps,
        beta=config.params.beta,

    )


if __name__ == '__main__':
    main()

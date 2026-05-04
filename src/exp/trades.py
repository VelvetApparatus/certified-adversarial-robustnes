from __future__ import print_function
import os
import argparse
import shutil

from src.config.trades import load_trades_config
from src.pkg import *
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

    train(
        "TRADES-training",
        cfg=config.training,
        model_cfg=config.model,
        norm_cfg=config.normalization,
        device=device,
        train_dataset_config=config.dataset,
        split_config=config.split,
        loss_fn=get_loss_fn(config.training.criterion),
        train_epoch_fn=trades_train_one_epoch,
        # kwargs
        step_size=config.params.step_size,
        epsilon=config.params.epsilon,
        perturb_steps=10,
        beta=config.params.beta,
        distance="l2"

    )


if __name__ == '__main__':
    main()

# src/exp/macer.py

import argparse

from src.config.macer import load_macer_training_config
from src.pkg import get_device, get_loss_fn
from src.train.common import train
from src.train.macer import macer_train_one_epoch

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", type=str, required=True)
args = arg_parser.parse_args()


def main():
    cfg = load_macer_training_config(args.config)

    device = get_device()
    loss_fn = get_loss_fn(cfg.training.criterion)

    train(
        name="macer",
        cfg=cfg.training,
        model_cfg=cfg.model,
        device=device,
        train_dataset_config=cfg.dataset,
        split_config=cfg.split,
        output_dir=cfg.training.save_dir,
        loss_fn=loss_fn,
        train_epoch_fn=macer_train_one_epoch,

        gauss_samples=cfg.params.gauss_samples,
        sigma=cfg.params.sigma,
        num_classes=cfg.params.num_classes,
        beta=cfg.params.beta,
        gamma=cfg.params.gamma,
        lbd=cfg.params.lbd,
        eps=cfg.params.eps,
    )


if __name__ == "__main__":
    main()

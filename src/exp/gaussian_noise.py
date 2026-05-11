import argparse
import os
import shutil

from src.config.gaussian import load_gaussian_train_config
from src.train.common import train
from src.eval.validation import evaluate_noisy
from src.train.gaussian_training import gaussian_train_one_epoch
from src.pkg import get_device, get_loss_fn

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", type=str, required=True)
args = arg_parser.parse_args()


def main():
    cfg = load_gaussian_train_config(args.config)
    shutil.copy(args.config, os.path.join(cfg.train.save_dir, "config.yaml"))
    device = get_device()
    train(
        name="gaussian_noise_training",
        cfg=cfg.train,
        norm_cfg=cfg.normalization,
        model=cfg.model,
        device=device,
        train_dataset_config=cfg.dataset,
        split_config=cfg.split,
        loss_fn=get_loss_fn(cfg.train.criterion),

        # train epoch
        train_epoch_fn=gaussian_train_one_epoch,
        # eval
        eval_fn=evaluate_noisy,


        # kwargs
        sigma=cfg.params.sigma,
        clean_loss_weight=cfg.params.clean_loss_weight,
        noisy_loss_weight=cfg.params.noisy_loss_weight,
        noise_ratio=cfg.params.noise_ratio,
        normalized_space=cfg.params.normalized_space,
    )


if __name__ == "__main__":
    main()

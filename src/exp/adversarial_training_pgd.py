import argparse
import os
import shutil

from src.eval.validation import evaluate_adversarial
from src.config.adversarial_training import load_adversarial_training_config
from src.adversaries.pgd import PGD
from src.train.common import train
from src.train.adversarial_training import adversarial_train_one_epoch
from src.pkg import get_device, get_loss_fn

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", type=str, required=True)
args = arg_parser.parse_args()


def main():
    cfg = load_adversarial_training_config(args.config)
    shutil.copy(args.config, os.path.join(cfg.training.save_dir, "config.yaml"))

    adversary = PGD(
        epsilon=cfg.pgd.epsilon,
        alpha=cfg.pgd.alpha,
        steps=cfg.pgd.steps,
        loss_fn=get_loss_fn(cfg.training.criterion),
        norm=cfg.pgd.norm,
    )

    device = get_device()
    train(
        name="aversarial_training_pgd",
        cfg=cfg.training,
        norm_cfg=cfg.normalization,
        model_cfg=cfg.model,
        device=device,
        train_dataset_config=cfg.dataset,
        split_config=cfg.split,
        loss_fn=get_loss_fn(cfg.training.criterion),

        # train epoch
        train_epoch_fn=adversarial_train_one_epoch,

        # eval
        eval_fn=evaluate_adversarial,

        # kwargs
        metric_prefix="pgd",
        adversary=adversary,
        adversarial_config=cfg
    )

    if __name__ == "__main__":
        main()

import argparse
import os
import shutil

from src.config.adversarial_training import load_adversarial_training_config
from src.adversaries.fgsm import FGSMAttack
from src.train.common import train
from src.train.adversarial_training import adversarial_train_one_epoch
from src.pkg import get_device, get_loss_fn

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", type=str, required=True)
args = arg_parser.parse_args()


def main():
    cfg = load_adversarial_training_config(args.config)
    os.makedirs(cfg.training.save_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(cfg.training.save_dir, "config.yaml"))

    # adversary = PGD(
    #     epsilon=cfg.pgd.epsilon,
    #     alpha=cfg.pgd.alpha,
    #     steps=cfg.pgd.steps,
    #     loss_fn=get_loss_fn(cfg.training.criterion),
    #     norm=cfg.pgd.norm,
    # )

    adversary = FGSMAttack(
        eps=cfg.fgsm.epsilon,
        loss_fn=get_loss_fn(cfg.training.criterion)
    )

    device = get_device()
    train(
        name="aversarial_training_fgsm",
        cfg=cfg.training,
        norm_cfg=cfg.normalization,
        model_cfg=cfg.model,
        device=device,
        train_dataset_config=cfg.dataset,
        split_config=cfg.split,
        loss_fn=get_loss_fn(cfg.training.criterion),
        train_epoch_fn=adversarial_train_one_epoch,
        adversary=adversary,
        adversarial_config=cfg
    )


if __name__ == "__main__":
    main()

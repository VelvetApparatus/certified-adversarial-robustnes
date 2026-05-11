import argparse

from src.eval.validation import evaluate_adversarial
from src.config.adversarial_training import load_adversarial_training_config
from src.robustness.adversaries.api import get_adversary
from src.train.common import train
from src.train.adversarial_training import adversarial_train_one_epoch
from src.pkg import get_device, get_loss_fn

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", type=str, required=True)
args = arg_parser.parse_args()


def main():
    cfg = load_adversarial_training_config(args.config)
    eval_adversary = get_adversary(cfg.pgd)

    device = get_device()
    train(
        name="aversarial_training_pgd",
        cfg=cfg.training,
        norm_cfg=cfg.normalization,
        model=cfg.model,
        device=device,
        train_dataset_config=cfg.dataset,
        split_config=cfg.split,
        loss_fn=get_loss_fn(cfg.training.criterion),
        config_path=args.config,

        # train epoch
        train_epoch_fn=adversarial_train_one_epoch,

        # eval
        eval_fn=evaluate_adversarial,

        training_kwargs={
            "pgd_cfg": cfg.pgd,
            "adversarial_config": cfg,
        },
        eval_kwargs={
            "metric_prefix": "pgd",
            "adversary": eval_adversary,
        },
    )

if __name__ == "__main__":
    main()

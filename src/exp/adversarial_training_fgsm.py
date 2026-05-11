import argparse

from src.config.adversarial_training import load_adversarial_training_config
from src.robustness.adversaries import FGSMAttack
from src.train.common import train
from src.train.adversarial_training import adversarial_train_one_epoch
from src.pkg import get_device, get_loss_fn
from src.eval.validation import evaluate_adversarial

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", type=str, required=True)
args = arg_parser.parse_args()


def main():
    cfg = load_adversarial_training_config(args.config)

    adversary = FGSMAttack(
        eps=cfg.fgsm.epsilon,
        loss_fn=get_loss_fn(cfg.training.criterion)
    )

    device = get_device()
    train(
        name="aversarial_training_fgsm",
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

        # eval epoch
        eval_fn=evaluate_adversarial,

        training_kwargs={
            "adversary": adversary,
            "adversarial_config": cfg,
        },
        eval_kwargs={
            "metric_prefix": "fgsm",
            "adversary": adversary,
        },
    )

if __name__ == "__main__":
    main()

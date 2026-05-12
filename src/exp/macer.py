import argparse

from src.config.macer import load_macer_training_config
from src.pkg import get_device, get_loss_fn
from src.eval.validation import evaluate_smoothed
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
        norm_cfg=cfg.normalization,
        model=cfg.model,
        device=device,
        train_dataset_config=cfg.dataset,
        split_config=cfg.split,
        loss_fn=loss_fn,
        config_path=args.config,

        # eval
        eval_fn=evaluate_smoothed,

        # train epoch
        train_epoch_fn=macer_train_one_epoch,

        training_kwargs={
            "params": cfg.params,
        },
        eval_kwargs={
            "sigma": cfg.params.sigma,
            "num_classes": cfg.model.num_classes,
            # todo: add normal config
            "samples": cfg.params.gauss_samples * 4,
            "beta": cfg.params.beta,
            "eps": cfg.params.eps,
        },

    )


if __name__ == "__main__":
    main()

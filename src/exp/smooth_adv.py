import argparse

from src.config.smooth_adv import load_smooth_adv_train_config
from src.model.api import get_model
from src.train.common import train
from src.eval.validation import evaluate_smoothed
from src.train.smooth_adv import smooth_adv_train_one_epoch
from src.pkg import get_device, get_loss_fn

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", type=str, required=True)
args = arg_parser.parse_args()


def main():
    config = load_smooth_adv_train_config(args.config)
    device = get_device()

    train(
        name="smooth_adv",
        cfg=config.training,
        norm_cfg=config.normalization,
        model=get_model(config.model, device),
        train_dataset_config=config.dataset,
        split_config=config.split,
        device=device,
        loss_fn=get_loss_fn(config.training.criterion),
        config_path=args.config,

        # train epoch fn
        train_epoch_fn=smooth_adv_train_one_epoch,

        # eval epoch fn
        eval_fn=evaluate_smoothed,

        training_kwargs={
            "params": config.params,
        },
        eval_kwargs={
            "sigma": config.params.sigma,
            "num_classes": config.model.num_classes,
            # todo to config
            "samples": 64,
            "beta": config.params.beta,
            "beta_scheduler": config.params.beta_scheduler,
            "eps": config.params.epsilon,
        },

    )


if __name__ == "__main__":
    main()

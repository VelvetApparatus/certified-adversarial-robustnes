import argparse

from src.config.smooth_adv_masked import load_smoothed_adv_masked_config
from src.eval.validation import evaluate_smoothed
from src.model.api import get_model
from src.pkg import *
from src.robustness.input.mask import MaskGen
from src.train.common import train
from src.train.smooth_adv_masked import train_smooth_adv_masked

parser = argparse.ArgumentParser(
    description='Smooth Masked Training'
)
parser.add_argument("--config", type=str, required=True, )
args = parser.parse_args()


def main():
    config = load_smoothed_adv_masked_config(args.config)
    device = get_device()
    set_seed(config.training.seed)

    mask_gen = MaskGen(
        p=config.input_mask.p,
        ratio=config.input_mask.ratio,
    )

    train(
        name="smooth_adv_masked",
        cfg=config.training,
        norm_cfg=config.normalization,
        model=get_model(config.model, device),
        train_dataset_config=config.dataset,
        split_config=config.split,
        device=device,
        loss_fn=get_loss_fn(config.training.criterion),
        config_path=args.config,

        # train epoch fn
        train_epoch_fn=train_smooth_adv_masked,

        # eval epoch fn
        eval_fn=evaluate_smoothed,

        training_kwargs={
            "mask_gen": mask_gen,
            "mask_warmup": config.input_mask.warmup_steps,
            "attack_cfg": config.attack,
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


if __name__ == '__main__':
    main()

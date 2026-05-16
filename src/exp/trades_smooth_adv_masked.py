import argparse

from src.config.trades_smooth_adv_masked import load_trades_smooth_adv_masked_config
from src.eval.validation import evaluate_adversarial
from src.pkg import get_device, get_loss_fn, set_seed
from src.robustness.adversaries.api import get_adversary
from src.robustness.input.mask import MaskGen
from src.train.common import train
from src.train.trades_smooth_adv_masked import train_trades_smooth_adv_masked


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRADES + SmoothAdv + Consistency + Input Masking training"
    )
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_trades_smooth_adv_masked_config(args.config)

    set_seed(config.training.seed)
    device = get_device()

    print("consistency_type =", config.params.consistency_type)
    print("consistency_weight =", config.params.consistency_weight)
    print("consistency_scheduler =", config.params.consistency_scheduler)

    mask_gen = MaskGen(
        p=config.input_mask.p,
        ratio=config.input_mask.ratio,
    )
    eval_adversary = get_adversary(config.evalPGD)

    train(
        name="trades_smooth_adv_masked",
        cfg=config.training,
        model=config.model,
        norm_cfg=config.normalization,
        device=device,
        train_dataset_config=config.dataset,
        split_config=config.split,
        loss_fn=get_loss_fn(config.training.criterion),
        config_path=args.config,
        train_epoch_fn=train_trades_smooth_adv_masked,
        eval_fn=evaluate_adversarial,
        training_kwargs={
            "trades_attack_cfg": config.trainPGD,
            "smooth_attack_cfg": config.attack,
            "params": config.params,
            "mask_gen": mask_gen,
            "mask_gen_warmup": config.input_mask.warmup_steps,
        },
        eval_kwargs={
            "metric_prefix": "pgd",
            "adversary": eval_adversary,
        },
    )


if __name__ == "__main__":
    main()

import argparse
import copy

from src.config.trades_awp_masked import load_trades_awp_masked_config
from src.eval.validation import evaluate_adversarial
from src.model.api import get_model
from src.pkg import (
    InputNormalizer,
    get_device,
    get_loss_fn,
    get_optimizer,
    set_seed,
)
from src.robustness.adversaries.pgd import PGD
from src.robustness.input.mask import MaskGen
from src.robustness.model.awp import TradesAWP
from src.train.common import train
from src.train.trades_awp_masked import train_trades_awp_masked


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRADES + AWP + Input Masking training",
    )
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_trades_awp_masked_config(args.config)

    set_seed(config.training.seed)
    device = get_device()

    model = get_model(config.model, device=device).to(device)
    if config.normalization.enabled:
        model = InputNormalizer(
            model=model,
            mean=config.normalization.mean,
            std=config.normalization.std,
        ).to(device)

    optimizer = get_optimizer(model, config.training.optimizer)

    proxy = copy.deepcopy(model).to(device)
    proxy_optim = get_optimizer(proxy, config.awp.proxy_optimizer)
    awp = TradesAWP(
        model=model,
        proxy=proxy,
        proxy_optim=proxy_optim,
        wcoef=config.awp.weights_diff_coef,
        weps=config.awp.weights_epsilon,
    )

    mask_gen = MaskGen(
        p=config.input_mask.p,
        ratio=config.input_mask.ratio,
    )

    eval_adversary = PGD(
        epsilon=config.evalPGD.epsilon,
        alpha=config.evalPGD.alpha,
        steps=config.evalPGD.steps,
        lossfn=config.evalPGD.loss_fn,
        norm=config.evalPGD.norm,
        random_start=config.evalPGD.random_start,
    )

    train(
        name="trades_awp_masked",
        cfg=config.training,
        norm_cfg=config.normalization,
        model=model,
        optimizer=optimizer,
        model_is_prepared=True,
        train_dataset_config=config.dataset,
        split_config=config.split,
        device=device,
        loss_fn=get_loss_fn(config.training.criterion),
        config_path=args.config,
        train_epoch_fn=train_trades_awp_masked,
        eval_fn=evaluate_adversarial,
        training_kwargs={
            "mask_gen": mask_gen,
            "mask_gen_warmup": config.input_mask.warmup_steps,
            "attack_cfg": config.trainPGD,
            "params": config.trades,
            "awp_adversary": awp,
            "awp_cfg": config.awp,
        },
        eval_kwargs={
            "adversary": eval_adversary,
            "metric_prefix": "pgd",
        },
    )


if __name__ == "__main__":
    main()

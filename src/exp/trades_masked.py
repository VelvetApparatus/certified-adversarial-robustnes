import argparse

from src.config.trades_masked import load_trades_masked_config
from src.model.api import get_model
from src.robustness.adversaries import PGD
from src.robustness.input.mask import MaskGen
from src.train.common import train
from src.eval.validation import evaluate_adversarial
from src.train.trades_masked import train_trades_masked
from src.pkg import *

parser = argparse.ArgumentParser(
    description='Trades Masked Training'
)
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()


def main():
    config = load_trades_masked_config(args.config)
    device = get_device()

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
        name="Trades Masked Training",
        cfg=config.training,
        norm_cfg=config.normalization,
        model=get_model(config.model, device=device),
        train_dataset_config=config.dataset,
        split_config=config.split,
        device=device,
        loss_fn=get_loss_fn(config.training.criterion),
        config_path=args.config,

        #     train epoch fn
        train_epoch_fn=train_trades_masked,
        training_kwargs={
            "mask_gen": mask_gen,
            "mask_gen_warmup": config.input_mask.warmup_steps,
            "attack_cfg": config.trainPGD,
            "params": config.trades,
        },

        #     eval epoch fn
        eval_fn=evaluate_adversarial,
        eval_kwargs={
            "adversary": eval_adversary,
            "metric_prefix": "pgd"

        }
    )

if __name__ == '__main__':
    main()

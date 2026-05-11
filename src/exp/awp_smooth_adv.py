import argparse
import copy

from src.config.smoothed_awp import load_smoothed_awp_config
from src.eval.validation import evaluate_smoothed
from src.model.api import get_model
from src.pkg import (
    get_device,
    set_seed,
    get_optimizer,
    get_loss_fn,
    InputNormalizer,
)
from src.robustness.model.awp import AWPCrossEntropy
from src.train.common import train
from src.train.smoothed_awp import train_smoothed_awp

parser = argparse.ArgumentParser(
    description="PyTorch SmoothAdv + AWP Training",
)
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()


def main():
    config = load_smoothed_awp_config(args.config)

    device = get_device()
    set_seed(config.training.seed)

    model = get_model(config.model, device).to(device)

    if config.normalization.enabled:
        model = InputNormalizer(
            model=model,
            mean=config.normalization.mean,
            std=config.normalization.std,
        ).to(device)

    optimizer = get_optimizer(model, config.training.optimizer)

    proxy = copy.deepcopy(model).to(device)
    proxy_optim = get_optimizer(proxy, config.awp.proxy_optimizer)

    awp = AWPCrossEntropy(
        model=model,
        proxy=proxy,
        proxy_optim=proxy_optim,
        wcoef=config.awp.weights_diff_coef,
        weps=config.awp.weights_epsilon,
    )

    print("SmoothAdv + AWP config")
    print(f"attack = {config.attack}")
    print(f"sigma = {config.params.sigma}")
    print(f"epsilon = {config.params.epsilon}")
    print(f"num_noise_vec = {config.params.num_noise_vec}")
    print(f"AWP warmup = {config.awp.warmup_steps}")

    train(
        name="SmoothAdv+AWP",
        cfg=config.training,
        model=model,
        optimizer=optimizer,
        norm_cfg=config.normalization,
        model_is_prepared=True,
        device=device,
        train_dataset_config=config.dataset,
        split_config=config.split,
        loss_fn=get_loss_fn(config.training.criterion),
        config_path=args.config,

        train_epoch_fn=train_smoothed_awp,
        eval_fn=evaluate_smoothed,

        training_kwargs={
            "awp": awp,
            "awp_warmup": config.awp.warmup_steps,
            "attack_cfg": config.attack,
            "params": config.params,
            "beta": None,
        },

        eval_kwargs={
            "sigma": config.params.sigma,
            "num_classes": config.eval_smoothed.get(
                "num_classes",
                config.model.num_classes,
            ),
            "samples": config.eval_smoothed.get("samples", 64),
            "beta": config.params.beta,
            "beta_scheduler": config.params.beta_scheduler,
            "eps": config.eval_smoothed.get(
                "eps",
                config.params.epsilon,
            ),
        },
    )


if __name__ == "__main__":
    main()

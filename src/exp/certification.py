import argparse
import datetime
import os
import shutil
from time import time
from tqdm import tqdm

import torch

from src.pkg import set_seed, get_device
from src.model.api import get_model
from src.db.api import get_dataset
from src.config.conf import load_certification_config
from src.certify.rs import Smooth

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", type=str, required=True)
args = arg_parser.parse_args()


def main():
    config = load_certification_config(args.config)

    set_seed(config.certification.seed)
    device = get_device()

    dataset = get_dataset(cfg=config.dataset)

    model = get_model(cfg=config.model, device=device)
    model.to(device)
    model.eval()

    smoothed = Smooth(
        base_classifier=model,
        # todo: add to config
        num_classes=10,
        sigma=config.certification.sigma,
        device=device,
    )

    run_name = "{}_{}_{}".format(
        config.model.name,
        config.dataset.name,
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    output_dir = os.path.join(config.certification.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(output_dir, "config.yaml"))

    log_path = os.path.join(output_dir, "log.tsv")

    # todo: add to config
    # max_samples = config.certification.max_samples
    # if max_samples is None:
    max_samples = 100

    num_samples = min(len(dataset), max_samples)

    with open(log_path, "w") as f:
        print(
            "idx\tlabel\tpredict\tradius\tcorrect\tabstain\ttime",
            file=f,
            flush=True,
        )

        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="certification"):
                x, y = dataset[i]

                x = x.to(device, non_blocking=True)

                if torch.is_tensor(y):
                    label = int(y.item())
                else:
                    label = int(y)

                time_before = time()

                prediction, radius = smoothed.certify(
                    x=x,
                    n0=config.certification.n0,
                    n=config.certification.n,
                    alpha=config.certification.alpha,
                    # todo: add to config
                    batch_size=1000,
                )

                after_time = time()

                prediction = int(prediction)
                abstain = int(prediction == Smooth.ABSTAIN)
                correct = int(prediction == label) if not abstain else 0

                time_elapsed = str(
                    datetime.timedelta(seconds=(after_time - time_before))
                )

                print(
                    "{}\t{}\t{}\t{:.6f}\t{}\t{}\t{}".format(
                        i,
                        label,
                        prediction,
                        radius,
                        correct,
                        abstain,
                        time_elapsed,
                    ),
                    file=f,
                    flush=True,
                )


if __name__ == "__main__":
    main()

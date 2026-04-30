import argparse
import datetime
import os
import shutil
import time

from torch.distributions import Normal
from tqdm import tqdm

import torch
from torch.nn.functional import (
    softmax, nll_loss
)

from src.config.macer import load_macer_config
from src.pkg import set_seed, get_device, get_optimizer, get_scheduler
from src.model.api import get_model
from src.db.api import get_dataset
from src.certify.rs import certify

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", type=str, required=True)
args = arg_parser.parse_args()


def train_epoch(
        dataloader,
        model,
        optimizer,
        device,
        gauss_samples,
        sigma,
        num_classes,
        beta,
        gamma,
        lbd,
):
    model.train()

    # normalization for icdf
    m = Normal(torch.tensor([0.0]).to(device),
               torch.tensor([1.0]).to(device))

    # metrics
    cl_total = 0.0
    rl_total = 0.0
    input_total = 0

    for idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        batch_size = len(inputs)
        input_total += batch_size

        # [B, C, H, W] -> [B * gauss_samples, C, H, W]
        new_shape = [batch_size * gauss_samples]
        new_shape.extend(inputs[0].shape)
        inputs = inputs.repeat((1, gauss_samples, 1, 1)).view(new_shape)

        # add noise to samples
        noise = torch.randn_like(inputs, device=device) * sigma
        noisy_inputs = inputs + noise

        outputs = model(noisy_inputs)
        outputs = outputs.reshape((batch_size, gauss_samples, num_classes))

        # classification loss (with surrogate loss: TRADES, MACER)

        # soft targets
        outputs_softmax = softmax(outputs, dim=2).mean(dim=1)

        # add 1e-10 to avoid NaN
        outputs_log_softmax = torch.log(outputs_softmax + 1e-10)
        classification_loss = nll_loss(
            input=outputs_log_softmax,
            target=targets,
            reduction="sum",
        )
        cl_total += classification_loss.item()

        # Robustness loss
        beta_outputs = outputs * beta
        beta_outputs_softmax = softmax(beta_outputs, dim=2).mean(dim=1)
        top2 = torch.topk(beta_outputs_softmax, k=2)

        top2_score, top2_idx = top2[0], top2[1]

        correct_mask = (top2_idx[:, 0] == targets)

        out0, out1 = top2_score[correct_mask, 0], top2_score[correct_mask, 1]

        # Calculate distance between two top predictions
        # Ф^-1(p_B) - Ф^-1(p_A)
        robustness_loss = m.icdf(out1) - m.icdf(out0)

        # Hinge filter
        hinge_filter = (~torch.isnan(robustness_loss) &
                        ~torch.isinf(robustness_loss) &
                        (torch.abs(robustness_loss) < gamma))
        out0, out1 = out0[hinge_filter], out1[hinge_filter]

        robustness_loss = m.icdf(out1) - m.icdf(out0) + gamma
        robustness_loss = robustness_loss.sum() * sigma / 2
        rl_total += robustness_loss.item()

        # final loss
        loss = classification_loss + lbd * robustness_loss
        loss /= batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cl_total /= input_total
    rl_total /= input_total

    return cl_total, rl_total


def main():
    config = load_macer_config(args.config)

    # logging
    run_name = "{}_{}_{}".format(
        config.model.name,
        config.train_dataset.name,
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    output_dir = os.path.join(config.params.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(output_dir, "config.yaml"))

    gauss_samples = config.params.gauss_samples
    sigma = config.params.sigma
    beta = config.params.beta
    # todo: make it reliable to dataset
    num_classes = config.params.num_classes
    gamma = config.params.gamma
    lbd = config.params.lbd
    epochs = config.params.epochs
    certificate_every_epoch = config.params.certificate_every_epoch
    certificate_epoch_threshold = config.params.certificate_epoch_threshold

    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    matdir = os.path.join(output_dir, "mat")
    os.makedirs(matdir, exist_ok=True)
    checkpoint = config["checkpoint"]

    # set seed
    set_seed(config.params.seed)

    # get device
    device = get_device()

    # model
    model = get_model(config["model"], device)
    model.to(device)

    # optimizer
    optimizer = get_optimizer(model, config.optimizer)

    # scheduler
    scheduler = get_scheduler(optimizer, config.scheduler)

    start_epoch = 0
    # checkpoint
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        scheduler.step(start_epoch)

    # dataset
    train_dataset = get_dataset(config.train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size,
        shuffle=True,
        num_workers=config.train_dataset.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    test_dataset = get_dataset(config.test_dataset)

    classification_errors = []
    robustness_errors = []

    for epoch in tqdm(range(start_epoch, epochs), "macer training"):

        cl_err, rb_err = train_epoch(
            dataloader=train_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            gauss_samples=gauss_samples,
            sigma=sigma,
            num_classes=num_classes,
            beta=beta,
            gamma=gamma,
            lbd=lbd,
        )

        scheduler.step()

        print(
            "Epoch: {}, Classification error: {:.4f}, Robustness error: {:.4f}".format(
                epoch, cl_err, rb_err)
        )

        classification_errors.append(cl_err)
        robustness_errors.append(rb_err)

        if epoch % certificate_every_epoch == 0 and epoch >= certificate_epoch_threshold:

            # Certify test
            print('===test(epoch={})==='.format(epoch))
            t1 = time.time()
            model.eval()
            # todo: config
            certify(model, device, test_dataset, num_classes,
                    mode='soft', start_img=config.params.cert_start, num_img=config.params.cert_num,
                    sigma=sigma, beta=beta,
                    matfile=(None if matdir is None else os.path.join(matdir, '{}.mat'.format(epoch))))
            t2 = time.time()
            print('Elapsed time: {}'.format(t2 - t1))

            if checkpoints_dir is not None:
                # Save checkpoint
                print('==> Saving {}.pth..'.format(epoch))
                try:
                    state = {
                        'net': model.state_dict(),
                        'epoch': epoch,
                    }
                    torch.save(state, '{}/{}.pth'.format(checkpoints_dir, epoch))
                except OSError:
                    print('OSError while saving {}.pth'.format(epoch))
                    print('Ignoring...')

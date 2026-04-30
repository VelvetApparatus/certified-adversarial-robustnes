from __future__ import print_function

import datetime
import os
import argparse
import shutil

import torch
import torch.nn.functional as F

from src.certify.rs import certify
from src.config.trades import load_trades_config
from src.db.api import get_dataset
from src.model.api import get_model
from src.pkg import *
from src.robustness.trades import trades_loss

parser = argparse.ArgumentParser(description='PyTorch TRADES Adversarial Training')
parser.add_argument("--config")
args = parser.parse_args()


def main():
    config = load_trades_config(args.config)

    device = get_device()
    set_seed(config.params.seed)

    model = get_model(config.model, device)
    model.to(device)

    # optimizer
    optimizer = get_optimizer(model, config.optimizer)

    # scheduler
    scheduler = get_scheduler(optimizer, config.scheduler)

    run_name = "{}_{}_{}".format(
        config.model.name,
        config.train_dataset.name,
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    output_dir = os.path.join(config.params.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(output_dir, "config.yaml"))

    start_epoch = 0

    # checkpoint
    if config.params.checkpoint is not None:
        checkpoint = torch.load(config.params.checkpoint)
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

    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    matdir = os.path.join(output_dir, "mat")
    os.makedirs(matdir, exist_ok=True)

    # params
    checkpoint = config.params.checkpoint
    certificate_every_epoch = config.params.certificate_every_epoch
    certificate_epoch_threshold = config.params.certificate_epoch_threshold
    num_classes = config.model.num_classes
    sigma = config.params.sigma
    beta = config.params.beta

    test_dataset = get_dataset(cfg=config.test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test_dataset.batch_size,
        shuffle=False,
        num_workers=config.test_dataset.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # checkpoint
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        scheduler.step(start_epoch)

    for epoch in range(1, args.epochs + 1):

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        optimizer.step()

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        if epoch % certificate_every_epoch == 0 and epoch >= certificate_epoch_threshold:

            # Certify test
            print('===test(epoch={})==='.format(epoch))
            t1 = datetime.time.time()
            model.eval()
            certify(model, device, test_dataset, num_classes,
                    mode='soft', start_img=config.params.cert_start, num_img=config.params.cert_num,
                    sigma=sigma, beta=beta,
                    matfile=(None if matdir is None else os.path.join(matdir, '{}.mat'.format(epoch))))
            t2 = datetime.time.time()
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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


if __name__ == '__main__':
    main()

#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG=configs/train/trades-smoothadv-consistency-resnet18-cifar10.yaml
python -m src.exp.trades_smooth_adv --config "$CONFIG"

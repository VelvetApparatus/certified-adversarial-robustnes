#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG=configs/train/trades-smoothadv-consistency-masked-resnet18-cifar10.yaml
python -m src.exp.trades_smooth_adv_masked --config "$CONFIG"

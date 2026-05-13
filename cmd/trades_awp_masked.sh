#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG=configs/train/trades-awp-masked-pgdmasked-resnet18-cifar10.yaml
python -m src.exp.trades_awp_masked --config "$CONFIG"

# CONFIG=configs/train/trades-awp-masked-pgdclean-resnet18-cifar10.yaml
# python -m src.exp.trades_awp_masked --config "$CONFIG"

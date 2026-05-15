#!/usr/bin/env bash
set -euo pipefail

echo "=== Evaluating adversarial PGD training model with PGD-50 x5 ==="
python -m src.exp.evaluate --config configs/eval/strong_pgd/resnet18_cifar10/adversarial-pgd-training.yaml

echo "=== Evaluating TRADES model with PGD-50 x5 ==="
python -m src.exp.evaluate --config configs/eval/strong_pgd/resnet18_cifar10/trades.yaml

echo "=== Evaluating TRADES AWP model with PGD-50 x5 ==="
python -m src.exp.evaluate --config configs/eval/strong_pgd/resnet18_cifar10/trades-awp.yaml

echo "=== Evaluating TRADES masked PGD-masked model with PGD-50 x5 ==="
python -m src.exp.evaluate --config configs/eval/strong_pgd/resnet18_cifar10/trades-masked-pgdmasked.yaml

echo "=== Evaluating TRADES AWP masked PGD-masked model with PGD-50 x5 ==="
python -m src.exp.evaluate --config configs/eval/strong_pgd/resnet18_cifar10/trades-awp-masked-pgdmasked.yaml

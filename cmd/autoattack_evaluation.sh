#!/usr/bin/env bash
set -euo pipefail

# echo "=== Evaluating adversarial PGD training model with AutoAttack ==="
# python -m src.exp.evaluate_autoattack --config configs/eval/autoattack/adv-PGD-resnet18-cifar10.yaml --no-wandb

# echo "=== Evaluating adversarial FGSM training model with AutoAttack ==="
# python -m src.exp.evaluate_autoattack --config configs/eval/autoattack/adv-FGSM-resnet18-cifar10.yaml --no-wandb

# echo "=== Evaluating gaussian training model with AutoAttack ==="
# python -m src.exp.evaluate_autoattack --config configs/eval/autoattack/gaussian-noise-resnet18-cifar10.yaml --no-wandb

# echo "=== Evaluating TRADES model with AutoAttack ==="
# python -m src.exp.evaluate_autoattack --config configs/eval/autoattack/trades-resnet18-cifar10.yaml --no-wandb

# echo "=== Evaluating MACER model with AutoAttack ==="
# python -m src.exp.evaluate_autoattack --config configs/eval/autoattack/macer-resnet18-cifar10.yaml --no-wandb

# echo "=== Evaluating baseline model with AutoAttack ==="
# python -m src.exp.evaluate_autoattack --config configs/eval/autoattack/evaluation-baseline-resnet18-cifar10.yaml --no-wandb

echo "=== Evaluating Smooth Adv with AutoAttack ==="
python -m src.exp.evaluate_autoattack --config configs/eval/autoattack/smooth-adv-resnet18-cifar10.yaml --no-wandb

echo "=== Evaluating Smooth Adv + AWP with AutoAttack ==="
python -m src.exp.evaluate_autoattack --config configs/eval/autoattack/AWP-smooth-adv-resnet18-cifar10.yaml --no-wandb

echo "=== Evaluating TRADES AWP with AutoAttack ==="
python -m src.exp.evaluate_autoattack --config configs/eval/autoattack/AWP-trades-resnet18-cifar10.yaml --no-wandb

# echo "=== Evaluating TRADES SmoothAdv Consistency Masked with AutoAttack ==="
# python -m src.exp.evaluate_autoattack --config configs/eval/autoattack/trades-smoothadv-consistency-masked-resnet18-cifar10.yaml --no-wandb

echo "=== Evaluating TRADES SmoothAdv Consistency Masked AWP with AutoAttack ==="
# Placeholder checkpoint path in config; update configs/eval/autoattack/trades-smoothadv-consistency-masked-awp-resnet18-cifar10.yaml after training.
# python -m src.exp.evaluate_autoattack --config configs/eval/autoattack/trades-smoothadv-consistency-masked-awp-resnet18-cifar10.yaml --no-wandb

echo "=== Evaluating TRADES Masked with AutoAttack ==="
python -m src.exp.evaluate_autoattack --config configs/eval/autoattack/trades-masked-pgdclean-resnet18-cifar10.yaml --no-wandb
python -m src.exp.evaluate_autoattack --config configs/eval/autoattack/trades-masked-pgdmasked-resnet18-cifar10.yaml --no-wandb

echo "=== Done ==="

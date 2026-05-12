#!/usr/bin/env bash
set -euo pipefail

# echo "=== Evaluating adversarial PGD training model ==="
# python -m src.exp.evaluate --config configs/eval/adv-PGD-resnet18-cifar10.yaml

#  echo "=== Evaluating adversarial FGSM training model ==="
#  python -m src.exp.evaluate --config configs/eval/adv-FGSM-resnet18-cifar10.yaml

# echo "=== Evaluating gaussian training model ==="
# python -m src.exp.evaluate --config configs/eval/gaussian-noise-resnet18-cifar10.yaml

# echo "=== Evaluating TRADES model ==="
# python -m src.exp.evaluate --config configs/eval/trades-resnet18-cifar10.yaml

# echo "=== Evaluating MACER model ==="
# python -m src.exp.evaluate --config configs/eval/macer-resnet18-cifar10.yaml

#echo "=== Evaluation baseline model ==="
#python -m src.exp.evaluate --config configs/eval/evaluation-baseline-resnet18-cifar10.yaml

# echo "=== Evaluation Smooth Adv ==="
# python -m src.exp.evaluate --config configs/eval/smooth-adv-resnet18-cifar10.yaml

echo "=== Evalutation Smooth Adv + AWP ==="
python -m src.exp.evaluate --config configs/eval/AWP-smooth-adv-resnet18-cifar10.yaml

echo "=== Evaluation TRADES AWP ==="
python -m src.exp.evaluate --config configs/eval/AWP-trades-resnet18-cifar10.yaml



echo "=== Done ==="
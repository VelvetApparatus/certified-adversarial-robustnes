#!/usr/bin/env bash
set -euo pipefail

#echo "=== Evaluating adversarial training model ==="
#python -m src.exp.evaluate --config configs/evaluation-adversarial-examples-resnet18-cifar10.yaml
#
#echo "=== Evaluating gaussian training model ==="
#python -m src.exp.evaluate --config configs/evaluation-gaussian-noise-resnet18-cifar10.yaml
#
#echo "=== Evaluating TRADES model ==="
#python -m src.exp.evaluate --config configs/evaluation-trades-resnet18-cifar10.yaml
#
#echo "=== Evaluating MACER model ==="
#python -m src.exp.evaluate --config configs/evaluation-macer-resnet18-cifar10.yaml

echo "=== Evaluation baseline model ==="
python -m src.exp.evaluate --config configs/evaluation-baseline-resnet18-cifar10.yaml

echo "=== Done ==="
#!/usr/bin/env bash
set -Eeuo pipefail

run_eval() {
  local name="$1"
  local config="$2"

  echo "============================================================"
  echo "Evaluating: $name"
  echo "Config:     $config"
  echo "Time:       $(date)"
  echo "============================================================"

  if [[ ! -f "$config" ]]; then
    echo "SKIP: config not found: $config"
    echo
    return 0
  fi

  python -m src.exp.evaluate --config "$config"

  echo "============================================================"
  echo "Finished:   $name"
  echo "Time:       $(date)"
  echo "============================================================"
  echo
}

# ============================================================
# Baselines
# ============================================================

# run_eval "baseline" \
#   "configs/eval/evaluation-baseline-resnet18-cifar10.yaml"

# run_eval "gaussian_noise" \
#   "configs/eval/gaussian-noise-resnet18-cifar10.yaml"

# ============================================================
# Empirical adversarial training
# ============================================================

# run_eval "adversarial_fgsm" \
#   "configs/eval/adv-FGSM-resnet18-cifar10.yaml"

# run_eval "adversarial_pgd" \
#   "configs/eval/adv-PGD-resnet18-cifar10.yaml"

# ============================================================
# TRADES / SmoothAdv / MACER
# ============================================================

# run_eval "trades" \
#   "configs/eval/trades-resnet18-cifar10.yaml"

# run_eval "smooth_adv" \
#   "configs/eval/smooth-adv-resnet18-cifar10.yaml"

# run_eval "macer" \
#   "configs/eval/macer-resnet18-cifar10.yaml"

# ============================================================
# AWP methods
# ============================================================

# run_eval "trades_awp" \
#   "configs/eval/AWP-trades-resnet18-cifar10.yaml"

# run_eval "smooth_adv_awp" \
#   "configs/eval/AWP-smooth-adv-resnet18-cifar10.yaml"

# ============================================================
# Masked methods
# ============================================================

# run_eval "smooth_adv_masked_pgd_clean" \
#   "configs/eval/smooth-adv-masked-pgdclean-resnet18-cifar10.yaml"

# run_eval "smooth_adv_masked_pgd_masked" \
#   "configs/eval/smooth-adv-masked-pgdmasked-resnet18-cifar10.yaml"

# run_eval "trades_masked_pgd_clean" \
#   "configs/eval/trades-masked-pgdclean-resnet18-cifar10.yaml"

# run_eval "trades_masked_pgd_masked" \
#   "configs/eval/trades-masked-pgdmasked-resnet18-cifar10.yaml"

# ============================================================
# Future combined methods
# ============================================================

# run_eval "trades_awp_masked" \
#   "configs/eval/trades-awp-masked-resnet18-cifar10.yaml"

# run_eval "smooth_adv_awp_masked" \
#   "configs/eval/smooth-adv-awp-masked-resnet18-cifar10.yaml"

# ============================================================
# Future combined methods
# ============================================================
# run_eval "trades-awp-masked" "configs/eval/trades-awp-masked-pgdmasked-resnet18-cifar10.yaml"
# run_eval "trades-smoothad-consistency"  "configs/eval/trades-smoothadv-consistency-resnet18-cifar10.yaml"
run_eval "trades-smoothadv-cons-masked" "configs/eval/trades-smoothadv-consistency-masked-resnet18-cifar10.yaml"



echo "============================================================"
echo "All available evaluations finished."
echo "Time: $(date)"
echo "============================================================"
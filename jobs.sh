#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="logs/train_all/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

run_job() {
  local name="$1"
  local script="$2"

  if [[ ! -f "$script" ]]; then
    echo "Script not found: $script"
    echo "Current dir: $(pwd)"
    exit 1
  fi

  echo "============================================================"
  echo "Starting: $name"
  echo "Script:   $script"
  echo "Log:      $LOG_DIR/${name}.log"
  echo "Time:     $(date)"
  echo "============================================================"

  bash "$script" 2>&1 | tee "$LOG_DIR/${name}.log"

  echo "============================================================"
  echo "Finished: $name"
  echo "Time:     $(date)"
  echo "============================================================"
  echo
}

# run_job "adversarial_train" "./cmd/adversarial_training_pgd.sh"
# run_job "adversarial_train_fgsm" "./cmd/adversarial_training_fgsm.sh"
# run_job "smooth_adv" "./cmd/smooth_adv.sh"
# run_job "awp trades" "./cmd/awp_trades.sh"
# run_job "awp smooth" "./cmd/smoothed_awp.sh"
# run_job "macer" "./cmd/macer.sh"
# run_job "trades" "./cmd/trades.sh"
# run_job "gaussian_training" "./cmd/gaussian_training.sh"
# run_job "smooth_adv_masked" "./cmd/smooth_adv_masked.sh"
# run_job "trades_masked" "./cmd/trades_masked.sh"
# run_job "evaluation" "./cmd/evaluation.sh"
# run_job "trades_awp_masked" "./cmd/trades_awp_masked.sh"
run_job "trades_smoothadv_consistency" "./cmd/trades_smoothadv_consistency.sh" 

echo "All training jobs finished successfully."
echo "Logs saved to: $LOG_DIR"
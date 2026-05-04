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

run_job "adversarial_train" "./cmd/adversarial_training.sh"
run_job "gaussian_training" "./cmd/gaussian_training.sh"
run_job "macer" "./cmd/macer.sh"
run_job "trades" "./cmd/trades.sh"
run_job "evaluation" "./cmd/evaluation.sh"

echo "All training jobs finished successfully."
echo "Logs saved to: $LOG_DIR"
#!/usr/bin/env bash
set -euo pipefail

# Runs Ray Tune hyperparameter orchestration with limited trial concurrency.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://${HEAD_PRIVATE_IP}:5000}"

python -m src.ray_tune.tune_pytorch_cpu \
  --ray-address "${RAY_ADDRESS:-auto}" \
  --max-concurrent-trials "${MAX_CONCURRENT_TRIALS:-2}" \
  --num-samples "${TUNE_NUM_SAMPLES:-8}" \
  --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
  "$@"

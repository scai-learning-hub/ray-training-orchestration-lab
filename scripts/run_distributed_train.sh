#!/usr/bin/env bash
set -euo pipefail

# Runs distributed CPU-only PyTorch training on Ray Train.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

HEAD_PRIVATE_IP="$(hostname -I | awk '{print $1}')"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://${HEAD_PRIVATE_IP}:5000}"

python -m src.ray_train.distributed_cpu_pytorch_train \
  --ray-address "${RAY_ADDRESS:-auto}" \
  --num-workers "${TRAIN_NUM_WORKERS:-2}" \
  --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
  "$@"

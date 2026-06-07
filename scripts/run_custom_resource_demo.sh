#!/usr/bin/env bash
set -euo pipefail

# Runs the Ray Core custom resource placement demo.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"

python -m src.ray_core.custom_resource_demo \
  --ray-address "${RAY_ADDRESS:-auto}" \
  --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
  "$@"

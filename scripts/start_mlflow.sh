#!/usr/bin/env bash
set -euo pipefail

# Starts the MLflow tracking server on the Ray head node.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALLOWED_HOSTS="${MLFLOW_ALLOWED_HOSTS:-*}"
CORS_ALLOWED_ORIGINS="${MLFLOW_CORS_ALLOWED_ORIGINS:-*}"

cd "${PROJECT_ROOT}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

mkdir -p mlartifacts

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "${ALLOWED_HOSTS}" \
  --cors-allowed-origins "${CORS_ALLOWED_ORIGINS}"

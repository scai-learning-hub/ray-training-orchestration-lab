#!/usr/bin/env bash
set -euo pipefail

# Ubuntu 22.04 environment bootstrap for the Ray CPU orchestration lab.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip build-essential

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --no-cache-dir --upgrade pip wheel
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install --no-cache-dir -r requirements.txt

mkdir -p outputs/datasets outputs/models outputs/ray_results mlartifacts

echo "Environment ready. Activate it with: source ${PROJECT_ROOT}/.venv/bin/activate"

#!/usr/bin/env bash
set -euo pipefail

# Usage: ./start_ray_worker.sh <HEAD_PRIVATE_IP> <CUSTOM_RESOURCE_NAME> [WORKER_PRIVATE_IP]
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <HEAD_PRIVATE_IP> <CUSTOM_RESOURCE_NAME> [WORKER_PRIVATE_IP]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HEAD_PRIVATE_IP="$1"
CUSTOM_RESOURCE_NAME="$2"
WORKER_PRIVATE_IP="${3:-$(hostname -I | awk '{print $1}')}"

cd "${PROJECT_ROOT}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

RESOURCES=$(printf '{"%s": 1, "training_worker": 1}' "${CUSTOM_RESOURCE_NAME}")

ray stop --force >/dev/null 2>&1 || true

ray start \
  --address="${HEAD_PRIVATE_IP}:6379" \
  --node-ip-address="${WORKER_PRIVATE_IP}" \
  --resources="${RESOURCES}"

echo "Ray worker joined ${HEAD_PRIVATE_IP}:6379"
echo "Worker private IP: ${WORKER_PRIVATE_IP}"
echo "Worker resources: ${RESOURCES}"

#!/usr/bin/env bash
set -euo pipefail

# Starts the Ray head node with the dashboard exposed externally.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HEAD_PRIVATE_IP="${1:-$(hostname -I | awk '{print $1}')}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"

cd "${PROJECT_ROOT}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

ray stop --force >/dev/null 2>&1 || true

ray start \
  --head \
  --node-ip-address="${HEAD_PRIVATE_IP}" \
  --port="${RAY_PORT}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${DASHBOARD_PORT}" \
  --disable-usage-stats \
  --resources='{"head_node": 1}'

echo "Ray head started at ${HEAD_PRIVATE_IP}:${RAY_PORT}"
echo "Dashboard: http://<HEAD_PUBLIC_IP>:${DASHBOARD_PORT}"
echo "Verify with: ray status"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_ID="memxd_$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="results/personamem/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting PersonaMem"
python -u runpersonamem.py --run-dir "${RUN_DIR}" "$@" 2>&1 | tee "${RUN_DIR}/run.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished PersonaMem"

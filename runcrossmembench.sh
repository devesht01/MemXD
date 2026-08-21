#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_ID="memxd_$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="results/crossmembench/${RUN_ID}"
mkdir -p "${RUN_DIR}/logs"

fmt_duration() {
  local s=$1
  printf '%dh%02dm%02ds' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

USERS=(
  u001 u002 u003 u004 u005 u006 u007 u008 u009 u010
  u011 u012 u013 u014 u015 u016 u017 u018 u019 u020
  u021 u022 u023 u024 u025 u026 u027 u028 u029 u030
  u031 u032 u033 u034 u035 u036 u037 u038 u039 u040
)

START_EPOCH=$(date +%s)
total=${#USERS[@]}
i=0
for user in "${USERS[@]}"; do
  i=$((i + 1))
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${user} (${i}/${total})"
  python -u runcrossmembench.py --run-dir "${RUN_DIR}" --user "${user}" "$@" 2>&1 | tee "${RUN_DIR}/logs/${user}.log"
  now=$(date +%s)
  elapsed=$((now - START_EPOCH))
  remaining=$((elapsed * (total - i) / i))
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${user} (${i}/${total}) elapsed=$(fmt_duration "${elapsed}") remaining=$(fmt_duration "${remaining}")" | tee -a "${RUN_DIR}/progress.log"
done

python -u analyze_crossmembench.py "${RUN_DIR}"

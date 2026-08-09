#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/home/chengcli/scix/workspace/UM-EARTH
RUNNER="$WORKSPACE/scripts/run_pte1b_extended_hres00_lead20_refined_56h_2gpu.sh"

# At 04:00 Eastern, use the previous Eastern calendar day's 00Z HRES cycle.
target_ymd=$(TZ=America/Detroit date -d yesterday +%Y%m%d)

echo "Execution time: $(TZ=America/Detroit date --iso-8601=seconds)"
echo "HRES base date: $target_ymd 00Z"
echo "Initial state: ${target_ymd} 00Z +20h"

exec /bin/bash "$RUNNER" "$target_ymd"

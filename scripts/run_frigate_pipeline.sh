#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=/home/chengcli/data/2025.FRIGATE
RUNS_DIR="$BASE_DIR/runs"
WORKSPACE=/home/chengcli/scix/workspace/UM-EARTH

target_ymd=${1:-20260414}
target_date=$(python - "$target_ymd" <<'PY'
from datetime import datetime
import sys

date = datetime.strptime(sys.argv[1], "%Y%m%d")
print(date.strftime("%Y-%m-%d"))
PY
)

run_dir="$RUNS_DIR/pte1b-$target_date"
forecast_input_dir="$run_dir/forecast_input"
tensor_dir="$forecast_input_dir/regridded_pte1b_${target_ymd}_00_tensors"
topography_dir="$run_dir/topography/products"
output_dir="$run_dir/forecast_refined_ghost_48h_2gpu"
session_name="pte1b_${target_ymd}_refined_48h_2gpu"
master_port=${MASTER_PORT:-$((29000 + 10#${target_ymd:4:4}))}

if [[ ! -f "$tensor_dir/regridded_pte1b_${target_ymd}_00.restart" ]]; then
  "$WORKSPACE/scripts/run_frigate_prepare.sh" "$target_ymd"
fi

mkdir -p "$output_dir"

cmd="cd $WORKSPACE && env CUDA_VISIBLE_DEVICES=0,1 PYTHONFAULTHANDLER=1 torchrun --nproc-per-node=2 --master-port $master_port $WORKSPACE/run_frigate_prediction.py -c $run_dir/pte1b.yaml -i $tensor_dir -o $output_dir --topography-dir $topography_dir --device cuda --refinement-mode staged --forcing-mode ghost --prediction-duration 108000 2>&1 | tee $output_dir/run.log"

tmux kill-session -t "$session_name" 2>/dev/null || true
tmux new-session -d -s "$session_name" "$cmd"

echo "Started session: $session_name"
echo "Master port: $master_port"
echo "Log: $output_dir/run.log"

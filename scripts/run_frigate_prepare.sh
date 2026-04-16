#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=/home/chengcli/data/2025.FRIGATE
RUNS_DIR="$BASE_DIR/runs"
WORKSPACE=/home/chengcli/scix/workspace/UM-EARTH
KML_FILE="$BASE_DIR/pte1b.kml"
TOPO_SOURCE_RUN="${TOPO_SOURCE_RUN:-$RUNS_DIR/pte1b-2026-04-01}"

target_ymd=${1:-20260414}
target_date=$(python - "$target_ymd" <<'PY'
from datetime import datetime
import sys

date = datetime.strptime(sys.argv[1], "%Y%m%d")
print(date.strftime("%Y-%m-%d"))
PY
)

forecast_dir="$BASE_DIR/ECWMF_prediction_data/$target_ymd"
run_dir="$RUNS_DIR/pte1b-$target_date"
forecast_input_dir="$run_dir/forecast_input"
topography_dir="$run_dir/topography/products"
raw_topography_dir="$run_dir/topography/raw/pte1b"
restart_bundle="$forecast_input_dir/regridded_pte1b_${target_ymd}_00_tensors/regridded_pte1b_${target_ymd}_00.restart"
source_merged_tif="$TOPO_SOURCE_RUN/topography/raw/pte1b/pte1b_merged_10m.tif"

echo "Preparing pte1b forecast input for $target_date"
echo "Forecast source: $forecast_dir"
echo "Run directory: $run_dir"

if [[ ! -d "$forecast_dir" ]]; then
  echo "Forecast input directory does not exist: $forecast_dir" >&2
  exit 1
fi

if [[ ! -f "$source_merged_tif" ]]; then
  echo "Source merged topography file not found: $source_merged_tif" >&2
  exit 1
fi

mkdir -p "$run_dir" "$forecast_input_dir" "$topography_dir" "$raw_topography_dir"

python - "$WORKSPACE" "$KML_FILE" "$run_dir/pte1b.yaml" "$target_date" <<'PY'
from pathlib import Path
import sys

workspace = Path(sys.argv[1])
kml_file = Path(sys.argv[2])
config_path = Path(sys.argv[3])
target_date = sys.argv[4]

sys.path.insert(0, str(workspace))

from um_earth.frigate_pipeline import build_prepared_domain, write_config
from um_earth.regions import load_region

region = load_region(region_kml=kml_file, location_id="pte1b")
prepared = build_prepared_domain(region)
write_config(config_path, region, prepared, date=target_date, nb2=2, nb3=1)
PY

cp "$source_merged_tif" "$raw_topography_dir/pte1b_merged_10m.tif"

python - "$WORKSPACE" "$KML_FILE" "$raw_topography_dir/pte1b_merged_10m.tif" "$topography_dir" <<'PY'
from pathlib import Path
import sys

workspace = Path(sys.argv[1])
kml_file = Path(sys.argv[2])
merged_tif = Path(sys.argv[3])
topography_dir = Path(sys.argv[4])

sys.path.insert(0, str(workspace))

from um_earth.frigate_pipeline import (
    DEFAULT_TARGET_RESOLUTIONS_KM,
    build_prepared_domain,
    build_resolution_products,
)
from um_earth.regions import load_region

region = load_region(region_kml=kml_file, location_id="pte1b")
prepared = build_prepared_domain(region)
build_resolution_products(
    merged_tif=merged_tif,
    out_dir=topography_dir,
    region_id=region.region_id,
    lat_bounds=(prepared.lat_min, prepared.lat_max),
    lon_bounds=(prepared.lon_min, prepared.lon_max),
    target_resolutions_km=DEFAULT_TARGET_RESOLUTIONS_KM,
)
PY

python "$WORKSPACE/prepare_initial_condition.py" \
  pte1b \
  --region-kml "$KML_FILE" \
  --config "$run_dir/pte1b.yaml" \
  --output-base "$forecast_input_dir" \
  --data-source forecast \
  --forecast-input-dir "$forecast_dir" \
  --forecast-cycle 00 \
  --forecast-leads 0 6 12 18 \
  --nY 2 --nX 1 \
  --timeout 7200

if [[ ! -f "$restart_bundle" ]]; then
  echo "Prepared restart bundle not found: $restart_bundle" >&2
  exit 1
fi

echo "Prepared restart bundle: $restart_bundle"

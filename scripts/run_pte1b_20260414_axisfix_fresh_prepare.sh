#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=/home/chengcli/data/2025.FRIGATE
RUN_DIR="$BASE_DIR/runs/pte1b-2026-04-14-axisfix-fresh"
WORKSPACE=/home/chengcli/scix/workspace/UM-EARTH
KML_FILE="$BASE_DIR/pte1b.kml"
FORECAST_DIR="$BASE_DIR/ECWMF_prediction_data/20260414"
SOURCE_MERGED_TIF="$BASE_DIR/runs/pte1b-2026-04-01/topography/raw/pte1b/pte1b_merged_10m.tif"

mkdir -p "$RUN_DIR/topography/raw/pte1b" "$RUN_DIR/topography/products" "$RUN_DIR/forecast_input"
cp "$SOURCE_MERGED_TIF" "$RUN_DIR/topography/raw/pte1b/pte1b_merged_10m.tif"

python - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, "/home/chengcli/scix/workspace/UM-EARTH")

from um_earth.frigate_pipeline import (
    DEFAULT_TARGET_RESOLUTIONS_KM,
    build_prepared_domain,
    build_resolution_products,
    write_config,
)
from um_earth.regions import load_region

base = Path("/home/chengcli/data/2025.FRIGATE")
run_dir = base / "runs" / "pte1b-2026-04-14-axisfix-fresh"
region = load_region(region_kml=base / "pte1b.kml", location_id="pte1b")
prepared = build_prepared_domain(region)

write_config(run_dir / "pte1b.yaml", region, prepared, date="2026-04-14", nb2=2, nb3=1)
build_resolution_products(
    merged_tif=run_dir / "topography" / "raw" / "pte1b" / "pte1b_merged_10m.tif",
    out_dir=run_dir / "topography" / "products",
    region_id=region.region_id,
    lat_bounds=(prepared.lat_min, prepared.lat_max),
    lon_bounds=(prepared.lon_min, prepared.lon_max),
    target_resolutions_km=DEFAULT_TARGET_RESOLUTIONS_KM,
)
PY

python "$WORKSPACE/prepare_initial_condition.py" \
  pte1b \
  --region-kml "$KML_FILE" \
  --config "$RUN_DIR/pte1b.yaml" \
  --output-base "$RUN_DIR/forecast_input" \
  --data-source forecast \
  --forecast-input-dir "$FORECAST_DIR" \
  --forecast-cycle 00 \
  --forecast-leads 0 6 12 18 \
  --nY 2 --nX 1 \
  --timeout 7200

echo "Prepared fresh April 14 axis-fix inputs under:"
echo "  $RUN_DIR"

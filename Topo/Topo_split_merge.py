#!/usr/bin/env python3
# Refine a merged DEM by an arbitrary integer factor Nx using interpolation,
# and save ONLY as TorchScript (.pt). 

# Usage:
#     python Topo_split_merge.py ws-site1 2x
#     python Topo_split_merge.py ws-site1 4x
#     python Topo_split_merge.py ws-site1 8x
#     python Topo_split_merge.py ws-site1 --test-window 512
#     python Topo_split_merge.py ws-site1 cubic (default linear)   

# Inputs:
#     - locations.csv (to get lat/lon bounds)
#     - merged DEM GeoTIFF:
#       <base-dir>/<location_id>/<location_id>_merged_10m.tif

# Outputs:
#     - TorchScript .pt (buffers):
#       <out-dir>/<location_id>/<location_id>_refined_<factor>x.pt

# Saved tensors (buffers):
#     - topography: (1, H, W) float32
#     - lat_bounds: (2,) float32
#     - lon_bounds: (2,) float32

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import io
import csv
import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator
import torch

# Path anchors (GitHub-safe)
SCRIPT_DIR = Path(__file__).resolve().parent      # e.g., UM-EARTH/Topo
PROJECT_ROOT = SCRIPT_DIR.parent                  # e.g., UM-EARTH 

DEFAULT_LOCATIONS = PROJECT_ROOT / "locations.csv"
DEFAULT_BASE_DIR  = PROJECT_ROOT / "Topo" / "Data" / "Raw"
DEFAULT_OUT_DIR   = PROJECT_ROOT / "Topo" / "Data" / "Split"


# TorchScript tensor saver (your preferred format)
def save_tensors(tensor_map: dict[str, torch.Tensor], filename: str):
    class TensorModule(torch.nn.Module):
        def __init__(self, tensors):
            super().__init__()
            for name, tensor in tensors.items():
                self.register_buffer(name, tensor)

    module = TensorModule(tensor_map)
    scripted = torch.jit.script(module)
    scripted.save(filename)




# TopographyData (minimal)

class TopographyData:
    def __init__(self, data, lat_bounds, lon_bounds, nlat, nlon):
        if data.shape != (nlat, nlon):
            raise ValueError(f"Shape mismatch: {data.shape} vs ({nlat},{nlon})")
        self.data = data
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.nlat = nlat
        self.nlon = nlon

    def __repr__(self):
        return f"TopographyData(shape={self.data.shape}, lat={self.lat_bounds}, lon={self.lon_bounds})"
    

# locations.csv -> bounds
def read_bounds_from_locations(locations_csv: Path, location_id: str):
    """
    Read (lat_bounds, lon_bounds) for a location_id from locations.csv.
    Skips comment lines starting with '#'.
    """
    if not locations_csv.exists():
        raise FileNotFoundError(locations_csv)

    with locations_csv.open("r", encoding="utf-8") as f:
        lines = [line for line in f if not line.strip().startswith("#") and line.strip()]

    reader = csv.DictReader(io.StringIO("".join(lines)), delimiter=",", skipinitialspace=True)
    reader.fieldnames = [h.strip() for h in reader.fieldnames]

    required = {"Name", "Latmin", "Latmax", "Lonmin", "Lonmax"}
    missing = required - set(reader.fieldnames)
    if missing:
        raise ValueError(f"locations.csv missing columns: {sorted(missing)}")

    for row in reader:
        row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        if row["Name"] == location_id:
            lat_bounds = (float(row["Latmin"]), float(row["Latmax"]))
            lon_bounds = (float(row["Lonmin"]), float(row["Lonmax"]))
            return lat_bounds, lon_bounds

    raise KeyError(f"location_id '{location_id}' not found in {locations_csv}")


# Read GeoTIFF -> TopographyData
def from_geotiff(tif_path: Path, lat_bounds, lon_bounds):
    with rasterio.open(str(tif_path)) as src:
        arr = src.read(1)  # (nlat, nlon)
        crs = src.crs
        res = src.res

    topo = TopographyData(arr, lat_bounds, lon_bounds, arr.shape[0], arr.shape[1])
    return topo, crs, res


# Interpolation utilities
def interpolate_to_shape(topo: TopographyData, target_nlat: int, target_nlon: int, method: str) -> TopographyData:
    # Interpolate topo.data to (target_nlat, target_nlon) over the same bounds.
    if method not in ("linear", "cubic"):
        raise ValueError("method must be 'linear' or 'cubic'")

    lat_old = np.linspace(topo.lat_bounds[0], topo.lat_bounds[1], topo.nlat)
    lon_old = np.linspace(topo.lon_bounds[0], topo.lon_bounds[1], topo.nlon)
    lat_new = np.linspace(topo.lat_bounds[0], topo.lat_bounds[1], target_nlat)
    lon_new = np.linspace(topo.lon_bounds[0], topo.lon_bounds[1], target_nlon)

    interp = RegularGridInterpolator(
        (lat_old, lon_old),
        topo.data,
        method=method,
        bounds_error=False,
        fill_value=None,
    )

    lat_m, lon_m = np.meshgrid(lat_new, lon_new, indexing="ij")
    pts = np.stack([lat_m.ravel(), lon_m.ravel()], axis=-1)
    new_data = interp(pts).reshape(target_nlat, target_nlon).astype(np.float32, copy=False)

    return TopographyData(
        data=new_data,
        lat_bounds=topo.lat_bounds,
        lon_bounds=topo.lon_bounds,
        nlat=target_nlat,
        nlon=target_nlon,
    )


def split_no_download_once(topo: TopographyData, method="linear") -> TopographyData:
    # One split step: 2x resolution via interpolation.
    return interpolate_to_shape(topo, topo.nlat * 2, topo.nlon * 2, method)


def parse_factor_pow2(factor_str: str) -> int:
    """
    Parse factor like 2x, 4x, 8x, 16x (power of two only).
    """
    s = factor_str.strip().lower()
    if not s.endswith("x"):
        raise ValueError("factor must look like 2x, 4x, 8x")

    try:
        factor = int(s[:-1])
    except ValueError:
        raise ValueError("factor must be an integer like 2x, 4x, 8x")

    if factor < 2 or (factor & (factor - 1)) != 0:
        raise ValueError("factor must be a power of two: 2x, 4x, 8x, ...")

    return factor



def largest_pow2_leq(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def estimate_bytes_float32(nlat: int, nlon: int) -> int:
    return int(nlat) * int(nlon) * 4


# CLI
def parse_args():
    ap = argparse.ArgumentParser(
        description="Refine merged DEM by power-of-two factor and save as TorchScript (.pt)."
    )
    ap.add_argument("location_id", help="e.g. ws-site1")
    ap.add_argument(
        "factor",
        help="Refinement factor (power of two), e.g. 2x, 4x, 8x (N >= 2)"
    )

    ap.add_argument(
         "--locations",
        type=Path,
        default=DEFAULT_LOCATIONS,
        help="Path to locations.csv (default: UM-EARTH/locations.csv)"
    )

    ap.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory containing <location>/<location>_merged_10m.tif (default: UM-EARTH/Topo/Data/Raw)"
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for TorchScript .pt (default: UM-EARTH/Topo/Data/Split)"
    )

    ap.add_argument(
        "--method",
        choices=["linear", "cubic"],
        default="linear",
        help="Interpolation method (default: linear). cubic is slower."
    )

    ap.add_argument(
        "--max-gb",
        type=float,
        default=6.0,
        help="Abort if estimated output array exceeds this many GB (float32). Default 6 GB."
    )
    ap.add_argument(
    "--test-window",
    type=int,
    default=None,
    help="Crop a NxN window from merged DEM for quick testing (e.g., 512)"
)


    return ap.parse_args()


# main

def main():
    args = parse_args()

    try:
        factor = parse_factor_pow2(args.factor)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Input DEM (only read)
    in_tif = args.base_dir / args.location_id / f"{args.location_id}_merged_10m.tif"
    if not in_tif.exists():
        print(f"ERROR: input not found: {in_tif}", file=sys.stderr)
        sys.exit(1)

    # bounds from locations.csv
    try:
        lat_bounds, lon_bounds = read_bounds_from_locations(args.locations, args.location_id)
    except Exception as e:
        print(f"ERROR reading bounds from locations.csv: {e}", file=sys.stderr)
        sys.exit(1)

    # output dir per location
    out_dir = args.out_dir / args.location_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Locations:", args.locations.resolve())
    print("Input DEM:", in_tif.resolve())
    print("Output dir:", out_dir.resolve())
    print("Location:", args.location_id)
    print("lat_bounds:", lat_bounds)
    print("lon_bounds:", lon_bounds)
    print("Requested factor:", f"{factor}x")
    print("Method:", args.method)

    topo, crs, res = from_geotiff(in_tif, lat_bounds, lon_bounds)
    if args.test_window is not None:
        n = args.test_window
        print(f"[TEST MODE] Cropping a {n}x{n} window for quick validation")

        if topo.nlat < n or topo.nlon < n:
            raise ValueError("Test window larger than DEM size")

        topo = TopographyData(
            data=topo.data[:n, :n],
            lat_bounds=topo.lat_bounds,   # bounds 对测试不重要
            lon_bounds=topo.lon_bounds,
            nlat=n,
            nlon=n,
        )

        print("Test topo shape:", topo.data.shape)

    print("Base:", topo)
    print("CRS:", crs)
    print("Resolution (x,y):", res)
    print("base dtype:", topo.data.dtype, "min/max:", float(np.nanmin(topo.data)), float(np.nanmax(topo.data)))

    # estimate output size
    out_nlat = topo.nlat * factor
    out_nlon = topo.nlon * factor
    est_gb = estimate_bytes_float32(out_nlat, out_nlon) / (1024**3)
    print(f"Estimated output array size (float32): {est_gb:.2f} GB")
    if est_gb > args.max_gb:
        print(f"ABORT: estimated output {est_gb:.2f} GB exceeds --max-gb {args.max_gb:.2f}.", file=sys.stderr)
        print("Tip: choose smaller factor, or increase --max-gb if you truly have enough RAM.", file=sys.stderr)
        sys.exit(2)

    # plan: split to pow2 then final interpolate if needed
    pow2 = largest_pow2_leq(factor)
    n_split = int(np.log2(pow2))
    residual = factor / pow2  # >= 1

    print(f"Split plan: reach {pow2}x via {n_split} split(s)")
    if residual != 1:
        print(f"Then final interpolate by residual factor {residual:.6f} to reach {factor}x exactly")

    # split steps
    for i in range(n_split):
        print(f"Split {i+1}/{n_split} ...")
        topo = split_no_download_once(topo, method=args.method)

    # final interpolate if needed
    if residual != 1:
        target_nlat = int(round(topo.nlat * residual))
        target_nlon = int(round(topo.nlon * residual))
        print(f"Final interpolate to target shape: ({target_nlat}, {target_nlon}) ...")
        topo = interpolate_to_shape(topo, target_nlat, target_nlon, method=args.method)

    # save TorchScript tensors
    topo_tensor = torch.from_numpy(topo.data).unsqueeze(0)  # (1, H, W)
    tensor_map = {
        "topography": topo_tensor.contiguous(),
        "lat_bounds": torch.tensor(topo.lat_bounds, dtype=torch.float32),
        "lon_bounds": torch.tensor(topo.lon_bounds, dtype=torch.float32),
    }

    out_pt = out_dir / f"{args.location_id}_refined_{factor}x.pt"
    print("Writing TorchScript:", out_pt.resolve())
    save_tensors(tensor_map, str(out_pt))

    print("Saved: ", out_pt.resolve())
    print("topography shape:", tuple(topo_tensor.shape), "dtype:", topo_tensor.dtype)
    print("DONE.")


if __name__ == "__main__":
    main()
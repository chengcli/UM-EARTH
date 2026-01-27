# This script performs the complete preprocessing workflow for USGS 3DEP DEM data:

# 1. Query the USGS TNM Access API to identify all 3DEP DEM tiles intersecting
#    a user-defined geographic region.
# 2. Download the corresponding 10 m (1/3 arc-second) DEM GeoTIFF tiles
#    (unless the download step is explicitly skipped).
# 3. Clip each DEM tile to the exact bounding box defined in locations.csv
#    using window-based raster clipping.
# 4. Merge all clipped tiles into a single, seamless GeoTIFF covering
#    the target region.

# Usage:
#     python Topo_download_crop.py <location-id> [options]

# To skip the download step and reuse existing DEM tiles:
#     python Topo_download_crop.py <location-id> --skip-download

# Notes:
# - Input DEM: USGS 3DEP National Elevation Dataset (NED), 10 m resolution
# - Coordinate Reference System (CRS): EPSG:4269 (geographic coordinates)
# - Units: degrees (horizontal), meters (elevation)
# - Bounding boxes are read from locations.csv
# - DEM tiles are clipped conservatively to guarantee full spatial coverage
# - Requires `seamless-3dep` for downloading USGS 3DEP GeoTIFF files
#   (install with: pip install seamless-3dep)



# 1. download DEM file
#!/usr/bin/env python3
from pathlib import Path
import sys, io, csv
import argparse
import subprocess
import requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import Window
from rasterio.merge import merge
from rasterio.transform import from_origin
import math
import torch



SCRIPT_DIR = Path(__file__).resolve().parent      # UM-EARTH/Topo
PROJECT_ROOT = SCRIPT_DIR.parent                 # UM-EARTH
DEFAULT_LOCATIONS = PROJECT_ROOT / "locations.csv"
DEFAULT_OUT = PROJECT_ROOT / "Topo" / "Data" / "Raw"

# 1.1 read locations.csv
def load_locations(locations_file):
    locations = {}

    with open(locations_file, "r", encoding="utf-8") as f:
        lines = [
            line for line in f
            if not line.strip().startswith("#") and line.strip()
        ]

    csv_data = io.StringIO("".join(lines))
    reader = csv.DictReader(csv_data, delimiter=",", skipinitialspace=True)
    reader.fieldnames = [h.strip() for h in reader.fieldnames]

    for row in reader:
        row = {k.strip(): v.strip() for k, v in row.items()}

        latmin = float(row["Latmin"])
        latmax = float(row["Latmax"])
        lonmin = float(row["Lonmin"])
        lonmax = float(row["Lonmax"])

        polygon = [
            [lonmin, latmin],
            [lonmin, latmax],
            [lonmax, latmax],
            [lonmax, latmin],
        ]

        locations[row["Name"]] = {
            "name": row["Description"],
            "polygon": polygon,
        }

    return locations


def polygon_to_bbox(polygon):
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    return (min(lons), min(lats), max(lons), max(lats))

def save_tensors(tensor_map: dict[str, torch.Tensor], filename: str):
    class TensorModule(torch.nn.Module):
        def __init__(self, tensors):
            super().__init__()
            for name, tensor in tensors.items():
                self.register_buffer(name, tensor)

    module = TensorModule(tensor_map)
    scripted = torch.jit.script(module)
    scripted.save(filename)


# 1.2 argparse：

def parse_args():
    ap = argparse.ArgumentParser(
        description="Download USGS 3DEP DEM using bbox from locations.csv (wget-based)"
    )
    ap.add_argument(
        "location_id",
        help="Location ID in locations.csv (e.g., ws-site1)"
    )
    ap.add_argument(
       "--locations",
        type=Path,
        default=DEFAULT_LOCATIONS,
        help="Path to locations.csv (default: UM-EARTH/locations.csv)"
    )
    ap.add_argument(
    "--skip-download",
    action="store_true",
    help="Skip download step and use existing tif files"
    )

    ap.add_argument(
         "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory (default: UM-EARTH/Topo/Data/Raw)"
    
    )
    ap.add_argument(
        "--save-pt",
        action="store_true",
        help="Also save coarse-mean topography as TorchScript (.pt)"
    )
    return ap.parse_args()



# 1.3 BBox → TNM API → wget
def main():
    args = parse_args()

    locations = load_locations(args.locations)

    if args.location_id not in locations:
        print(f"ERROR: {args.location_id} not found in locations.csv")
        print("Available:", ", ".join(locations.keys()))
        sys.exit(1)

    info = locations[args.location_id]
    bbox = polygon_to_bbox(info["polygon"])
    download_bbox = (
        bbox[0], # left
        bbox[1],   # bottom
        bbox[2], # right
        bbox[3], # Top
        )
    print("Locations file:", args.locations.resolve())
    print("Output dir:    ", args.out.resolve())
    print("Location:      ", args.location_id, "-", info["name"])
    print("BBox (W,S,E,N):", bbox)

    args.out.mkdir(parents=True, exist_ok=True)


    # TNM Access API 

    endpoint = "https://tnmaccess.nationalmap.gov/api/v1/products"
    params = {
        "datasets": "National Elevation Dataset (NED) 1/3 arc-second Current",
        "bbox": f"{download_bbox[0]},{download_bbox[1]},{download_bbox[2]},{download_bbox[3]}",
        "max": 200,
    }


    print("\nQuerying USGS TNM API...")
    r = requests.get(endpoint, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    items = data.get("items", [])
    if not items:
        print("No DEM products returned.")
        sys.exit(1)

    print(f"Found {len(items)} DEM file(s). Starting download...\n")


    # Download DEM tiles (optional)

    save_dir = args.out / args.location_id
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Save to location:", save_dir.resolve())

    if args.skip_download:
        print("\n[SKIP] Download step skipped. Using existing tif files.")
    else:
        print(f"\nFound {len(items)} DEM file(s). Starting download...\n")

        for i, item in enumerate(items, 1):
            url = item.get("downloadURL")
            if not url:
                continue

            print("=" * 70)
            print(f"[{i}/{len(items)}] {url}")

            subprocess.run(
                ["wget", "-c", url, "-P", str(save_dir)],
                check=True
        )

    print("\nAll downloads completed.")



    # 2) Clip each DEM tile to the bounding box
    import rasterio
    import math
    from rasterio.windows import Window
    from rasterio.windows import from_bounds
    from rasterio.merge import merge

    print("\nClipping tiles to bbox window...")

    # Search recursively for GeoTIFF files (in case of subdirectories)
    raw_tifs = sorted(list(save_dir.rglob("*.tif")) + list(save_dir.rglob("*.tiff")))
    if not raw_tifs:
        print(f"No tif files found in {save_dir}")
        sys.exit(1)

    clipped_tifs = []

    for fp in raw_tifs:
        clip_fp = save_dir / f"clip_{fp.name}"

        with rasterio.open(fp) as src:
            # 1) bbox -> float window
            win0 = from_bounds(*bbox, transform=src.transform)

            # Expand window outward to guarantee full coverage
            win = Window(
                col_off=math.floor(win0.col_off),
                row_off=math.floor(win0.row_off),
                width=math.ceil(win0.width),
                height=math.ceil(win0.height),
            )

            # Clamp window to image bounds
            from rasterio.errors import WindowError

            try:
                win = win.intersection(Window(0, 0, src.width, src.height))
            except WindowError:
                print(f"  skip (no overlap after clamp): {fp.name}")
                continue

            if win.width <= 0 or win.height <= 0:
                print(f"  skip (no overlap): {fp.name}")
                continue

            data = src.read(1, window=win)
            if data.size == 0:
                print(f"  skip (empty): {fp.name}")
                continue

            new_transform = rasterio.windows.transform(win, src.transform)

            profile = src.profile.copy()
            profile.update(
                height=data.shape[0],
                width=data.shape[1],
                transform=new_transform,
                count=1,
            )

        # Write clipped tile
        with rasterio.open(clip_fp, "w", **profile) as dst:
            dst.write(data, 1)

        # Sanity check: southern boundary coverage
        with rasterio.open(fp) as src:
            src_b = src.bounds   

        with rasterio.open(clip_fp) as chk:
            b = chk.bounds

            # src_b.bottom <= bbox[1] <= src_b.top
            if (src_b.bottom <= bbox[1] <= src_b.top):
                if b.bottom > bbox[1] + 1e-6:
                    raise RuntimeError(
                        f"Clip south bound {b.bottom} is north of requested {bbox[1]} "
                        f"for {clip_fp.name}"
                    )       


        clipped_tifs.append(clip_fp)
        print(f"  clipped: {clip_fp.name}")

    if not clipped_tifs:
        print("No clipped tiles were produced.")
        sys.exit(1)

    print(f"Clipped {len(clipped_tifs)} tile(s).")


    # 3) Merge all clipped tiles into a single GeoTIFF

    print("\nMerging clipped tiles...")

    datasets = [rasterio.open(str(p)) for p in clipped_tifs]
    mosaic, out_transform = merge(datasets)

    out_profile = datasets[0].profile.copy()
    out_profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=out_transform,
    )

    for ds in datasets:
        ds.close()

    merged_fp = save_dir / f"{args.location_id}_merged_10m.tif"
    with rasterio.open(merged_fp, "w", **out_profile) as dst:
        dst.write(mosaic)

    print("\nDONE.")
    print("Merged DEM:", merged_fp.resolve())
    
    # 4 Coarsen merged DEM to an NxN lat/lon grid by computing cell mean
    print("\nComputing coarse-grid cell mean from merged DEM...")

    lon_min, lat_min, lon_max, lat_max = bbox  # bbox = (W,S,E,N)

    nx = ny = 60  

    lon_edges = np.linspace(lon_min, lon_max, nx + 1)
    lat_edges = np.linspace(lat_min, lat_max, ny + 1)

    cell_mean = np.full((ny, nx), np.nan, dtype=np.float32)

    with rasterio.open(merged_fp) as ds:
        crs = ds.crs  # inherit DEM CRS (EPSG:4269)

        for j in range(ny):  # j=0 is the southernmost cell
            for i in range(nx):
                w, e = lon_edges[i], lon_edges[i + 1]
                s, n = lat_edges[j], lat_edges[j + 1]

                win = from_bounds(w, s, e, n, ds.transform)

                # Make sure window is within raster bounds
                try:
                    win = win.intersection(Window(0, 0, ds.width, ds.height))
                except Exception:
                    continue

                if win.width <= 0 or win.height <= 0:
                    continue

                data = ds.read(1, window=win, masked=True)

                # If fully masked or empty, keep NaN
                if data.size > 0 and (not np.all(data.mask)):
                    cell_mean[j, i] = float(data.mean())

    # Write GeoTIFF (north-up expects first row = north), so flip vertically
    cell_mean_northup = np.flipud(cell_mean).copy()

    dx = (lon_max - lon_min) / nx
    dy = (lat_max - lat_min) / ny
    transform = from_origin(lon_min, lat_max, dx, dy)  # (west, north, xsize, ysize)

    out_tif = save_dir / f"{args.location_id}_topo_{nx}x{ny}_mean.tif"

    profile2 = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": -9999.0,
    }

    to_write = np.where(
        np.isfinite(cell_mean_northup),
        cell_mean_northup,
        profile2["nodata"]
    ).astype(np.float32)

    with rasterio.open(out_tif, "w", **profile2) as dst:
        dst.write(to_write, 1)

    print("Saved coarse-mean GeoTIFF:", out_tif.resolve())
    print("dx, dy (deg):", dx, dy)
    print(f"{nx}x{ny} (south->north indexing) array:\n", cell_mean)
    
    # 5 Save coarse-mean as TorchScript .pt for model ingestion
    if args.save_pt:
        topo_tensor = torch.from_numpy(cell_mean_northup).unsqueeze(0).contiguous()  # (1, H, W)

        tensor_map = {
            "topography": topo_tensor,
            "lat_bounds": torch.tensor([lat_min, lat_max], dtype=torch.float32),
            "lon_bounds": torch.tensor([lon_min, lon_max], dtype=torch.float32),
        }

        out_pt = save_dir / f"{args.location_id}_topo_{nx}x{ny}_mean.pt"
        print("Writing TorchScript:", out_pt.resolve())
        save_tensors(tensor_map, str(out_pt))

        print("Saved:", out_pt.resolve())
        print("topography shape:", tuple(topo_tensor.shape), "dtype:", topo_tensor.dtype)

if __name__ == "__main__":
    main()
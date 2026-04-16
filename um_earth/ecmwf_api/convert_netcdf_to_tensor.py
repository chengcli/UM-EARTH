#!/usr/bin/env python3
"""
Convert NetCDF Block Files to PyTorch Tensor Files (Step 6)

This script converts decomposed NetCDF files from Step 5 into PyTorch tensor files
for use in dynamic atmospheric simulations. It reads each block file, processes the
variables, and saves them as LibTorch-compatible tensors.

The script performs the following operations:
1. Reads variables from NetCDF file: rho, w, u, v, p, q, ciwc, clwc, cswc, crwc
2. Combines cloud variables:
   - ciwc + clwc -> q2 (cloud ice + cloud liquid water)
   - cswc + crwc -> q3 (cloud snow + cloud rain water)
3. Aggregates variables into hydro_w tensor with ordering:
   (rho, w, u, v, p, q, q2, q3) -> shape (time, nvar=8, x3, x2, x1)
4. Saves tensor to .part file using torch.jit.script for LibTorch compatibility

Usage:
    python convert_netcdf_to_tensor.py <input_file.nc> [--output OUTPUT.part]
    python convert_netcdf_to_tensor.py <input_dir> [--output-dir OUTPUT_DIR]

Examples:
    # Convert a single file
    python convert_netcdf_to_tensor.py regridded_block_0_0.nc
    
    # Convert a single file with custom output
    python convert_netcdf_to_tensor.py regridded_block_0_0.nc --output block0.part
    
    # Convert all .nc files in a directory
    python convert_netcdf_to_tensor.py ./blocks/ --output-dir ./tensors/

Output:
    - .part files with a similar basename as the input .nc files
    - Each file contains a TensorModule with 'hydro_w' tensor
    - hydro_w shape: (time, nvar=8, x3, x2, x1)
    - Variable order in nvar dimension: rho, w, u, v, p, q, q2, q3
"""

import argparse
import os
import sys
import tarfile
from typing import Dict, Optional
from pathlib import Path
import re

import yaml
from scipy.interpolate import RegularGridInterpolator

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip install torch")

try:
    from netCDF4 import Dataset
    import numpy as np
except ImportError:
    raise ImportError("netCDF4 and numpy are required. Install with: pip install netCDF4 numpy")


def save_tensors(tensor_map: Dict[str, torch.Tensor], filename: str) -> None:
    """
    Save tensors to a file using torch.jit for LibTorch compatibility.
    
    Args:
        tensor_map: Dictionary mapping tensor names to torch tensors
        filename: Output .part file path
    """
    class TensorModule(torch.nn.Module):
        def __init__(self, tensors):
            super().__init__()
            for name, tensor in tensors.items():
                self.register_buffer(name, tensor)
    
    module = TensorModule(tensor_map)
    scripted = torch.jit.script(module)  # Needed for LibTorch compatibility
    scripted.save(filename)


BLOCK_PATTERN = re.compile(r"^(?P<stem>.+)_block_(?P<i2>\d+)_(?P<i3>\d+)$")


def parse_block_indices(base_name: str) -> tuple[str, int, int] | None:
    match = BLOCK_PATTERN.match(base_name)
    if match is None:
        return None
    return match.group("stem"), int(match.group("i2")), int(match.group("i3"))


def block_rank(i2: int, i3: int, n_blocks_x3: int) -> int:
    return i2 * n_blocks_x3 + i3


def block_part_name(stem: str, rank: int, file_number: int = 0) -> str:
    return f"{stem}.block{rank}.{file_number:05d}.part"


def bundle_restart_parts(part_files: list[str], restart_file: str) -> str:
    restart_path = Path(restart_file)
    restart_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(restart_path, "w") as tar:
        for path in sorted(part_files):
            tar.add(path, arcname=Path(path).name)
    return str(restart_path)


def load_topography_buffers(topo_file: str) -> dict[str, torch.Tensor]:
    module = torch.jit.load(str(topo_file))
    data: dict[str, torch.Tensor] = {}
    for name, buf in module.named_buffers(recurse=True):
        data[name] = buf.detach().cpu()
    for name, param in module.named_parameters(recurse=True):
        data[name] = param.detach().cpu()
    return data


def orient_topography_array(topo_data: dict[str, torch.Tensor]) -> np.ndarray:
    topo = topo_data["topography"].detach().cpu().numpy().astype(np.float64)
    row0_is_north = topo_data.get("row0_is_north")
    if row0_is_north is not None and bool(row0_is_north.item()):
        topo = np.flip(topo, axis=0)
    return topo


def load_geometry_from_config(config_file: str) -> dict:
    with open(config_file, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    geometry = config["geometry"]
    bounds = geometry["bounds"]
    cells = geometry["cells"]
    return {
        "bounds": {
            "x2min": float(bounds["x2min"]),
            "x2max": float(bounds["x2max"]),
            "x3min": float(bounds["x3min"]),
            "x3max": float(bounds["x3max"]),
        },
        "cells": {
            "nx2": int(cells["nx2"]),
            "nx3": int(cells["nx3"]),
            "nghost": int(cells["nghost"]),
        },
    }


def compute_interior_horizontal_centers(geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    bounds = geometry["bounds"]
    cells = geometry["cells"]
    nghost = cells["nghost"]
    nx2_total = cells["nx2"] + 2 * nghost
    nx3_total = cells["nx3"] + 2 * nghost

    x2f = np.linspace(bounds["x2min"], bounds["x2max"], nx2_total + 1, dtype=np.float64)
    x3f = np.linspace(bounds["x3min"], bounds["x3max"], nx3_total + 1, dtype=np.float64)
    x2 = 0.5 * (x2f[:-1] + x2f[1:])
    x3 = 0.5 * (x3f[:-1] + x3f[1:])
    return x2[nghost:-nghost], x3[nghost:-nghost]


def topography_on_block_grid(
    topo_file: str,
    config_file: str,
    x2_block: np.ndarray,
    x3_block: np.ndarray,
) -> np.ndarray:
    topo_data = load_topography_buffers(topo_file)
    topo_x3x2 = orient_topography_array(topo_data)
    x2_global, x3_global = compute_interior_horizontal_centers(load_geometry_from_config(config_file))

    if topo_x3x2.shape != (len(x3_global), len(x2_global)):
        raise ValueError(
            f"Topography shape {topo_x3x2.shape} does not match config-derived "
            f"(x3, x2)=({len(x3_global)}, {len(x2_global)})"
        )

    interp = RegularGridInterpolator(
        (x3_global, x2_global),
        topo_x3x2,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    x2_query = np.clip(np.asarray(x2_block, dtype=np.float64), x2_global[0], x2_global[-1])
    x3_query = np.clip(np.asarray(x3_block, dtype=np.float64), x3_global[0], x3_global[-1])
    x3_mesh, x2_mesh = np.meshgrid(x3_query, x2_query, indexing="ij")
    return interp(np.stack([x3_mesh.ravel(), x2_mesh.ravel()], axis=-1)).reshape(len(x3_query), len(x2_query))


def zero_velocity_below_topography(
    variables: Dict[str, np.ndarray],
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    *,
    topo_file: str,
    config_file: str,
) -> int:
    topo_x3x2 = topography_on_block_grid(topo_file, config_file, x2, x3)
    topo_x2x3 = topo_x3x2.T
    below_topo = np.asarray(x1, dtype=np.float64)[:, np.newaxis, np.newaxis] < topo_x2x3[np.newaxis, :, :]
    zeroed = int(np.count_nonzero(below_topo))
    if zeroed == 0:
        return 0

    for name in ("w", "u", "v"):
        variables[name] = np.where(below_topo[np.newaxis, ...], 0.0, variables[name])
    return zeroed


def convert_netcdf_to_tensor(
    input_file: str,
    output_file: Optional[str] = None,
    *,
    topo_file: Optional[str] = None,
    config_file: Optional[str] = None,
) -> str:
    """
    Convert a single NetCDF block file to PyTorch tensor file.
    
    Args:
        input_file: Path to input NetCDF file from Step 5
        output_file: Optional output .part file path. If None, uses same basename.
    
    Returns:
        Path to output .part file
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required variables are missing
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Determine output file path
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.dirname(input_file)
        output_file = os.path.join(output_dir, f"{base_name}.part")
    
    print(f"Converting {input_file} -> {output_file}")
    
    # Read NetCDF file
    with Dataset(input_file, 'r') as nc:
        # Get dimensions
        n_time = len(nc.dimensions['time'])
        n_x1 = len(nc.dimensions['x1'])
        n_x2 = len(nc.dimensions['x2'])
        n_x3 = len(nc.dimensions['x3'])
        
        print(f"  Dimensions: time={n_time}, x1={n_x1}, x2={n_x2}, x3={n_x3}")
        
        # Required variables (on cell centers with dimensions: time, x1, x2, x3)
        required_vars = ['rho', 'w', 'v', 'u', 'q']
        cloud_vars = ['ciwc', 'clwc', 'cswc', 'crwc']
        x1_values = np.asarray(nc.variables["x1"][:], dtype=np.float64)
        x2_values = np.asarray(nc.variables["x2"][:], dtype=np.float64)
        x3_values = np.asarray(nc.variables["x3"][:], dtype=np.float64)
        
        # Check for pressure variable - could be 'p', 'pressure', or 'pressure_level'
        pressure_var = None
        if 'p' in nc.variables:
            pressure_var = 'p'
        elif 'pressure' in nc.variables:
            pressure_var = 'pressure'
        elif 'pressure_level' in nc.variables:
            # pressure_level might be on interfaces (x1f), need to handle carefully
            pressure_var = 'pressure_level'
        else:
            raise ValueError("No pressure variable found. Expected 'p', 'pressure', or 'pressure_level'")
        
        # Check for missing required variables
        missing_vars = [var for var in required_vars if var not in nc.variables]
        if missing_vars:
            available_vars = list(nc.variables.keys())
            raise ValueError(
                f"Missing required variables: {missing_vars}. "
                f"Available variables: {available_vars}"
            )
        
        # Check for missing cloud variables
        missing_cloud = [var for var in cloud_vars if var not in nc.variables]
        if missing_cloud:
            print(f"  Warning: Missing cloud variables: {missing_cloud}")
            print(f"  Will use zeros for missing variables")
        
        # Load variables into numpy arrays
        # Note: NetCDF has shape (time, x1, x2, x3)
        variables = {}
        
        for var_name in required_vars:
            var_data = nc.variables[var_name][:]
            if var_data.shape != (n_time, n_x1, n_x2, n_x3):
                raise ValueError(
                    f"Variable '{var_name}' has unexpected shape {var_data.shape}, "
                    f"expected ({n_time}, {n_x1}, {n_x2}, {n_x3})"
                )
            variables[var_name] = var_data
            print(f"  Loaded {var_name}: shape {var_data.shape}")
        
        # Load pressure
        p_data = nc.variables[pressure_var][:]
        
        # Handle pressure_level which might be on interfaces
        if pressure_var == 'pressure_level' and len(p_data.shape) == 4:
            if p_data.shape[1] == n_x1 + 1:
                # Pressure on interfaces, need to average to cell centers
                print(f"  Pressure on interfaces, averaging to cell centers")
                p_data = 0.5 * (p_data[:, :-1, :, :] + p_data[:, 1:, :, :])
        
        if p_data.shape != (n_time, n_x1, n_x2, n_x3):
            raise ValueError(
                f"Pressure variable '{pressure_var}' has unexpected shape {p_data.shape}, "
                f"expected ({n_time}, {n_x1}, {n_x2}, {n_x3})"
            )
        variables['p'] = p_data
        print(f"  Loaded pressure: shape {p_data.shape}")
        
        # Load cloud variables (use zeros if missing)
        for var_name in cloud_vars:
            if var_name in nc.variables:
                var_data = nc.variables[var_name][:]
                if var_data.shape != (n_time, n_x1, n_x2, n_x3):
                    raise ValueError(
                        f"Variable '{var_name}' has unexpected shape {var_data.shape}, "
                        f"expected ({n_time}, {n_x1}, {n_x2}, {n_x3})"
                    )
                variables[var_name] = var_data
                print(f"  Loaded {var_name}: shape {var_data.shape}")
            else:
                variables[var_name] = np.zeros((n_time, n_x1, n_x2, n_x3), dtype=np.float32)
                print(f"  Using zeros for missing {var_name}")

    if topo_file is not None or config_file is not None:
        if topo_file is None or config_file is None:
            raise ValueError("Both topo_file and config_file are required when zeroing velocities below topography")
        zeroed = zero_velocity_below_topography(
            variables,
            x1_values,
            x2_values,
            x3_values,
            topo_file=topo_file,
            config_file=config_file,
        )
        print(f"  Zeroed u/v/w below topography in {zeroed} cell centers per time slice")
    
    # Step 1: Combine cloud variables
    # q2 = ciwc + clwc (cloud ice + cloud liquid water)
    # q3 = cswc + crwc (cloud snow + cloud rain water)
    q2 = variables['ciwc'] + variables['clwc']
    q3 = variables['cswc'] + variables['crwc']
    
    print(f"  Combined ciwc + clwc -> q2")
    print(f"  Combined cswc + crwc -> q3")
    
    # Step 2: Aggregate into hydro_w tensor
    # Order: (rho, w, u, v, p, q, q2, q3)
    # Output shape: (time, nvar=8, x3, x2, x1)
    # Note: Need to reorder axes from (time, x1, x2, x3) to (time, x1, x2, x3) -> (time, x3, x2, x1)
    
    var_list = [
        variables['rho'],
        variables['w'],
        variables['u'],
        variables['v'],
        variables['p'],
        variables['q'],
        q2,
        q3
    ]
    
    # Stack along new axis to get shape (time, nvar=8, x1, x2, x3)
    hydro_w_np = np.stack(var_list, axis=1)
    
    # Reorder axes from (time, nvar, x1, x2, x3) to (time, nvar, x3, x2, x1)
    # This is: (0, 1, 2, 3, 4) -> (0, 1, 4, 3, 2)
    hydro_w_np = np.transpose(hydro_w_np, (0, 1, 4, 3, 2))
    
    print(f"  Created hydro_w tensor: shape {hydro_w_np.shape}")
    print(f"  Variable order: rho, w, u, v, p, q, q2, q3")
    print(f"  Axis order: (time, nvar, x3, x2, x1)")
    
    # Convert to PyTorch tensor
    hydro_w_tensor = torch.from_numpy(hydro_w_np.copy())
    
    # Save to file
    tensor_map = {'hydro_w': hydro_w_tensor}
    save_tensors(tensor_map, output_file)
    
    print(f"  Saved to {output_file}")
    print(f"  Tensor shape: {hydro_w_tensor.shape}")
    print(f"  Tensor dtype: {hydro_w_tensor.dtype}")
    
    return output_file


def convert_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    pattern: str = "*.nc",
    *,
    n_blocks_x2: int = 1,
    n_blocks_x3: int = 1,
    bundle_restart: Optional[str] = None,
    topo_file: Optional[str] = None,
    config_file: Optional[str] = None,
) -> list:
    """
    Convert all NetCDF files in a directory to PyTorch tensors.
    
    Args:
        input_dir: Directory containing input .nc files
        output_dir: Optional output directory. If None, uses input_dir
        pattern: Glob pattern for finding NetCDF files (default: "*.nc")
    
    Returns:
        List of output .part file paths
        
    Raises:
        FileNotFoundError: If input directory doesn't exist
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Create output directory if specified
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    # Find all NetCDF files
    nc_files = sorted(input_path.glob(pattern))
    
    if not nc_files:
        print(f"Warning: No files matching '{pattern}' found in {input_dir}")
        return []
    
    print(f"Found {len(nc_files)} NetCDF files to convert")
    
    output_files = []
    for nc_file in nc_files:
        base_name = nc_file.stem  # filename without extension
        parsed = parse_block_indices(base_name)
        if parsed is not None and (n_blocks_x2 > 1 or n_blocks_x3 > 1):
            stem, i2, i3 = parsed
            rank = block_rank(i2, i3, n_blocks_x3)
            output_file = output_path / block_part_name(stem, rank)
        else:
            output_file = output_path / f"{base_name}.part"
        
        try:
            convert_netcdf_to_tensor(
                str(nc_file),
                str(output_file),
                topo_file=topo_file,
                config_file=config_file,
            )
            output_files.append(str(output_file))
        except Exception as e:
            print(f"Error converting {nc_file}: {e}")
            print(f"Skipping {nc_file}")
            continue
    
    if bundle_restart is not None and output_files:
        bundle_restart_parts(output_files, bundle_restart)
        print(f"Bundled {len(output_files)} part files into {bundle_restart}")

    print(f"\nSuccessfully converted {len(output_files)} files")
    return output_files


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert NetCDF block files to PyTorch tensor files (Step 6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single file
  python convert_netcdf_to_tensor.py regridded_block_0_0.nc
  
  # Convert with custom output
  python convert_netcdf_to_tensor.py regridded_block_0_0.nc --output block_0_0.part
  
  # Convert all files in directory
  python convert_netcdf_to_tensor.py ./blocks/ --output-dir ./tensors/
"""
    )
    
    parser.add_argument(
        'input',
        help='Input NetCDF file or directory containing .nc files'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output .part file (for single file mode) or output directory (for directory mode)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory (alternative to --output for directory mode)'
    )
    
    parser.add_argument(
        '--pattern',
        default='*.nc',
        help='Glob pattern for finding NetCDF files in directory mode (default: *.nc)'
    )
    parser.add_argument(
        '--n-blocks-x2',
        type=int,
        default=1,
        help='Number of x2 blocks represented by the input block files (default: 1)'
    )
    parser.add_argument(
        '--n-blocks-x3',
        type=int,
        default=1,
        help='Number of x3 blocks represented by the input block files (default: 1)'
    )
    parser.add_argument(
        '--bundle-restart',
        help='Optional output restart tarball bundling all generated .part files'
    )
    parser.add_argument(
        '--topography-file',
        help='Optional low-resolution topography tensor (.pt). When provided with --config-file, zeroes u/v/w below terrain before writing hydro_w.'
    )
    parser.add_argument(
        '--config-file',
        help='FRIGATE YAML config used to reconstruct the low-resolution x2/x3 grid for topography masking.'
    )
    
    args = parser.parse_args()
    
    # Determine if input is file or directory
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    if input_path.is_file():
        # Single file mode
        if args.output_dir is not None:
            print("Warning: --output-dir is ignored in single file mode. Use --output instead.")
        
        output_file = args.output
        try:
            convert_netcdf_to_tensor(
                str(input_path),
                output_file,
                topo_file=args.topography_file,
                config_file=args.config_file,
            )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif input_path.is_dir():
        # Directory mode
        output_dir = args.output_dir or args.output
        
        try:
            output_files = convert_directory(
                str(input_path),
                output_dir,
                args.pattern,
                n_blocks_x2=args.n_blocks_x2,
                n_blocks_x3=args.n_blocks_x3,
                bundle_restart=args.bundle_restart,
                topo_file=args.topography_file,
                config_file=args.config_file,
            )
            if not output_files:
                sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    else:
        print(f"Error: Input path is neither a file nor directory: {args.input}")
        sys.exit(1)


if __name__ == '__main__':
    main()

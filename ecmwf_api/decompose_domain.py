#!/usr/bin/env python3
"""
Domain Decomposition for Regridded NetCDF Files

This script breaks one regridded netCDF file (after step4) into multiple netCDF files
using domain decomposition. The decomposition is done in the horizontal directions (x2, x3)
only, with no decomposition in the vertical direction (x1).

Each block includes ghost zones on all sides to facilitate parallel computation with
proper boundary conditions.

The script supports parallel processing using multiple CPU cores to significantly
improve performance for large files.

Usage:
    python decompose_domain.py <input_file> <n_blocks_x2> <n_blocks_x3> [--output-dir DIR] [--num-workers N]
    
    # Use all CPU cores (default)
    python decompose_domain.py regridded_ann-arbor_20251102.nc 4 4 --output-dir ./blocks/
    
    # Use specific number of workers
    python decompose_domain.py regridded_ann-arbor_20251102.nc 4 4 --num-workers 4

The script:
  1. Reads the regridded NetCDF file with ghost zones
  2. Extracts ghost zone size from metadata (nghost)
  3. Calculates interior domain size (without ghost zones)
  4. Divides interior domain into blocks
  5. Extracts each block with ghost zones overlapping with neighbors (in parallel)
  6. Saves each block as a separate NetCDF file

Output files are named: {base_name}_block_{i2}_{i3}.nc
  where i2 and i3 are the block indices (0-based)

Example:
  Input: regridded_ann-arbor_20251102.nc (206x206 total, 200x200 interior, nghost=3)
  Blocks: 4x4
  Each interior block: 50x50 cells
  Each block with ghosts: 56x56 cells (50 + 2*3 ghost zones)
  Output: regridded_ann-arbor_20251102_block_0_0.nc, ..., regridded_ann-arbor_20251102_block_3_3.nc
"""

import argparse
import os
import sys
from typing import Tuple, Dict, List
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count

try:
    from netCDF4 import Dataset
except ImportError:
    raise ImportError("netCDF4 is required. Install with: pip install netCDF4")


def read_netcdf_metadata(filename: str) -> Dict:
    """
    Read metadata from the regridded NetCDF file.
    
    Args:
        filename: Path to input NetCDF file
        
    Returns:
        Dictionary containing metadata including dimensions and ghost zone info
    """
    with Dataset(filename, 'r') as ncfile:
        metadata = {}
        
        # Get dimensions
        metadata['dims'] = {
            'time': len(ncfile.dimensions['time']),
            'x1': len(ncfile.dimensions['x1']),
            'x2': len(ncfile.dimensions['x2']),
            'x3': len(ncfile.dimensions['x3']),
        }
        
        # Check for interface dimensions
        if 'x1f' in ncfile.dimensions:
            metadata['dims']['x1f'] = len(ncfile.dimensions['x1f'])
        if 'x2f' in ncfile.dimensions:
            metadata['dims']['x2f'] = len(ncfile.dimensions['x2f'])
        if 'x3f' in ncfile.dimensions:
            metadata['dims']['x3f'] = len(ncfile.dimensions['x3f'])
        
        # Get ghost zone size
        if hasattr(ncfile, 'nghost'):
            metadata['nghost'] = int(ncfile.nghost)
        else:
            raise ValueError(
                "Input file missing 'nghost' global attribute. "
                "This file may not be a properly formatted regridded file."
            )
        
        # Get interior dimensions
        if hasattr(ncfile, 'nx1_interior'):
            metadata['nx1_interior'] = int(ncfile.nx1_interior)
        if hasattr(ncfile, 'nx2_interior'):
            metadata['nx2_interior'] = int(ncfile.nx2_interior)
        if hasattr(ncfile, 'nx3_interior'):
            metadata['nx3_interior'] = int(ncfile.nx3_interior)
        
        # Get variable names
        metadata['variables'] = list(ncfile.variables.keys())
        
        # Get coordinate names
        metadata['coords'] = ['time', 'x1', 'x2', 'x3']
        if 'x1f' in ncfile.variables:
            metadata['coords'].extend(['x1f', 'x2f', 'x3f'])
        
        # Get global attributes
        metadata['global_attrs'] = {attr: getattr(ncfile, attr) 
                                   for attr in ncfile.ncattrs()}
        
        return metadata


def calculate_block_boundaries(
    n_interior: int,
    n_blocks: int,
    nghost: int
) -> List[Tuple[int, int]]:
    """
    Calculate start and end indices for each block in one dimension.
    
    Each block contains:
    - Interior cells: approximately n_interior / n_blocks
    - Ghost cells: nghost on each side (overlapping with neighbors)
    
    The full grid has structure:
    [ghost_left (nghost) | interior (n_interior) | ghost_right (nghost)]
    Total cells: nghost + n_interior + nghost
    
    For example, with 100 interior cells, 4 blocks, and 3 ghost cells:
    - Full grid: [0:106] (3 ghost + 100 interior + 3 ghost)
    - Interior region: [3:103]
    - Block 0 interior: [3:28] (25 cells)
    - Block 0 with ghosts: [0:31] (includes left ghosts + interior + right ghosts)
    
    Args:
        n_interior: Number of interior cells (without ghost zones)
        n_blocks: Number of blocks to decompose into
        nghost: Number of ghost cells on each side
        
    Returns:
        List of (start_idx, end_idx) tuples for each block
        Indices are in the full grid (including original ghost zones)
    """
    # Calculate interior cells per block
    cells_per_block = n_interior // n_blocks
    remainder = n_interior % n_blocks
    
    # Distribute remainder cells to first blocks
    block_sizes = [cells_per_block + (1 if i < remainder else 0) 
                   for i in range(n_blocks)]
    
    # Calculate boundaries for the interior part of each block
    # These are positions in the interior-only coordinate system [0, n_interior)
    interior_starts = []
    pos = 0
    for size in block_sizes:
        interior_starts.append(pos)
        pos += size
    
    # Convert to full grid indices
    full_boundaries = []
    for i in range(n_blocks):
        # Interior start and end in the interior coordinate system
        int_start = interior_starts[i]
        int_end = int_start + block_sizes[i]
        
        # Convert to full grid coordinates
        # Interior region in full grid: [nghost, nghost + n_interior)
        # So interior position p maps to full grid position nghost + p
        
        # Start of block (including left ghosts)
        full_start = nghost + int_start - nghost  # = int_start
        # But first block starts at 0 to include original left ghosts
        if i == 0:
            full_start = 0
        
        # End of block (including right ghosts)  
        full_end = nghost + int_end + nghost  # = int_end + 2*nghost
        # But last block ends at total size to include original right ghosts
        if i == n_blocks - 1:
            full_end = nghost + n_interior + nghost
        
        full_boundaries.append((full_start, full_end))
    
    return full_boundaries


def _extract_block_wrapper(args: Tuple) -> str:
    """
    Wrapper function for extract_block that unpacks arguments.
    This is needed for multiprocessing.Pool.map() compatibility.
    
    Args:
        args: Tuple containing all arguments for extract_block
        
    Returns:
        Output file path
    """
    (input_file, output_file, i2, j3, x2_bounds, x3_bounds, metadata) = args
    extract_block(input_file, output_file, i2, j3, x2_bounds, x3_bounds, metadata)
    return output_file


def extract_block(
    input_file: str,
    output_file: str,
    i2: int,
    j3: int,
    x2_bounds: Tuple[int, int],
    x3_bounds: Tuple[int, int],
    metadata: Dict
) -> None:
    """
    Extract a single block from the input file and save to output file.
    
    Args:
        input_file: Path to input NetCDF file
        output_file: Path to output NetCDF file
        i2: Block index in x2 direction
        j3: Block index in x3 direction
        x2_bounds: (start, end) indices in x2 direction
        x3_bounds: (start, end) indices in x3 direction
        metadata: Metadata from input file
    """
    x2_start, x2_end = x2_bounds
    x3_start, x3_end = x3_bounds
    
    with Dataset(input_file, 'r') as nc_in:
        with Dataset(output_file, 'w', format='NETCDF4') as nc_out:
            # Get dimensions
            n_time = metadata['dims']['time']
            n_x1 = metadata['dims']['x1']
            n_x2_block = x2_end - x2_start
            n_x3_block = x3_end - x3_start
            
            # Create dimensions
            nc_out.createDimension('time', n_time)
            nc_out.createDimension('x1', n_x1)  # No vertical decomposition
            nc_out.createDimension('x2', n_x2_block)
            nc_out.createDimension('x3', n_x3_block)
            
            # Create interface dimensions if they exist
            if 'x1f' in metadata['dims']:
                nc_out.createDimension('x1f', metadata['dims']['x1f'])
                nc_out.createDimension('x2f', n_x2_block + 1)
                nc_out.createDimension('x3f', n_x3_block + 1)
            
            # Copy coordinate variables
            # Time (no decomposition)
            if 'time' in nc_in.variables:
                time_in = nc_in.variables['time']
                time_out = nc_out.createVariable('time', time_in.dtype, ('time',))
                time_out[:] = time_in[:]
                time_out.setncatts({k: time_in.getncattr(k) 
                                   for k in time_in.ncattrs()})
            
            # x1 (no vertical decomposition)
            if 'x1' in nc_in.variables:
                x1_in = nc_in.variables['x1']
                x1_out = nc_out.createVariable('x1', x1_in.dtype, ('x1',))
                x1_out[:] = x1_in[:]
                x1_out.setncatts({k: x1_in.getncattr(k) 
                                 for k in x1_in.ncattrs()})
            
            # x2 (horizontal decomposition)
            if 'x2' in nc_in.variables:
                x2_in = nc_in.variables['x2']
                x2_out = nc_out.createVariable('x2', x2_in.dtype, ('x2',))
                x2_out[:] = x2_in[x2_start:x2_end]
                x2_out.setncatts({k: x2_in.getncattr(k) 
                                 for k in x2_in.ncattrs()})
            
            # x3 (horizontal decomposition)
            if 'x3' in nc_in.variables:
                x3_in = nc_in.variables['x3']
                x3_out = nc_out.createVariable('x3', x3_in.dtype, ('x3',))
                x3_out[:] = x3_in[x3_start:x3_end]
                x3_out.setncatts({k: x3_in.getncattr(k) 
                                 for k in x3_in.ncattrs()})
            
            # Interface coordinates
            if 'x1f' in nc_in.variables:
                x1f_in = nc_in.variables['x1f']
                x1f_out = nc_out.createVariable('x1f', x1f_in.dtype, ('x1f',))
                x1f_out[:] = x1f_in[:]
                x1f_out.setncatts({k: x1f_in.getncattr(k) 
                                  for k in x1f_in.ncattrs()})
            
            if 'x2f' in nc_in.variables:
                x2f_in = nc_in.variables['x2f']
                x2f_out = nc_out.createVariable('x2f', x2f_in.dtype, ('x2f',))
                x2f_out[:] = x2f_in[x2_start:x2_end+1]
                x2f_out.setncatts({k: x2f_in.getncattr(k) 
                                  for k in x2f_in.ncattrs()})
            
            if 'x3f' in nc_in.variables:
                x3f_in = nc_in.variables['x3f']
                x3f_out = nc_out.createVariable('x3f', x3f_in.dtype, ('x3f',))
                x3f_out[:] = x3f_in[x3_start:x3_end+1]
                x3f_out.setncatts({k: x3f_in.getncattr(k) 
                                  for k in x3f_in.ncattrs()})
            
            # Copy data variables
            coord_vars = ['time', 'x1', 'x2', 'x3', 'x1f', 'x2f', 'x3f']
            for var_name in nc_in.variables:
                if var_name in coord_vars:
                    continue  # Already copied
                
                var_in = nc_in.variables[var_name]
                dims = var_in.dimensions
                
                # Determine how to slice based on dimensions
                # Variables can be on cell centers or interfaces
                if dims == ('time', 'x1', 'x2', 'x3'):
                    # Regular variable on cell centers
                    var_out = nc_out.createVariable(
                        var_name, var_in.dtype, dims,
                        zlib=True, complevel=4
                    )
                    var_out[:] = var_in[:, :, x2_start:x2_end, x3_start:x3_end]
                    
                elif dims == ('time', 'x1f', 'x2', 'x3'):
                    # Variable on vertical interfaces (like pressure_level)
                    var_out = nc_out.createVariable(
                        var_name, var_in.dtype, dims,
                        zlib=True, complevel=4
                    )
                    var_out[:] = var_in[:, :, x2_start:x2_end, x3_start:x3_end]
                    
                else:
                    # Other variables - copy structure but may need different slicing
                    print(f"Warning: Variable {var_name} has unexpected dimensions {dims}, skipping")
                    continue
                
                # Copy attributes
                var_out.setncatts({k: var_in.getncattr(k) 
                                  for k in var_in.ncattrs()})
            
            # Copy global attributes
            for attr_name, attr_value in metadata['global_attrs'].items():
                setattr(nc_out, attr_name, attr_value)
            
            # Update global attributes for this block
            nc_out.block_index_x2 = i2
            nc_out.block_index_x3 = j3
            nc_out.block_x2_start = x2_start
            nc_out.block_x2_end = x2_end
            nc_out.block_x3_start = x3_start
            nc_out.block_x3_end = x3_end
            
            # Add decomposition history
            decomp_history = (
                f"\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}: "
                f"Domain decomposed into blocks. "
                f"This is block ({i2}, {j3}). "
                f"x2 range: [{x2_start}:{x2_end}], "
                f"x3 range: [{x3_start}:{x3_end}]."
            )
            
            if hasattr(nc_out, 'history'):
                nc_out.history = nc_out.history + decomp_history
            else:
                nc_out.history = decomp_history


def decompose_domain(
    input_file: str,
    n_blocks_x2: int,
    n_blocks_x3: int,
    output_dir: str = None,
    num_workers: int = None
) -> List[str]:
    """
    Decompose the domain into multiple blocks.
    
    Args:
        input_file: Path to input regridded NetCDF file
        n_blocks_x2: Number of blocks in x2 direction
        n_blocks_x3: Number of blocks in x3 direction
        output_dir: Output directory (default: same as input file)
        num_workers: Number of parallel workers (default: all CPU cores)
        
    Returns:
        List of output file paths
    """
    print("="*70)
    print("Domain Decomposition for Regridded NetCDF Files")
    print("="*70)
    
    # Read metadata
    print(f"\n1. Reading input file: {input_file}")
    metadata = read_netcdf_metadata(input_file)
    
    nghost = metadata['nghost']
    n_time = metadata['dims']['time']
    n_x1 = metadata['dims']['x1']
    n_x2_total = metadata['dims']['x2']
    n_x3_total = metadata['dims']['x3']
    
    print(f"   Dimensions: time={n_time}, x1={n_x1}, x2={n_x2_total}, x3={n_x3_total}")
    print(f"   Ghost zones: {nghost} cells on each side")
    
    # Calculate interior dimensions
    if 'nx2_interior' in metadata and 'nx3_interior' in metadata:
        n_x2_interior = metadata['nx2_interior']
        n_x3_interior = metadata['nx3_interior']
    else:
        # Calculate from total and ghost
        n_x2_interior = n_x2_total - 2 * nghost
        n_x3_interior = n_x3_total - 2 * nghost
    
    print(f"   Interior dimensions: x2={n_x2_interior}, x3={n_x3_interior}")
    print(f"   Decomposition: {n_blocks_x2} blocks in x2, {n_blocks_x3} blocks in x3")
    
    # Calculate block boundaries
    print("\n2. Calculating block boundaries...")
    x2_boundaries = calculate_block_boundaries(n_x2_interior, n_blocks_x2, nghost)
    x3_boundaries = calculate_block_boundaries(n_x3_interior, n_blocks_x3, nghost)
    
    print(f"   x2 blocks: {[f'[{s}:{e}]' for s, e in x2_boundaries]}")
    print(f"   x3 blocks: {[f'[{s}:{e}]' for s, e in x3_boundaries]}")
    
    # Prepare output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base name for output files
    base_name = os.path.basename(input_file).replace('.nc', '')
    
    # Extract blocks
    print(f"\n3. Extracting blocks to: {output_dir}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    
    total_blocks = n_blocks_x2 * n_blocks_x3
    
    # Use parallel processing if more than one worker
    if num_workers > 1 and total_blocks > 1:
        print(f"   Using {num_workers} parallel workers for {total_blocks} blocks")
        
        # Prepare arguments for all blocks
        block_args = []
        for i2 in range(n_blocks_x2):
            for j3 in range(n_blocks_x3):
                output_file = os.path.join(
                    output_dir,
                    f"{base_name}_block_{i2}_{j3}.nc"
                )
                
                block_args.append((
                    input_file,
                    output_file,
                    i2,
                    j3,
                    x2_boundaries[i2],
                    x3_boundaries[j3],
                    metadata
                ))
        
        # Process blocks in parallel
        with Pool(processes=num_workers) as pool:
            output_files = pool.map(_extract_block_wrapper, block_args)
        
        # Print summary of completed blocks
        for i, (i2, j3) in enumerate([(args[2], args[3]) for args in block_args], 1):
            x2_start, x2_end = x2_boundaries[i2]
            x3_start, x3_end = x3_boundaries[j3]
            n_x2_block = x2_end - x2_start
            n_x3_block = x3_end - x3_start
            print(f"   Block ({i2},{j3}): {i}/{total_blocks} - "
                  f"x2[{x2_start}:{x2_end}]({n_x2_block}), "
                  f"x3[{x3_start}:{x3_end}]({n_x3_block}) [completed]")
    else:
        # Sequential processing (single worker or single block)
        if num_workers == 1:
            print(f"   Using single worker (sequential processing)")
        
        output_files = []
        block_count = 0
        
        for i2 in range(n_blocks_x2):
            for j3 in range(n_blocks_x3):
                block_count += 1
                
                # Generate output filename
                output_file = os.path.join(
                    output_dir,
                    f"{base_name}_block_{i2}_{j3}.nc"
                )
                
                x2_start, x2_end = x2_boundaries[i2]
                x3_start, x3_end = x3_boundaries[j3]
                
                n_x2_block = x2_end - x2_start
                n_x3_block = x3_end - x3_start
                
                print(f"   Block ({i2},{j3}): {block_count}/{total_blocks} - "
                      f"x2[{x2_start}:{x2_end}]({n_x2_block}), "
                      f"x3[{x3_start}:{x3_end}]({n_x3_block})")
                
                # Extract and save block
                extract_block(
                    input_file,
                    output_file,
                    i2,
                    j3,
                    x2_boundaries[i2],
                    x3_boundaries[j3],
                    metadata
                )
                
                output_files.append(output_file)
    
    print("\n" + "="*70)
    print("Domain decomposition completed successfully!")
    print("="*70)
    print(f"\nCreated {len(output_files)} block files:")
    print(f"  Block dimensions: approximately {n_x2_interior//n_blocks_x2}x{n_x3_interior//n_blocks_x3} interior cells")
    print(f"  Each block includes {nghost} ghost cells on each side")
    print(f"  Output directory: {output_dir}")
    
    return output_files


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Decompose regridded NetCDF file into multiple blocks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs domain decomposition on regridded NetCDF files from the
ECMWF data pipeline. The decomposition is done in horizontal directions only
(x2, x3), with each block including ghost zones for parallel computation.

Example:
    # Decompose into 4x4 blocks (using all CPU cores)
    python decompose_domain.py regridded_ann-arbor_20251102.nc 4 4
    
    # Decompose using 4 parallel workers
    python decompose_domain.py regridded_data.nc 4 4 --num-workers 4
    
    # Sequential processing (single worker)
    python decompose_domain.py regridded_data.nc 2 2 --num-workers 1
    
    # Decompose into 2x3 blocks with custom output directory
    python decompose_domain.py regridded_data.nc 2 3 --output-dir ./blocks/

Output files are named: {input_base}_block_{i2}_{j3}.nc
  where i2 and j3 are the block indices (0-based)

Requirements:
  - Input file must be from step 3 or 4 of the ECMWF pipeline
  - Input file must have 'nghost' global attribute
  - netCDF4 Python library
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input regridded NetCDF file'
    )
    
    parser.add_argument(
        'n_blocks_x2',
        type=int,
        help='Number of blocks in x2 (Y, North-South) direction'
    )
    
    parser.add_argument(
        'n_blocks_x3',
        type=int,
        help='Number of blocks in x3 (X, East-West) direction'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for block files (default: same as input file)'
    )
    
    parser.add_argument(
        '--num-workers', '-n',
        type=int,
        default=None,
        help='Number of parallel workers (default: use all CPU cores)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_file):
        print(f"✗ Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    if args.n_blocks_x2 < 1:
        print(f"✗ Error: n_blocks_x2 must be at least 1, got {args.n_blocks_x2}")
        sys.exit(1)
    
    if args.n_blocks_x3 < 1:
        print(f"✗ Error: n_blocks_x3 must be at least 1, got {args.n_blocks_x3}")
        sys.exit(1)
    
    if args.num_workers is not None and args.num_workers < 1:
        print(f"✗ Error: num_workers must be at least 1, got {args.num_workers}")
        sys.exit(1)
    
    try:
        decompose_domain(
            args.input_file,
            args.n_blocks_x2,
            args.n_blocks_x3,
            args.output_dir,
            args.num_workers
        )
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

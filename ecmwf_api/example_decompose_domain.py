#!/usr/bin/env python3
"""
Example usage of decompose_domain.py

This example demonstrates how to use the domain decomposition script
to break a regridded NetCDF file into multiple blocks.
"""

import os
import sys
from decompose_domain import decompose_domain, read_netcdf_metadata

# Add path for imports
sys.path.insert(0, os.path.dirname(__file__))


def main():
    """Main function demonstrating decompose_domain usage."""
    
    print("="*70)
    print("Example: Domain Decomposition")
    print("="*70)
    
    # Example 1: Basic decomposition
    print("\nExample 1: Decompose into 4x4 blocks")
    print("-" * 70)
    
    input_file = "regridded_ann-arbor_20251102.nc"
    
    # Check if example file exists
    if not os.path.exists(input_file):
        print(f"Note: Example file '{input_file}' not found.")
        print("This example assumes you have run steps 1-4 of the pipeline.")
        print("\nTo create an example file:")
        print("  1. python fetch_era5_pipeline.py example_ann_arbor.yaml")
        print("  2. python calculate_density.py --input-dir <output_dir> --output-dir <output_dir>")
        print("  3. python regrid_era5_to_cartesian.py example_ann_arbor.yaml <output_dir>")
        print("  4. (optional) python compute_hydrostatic_pressure.py example_ann_arbor.yaml regridded_*.nc")
        print("\nFor this example, we'll show the command syntax:\n")
    
    # Show command
    print("Command:")
    print(f"  python decompose_domain.py {input_file} 4 4\n")
    
    print("This will:")
    print("  - Read the regridded NetCDF file")
    print("  - Divide the horizontal domain into 4x4 = 16 blocks")
    print("  - Each block includes ghost zones for boundary conditions")
    print("  - Create 16 output files: {input}_block_0_0.nc through {input}_block_3_3.nc")
    
    # Example 2: Custom output directory
    print("\n" + "="*70)
    print("Example 2: Decompose with custom output directory")
    print("-" * 70)
    
    print("Command:")
    print(f"  python decompose_domain.py {input_file} 2 3 --output-dir ./blocks/\n")
    
    print("This will:")
    print("  - Decompose into 2x3 = 6 blocks")
    print("  - Save output files to ./blocks/ directory")
    print("  - Create: ./blocks/regridded_ann-arbor_20251102_block_i_j.nc")
    
    # Example 3: If file exists, show actual metadata
    if os.path.exists(input_file):
        print("\n" + "="*70)
        print("Example 3: Actual file decomposition")
        print("-" * 70)
        
        # Read metadata
        metadata = read_netcdf_metadata(input_file)
        
        print(f"\nInput file: {input_file}")
        print(f"  Dimensions:")
        print(f"    time = {metadata['dims']['time']}")
        print(f"    x1 (vertical) = {metadata['dims']['x1']}")
        print(f"    x2 (Y, North-South) = {metadata['dims']['x2']}")
        print(f"    x3 (X, East-West) = {metadata['dims']['x3']}")
        
        if 'nghost' in metadata:
            print(f"  Ghost zones: {metadata['nghost']} cells on each side")
        
        if 'nx2_interior' in metadata and 'nx3_interior' in metadata:
            nx2_int = metadata['nx2_interior']
            nx3_int = metadata['nx3_interior']
            print(f"  Interior dimensions: {nx2_int} x {nx3_int}")
        
        print(f"\n  Variables: {', '.join([v for v in metadata['variables'] if v not in ['time', 'x1', 'x2', 'x3', 'x1f', 'x2f', 'x3f']])}")
        
        # Perform actual decomposition
        print("\nPerforming 4x4 decomposition...")
        output_dir = "./example_blocks"
        
        output_files = decompose_domain(
            input_file,
            4,  # 4 blocks in x2
            4,  # 4 blocks in x3
            output_dir
        )
        
        print(f"\nCreated {len(output_files)} block files in {output_dir}/")
        print("\nFirst few blocks:")
        for f in output_files[:3]:
            print(f"  {os.path.basename(f)}")
        if len(output_files) > 3:
            print(f"  ... and {len(output_files) - 3} more")
    
    # Example 4: Python API usage
    print("\n" + "="*70)
    print("Example 4: Using decompose_domain in Python scripts")
    print("-" * 70)
    
    print("""
Python code:

    from decompose_domain import decompose_domain
    
    # Decompose file
    output_files = decompose_domain(
        input_file='regridded_data.nc',
        n_blocks_x2=4,
        n_blocks_x3=4,
        output_dir='./blocks'
    )
    
    # Process each block
    for block_file in output_files:
        # Your processing code here
        print(f"Processing {block_file}")
""")
    
    # Example 5: Common configurations
    print("\n" + "="*70)
    print("Example 5: Common decomposition configurations")
    print("-" * 70)
    
    print("""
For different domain sizes and computational resources:

1. Small domain or single-node computation:
   python decompose_domain.py input.nc 1 1
   (No decomposition, useful for testing)

2. Medium domain on multi-core workstation:
   python decompose_domain.py input.nc 2 2
   (4 blocks, can use 4 CPU cores)

3. Large domain on cluster (16 nodes):
   python decompose_domain.py input.nc 4 4
   (16 blocks, one per node)

4. Very large domain on supercomputer:
   python decompose_domain.py input.nc 8 8
   (64 blocks, for massive parallelism)

5. Elongated domain (e.g., coastal simulation):
   python decompose_domain.py input.nc 2 8
   (16 blocks, more in one direction)

Choose n_blocks based on:
- Available computational resources
- Domain aspect ratio
- Memory constraints per process
- Communication overhead vs. computation time
""")
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)


if __name__ == "__main__":
    main()

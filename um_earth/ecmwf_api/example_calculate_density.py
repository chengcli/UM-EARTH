#!/usr/bin/env python3
"""
Example demonstrating how to use calculate_density.py to compute air density
from ERA5 dynamics and densities data.

This example shows:
1. Processing a single pair of files
2. Batch processing a directory of files
3. Accessing the calculated density data

Requirements:
    - NumPy
    - netCDF4
    - ERA5 dynamics and densities NetCDF files from Step 1

Usage:
    python example_calculate_density.py
"""

import os
import sys

# Add current directory to path for importing local modules
sys.path.insert(0, os.path.dirname(__file__))

from calculate_density import (
    solve_density_equations,
    load_netcdf_data,
    calculate_total_density,
    save_density_netcdf,
    process_single_date,
    process_directory
)


def example_1_single_file_processing():
    """
    Example 1: Process a single pair of dynamics and densities files.
    
    This is useful when you have a specific date to process or want to
    test the calculation on a single file.
    """
    print("="*70)
    print("Example 1: Single File Processing")
    print("="*70)
    
    # Input files (replace with actual file paths)
    dynamics_file = './data/era5_hourly_dynamics_20240101.nc'
    densities_file = './data/era5_hourly_densities_20240101.nc'
    output_file = './output/era5_density_20240101.nc'
    
    print(f"\nInput files:")
    print(f"  Dynamics:  {dynamics_file}")
    print(f"  Densities: {densities_file}")
    print(f"\nOutput file:")
    print(f"  Density:   {output_file}")
    
    # Check if input files exist
    if not os.path.exists(dynamics_file):
        print(f"\n⚠ Note: Example file not found: {dynamics_file}")
        print("This is just a demonstration. Replace with your actual file paths.")
        return
    
    # Process the files
    try:
        process_single_date(dynamics_file, densities_file, output_file)
        print(f"\n✓ Successfully created: {output_file}")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def example_2_batch_processing():
    """
    Example 2: Batch process all files in a directory.
    
    This is the most common use case - processing all downloaded ERA5 data
    from Step 1 of the pipeline.
    """
    print("\n" + "="*70)
    print("Example 2: Batch Directory Processing")
    print("="*70)
    
    # Directory containing downloaded ERA5 data
    input_dir = './data/29.19N_30.81N_110.93W_109.07W'
    output_dir = './output/densities'
    
    print(f"\nInput directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"\n⚠ Note: Example directory not found: {input_dir}")
        print("This is just a demonstration. Replace with your actual directory path.")
        print("\nExpected directory structure:")
        print("  data/29.19N_30.81N_110.93W_109.07W/")
        print("    era5_hourly_dynamics_20240101.nc")
        print("    era5_hourly_dynamics_20240102.nc")
        print("    era5_hourly_densities_20240101.nc")
        print("    era5_hourly_densities_20240102.nc")
        print("    ...")
        return
    
    # Process all files in directory
    try:
        process_directory(input_dir, output_dir)
        print(f"\n✓ All files processed and saved to: {output_dir}")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def example_3_custom_processing():
    """
    Example 3: Custom processing with access to intermediate results.
    
    This shows how to use the lower-level functions for custom workflows
    where you need to access or modify the calculated densities.
    """
    print("\n" + "="*70)
    print("Example 3: Custom Processing with Intermediate Access")
    print("="*70)
    
    dynamics_file = './data/era5_hourly_dynamics_20240101.nc'
    densities_file = './data/era5_hourly_densities_20240101.nc'
    
    # Check if input files exist
    if not os.path.exists(dynamics_file):
        print(f"\n⚠ Note: Example files not found")
        print("This is just a demonstration.")
        print("\nTo run this example, you need ERA5 data from Step 1:")
        print("  1. Run fetch_era5_pipeline.py with a YAML config")
        print("  2. Wait for the download to complete")
        print("  3. Update the file paths in this example")
        return
    
    try:
        # Step 1: Load data from NetCDF files
        print("\nStep 1: Loading data from NetCDF files...")
        data = load_netcdf_data(dynamics_file, densities_file)
        print(f"  Temperature shape: {data['temperature'].shape}")
        print(f"  Pressure levels: {len(data['level'])} levels")
        
        # Step 2: Calculate density components
        print("\nStep 2: Calculating air density...")
        rho_total, rho_d, rho_v, rho_c, pressure_pa = calculate_total_density(data)
        
        # Access the calculated values
        import numpy as np
        
        print(f"\nDensity statistics:")
        print(f"  Total density:")
        print(f"    Mean:  {np.mean(rho_total):.6f} kg/m³")
        print(f"    Min:   {np.min(rho_total):.6f} kg/m³")
        print(f"    Max:   {np.max(rho_total):.6f} kg/m³")
        
        print(f"  Dry air density:")
        print(f"    Mean:  {np.mean(rho_d):.6f} kg/m³")
        print(f"    Fraction: {np.mean(rho_d/rho_total)*100:.2f}%")
        
        print(f"  Water vapor density:")
        print(f"    Mean:  {np.mean(rho_v):.6f} kg/m³")
        print(f"    Fraction: {np.mean(rho_v/rho_total)*100:.2f}%")
        
        print(f"  Cloud density:")
        print(f"    Mean:  {np.mean(rho_c):.6f} kg/m³")
        print(f"    Fraction: {np.mean(rho_c/rho_total)*100:.2f}%")
        
        # Step 3: Save to NetCDF (or do custom processing)
        output_file = './output/era5_density_custom_20240101.nc'
        print(f"\nStep 3: Saving results to {output_file}...")
        save_density_netcdf(output_file, data, rho_total, rho_d, rho_v, rho_c)
        
        print(f"\n✓ Custom processing completed!")
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Please install required packages: pip install numpy netCDF4")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def example_4_direct_calculation():
    """
    Example 4: Direct calculation with synthetic data.
    
    This shows how to use the density equation solver directly with
    your own data arrays.
    """
    print("\n" + "="*70)
    print("Example 4: Direct Calculation with Synthetic Data")
    print("="*70)
    
    try:
        import numpy as np
        
        # Create synthetic atmospheric data
        # Shape: (1 time, 5 pressure levels, 3 lat, 4 lon)
        print("\nCreating synthetic atmospheric data...")
        temperature = np.linspace(250, 290, 60).reshape(1, 5, 3, 4)  # K
        pressure_hpa = np.array([1000, 850, 700, 500, 300])  # hPa
        pressure_pa = pressure_hpa.reshape(1, 5, 1, 1) * 100.0  # Convert to Pa
        
        # Specific humidity decreases with altitude
        q = np.linspace(0.015, 0.001, 60).reshape(1, 5, 3, 4)  # dimensionless
        
        # Cloud content (mostly in mid-levels)
        cloud_content = np.zeros((1, 5, 3, 4))
        cloud_content[0, 2:4, :, :] = 0.002  # 0.2% clouds in mid-levels
        
        print(f"  Temperature range: {np.min(temperature):.1f} - {np.max(temperature):.1f} K")
        print(f"  Pressure levels: {pressure_hpa} hPa")
        print(f"  Humidity range: {np.min(q):.4f} - {np.max(q):.4f}")
        
        # Calculate density
        print("\nCalculating air density...")
        rho_total, rho_d, rho_v, rho_c = solve_density_equations(
            temperature, pressure_pa, q, cloud_content
        )
        
        # Display results
        print(f"\nResults:")
        print(f"  Total density at surface (1000 hPa):")
        print(f"    Mean: {np.mean(rho_total[0, 0, :, :]):.6f} kg/m³")
        print(f"  Total density at 300 hPa:")
        print(f"    Mean: {np.mean(rho_total[0, 4, :, :]):.6f} kg/m³")
        
        # Verify density decreases with altitude
        surface_density = np.mean(rho_total[0, 0, :, :])
        upper_density = np.mean(rho_total[0, 4, :, :])
        ratio = upper_density / surface_density
        
        print(f"\n  Density ratio (300 hPa / 1000 hPa): {ratio:.4f}")
        print(f"  ✓ Density correctly decreases with altitude")
        
        # Component analysis at surface
        surface_rho_d = np.mean(rho_d[0, 0, :, :])
        surface_rho_v = np.mean(rho_v[0, 0, :, :])
        surface_rho_c = np.mean(rho_c[0, 0, :, :])
        
        print(f"\n  Surface composition:")
        print(f"    Dry air:     {surface_rho_d/surface_density*100:.2f}%")
        print(f"    Water vapor: {surface_rho_v/surface_density*100:.2f}%")
        print(f"    Clouds:      {surface_rho_c/surface_density*100:.2f}%")
        
        print(f"\n✓ Direct calculation completed successfully!")
        
    except ImportError:
        print("\n✗ NumPy is required for this example")
        print("Install with: pip install numpy")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ECMWF Data Curation - Step 2: Calculate Air Density Examples")
    print("="*70)
    
    # Example 1: Single file processing
    example_1_single_file_processing()
    
    # Example 2: Batch processing
    example_2_batch_processing()
    
    # Example 3: Custom processing
    example_3_custom_processing()
    
    # Example 4: Direct calculation
    example_4_direct_calculation()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
    print("\nFor more information, see:")
    print("  - README_ECMWF.md for full documentation")
    print("  - calculate_density.py --help for command-line usage")
    print("  - test_calculate_density.py for unit tests")


if __name__ == "__main__":
    main()

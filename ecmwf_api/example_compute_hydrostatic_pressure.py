#!/usr/bin/env python3
"""
Example demonstrating Step 4: Computing hydrostatic pressure.

This example creates synthetic regridded data, runs Step 4 to compute
hydrostatically balanced pressure, and visualizes the results.
"""

import os
import sys
import tempfile
import numpy as np
import yaml

# Add current directory to path for importing local modules
sys.path.insert(0, os.path.dirname(__file__))

from compute_hydrostatic_pressure import (
    parse_yaml_config,
    extract_gravity,
    compute_hydrostatic_pressure,
    augment_netcdf_with_pressure,
)


def create_example_config(config_file: str, gravity: float = -9.8) -> None:
    """
    Create an example YAML configuration file.
    
    Args:
        config_file: Path to output YAML file
        gravity: Gravity value (negative for downward)
    """
    config = {
        'forcing': {
            'const-gravity': {
                'grav1': gravity
            }
        },
        'geometry': {
            'type': 'cartesian',
            'bounds': {
                'x1min': 0.0,
                'x1max': 10000.0,
                'x2min': -5000.0,
                'x2max': 5000.0,
                'x3min': -5000.0,
                'x3max': 5000.0
            },
            'cells': {
                'nx1': 20,
                'nx2': 10,
                'nx3': 10,
                'nghost': 2
            },
            'center_latitude': 35.0,
            'center_longitude': -106.0
        }
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created example config: {config_file}")


def create_example_netcdf(nc_file: str, nx1: int = 20, nx2: int = 10, nx3: int = 10) -> None:
    """
    Create an example regridded NetCDF file with synthetic data.
    
    Args:
        nc_file: Path to output NetCDF file
        nx1: Number of vertical cells
        nx2: Number of Y cells
        nx3: Number of X cells
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError("netCDF4 is required. Install with: pip install netCDF4")
    
    print(f"\nCreating example NetCDF file: {nc_file}")
    
    T = 1  # One time step
    Z = nx1
    Y = nx2
    X = nx3
    Zf = Z + 1
    
    # Create vertical coordinate (0 to 10 km)
    x1f = np.linspace(0, 10000, Zf)
    x1 = 0.5 * (x1f[:-1] + x1f[1:])
    
    # Create horizontal coordinates
    x2 = np.linspace(-5000, 5000, Y)
    x3 = np.linspace(-5000, 5000, X)
    
    # Create a realistic atmosphere:
    # - Exponentially decreasing density with height
    # - Exponentially decreasing pressure with height
    
    # Surface values
    p_surface = 101325.0  # Pa
    rho_surface = 1.225   # kg/m³
    scale_height = 8500.0  # meters
    gravity = 9.8  # m/s²
    
    # Density at cell centers (exponential profile)
    rho = np.zeros((T, Z, Y, X))
    for i in range(Z):
        z_center = x1[i]
        rho[0, i, :, :] = rho_surface * np.exp(-z_center / scale_height)
    
    # Initial pressure at interfaces (exponential profile)
    pressure_level = np.zeros((T, Zf, Y, X))
    for i in range(Zf):
        z_interface = x1f[i]
        pressure_level[0, i, :, :] = p_surface * np.exp(-z_interface / scale_height)
    
    # Add some small random perturbations to make it more realistic
    np.random.seed(42)
    rho += np.random.randn(T, Z, Y, X) * 0.01 * rho
    pressure_level += np.random.randn(T, Zf, Y, X) * 0.01 * pressure_level
    
    # Ensure positive values
    rho = np.maximum(rho, 1e-6)
    pressure_level = np.maximum(pressure_level, 1.0)
    
    # Create NetCDF file
    with Dataset(nc_file, 'w', format='NETCDF4') as ncfile:
        # Create dimensions
        ncfile.createDimension('time', T)
        ncfile.createDimension('x1', Z)
        ncfile.createDimension('x1f', Zf)
        ncfile.createDimension('x2', Y)
        ncfile.createDimension('x3', X)
        
        # Create coordinate variables
        time_var = ncfile.createVariable('time', 'f8', ('time',))
        x1_var = ncfile.createVariable('x1', 'f8', ('x1',))
        x1f_var = ncfile.createVariable('x1f', 'f8', ('x1f',))
        x2_var = ncfile.createVariable('x2', 'f8', ('x2',))
        x3_var = ncfile.createVariable('x3', 'f8', ('x3',))
        
        # Set coordinate values
        time_var[:] = [0]
        x1_var[:] = x1
        x1f_var[:] = x1f
        x2_var[:] = x2
        x3_var[:] = x3
        
        # Set coordinate attributes
        time_var.units = 'hours since 1900-01-01 00:00:00'
        x1_var.units = 'meters'
        x1f_var.units = 'meters'
        x2_var.units = 'meters'
        x3_var.units = 'meters'
        
        x1_var.long_name = 'cell center height'
        x1f_var.long_name = 'cell interface height'
        x2_var.long_name = 'Y coordinate'
        x3_var.long_name = 'X coordinate'
        
        # Create data variables
        rho_var = ncfile.createVariable('rho', 'f4', ('time', 'x1', 'x2', 'x3'),
                                       zlib=True, complevel=4)
        plev_var = ncfile.createVariable('pressure_level', 'f4', 
                                        ('time', 'x1f', 'x2', 'x3'),
                                        zlib=True, complevel=4)
        
        # Set data
        rho_var[:] = rho.astype('f4')
        plev_var[:] = pressure_level.astype('f4')
        
        # Set attributes
        rho_var.units = 'kg m-3'
        rho_var.long_name = 'Air density'
        rho_var.standard_name = 'air_density'
        
        plev_var.units = 'Pa'
        plev_var.long_name = 'Pressure at cell interfaces'
        plev_var.standard_name = 'air_pressure'
        
        # Set global attributes
        ncfile.title = 'Example regridded ERA5 data'
        ncfile.description = 'Synthetic atmosphere with exponential density and pressure profiles'
    
    print(f"  Created NetCDF with shape: T={T}, Z={Z}, Y={Y}, X={X}")
    print(f"  Density range: [{np.min(rho):.3f}, {np.max(rho):.3f}] kg/m³")
    print(f"  Pressure range: [{np.min(pressure_level):.1f}, {np.max(pressure_level):.1f}] Pa")


def analyze_results(nc_file: str) -> None:
    """
    Analyze and compare the pressure fields in the NetCDF file.
    
    Args:
        nc_file: Path to NetCDF file with computed pressure
    """
    try:
        import xarray as xr
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nSkipping visualization (matplotlib not available)")
        return
    
    print("\nAnalyzing results...")
    
    ds = xr.open_dataset(nc_file)
    
    # Check that 'p' variable exists
    if 'p' not in ds.data_vars:
        print("  ERROR: 'p' variable not found in NetCDF file")
        return
    
    # Extract data for first time step and middle horizontal location
    t_idx = 0
    y_idx = len(ds['x2']) // 2
    x_idx = len(ds['x3']) // 2
    
    rho_profile = ds['rho'].isel(time=t_idx, x2=y_idx, x3=x_idx).values
    p_profile = ds['p'].isel(time=t_idx, x2=y_idx, x3=x_idx).values
    pressure_level_profile = ds['pressure_level'].isel(time=t_idx, x2=y_idx, x3=x_idx).values
    
    x1 = ds['x1'].values
    x1f = ds['x1f'].values
    
    print(f"\nVertical profiles at center location (y={y_idx}, x={x_idx}):")
    print(f"  Altitude range: {x1[0]:.0f} to {x1[-1]:.0f} m")
    print(f"  Density range: {np.min(rho_profile):.4f} to {np.max(rho_profile):.4f} kg/m³")
    print(f"  Cell center pressure range: {np.min(p_profile):.1f} to {np.max(p_profile):.1f} Pa")
    print(f"  Interface pressure range: {np.min(pressure_level_profile):.1f} to {np.max(pressure_level_profile):.1f} Pa")
    
    # Check hydrostatic balance
    gravity = 9.8  # m/s²
    dz = np.diff(x1f)
    
    # Recompute interface pressures from hydrostatic balance
    pf_check = np.zeros(len(x1f))
    pf_check[-1] = pressure_level_profile[-1]
    for i in range(len(x1) - 1, -1, -1):
        pf_check[i] = pf_check[i + 1] + rho_profile[i] * gravity * dz[i]
    
    # Compute geometric mean
    p_check = np.sqrt(pf_check[:-1] * pf_check[1:])
    
    # Compare with stored pressure
    error = np.abs(p_profile - p_check)
    rel_error = error / p_check * 100
    
    print(f"\nHydrostatic balance verification:")
    print(f"  Max absolute error: {np.max(error):.3e} Pa")
    print(f"  Max relative error: {np.max(rel_error):.3e} %")
    print(f"  Mean relative error: {np.mean(rel_error):.3e} %")
    
    if np.max(rel_error) < 0.01:
        print("  ✓ Pressure field matches hydrostatic balance within tolerance")
    else:
        print("  ✗ WARNING: Pressure field deviates from hydrostatic balance")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Plot 1: Vertical profiles
    ax1 = axes[0]
    ax1.plot(pressure_level_profile / 100, x1f / 1000, 'o-', label='Interface pressure', markersize=4)
    ax1.plot(p_profile / 100, x1 / 1000, 's-', label='Cell center pressure', markersize=4)
    ax1.set_xlabel('Pressure (hPa)')
    ax1.set_ylabel('Altitude (km)')
    ax1.set_title('Pressure Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Plot 2: Density profile
    ax2 = axes[1]
    ax2.plot(rho_profile, x1 / 1000, 'o-', color='C2', markersize=4)
    ax2.set_xlabel('Density (kg/m³)')
    ax2.set_ylabel('Altitude (km)')
    ax2.set_title('Density Profile')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Relative error
    ax3 = axes[2]
    ax3.plot(rel_error, x1 / 1000, 'o-', color='C3', markersize=4)
    ax3.set_xlabel('Relative Error (%)')
    ax3.set_ylabel('Altitude (km)')
    ax3.set_title('Hydrostatic Balance Error')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.dirname(nc_file)
    plot_file = os.path.join(output_dir, 'hydrostatic_pressure_example.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {plot_file}")
    
    ds.close()


def main():
    """Run the example."""
    print("="*70)
    print("Example: Computing Hydrostatic Pressure - Step 4")
    print("="*70)
    
    # Create temporary directory for example files
    temp_dir = tempfile.mkdtemp(prefix='ecmwf_step4_example_')
    print(f"\nWorking directory: {temp_dir}")
    
    try:
        # Create example configuration
        config_file = os.path.join(temp_dir, 'example_config.yaml')
        create_example_config(config_file, gravity=-9.8)
        
        # Create example NetCDF file
        nc_file = os.path.join(temp_dir, 'example_regridded.nc')
        create_example_netcdf(nc_file, nx1=20, nx2=10, nx3=10)
        
        # Parse config and extract gravity
        print("\n" + "-"*70)
        print("Step 1: Reading configuration")
        print("-"*70)
        config = parse_yaml_config(config_file)
        gravity = extract_gravity(config)
        print(f"Extracted gravity: {gravity:.6f} m/s²")
        
        # Load NetCDF data
        print("\n" + "-"*70)
        print("Step 2: Loading NetCDF data")
        print("-"*70)
        try:
            import xarray as xr
            ds = xr.open_dataset(nc_file)
            print(f"Loaded dataset with variables: {list(ds.data_vars.keys())}")
            print(f"Dimensions: {dict(ds.dims)}")
        except ImportError:
            print("xarray not available, skipping data loading check")
            ds = None
        
        # Compute hydrostatic pressure
        print("\n" + "-"*70)
        print("Step 3: Computing hydrostatic pressure")
        print("-"*70)
        if ds is not None:
            pressure = compute_hydrostatic_pressure(ds, gravity)
            ds.close()
            
            # Augment NetCDF file
            print("\n" + "-"*70)
            print("Step 4: Augmenting NetCDF file")
            print("-"*70)
            augment_netcdf_with_pressure(nc_file, pressure)
            
            # Analyze results
            print("\n" + "-"*70)
            print("Step 5: Analyzing results")
            print("-"*70)
            analyze_results(nc_file)
        
        print("\n" + "="*70)
        print("Example completed successfully!")
        print("="*70)
        print(f"\nOutput files:")
        print(f"  Config: {config_file}")
        print(f"  NetCDF: {nc_file}")
        print(f"\nTo clean up:")
        print(f"  rm -rf {temp_dir}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

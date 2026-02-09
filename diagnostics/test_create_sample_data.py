#!/usr/bin/env python3
"""
Create sample test data for diagnostic plotting tests.

This script creates:
1. Sample NetCDF files (out1.nc and out2.nc) with test data
2. Sample topography files (.pt) at different resolutions
"""

import os
import numpy as np
import xarray as xr
import torch

def create_sample_netcdf(output_dir, nx=60, ny=60, nz=20, nt=3):
    """
    Create sample NetCDF files for testing.
    
    Parameters
    ----------
    output_dir : str
        Directory to save test files
    nx, ny : int
        Horizontal grid size
    nz : int
        Vertical grid size
    nt : int
        Number of time steps
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create coordinate arrays
    x1 = np.linspace(0, 5000, nz)  # Height: 0-5km in meters
    x2 = np.linspace(0, 20000, nx)  # X: 0-20km in meters
    x3 = np.linspace(0, 20000, ny)  # Y: 0-20km in meters
    time = np.linspace(0, 7200, nt)  # Time: 0-2 hours in seconds
    
    # Create meshgrids
    X2, X3 = np.meshgrid(x2, x3, indexing='ij')
    
    # Create out1.nc with dynamics variables
    print("Creating out1.nc...")
    
    # Generate some test data with spatial patterns
    press_data = np.zeros((nt, nz, ny, nx))
    vel1_data = np.zeros((nt, nz, ny, nx))
    vel2_data = np.zeros((nt, nz, ny, nx))
    vel3_data = np.zeros((nt, nz, ny, nx))
    
    for t in range(nt):
        for k in range(nz):
            # Pressure: decreasing with height, with spatial variation
            press_surface = 101325 - 50 * np.sin(2*np.pi*X2/20000) * np.cos(2*np.pi*X3/20000)
            press_data[t, k, :, :] = press_surface * np.exp(-x1[k]/8000)
            
            # Vertical velocity: some updrafts and downdrafts
            vel1_data[t, k, :, :] = 0.5 * np.sin(2*np.pi*X2/10000) * np.sin(2*np.pi*X3/10000) * (1 - x1[k]/5000)
            
            # Horizontal velocities: some wind patterns
            vel2_data[t, k, :, :] = 5.0 * np.cos(2*np.pi*X3/20000) * (1 - x1[k]/5000)
            vel3_data[t, k, :, :] = 3.0 * np.sin(2*np.pi*X2/20000) * (1 - x1[k]/5000)
    
    ds1 = xr.Dataset(
        {
            'press': (['time', 'x1', 'x3', 'x2'], press_data),
            'vel1': (['time', 'x1', 'x3', 'x2'], vel1_data),
            'vel2': (['time', 'x1', 'x3', 'x2'], vel2_data),
            'vel3': (['time', 'x1', 'x3', 'x2'], vel3_data),
        },
        coords={
            'time': time,
            'x1': x1,
            'x2': x2,
            'x3': x3,
        }
    )
    
    out1_file = os.path.join(output_dir, 'test_out1.nc')
    ds1.to_netcdf(out1_file)
    print(f"  Saved: {out1_file}")
    
    # Create out2.nc with thermodynamic variables
    print("Creating out2.nc...")
    
    temp_data = np.zeros((nt, nz, ny, nx))
    theta_v_data = np.zeros((nt, nz, ny, nx))
    rh_data = np.zeros((nt, nz, ny, nx))
    path_H2O_data = np.zeros((nt, ny, nx))
    path_H2O_l_data = np.zeros((nt, ny, nx))
    
    for t in range(nt):
        for k in range(nz):
            # Temperature: decreasing with height
            temp_surface = 288 + 5 * np.sin(2*np.pi*X2/20000) * np.cos(2*np.pi*X3/20000)
            temp_data[t, k, :, :] = temp_surface - 6.5 * x1[k] / 1000  # 6.5 K/km lapse rate
            
            # Virtual potential temperature
            theta_v_data[t, k, :, :] = temp_data[t, k, :, :] * (101325 / press_data[t, k, :, :])**(0.286)
            
            # Relative humidity: higher at surface, lower aloft
            rh_data[t, k, :, :] = 0.7 * (1 - x1[k]/5000) + 0.2
        
        # Water paths (2D fields)
        path_H2O_data[t, :, :] = 20 + 10 * np.sin(2*np.pi*X2/20000) * np.cos(2*np.pi*X3/20000)
        path_H2O_l_data[t, :, :] = 0.5 + 0.3 * np.sin(2*np.pi*X2/15000) * np.cos(2*np.pi*X3/15000)
    
    ds2 = xr.Dataset(
        {
            'temp': (['time', 'x1', 'x3', 'x2'], temp_data),
            'theta_v': (['time', 'x1', 'x3', 'x2'], theta_v_data),
            'rh_H2O_l_': (['time', 'x1', 'x3', 'x2'], rh_data),
            'path_H2O': (['time', 'x3', 'x2'], path_H2O_data),
            'path_H2O_l_': (['time', 'x3', 'x2'], path_H2O_l_data),
        },
        coords={
            'time': time,
            'x1': x1,
            'x2': x2,
            'x3': x3,
        }
    )
    
    out2_file = os.path.join(output_dir, 'test_out2.nc')
    ds2.to_netcdf(out2_file)
    print(f"  Saved: {out2_file}")
    
    return out1_file, out2_file


def create_sample_topography(output_dir, location='test-site'):
    """
    Create sample topography files at different resolutions.
    
    Parameters
    ----------
    output_dir : str
        Directory to save topography files
    location : str
        Location prefix for files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define resolutions
    resolutions = [
        (60, 60, 0),
        (120, 120, 1),
        (240, 240, 2),
        (480, 480, 3)
    ]
    
    print("Creating topography files...")
    
    for nx, ny, level in resolutions:
        # Create synthetic topography: some hills
        x = np.linspace(0, 20000, nx)
        y = np.linspace(0, 20000, ny)
        X, Y = np.meshgrid(x, y)
        
        # Create some gaussian hills
        topo = (500 * np.exp(-((X-7000)**2 + (Y-7000)**2) / (3000**2)) +
                300 * np.exp(-((X-13000)**2 + (Y-13000)**2) / (4000**2)) +
                200 * np.exp(-((X-10000)**2 + (Y-15000)**2) / (2000**2)))
        
        # Convert to torch tensor
        topo_tensor = torch.from_numpy(topo.astype(np.float32))
        
        # Create data dictionary
        data = {
            'topography': topo_tensor,
            'lat_bounds': torch.tensor([40.0, 41.0], dtype=torch.float32),
            'lon_bounds': torch.tensor([-105.0, -104.0], dtype=torch.float32),
        }
        
        # Save
        topo_file = os.path.join(output_dir, f'{location}_{level}.pt')
        torch.save(data, topo_file)
        print(f"  Saved: {topo_file} (shape: {nx}x{ny})")
    
    return output_dir


def main():
    """Create all test data."""
    # Create test directory
    test_dir = '/tmp/test_diagnostics'
    os.makedirs(test_dir, exist_ok=True)
    
    print("="*60)
    print("Creating sample test data")
    print("="*60)
    print()
    
    # Create NetCDF files
    out1, out2 = create_sample_netcdf(test_dir)
    print()
    
    # Create topography files
    topo_dir = create_sample_topography(os.path.join(test_dir, 'topo'))
    print()
    
    print("="*60)
    print("Test data created successfully!")
    print("="*60)
    print(f"NetCDF files: {test_dir}")
    print(f"Topography files: {topo_dir}")
    print()
    print("To test plotting:")
    print(f"  cd diagnostics")
    print(f"  python plot_surface_pressure.py {out1} -o /tmp/test_surface_pressure.png --topo-dir {topo_dir} --location test-site")
    print(f"  python generate_all_plots.py {test_dir} --topo-dir {topo_dir} --location test-site --all-times --no-cleanup")


if __name__ == '__main__':
    main()

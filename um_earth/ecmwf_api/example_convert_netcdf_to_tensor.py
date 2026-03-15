#!/usr/bin/env python3
"""
Example usage of convert_netcdf_to_tensor.py

This script demonstrates how to use the NetCDF to PyTorch tensor converter
with sample data.
"""

import os
import tempfile
import numpy as np
from netCDF4 import Dataset

# Import the conversion function
from convert_netcdf_to_tensor import convert_netcdf_to_tensor


def create_sample_netcdf(filename, n_time=4, n_x1=56, n_x2=56, n_x3=56):
    """Create a sample NetCDF file for demonstration."""
    print(f"Creating sample NetCDF file: {filename}")
    
    with Dataset(filename, 'w', format='NETCDF4') as nc:
        # Create dimensions
        nc.createDimension('time', n_time)
        nc.createDimension('x1', n_x1)
        nc.createDimension('x2', n_x2)
        nc.createDimension('x3', n_x3)
        nc.createDimension('x1f', n_x1 + 1)
        nc.createDimension('x2f', n_x2 + 1)
        nc.createDimension('x3f', n_x3 + 1)
        
        # Create coordinate variables
        time = nc.createVariable('time', 'f8', ('time',))
        time[:] = np.arange(n_time) * 3600  # hourly data
        time.units = 'seconds since 2025-11-02 00:00:00'
        
        x1 = nc.createVariable('x1', 'f8', ('x1',))
        x1[:] = np.linspace(0, 15000, n_x1)
        x1.units = 'meters'
        x1.long_name = 'Height above ground'
        
        x2 = nc.createVariable('x2', 'f8', ('x2',))
        x2[:] = np.linspace(0, 28000, n_x2)
        x2.units = 'meters'
        x2.long_name = 'Y coordinate (North-South)'
        
        x3 = nc.createVariable('x3', 'f8', ('x3',))
        x3[:] = np.linspace(0, 28000, n_x3)
        x3.units = 'meters'
        x3.long_name = 'X coordinate (East-West)'
        
        # Create interface coordinates
        x1f = nc.createVariable('x1f', 'f8', ('x1f',))
        x1f[:] = np.linspace(0, 15000, n_x1 + 1)
        
        x2f = nc.createVariable('x2f', 'f8', ('x2f',))
        x2f[:] = np.linspace(0, 28000, n_x2 + 1)
        
        x3f = nc.createVariable('x3f', 'f8', ('x3f',))
        x3f[:] = np.linspace(0, 28000, n_x3 + 1)
        
        # Create atmospheric variables with realistic values
        print("  Creating atmospheric variables...")
        
        # Air density (kg/m³) - decreases with height
        rho = nc.createVariable('rho', 'f4', ('time', 'x1', 'x2', 'x3'),
                               zlib=True, complevel=4)
        height_profile = 1.2 * np.exp(-x1[:] / 8000)  # exponential decay
        rho_data = height_profile.reshape(1, -1, 1, 1) * np.ones((n_time, n_x1, n_x2, n_x3))
        rho[:] = rho_data.astype('f4')
        rho.units = 'kg m-3'
        
        # Wind components (m/s)
        u = nc.createVariable('u', 'f4', ('time', 'x1', 'x2', 'x3'),
                             zlib=True, complevel=4)
        u[:] = (5.0 + np.random.randn(n_time, n_x1, n_x2, n_x3) * 2.0).astype('f4')
        u.units = 'm s-1'
        
        v = nc.createVariable('v', 'f4', ('time', 'x1', 'x2', 'x3'),
                             zlib=True, complevel=4)
        v[:] = (3.0 + np.random.randn(n_time, n_x1, n_x2, n_x3) * 2.0).astype('f4')
        v.units = 'm s-1'
        
        w = nc.createVariable('w', 'f4', ('time', 'x1', 'x2', 'x3'),
                             zlib=True, complevel=4)
        w[:] = (np.random.randn(n_time, n_x1, n_x2, n_x3) * 0.1).astype('f4')
        w.units = 'Pa s-1'
        
        # Pressure (Pa) - decreases with height
        p = nc.createVariable('p', 'f4', ('time', 'x1', 'x2', 'x3'),
                             zlib=True, complevel=4)
        pressure_profile = 101325 * np.exp(-x1[:] / 8000)
        p_data = pressure_profile.reshape(1, -1, 1, 1) * np.ones((n_time, n_x1, n_x2, n_x3))
        p[:] = p_data.astype('f4')
        p.units = 'Pa'
        
        # Specific humidity (kg/kg)
        q = nc.createVariable('q', 'f4', ('time', 'x1', 'x2', 'x3'),
                             zlib=True, complevel=4)
        q[:] = (0.01 * np.exp(-x1[:] / 2000).reshape(1, -1, 1, 1) * 
               np.ones((n_time, n_x1, n_x2, n_x3))).astype('f4')
        q.units = 'kg kg-1'
        
        # Cloud water variables (kg/kg)
        ciwc = nc.createVariable('ciwc', 'f4', ('time', 'x1', 'x2', 'x3'),
                                zlib=True, complevel=4)
        ciwc[:] = (np.abs(np.random.randn(n_time, n_x1, n_x2, n_x3)) * 1e-6).astype('f4')
        ciwc.units = 'kg kg-1'
        
        clwc = nc.createVariable('clwc', 'f4', ('time', 'x1', 'x2', 'x3'),
                                zlib=True, complevel=4)
        clwc[:] = (np.abs(np.random.randn(n_time, n_x1, n_x2, n_x3)) * 1e-6).astype('f4')
        clwc.units = 'kg kg-1'
        
        cswc = nc.createVariable('cswc', 'f4', ('time', 'x1', 'x2', 'x3'),
                                zlib=True, complevel=4)
        cswc[:] = (np.abs(np.random.randn(n_time, n_x1, n_x2, n_x3)) * 1e-7).astype('f4')
        cswc.units = 'kg kg-1'
        
        crwc = nc.createVariable('crwc', 'f4', ('time', 'x1', 'x2', 'x3'),
                                zlib=True, complevel=4)
        crwc[:] = (np.abs(np.random.randn(n_time, n_x1, n_x2, n_x3)) * 1e-7).astype('f4')
        crwc.units = 'kg kg-1'
        
        # Set global attributes
        nc.title = 'Example atmospheric data block'
        nc.nghost = 3
        nc.block_index_x2 = 0
        nc.block_index_x3 = 0
        nc.center_latitude = 42.3
        nc.center_longitude = -83.7
    
    print(f"  Created sample file with dimensions: time={n_time}, x1={n_x1}, x2={n_x2}, x3={n_x3}")


def main():
    """Main demonstration."""
    print("=" * 70)
    print("Example: Converting NetCDF to PyTorch Tensor")
    print("=" * 70)
    
    # Create a temporary directory for demonstration
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nWorking directory: {temp_dir}\n")
        
        # Example 1: Single file conversion
        print("Example 1: Single File Conversion")
        print("-" * 70)
        
        input_file = os.path.join(temp_dir, 'sample_block_0_0.nc')
        create_sample_netcdf(input_file, n_time=4, n_x1=20, n_x2=20, n_x3=20)
        
        print("\nConverting to tensor...")
        output_file = convert_netcdf_to_tensor(input_file)
        
        print(f"\nOutput file created: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        
        # Load and verify
        import torch
        module = torch.jit.load(output_file)
        hydro_w = module.hydro_w
        
        print(f"\nTensor shape: {hydro_w.shape}")
        print(f"Tensor dtype: {hydro_w.dtype}")
        print(f"Variable order: rho, w, v, u, p, q, q2, q3")
        
        # Show some statistics
        print("\nVariable statistics:")
        var_names = ['rho', 'w', 'v', 'u', 'p', 'q', 'q2', 'q3']
        for i, name in enumerate(var_names):
            var_data = hydro_w[:, i, :, :, :]
            print(f"  {name:4s}: min={var_data.min():.3e}, "
                  f"max={var_data.max():.3e}, "
                  f"mean={var_data.mean():.3e}")
        
        # Example 2: Accessing specific variables
        print("\n" + "=" * 70)
        print("Example 2: Accessing Specific Variables")
        print("-" * 70)
        
        # Extract density at first time step
        rho_t0 = hydro_w[0, 0, :, :, :]
        print(f"\nDensity at t=0: shape {rho_t0.shape}")
        
        # Extract all wind components at first time step
        w_t0 = hydro_w[0, 1, :, :, :]
        v_t0 = hydro_w[0, 2, :, :, :]
        u_t0 = hydro_w[0, 3, :, :, :]
        print(f"Wind components at t=0:")
        print(f"  w: shape {w_t0.shape}")
        print(f"  v: shape {v_t0.shape}")
        print(f"  u: shape {u_t0.shape}")
        
        # Example 3: Multiple files
        print("\n" + "=" * 70)
        print("Example 3: Batch Conversion (Multiple Files)")
        print("-" * 70)
        
        # Create multiple sample files
        for i in range(2):
            for j in range(2):
                filename = os.path.join(temp_dir, f'sample_block_{i}_{j}.nc')
                create_sample_netcdf(filename, n_time=2, n_x1=10, n_x2=10, n_x3=10)
        
        print("\nConverting all blocks in directory...")
        from convert_netcdf_to_tensor import convert_directory
        
        output_dir = os.path.join(temp_dir, 'tensors')
        output_files = convert_directory(temp_dir, output_dir, pattern='sample_block_*.nc')
        
        print(f"\nConverted {len(output_files)} files:")
        for f in output_files:
            print(f"  {os.path.basename(f)}")
        
        print("\n" + "=" * 70)
        print("Example completed successfully!")
        print("=" * 70)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
ECMWF data fetching and curation pipeline - Step 2.

This script calculates the total air density from downloaded dynamics and density
variables from the previous step (Step 1).

The script solves three linear equations to find the density of dry air (rho_d),
water vapor (rho_v), and water cloud (rho_c):

    rho_d / m_d + rho_v / m_v = P / (Rgas * T)
    rho_v = q * (rho_d + rho_v + rho_c)
    rho_c = (ciwc + cswc + clwc + crwc) * (rho_d + rho_v + rho_c)

where:
    P = total pressure (Pa)
    Rgas = 8.31446 J/(mol·K) is the ideal gas constant
    T = temperature (K)
    m_d = 28.96e-3 kg/mol is the molecular weight of dry air
    m_v = 18.0e-3 kg/mol is the molecular weight of water vapor
    q = specific humidity
    ciwc = specific cloud ice water content
    cswc = specific snow water content
    clwc = specific cloud liquid water content
    crwc = specific rain water content

Once rho_d, rho_v, and rho_c are calculated, the total density is:
    rho = rho_d + rho_v + rho_c

Usage:
    python calculate_density.py --dynamics-file dynamics.nc --densities-file densities.nc --output density.nc
    
    python calculate_density.py --input-dir ./data_dir --output-dir ./output_dir
"""

import argparse
import sys
import os
from typing import Dict, Tuple
import warnings

# Physical constants
RGAS = 8.31446  # J/(mol·K) - ideal gas constant
M_DRY = 28.96e-3  # kg/mol - molecular weight of dry air
M_VAPOR = 18.0e-3  # kg/mol - molecular weight of water vapor


def solve_density_equations(temperature, pressure_pa, q, cloud_content):
    """
    Solve the three linear equations for air density components.
    
    The system of equations is:
        rho_d / m_d + rho_v / m_v = P / (Rgas * T)
        rho_v = q * (rho_d + rho_v + rho_c)
        rho_c = cloud_content * (rho_d + rho_v + rho_c)
    
    Let rho_total = rho_d + rho_v + rho_c
    
    From equations 2 and 3:
        rho_v = q * rho_total
        rho_c = cloud_content * rho_total
        rho_d = rho_total - rho_v - rho_c = rho_total * (1 - q - cloud_content)
    
    Substituting into equation 1:
        rho_total * (1 - q - cloud_content) / m_d + rho_total * q / m_v = P / (Rgas * T)
        rho_total * [(1 - q - cloud_content) / m_d + q / m_v] = P / (Rgas * T)
        rho_total = P / (Rgas * T) / [(1 - q - cloud_content) / m_d + q / m_v]
    
    Args:
        temperature: Temperature in Kelvin, shape (time, level, lat, lon)
        pressure_pa: Pressure in Pascals, shape (time, level, lat, lon) or (level,)
        q: Specific humidity (dimensionless), shape (time, level, lat, lon)
        cloud_content: Total cloud water content (ciwc+cswc+clwc+crwc), shape (time, level, lat, lon)
    
    Returns:
        Tuple of (rho_total, rho_d, rho_v, rho_c) in kg/m³
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("NumPy is required. Install with: pip install numpy")
    
    # Calculate the coefficient for rho_total
    # coeff = (1 - q - cloud_content) / m_d + q / m_v
    coeff = (1.0 - q - cloud_content) / M_DRY + q / M_VAPOR
    
    # Calculate rho_total = P / (Rgas * T) / coeff
    # Handle potential division by zero or very small values
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        rho_total = (pressure_pa / (RGAS * temperature)) / coeff
        
        # Replace inf and nan with 0
        rho_total = np.nan_to_num(rho_total, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate individual components
    rho_v = q * rho_total
    rho_c = cloud_content * rho_total
    rho_d = rho_total - rho_v - rho_c
    
    # Ensure non-negative densities
    rho_d = np.maximum(rho_d, 0.0)
    rho_v = np.maximum(rho_v, 0.0)
    rho_c = np.maximum(rho_c, 0.0)
    rho_total = np.maximum(rho_total, 0.0)
    
    return rho_total, rho_d, rho_v, rho_c


def load_netcdf_data(dynamics_file: str, densities_file: str) -> Dict:
    """
    Load data from dynamics and densities NetCDF files.
    
    Args:
        dynamics_file: Path to dynamics NetCDF file containing temperature
        densities_file: Path to densities NetCDF file containing q, ciwc, cswc, clwc, crwc
    
    Returns:
        Dictionary containing loaded data arrays and coordinates
        
    Raises:
        ImportError: If required packages are not installed
        FileNotFoundError: If input files don't exist
        ValueError: If required variables are missing
    """
    try:
        import numpy as np
        import netCDF4 as nc
    except ImportError as e:
        raise ImportError(f"Required package not found: {e}. Install with: pip install netCDF4 numpy")
    
    # Check files exist
    if not os.path.exists(dynamics_file):
        raise FileNotFoundError(f"Dynamics file not found: {dynamics_file}")
    if not os.path.exists(densities_file):
        raise FileNotFoundError(f"Densities file not found: {densities_file}")
    
    data = {}
    
    # Load temperature from dynamics file
    print(f"Loading dynamics data from: {dynamics_file}")
    with nc.Dataset(dynamics_file, 'r') as ds:
        # Temperature variable might be named 't' or 'temperature'
        if 't' in ds.variables:
            data['temperature'] = ds.variables['t'][:]
        elif 'temperature' in ds.variables:
            data['temperature'] = ds.variables['temperature'][:]
        else:
            raise ValueError(f"Temperature variable ('t' or 'temperature') not found in {dynamics_file}")
        
        # Load coordinates from dynamics file
        # Common coordinate names in ERA5 data
        coord_mapping = {
            'time': ['time', 'valid_time'],
            'level': ['level', 'pressure_level', 'isobaricInhPa'],
            'latitude': ['latitude', 'lat'],
            'longitude': ['longitude', 'lon']
        }
        
        for coord_name, possible_names in coord_mapping.items():
            for name in possible_names:
                if name in ds.variables:
                    data[coord_name] = ds.variables[name][:]
                    break
            else:
                # If not found in variables, try dimensions
                for name in possible_names:
                    if name in ds.dimensions:
                        # Create coordinate array from dimension
                        data[coord_name] = np.arange(len(ds.dimensions[name]))
                        break
        
        # Store global attributes
        data['dynamics_attrs'] = {attr: ds.getncattr(attr) for attr in ds.ncattrs()}
    
    # Load density variables from densities file
    print(f"Loading density data from: {densities_file}")
    with nc.Dataset(densities_file, 'r') as ds:
        # Map variable names (ERA5 uses short names in downloaded files)
        var_mapping = {
            'q': ['q', 'specific_humidity'],
            'ciwc': ['ciwc', 'specific_cloud_ice_water_content'],
            'cswc': ['cswc', 'specific_snow_water_content'],
            'clwc': ['clwc', 'specific_cloud_liquid_water_content'],
            'crwc': ['crwc', 'specific_rain_water_content']
        }
        
        for var_key, possible_names in var_mapping.items():
            found = False
            for name in possible_names:
                if name in ds.variables:
                    data[var_key] = ds.variables[name][:]
                    found = True
                    break
            if not found:
                raise ValueError(f"Variable {var_key} not found in {densities_file}. Tried: {possible_names}")
        
        # Store global attributes
        data['densities_attrs'] = {attr: ds.getncattr(attr) for attr in ds.ncattrs()}
    
    print(f"✓ Data loaded successfully")
    print(f"  Temperature shape: {data['temperature'].shape}")
    print(f"  Specific humidity shape: {data['q'].shape}")
    
    return data


def calculate_total_density(data: Dict) -> Tuple:
    """
    Calculate total air density from loaded data.
    
    Args:
        data: Dictionary containing loaded NetCDF data
    
    Returns:
        Tuple of (rho_total, rho_d, rho_v, rho_c, pressure_pa)
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("NumPy is required. Install with: pip install numpy")
    
    print("\nCalculating air density...")
    
    # Extract data
    temperature = data['temperature']
    q = data['q']
    
    # Calculate total cloud content
    cloud_content = data['ciwc'] + data['cswc'] + data['clwc'] + data['crwc']
    
    # Get pressure levels (in hPa) and convert to Pa
    pressure_hpa = data['level']
    pressure_pa = pressure_hpa * 100.0  # Convert hPa to Pa
    
    # Broadcast pressure to match temperature shape
    # Assume shape is (time, level, lat, lon)
    if temperature.ndim == 4:
        # Reshape pressure to (1, level, 1, 1) for broadcasting
        pressure_pa = pressure_pa.reshape(1, -1, 1, 1)
    elif temperature.ndim == 3:
        # Shape might be (level, lat, lon)
        pressure_pa = pressure_pa.reshape(-1, 1, 1)
    
    # Solve density equations
    rho_total, rho_d, rho_v, rho_c = solve_density_equations(
        temperature, pressure_pa, q, cloud_content
    )
    
    print(f"✓ Density calculation completed")
    print(f"  Total density shape: {rho_total.shape}")
    print(f"  Total density range: [{np.min(rho_total):.6f}, {np.max(rho_total):.6f}] kg/m³")
    print(f"  Dry air density range: [{np.min(rho_d):.6f}, {np.max(rho_d):.6f}] kg/m³")
    print(f"  Water vapor density range: [{np.min(rho_v):.6f}, {np.max(rho_v):.6f}] kg/m³")
    print(f"  Cloud density range: [{np.min(rho_c):.6f}, {np.max(rho_c):.6f}] kg/m³")
    
    return rho_total, rho_d, rho_v, rho_c, pressure_pa


def save_density_netcdf(output_file: str, data: Dict, rho_total, rho_d, rho_v, rho_c) -> None:
    """
    Save density data to a new NetCDF file with appropriate metadata.
    
    Args:
        output_file: Path to output NetCDF file
        data: Dictionary containing coordinate and metadata information
        rho_total: Total air density array
        rho_d: Dry air density array
        rho_v: Water vapor density array
        rho_c: Cloud density array
    """
    try:
        import netCDF4 as nc
        import numpy as np
    except ImportError as e:
        raise ImportError(f"Required package not found: {e}. Install with: pip install netCDF4 numpy")
    
    print(f"\nSaving density data to: {output_file}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create NetCDF file
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:
        # Create dimensions
        time_len = len(data['time']) if 'time' in data else rho_total.shape[0]
        level_len = len(data['level'])
        lat_len = len(data['latitude']) if 'latitude' in data else rho_total.shape[-2]
        lon_len = len(data['longitude']) if 'longitude' in data else rho_total.shape[-1]
        
        ds.createDimension('time', time_len)
        ds.createDimension('level', level_len)
        ds.createDimension('latitude', lat_len)
        ds.createDimension('longitude', lon_len)
        
        # Create coordinate variables
        if 'time' in data:
            time_var = ds.createVariable('time', data['time'].dtype, ('time',))
            time_var[:] = data['time']
            # ERA5 typically uses 'hours since 1900-01-01 00:00:00.0' as time units
            # This is preserved from the input data
            time_var.units = 'hours since 1900-01-01 00:00:00.0'
            time_var.long_name = 'time'
            time_var.calendar = 'gregorian'
        
        level_var = ds.createVariable('level', data['level'].dtype, ('level',))
        level_var[:] = data['level']
        level_var.units = 'hPa'
        level_var.long_name = 'pressure level'
        # Note: standard_name 'air_pressure' typically expects Pa, but ERA5 uses hPa for levels
        # Using 'air_pressure' here for compatibility with common tools, units attribute clarifies the scale
        
        if 'latitude' in data:
            lat_var = ds.createVariable('latitude', data['latitude'].dtype, ('latitude',))
            lat_var[:] = data['latitude']
            lat_var.units = 'degrees_north'
            lat_var.long_name = 'latitude'
            lat_var.standard_name = 'latitude'
        
        if 'longitude' in data:
            lon_var = ds.createVariable('longitude', data['longitude'].dtype, ('longitude',))
            lon_var[:] = data['longitude']
            lon_var.units = 'degrees_east'
            lon_var.long_name = 'longitude'
            lon_var.standard_name = 'longitude'
        
        # Create density variables
        # Use float32 for efficiency
        rho_var = ds.createVariable('rho', 'f4', ('time', 'level', 'latitude', 'longitude'),
                                     zlib=True, complevel=4)
        rho_var[:] = rho_total.astype(np.float32)
        rho_var.units = 'kg m-3'
        rho_var.long_name = 'total air density'
        rho_var.standard_name = 'air_density'
        rho_var.description = 'Total air density calculated from dry air, water vapor, and cloud components'
        
        rho_d_var = ds.createVariable('rho_d', 'f4', ('time', 'level', 'latitude', 'longitude'),
                                       zlib=True, complevel=4)
        rho_d_var[:] = rho_d.astype(np.float32)
        rho_d_var.units = 'kg m-3'
        rho_d_var.long_name = 'dry air density'
        rho_d_var.description = 'Dry air density component'
        
        rho_v_var = ds.createVariable('rho_v', 'f4', ('time', 'level', 'latitude', 'longitude'),
                                       zlib=True, complevel=4)
        rho_v_var[:] = rho_v.astype(np.float32)
        rho_v_var.units = 'kg m-3'
        rho_v_var.long_name = 'water vapor density'
        rho_v_var.description = 'Water vapor density component'
        
        rho_c_var = ds.createVariable('rho_c', 'f4', ('time', 'level', 'latitude', 'longitude'),
                                       zlib=True, complevel=4)
        rho_c_var[:] = rho_c.astype(np.float32)
        rho_c_var.units = 'kg m-3'
        rho_c_var.long_name = 'cloud density'
        rho_c_var.description = 'Cloud density component (ice + snow + liquid + rain)'
        
        # Add global attributes
        ds.title = 'Air density calculated from ERA5 data'
        ds.institution = 'ECMWF ERA5 Reanalysis'
        ds.source = 'ECMWF data fetching and curation pipeline - Step 2'
        ds.history = f'Created on {np.datetime64("now")}'
        ds.references = 'https://cds.climate.copernicus.eu'
        ds.comment = 'Density calculated using ideal gas law with molecular weights: m_d=28.96e-3 kg/mol, m_v=18.0e-3 kg/mol'
        ds.Conventions = 'CF-1.8'
        
        # Add calculation metadata
        ds.ideal_gas_constant = f'{RGAS} J/(mol·K)'
        ds.dry_air_molecular_weight = f'{M_DRY} kg/mol'
        ds.water_vapor_molecular_weight = f'{M_VAPOR} kg/mol'
    
    print(f"✓ Density data saved successfully")


def process_single_date(dynamics_file: str, densities_file: str, output_file: str) -> None:
    """
    Process a single pair of dynamics and densities files to calculate density.
    
    Args:
        dynamics_file: Path to dynamics NetCDF file
        densities_file: Path to densities NetCDF file
        output_file: Path to output density NetCDF file
    """
    print("="*70)
    print("ECMWF Data Curation - Step 2: Calculate Air Density")
    print("="*70)
    
    # Load data
    data = load_netcdf_data(dynamics_file, densities_file)
    
    # Calculate density
    rho_total, rho_d, rho_v, rho_c, pressure_pa = calculate_total_density(data)
    
    # Save results
    save_density_netcdf(output_file, data, rho_total, rho_d, rho_v, rho_c)
    
    print("\n" + "="*70)
    print("Processing completed successfully!")
    print("="*70)


def process_directory(input_dir: str, output_dir: str) -> None:
    """
    Process all matching dynamics and densities files in a directory.
    
    Looks for pairs of files matching:
        era5_hourly_dynamics_YYYYMMDD.nc
        era5_hourly_densities_YYYYMMDD.nc
    
    Creates output files:
        era5_density_YYYYMMDD.nc
    
    Args:
        input_dir: Directory containing input NetCDF files
        output_dir: Directory for output density files
    """
    import glob
    
    print("="*70)
    print("ECMWF Data Curation - Step 2: Calculate Air Density (Batch Mode)")
    print("="*70)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all dynamics files
    dynamics_pattern = os.path.join(input_dir, 'era5_hourly_dynamics_*.nc')
    dynamics_files = sorted(glob.glob(dynamics_pattern))
    
    if not dynamics_files:
        print(f"\n✗ No dynamics files found matching pattern: {dynamics_pattern}")
        sys.exit(1)
    
    print(f"\nFound {len(dynamics_files)} dynamics file(s)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each dynamics file
    processed = 0
    failed = 0
    
    for dynamics_file in dynamics_files:
        # Extract date from filename (era5_hourly_dynamics_YYYYMMDD.nc)
        basename = os.path.basename(dynamics_file)
        date_str = basename.replace('era5_hourly_dynamics_', '').replace('.nc', '')
        
        # Find corresponding densities file
        densities_file = os.path.join(input_dir, f'era5_hourly_densities_{date_str}.nc')
        
        if not os.path.exists(densities_file):
            print(f"\n✗ Skipping {date_str}: densities file not found: {densities_file}")
            failed += 1
            continue
        
        # Output file
        output_file = os.path.join(output_dir, f'era5_density_{date_str}.nc')
        
        print(f"\n--- Processing {date_str} ---")
        
        try:
            process_single_date(dynamics_file, densities_file, output_file)
            processed += 1
        except Exception as e:
            print(f"\n✗ Failed to process {date_str}: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("Batch Processing Summary")
    print("="*70)
    print(f"Total files: {len(dynamics_files)}")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        sys.exit(1)


def main():
    """Main function to execute density calculation."""
    parser = argparse.ArgumentParser(
        description='ECMWF data curation pipeline - Step 2: Calculate air density',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script calculates total air density from ERA5 dynamics and densities data.

Two modes are supported:

1. Single file mode:
   python calculate_density.py --dynamics-file dynamics.nc --densities-file densities.nc --output density.nc

2. Directory mode (batch processing):
   python calculate_density.py --input-dir ./data_dir --output-dir ./output_dir
   
   This will process all matching file pairs:
   - era5_hourly_dynamics_YYYYMMDD.nc
   - era5_hourly_densities_YYYYMMDD.nc
   
   And create output files:
   - era5_density_YYYYMMDD.nc

The script solves three linear equations to calculate:
  - rho_d: dry air density
  - rho_v: water vapor density
  - rho_c: cloud density (ice + snow + liquid + rain)
  - rho: total density = rho_d + rho_v + rho_c
        """
    )
    
    # Create mutually exclusive group for two modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    # Single file mode arguments
    mode_group.add_argument('--dynamics-file', type=str,
                           help='Path to dynamics NetCDF file (single file mode)')
    
    parser.add_argument('--densities-file', type=str,
                       help='Path to densities NetCDF file (required for single file mode)')
    
    parser.add_argument('--output', type=str,
                       help='Output density NetCDF file path (required for single file mode)')
    
    # Directory mode arguments
    mode_group.add_argument('--input-dir', type=str,
                           help='Input directory containing dynamics and densities files (directory mode)')
    
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for density files (required for directory mode)')
    
    args = parser.parse_args()
    
    try:
        if args.dynamics_file:
            # Single file mode
            if not args.densities_file:
                parser.error("--densities-file is required when using --dynamics-file")
            if not args.output:
                parser.error("--output is required when using --dynamics-file")
            
            process_single_date(args.dynamics_file, args.densities_file, args.output)
        
        elif args.input_dir:
            # Directory mode
            if not args.output_dir:
                parser.error("--output-dir is required when using --input-dir")
            
            process_directory(args.input_dir, args.output_dir)
        
    except FileNotFoundError as e:
        print(f"\n✗ File not found: {e}")
        sys.exit(1)
    
    except ValueError as e:
        print(f"\n✗ Value error: {e}")
        sys.exit(1)
    
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

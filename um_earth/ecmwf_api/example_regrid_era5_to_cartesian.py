#!/usr/bin/env python3
"""
Example: Using Step 3 of the ECMWF data curation pipeline

This example demonstrates how to use regrid_era5_to_cartesian.py to regrid
ERA5 data from pressure-lat-lon grids to Cartesian cell-centered coordinates
with ghost zones.

The example creates synthetic data to show the complete workflow without
requiring actual ERA5 data files.
"""

import os
import tempfile
import numpy as np
import yaml


def create_example_yaml_config(output_file='example_config.yaml'):
    """
    Create an example YAML configuration file.
    
    This configuration specifies a Cartesian grid with:
    - Domain bounds including ghost zones
    - Number of interior cells (excluding ghost zones)
    - Number of ghost cells on each side
    - Center latitude/longitude for coordinate mapping
    
    Args:
        output_file: Path to output YAML file
        
    Returns:
        Path to created YAML file
    """
    config = {
        'geometry': {
            'type': 'cartesian',
            'bounds': {
                # Domain extends including ghost zones
                # x1 is Z direction (vertical), x2 is Y, x3 is X
                'x1min': 0.0,       # meters (bottom)
                'x1max': 10000.0,   # meters (top, ~10 km)
                'x2min': -20000.0,  # meters (south)
                'x2max': 20000.0,   # meters (north)
                'x3min': -30000.0,  # meters (west)
                'x3max': 30000.0,   # meters (east)
            },
            'cells': {
                # Number of interior cells (excluding ghost zones)
                'nx1': 100,  # Z direction
                'nx2': 200,  # Y direction
                'nx3': 300,  # X direction
                # Number of ghost cells on each side
                'nghost': 3
            },
            # Center of the computational domain in lat-lon
            'center_latitude': 32.5,    # degrees north
            'center_longitude': -106.3  # degrees east (White Sands, NM)
        },
        'integration': {
            'start-date': '2024-01-01',
            'end-date': '2024-01-01'
        }
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created example configuration: {output_file}")
    print("\nConfiguration details:")
    print(f"  Domain center: ({config['geometry']['center_latitude']}°, "
          f"{config['geometry']['center_longitude']}°)")
    print(f"  Interior cells: {config['geometry']['cells']['nx1']} x "
          f"{config['geometry']['cells']['nx2']} x {config['geometry']['cells']['nx3']}")
    print(f"  Ghost zones: {config['geometry']['cells']['nghost']} cells per side")
    print(f"  Total cells: {config['geometry']['cells']['nx1'] + 2*config['geometry']['cells']['nghost']} x "
          f"{config['geometry']['cells']['nx2'] + 2*config['geometry']['cells']['nghost']} x "
          f"{config['geometry']['cells']['nx3'] + 2*config['geometry']['cells']['nghost']}")
    
    return output_file


def create_mock_era5_data(data_dir):
    """
    Create mock ERA5 data files for demonstration.
    
    In a real scenario, these files would be created by:
    - Step 1: fetch_era5_pipeline.py (downloads dynamics and densities)
    - Step 2: calculate_density.py (computes total density)
    
    Args:
        data_dir: Directory to create mock files in
    """
    try:
        import xarray as xr
        from netCDF4 import Dataset
    except ImportError:
        print("Warning: xarray and netCDF4 required for creating mock data")
        return False
    
    date_str = '20240101'
    
    # Define ERA5-like dimensions
    time = np.arange(0, 24)  # 24 hours
    plev = np.array([100000., 92500., 85000., 70000., 50000., 30000., 20000., 10000.])  # Pa
    lats = np.linspace(32.0, 33.0, 20)  # degrees
    lons = np.linspace(-107.0, -106.0, 30)  # degrees
    
    T, P, Lat, Lon = len(time), len(plev), len(lats), len(lons)
    
    print(f"\nCreating mock ERA5 data files in: {data_dir}")
    print(f"  Dimensions: time={T}, pressure={P}, lat={Lat}, lon={Lon}")
    
    # Create dynamics file
    dynamics_file = os.path.join(data_dir, f'era5_hourly_dynamics_{date_str}.nc')
    with Dataset(dynamics_file, 'w') as ds:
        # Create dimensions
        ds.createDimension('time', T)
        ds.createDimension('level', P)
        ds.createDimension('latitude', Lat)
        ds.createDimension('longitude', Lon)
        
        # Create coordinate variables
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var[:] = time
        time_var.units = 'hours since 2024-01-01 00:00:00'
        
        level_var = ds.createVariable('level', 'f8', ('level',))
        level_var[:] = plev
        level_var.units = 'Pa'
        
        lat_var = ds.createVariable('latitude', 'f8', ('latitude',))
        lat_var[:] = lats
        lat_var.units = 'degrees_north'
        
        lon_var = ds.createVariable('longitude', 'f8', ('longitude',))
        lon_var[:] = lons
        lon_var.units = 'degrees_east'
        
        # Create data variables
        # Temperature (K)
        t_var = ds.createVariable('t', 'f4', ('time', 'level', 'latitude', 'longitude'))
        t_var[:] = 280.0 + np.random.randn(T, P, Lat, Lon) * 10.0
        t_var.units = 'K'
        t_var.long_name = 'Temperature'
        
        # U wind (m/s)
        u_var = ds.createVariable('u', 'f4', ('time', 'level', 'latitude', 'longitude'))
        u_var[:] = 5.0 + np.random.randn(T, P, Lat, Lon) * 2.0
        u_var.units = 'm s-1'
        u_var.long_name = 'U component of wind'
        
        # V wind (m/s)
        v_var = ds.createVariable('v', 'f4', ('time', 'level', 'latitude', 'longitude'))
        v_var[:] = 3.0 + np.random.randn(T, P, Lat, Lon) * 2.0
        v_var.units = 'm s-1'
        v_var.long_name = 'V component of wind'
        
        # Vertical velocity (Pa/s)
        w_var = ds.createVariable('w', 'f4', ('time', 'level', 'latitude', 'longitude'))
        w_var[:] = np.random.randn(T, P, Lat, Lon) * 0.1
        w_var.units = 'Pa s-1'
        w_var.long_name = 'Vertical velocity'
        
        # Geopotential (m^2/s^2)
        z_var = ds.createVariable('z', 'f4', ('time', 'level', 'latitude', 'longitude'))
        # Approximate geopotential from pressure
        for ip, p in enumerate(plev):
            z_var[:, ip, :, :] = -7000.0 * 9.81 * np.log(p / 101325.0)
        z_var.units = 'm2 s-2'
        z_var.long_name = 'Geopotential'
    
    print(f"  Created: {os.path.basename(dynamics_file)}")
    
    # Create densities file
    densities_file = os.path.join(data_dir, f'era5_hourly_densities_{date_str}.nc')
    with Dataset(densities_file, 'w') as ds:
        # Create dimensions
        ds.createDimension('time', T)
        ds.createDimension('level', P)
        ds.createDimension('latitude', Lat)
        ds.createDimension('longitude', Lon)
        
        # Create coordinate variables (same as dynamics)
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var[:] = time
        
        level_var = ds.createVariable('level', 'f8', ('level',))
        level_var[:] = plev
        
        lat_var = ds.createVariable('latitude', 'f8', ('latitude',))
        lat_var[:] = lats
        
        lon_var = ds.createVariable('longitude', 'f8', ('longitude',))
        lon_var[:] = lons
        
        # Specific humidity (kg/kg)
        q_var = ds.createVariable('q', 'f4', ('time', 'level', 'latitude', 'longitude'))
        q_var[:] = 0.01 + np.random.randn(T, P, Lat, Lon) * 0.002
        q_var[:] = np.maximum(q_var[:], 0.0)  # Ensure positive
        q_var.units = 'kg kg-1'
        q_var.long_name = 'Specific humidity'
        
        # Cloud water contents (kg/kg) - typically small
        for var_name, long_name in [('ciwc', 'Cloud ice water content'),
                                     ('cswc', 'Snow water content'),
                                     ('clwc', 'Cloud liquid water content'),
                                     ('crwc', 'Rain water content')]:
            var = ds.createVariable(var_name, 'f4', ('time', 'level', 'latitude', 'longitude'))
            var[:] = np.abs(np.random.randn(T, P, Lat, Lon) * 1e-6)
            var.units = 'kg kg-1'
            var.long_name = long_name
    
    print(f"  Created: {os.path.basename(densities_file)}")
    
    # Create computed density file
    density_file = os.path.join(data_dir, f'era5_density_{date_str}.nc')
    with Dataset(density_file, 'w') as ds:
        # Create dimensions
        ds.createDimension('time', T)
        ds.createDimension('level', P)
        ds.createDimension('latitude', Lat)
        ds.createDimension('longitude', Lon)
        
        # Create coordinate variables
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var[:] = time
        
        level_var = ds.createVariable('level', 'f8', ('level',))
        level_var[:] = plev
        
        lat_var = ds.createVariable('latitude', 'f8', ('latitude',))
        lat_var[:] = lats
        
        lon_var = ds.createVariable('longitude', 'f8', ('longitude',))
        lon_var[:] = lons
        
        # Compute approximate density from ideal gas law: rho = p / (R * T)
        R_air = 287.05  # J/(kg·K)
        rho_var = ds.createVariable('rho', 'f4', ('time', 'level', 'latitude', 'longitude'))
        
        # Load temperature from dynamics file to compute density
        with Dataset(dynamics_file, 'r') as ds_dyn:
            temperature = ds_dyn['t'][:]
        
        for ip, p in enumerate(plev):
            rho_var[:, ip, :, :] = p / (R_air * temperature[:, ip, :, :])
        
        rho_var.units = 'kg m-3'
        rho_var.long_name = 'Air density'
    
    print(f"  Created: {os.path.basename(density_file)}")
    
    return True


def demonstrate_usage():
    """
    Demonstrate the usage of regrid_era5_to_cartesian.py.
    """
    print("="*70)
    print("Example: ECMWF Step 3 - Regrid ERA5 to Cartesian Coordinates")
    print("="*70)
    
    # Create temporary directory for example files
    temp_dir = tempfile.mkdtemp(prefix='era5_example_')
    
    try:
        # Step 1: Create example configuration
        print("\nStep 1: Create YAML configuration")
        config_file = os.path.join(temp_dir, 'example_config.yaml')
        create_example_yaml_config(config_file)
        
        # Step 2: Create mock ERA5 data
        print("\nStep 2: Create mock ERA5 data files")
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        if not create_mock_era5_data(data_dir):
            print("\nSkipping demonstration - required packages not available")
            return
        
        # Step 3: Run regridding
        print("\nStep 3: Run regridding script")
        output_file = os.path.join(temp_dir, 'regridded_era5.nc')
        
        print(f"\nCommand to run:")
        print(f"  python regrid_era5_to_cartesian.py {config_file} {data_dir} --output {output_file}")
        
        # Import and run the main function
        try:
            from regrid_era5_to_cartesian import regrid_era5_to_cartesian
            
            print("\nRunning regridding...")
            regrid_era5_to_cartesian(config_file, data_dir, output_file)
            
            # Step 4: Verify output
            print("\nStep 4: Verify output file")
            if os.path.exists(output_file):
                from netCDF4 import Dataset
                
                with Dataset(output_file, 'r') as ds:
                    print(f"\nOutput file created successfully: {output_file}")
                    print(f"\nDimensions:")
                    for dim_name, dim in ds.dimensions.items():
                        print(f"  {dim_name}: {len(dim)}")
                    
                    print(f"\nVariables:")
                    for var_name in ds.variables:
                        if var_name not in ['time', 'x1', 'x1f', 'x2', 'x2f', 'x3', 'x3f']:
                            var = ds.variables[var_name]
                            print(f"  {var_name}: shape {var.shape}, "
                                  f"range [{np.min(var[:]):.3e}, {np.max(var[:]):.3e}]")
                    
                    print(f"\nGlobal attributes:")
                    print(f"  Center: ({ds.center_latitude}°, {ds.center_longitude}°)")
                    print(f"  Interior cells: {ds.nx1_interior} x {ds.nx2_interior} x {ds.nx3_interior}")
                    print(f"  Ghost zones: {ds.nghost} cells per side")
                    
                    print(f"\n✓ Success! Output file contains regridded data on Cartesian grid")
                    print(f"  with {ds.nghost} ghost cells on each side.")
                
        except ImportError as e:
            print(f"\nCannot run demonstration: {e}")
            print("\nTo run the regridding yourself:")
            print(f"  python regrid_era5_to_cartesian.py {config_file} {data_dir} --output {output_file}")
    
    finally:
        # Clean up
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")


def main():
    """Main function."""
    demonstrate_usage()
    
    print("\n" + "="*70)
    print("Usage Summary")
    print("="*70)
    print("""
To use Step 3 of the ECMWF pipeline:

1. Prepare YAML configuration file with geometry specification:
   - Cartesian domain bounds (including ghost zones)
   - Number of interior cells (nx1, nx2, nx3)
   - Number of ghost cells (nghost)
   - Center latitude and longitude

2. Ensure ERA5 data files exist from steps 1 and 2:
   - era5_hourly_dynamics_YYYYMMDD.nc (from step 1)
   - era5_hourly_densities_YYYYMMDD.nc (from step 1)
   - era5_density_YYYYMMDD.nc (from step 2)

3. Run the regridding script:
   python regrid_era5_to_cartesian.py config.yaml data_dir/ --output output.nc

4. Output NetCDF file contains:
   - All ERA5 variables regridded to Cartesian grid
   - Cell center coordinates: x1, x2, x3
   - Cell interface coordinates: x1f, x2f, x3f
   - Variables ordered as (time, x1, x2, x3)
   - Ghost zones included in all dimensions
   - Complete metadata and coordinate information

For more information, see the script documentation:
   python regrid_era5_to_cartesian.py --help
    """)


if __name__ == "__main__":
    main()

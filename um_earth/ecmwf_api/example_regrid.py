"""
Example: Regridding ECMWF ERA5 data from pressure-lat-lon to distance grids

This example demonstrates how to use the regridding functions to convert
ECMWF ERA5 data from pressure-lat-lon grids to distance grids (height-Y-X).

The pipeline:
1. Compute layer thickness from pressure levels and density
2. Add layer thickness to topographic elevation to get absolute heights
3. Interpolate variables vertically to uniform height grid
4. Convert lat/lon to local Cartesian coordinates (Y, X)
5. Interpolate horizontally to desired output grid
"""

import numpy as np
from regrid import (
    compute_height_grid,
    regrid_pressure_to_height,
    regrid_topography,
    save_regridded_data_to_netcdf,
    save_topography_to_netcdf,
)


def create_synthetic_era5_data():
    """
    Create synthetic ECMWF ERA5-like data for demonstration.
    
    In practice, you would load this from NetCDF files using:
        import xarray as xr
        ds = xr.open_dataset('era5_data.nc')
    """
    # Input dimensions
    T = 5   # 5 time steps
    P = 8   # 8 pressure levels
    Lat = 50  # 50 latitude points
    Lon = 60  # 60 longitude points
    
    # Pressure levels (Pa) - standard ERA5 levels
    plev = np.array([
        100000.,  # 1000 hPa (surface)
        92500.,   # 925 hPa
        85000.,   # 850 hPa
        70000.,   # 700 hPa
        50000.,   # 500 hPa
        30000.,   # 300 hPa
        20000.,   # 200 hPa
        10000.,   # 100 hPa
    ])
    
    # Latitude and longitude grids (degrees)
    # Example: White Sands, New Mexico region
    lats = np.linspace(32.0, 33.5, Lat)
    lons = np.linspace(-106.8, -105.8, Lon)
    
    # Create synthetic temperature field (T, P, Lat, Lon)
    # Temperature decreases with height (pressure) and varies with location
    temp_tpll = np.zeros((T, P, Lat, Lon))
    for t in range(T):
        for p_idx, p in enumerate(plev):
            # Temperature lapse rate approximation
            T_surface = 300.0 - 5.0 * t  # K, varying with time
            lapse_rate = 0.0065  # K/m
            # Approximate height from pressure using barometric formula
            h = -7000.0 * np.log(p / 101325.0)  # meters
            T_at_p = T_surface - lapse_rate * h
            
            # Add spatial variation
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
            spatial_var = 5.0 * np.sin(2 * np.pi * lat_grid / 5.0) * np.cos(2 * np.pi * lon_grid / 5.0)
            
            temp_tpll[t, p_idx, :, :] = T_at_p + spatial_var + np.random.randn(Lat, Lon) * 1.0
    
    # Compute density from ideal gas law: rho = p / (R * T)
    R_air = 287.05  # J/(kg·K) for dry air
    rho_tpll = np.zeros_like(temp_tpll)
    for t in range(T):
        for p_idx, p in enumerate(plev):
            rho_tpll[t, p_idx, :, :] = p / (R_air * temp_tpll[t, p_idx, :, :])
    
    # Create synthetic topography (Lat, Lon)
    # White Sands is around 1200-1300 m elevation
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    topo_ll = 1250.0 + 100.0 * np.sin(2 * np.pi * lat_grid) * np.cos(2 * np.pi * lon_grid)
    topo_ll += np.random.randn(Lat, Lon) * 20.0  # Add noise
    
    return {
        'temperature': temp_tpll,
        'density': rho_tpll,
        'topography': topo_ll,
        'pressure_levels': plev,
        'latitudes': lats,
        'longitudes': lons,
    }


def main():
    """Main example demonstrating ECMWF ERA5 regridding."""
    print("=" * 70)
    print("ECMWF ERA5 Regridding Example")
    print("=" * 70)
    
    # Step 1: Load or create ERA5 data
    print("\n1. Loading ECMWF ERA5 data...")
    data = create_synthetic_era5_data()
    
    temp_tpll = data['temperature']
    rho_tpll = data['density']
    topo_ll = data['topography']
    plev = data['pressure_levels']
    lats = data['latitudes']
    lons = data['longitudes']
    
    T, P, Lat, Lon = temp_tpll.shape
    print(f"   Input shape: T={T}, P={P}, Lat={Lat}, Lon={Lon}")
    print(f"   Latitude range: [{lats[0]:.2f}, {lats[-1]:.2f}] degrees")
    print(f"   Longitude range: [{lons[0]:.2f}, {lons[-1]:.2f}] degrees")
    print(f"   Pressure range: [{plev[-1]/100:.0f}, {plev[0]/100:.0f}] hPa")
    print(f"   Temperature range: [{temp_tpll.min():.2f}, {temp_tpll.max():.2f}] K")
    print(f"   Topography range: [{topo_ll.min():.2f}, {topo_ll.max():.2f}] m")
    
    # Step 2: Define output grid
    print("\n2. Defining output distance grids...")
    # Height grid (vertical coordinate)
    x1f = np.linspace(0., 15000., 60)  # 0-15 km height, 60 levels
    
    # Horizontal grids (Y, X coordinates in meters)
    # ±20 km in both directions, centered on the input domain
    x2f = np.linspace(-20000., 20000., 80)  # Y (North-South)
    x3f = np.linspace(-30000., 30000., 100) # X (East-West)
    
    print(f"   Output shape: Z={len(x1f)}, Y={len(x2f)}, X={len(x3f)}")
    print(f"   Height range: [{x1f[0]:.0f}, {x1f[-1]:.0f}] m")
    print(f"   Y range: [{x2f[0]:.0f}, {x2f[-1]:.0f}] m")
    print(f"   X range: [{x3f[0]:.0f}, {x3f[-1]:.0f}] m")
    
    # Step 3: Set physical constants
    planet_grav = 9.81       # m/s^2 (Earth gravity)
    planet_radius = 6371.e3  # m (Earth radius)
    
    # Step 4: Compute height grid once (efficient for multiple variables)
    print("\n3. Computing height grid (once for all variables)...")
    
    try:
        z_tpll = compute_height_grid(rho_tpll, topo_ll, plev, planet_grav)
        print(f"   Height grid shape: {z_tpll.shape}")
        print(f"   Height range: [{z_tpll.min():.2f}, {z_tpll.max():.2f}] m")
        
    except ValueError as e:
        print(f"   Error computing height grid: {e}")
        return
    
    # Step 5: Regrid topography
    print("\n4. Regridding topography...")
    
    topo_yx = regrid_topography(
        topo_ll,
        lats,
        lons,
        x2f,
        x3f,
        planet_radius,
        bounds_error=False
    )
    
    print(f"   Output shape: {topo_yx.shape}")
    print(f"   Output topography range: [{np.nanmin(topo_yx):.2f}, {np.nanmax(topo_yx):.2f}] m")
    
    # Step 6: Regrid temperature using pre-computed heights with parallelization
    print("\n5. Regridding temperature field (using pre-computed heights and parallel processing)...")
    print("   Using automatic parallelization (n_jobs=None) for optimal performance...")
    
    try:
        temp_tzyx = regrid_pressure_to_height(
            temp_tpll,
            rho_tpll,
            topo_ll,
            plev,
            lats,
            lons,
            x1f,
            x2f,
            x3f,
            planet_grav,
            planet_radius,
            bounds_error=False,  # Allow NaNs at boundaries
            z_tpll=z_tpll,  # Use pre-computed heights for efficiency
            n_jobs=None  # Auto-parallelization: uses multiple CPUs for large datasets
        )
        
        print(f"   Output shape: {temp_tzyx.shape}")
        print(f"   Output temperature range: [{np.nanmin(temp_tzyx):.2f}, {np.nanmax(temp_tzyx):.2f}] K")
        print(f"   Valid data fraction: {np.sum(~np.isnan(temp_tzyx)) / temp_tzyx.size:.1%}")
        
    except ValueError as e:
        print(f"   Error during regridding: {e}")
        return
    
    # Step 7: Regrid density using same pre-computed heights (efficient!)
    print("\n6. Regridding density field (using pre-computed heights and parallel processing)...")
    
    rho_tzyx = regrid_pressure_to_height(
        rho_tpll,
        rho_tpll,  # Using density itself for the computation
        topo_ll,
        plev,
        lats,
        lons,
        x1f,
        x2f,
        x3f,
        planet_grav,
        planet_radius,
        bounds_error=False,
        z_tpll=z_tpll,  # Reuse pre-computed heights - no recomputation!
        n_jobs=None  # Auto-parallelization
    )
    
    print(f"   Output shape: {rho_tzyx.shape}")
    print(f"   Output density range: [{np.nanmin(rho_tzyx):.3f}, {np.nanmax(rho_tzyx):.3f}] kg/m^3")
    print(f"\n   Note: Heights computed once, reused for both variables, and processed in parallel!")
    
    # Alternative: Step 6b - Regrid multiple variables in one call (even more efficient!)
    print("\n6b. Alternative approach: Regrid multiple variables at once...")
    print("    This parallelizes across variables for even better performance!")
    
    from regrid import regrid_multiple_variables
    
    variables_to_regrid = {
        'temperature': temp_tpll,
        'density': rho_tpll,
    }
    
    results = regrid_multiple_variables(
        variables_to_regrid,
        rho_tpll,
        topo_ll,
        plev,
        lats,
        lons,
        x1f,
        x2f,
        x3f,
        planet_grav,
        planet_radius,
        bounds_error=False,
        z_tpll=z_tpll,  # Reuse pre-computed heights
        n_jobs=-1       # Use all CPUs (prioritizes across variables)
    )
    
    print(f"   Regridded {len(results)} variables in parallel")
    print(f"   Temperature shape: {results['temperature'].shape}")
    print(f"   Density shape: {results['density'].shape}")
    print(f"   Note: This approach is ideal when regridding many variables!")
    
    # Use results from multi-variable function for saving
    temp_tzyx = results['temperature']
    rho_tzyx = results['density']
    
    # Step 8: Save to NetCDF files
    print("\n7. Saving regridded data to NetCDF files...")
    
    # Prepare variables dictionary
    variables = {
        'temperature': temp_tzyx,
        'density': rho_tzyx,
    }
    
    # Prepare coordinates dictionary
    coordinates = {
        'time': np.arange(T, dtype=float),  # In practice, use actual time values
        'x1f': x1f,
        'x2f': x2f,
        'x3f': x3f,
    }
    
    # Prepare metadata
    metadata = {
        'source': 'ECMWF ERA5 Reanalysis',
        'region': 'White Sands, New Mexico',
        'temperature_units': 'K',
        'temperature_long_name': 'Air Temperature',
        'temperature_standard_name': 'air_temperature',
        'density_units': 'kg m-3',
        'density_long_name': 'Air Density',
        'density_standard_name': 'air_density',
        'time_units': 'hours since 2025-01-01 00:00:00',
    }
    
    # Processing history
    processing_history = (
        f"Regridded from ECMWF ERA5 pressure-lat-lon grids to Cartesian distance grids. "
        f"Original domain: lat [{lats[0]:.2f}, {lats[-1]:.2f}] deg, "
        f"lon [{lons[0]:.2f}, {lons[-1]:.2f}] deg, "
        f"pressure [{plev[-1]/100:.0f}, {plev[0]/100:.0f}] hPa. "
        f"Output domain: x1 [{x1f[0]:.0f}, {x1f[-1]:.0f}] m, "
        f"x2 [{x2f[0]:.0f}, {x2f[-1]:.0f}] m, "
        f"x3 [{x3f[0]:.0f}, {x3f[-1]:.0f}] m. "
        f"Planet radius: {planet_radius/1000:.0f} km, gravity: {planet_grav:.2f} m/s^2."
    )
    
    try:
        # Save atmospheric data
        save_regridded_data_to_netcdf(
            'regridded_era5_data.nc',
            variables,
            coordinates,
            metadata,
            processing_history
        )
        print("   ✓ Saved atmospheric data to: regridded_era5_data.nc")
        
        # Save topography
        save_topography_to_netcdf(
            'regridded_topography.nc',
            topo_yx,
            x2f,
            x3f,
            metadata={'source': 'Regridded from original topography'},
            processing_history=processing_history
        )
        print("   ✓ Saved topography to: regridded_topography.nc")
        
    except Exception as e:
        print(f"   Warning: Could not save NetCDF files: {e}")
        print("   (netCDF4 package may not be installed)")
    
    # Step 9: Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Successfully regridded ECMWF ERA5 data:")
    print(f"  Input:  (T={T}, P={P}, Lat={Lat}, Lon={Lon})")
    print(f"  Output: (T={T}, Z={len(x1f)}, Y={len(x2f)}, X={len(x3f)})")
    print(f"\nThe regridded data is now on a regular Cartesian grid suitable for")
    print(f"atmospheric modeling and analysis.")
    print("\nOutput files:")
    print("  - regridded_era5_data.nc: Atmospheric variables (temperature, density, etc.)")
    print("  - regridded_topography.nc: Topographic elevation")
    print("\nMetadata preserved:")
    print("  - Source information")
    print("  - Variable units and descriptions")
    print("  - Processing history")
    print("  - Coordinate system information")
    print("=" * 70)


if __name__ == "__main__":
    main()

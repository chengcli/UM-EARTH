"""
ECMWF ERA5 Regridding Module

This module provides functions to regrid ECMWF ERA5 data from pressure-lat-lon
grids to uniform Cartesian distance grids (height-Y-X) for atmospheric modeling.

Functions:
    compute_dz_from_plev: Compute layer thickness from pressure levels
    compute_heights_from_dz: Compute absolute heights from layer thickness and topography
    compute_height_grid: Compute height grid once for reuse with multiple variables
    latlon_to_xy: Convert lat/lon to local Cartesian coordinates
    vertical_interp_to_z: Vertical interpolation to uniform height grid
    horizontal_regrid_xy: Horizontal regridding on regular grids
    regrid_pressure_to_height: Complete pipeline from (T,P,Lat,Lon) to (T,Z,Y,X)
    regrid_multiple_variables: Regrid multiple variables in parallel for improved performance
    regrid_topography: Regrid topographic elevation to distance grids
    save_regridded_data_to_netcdf: Save regridded atmospheric data to NetCDF with metadata
    save_topography_to_netcdf: Save regridded topography to NetCDF with metadata
"""

from typing import Tuple, Optional, Dict, Any
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Parallelization thresholds
_VERTICAL_INTERP_PARALLEL_THRESHOLD = 100  # Minimum columns for parallel vertical interpolation
_HORIZONTAL_REGRID_PARALLEL_THRESHOLD = 10  # Minimum slices for parallel horizontal regridding

def compute_dz_from_plev(
    plev: np.ndarray,   # (P,) pressure levels [Pa], typically decreasing with height
    rho: np.ndarray,    # (T, P, Lat, Lon) density [kg/m^3]
    grav: float,        # gravity [m/s^2]
) -> np.ndarray:
    """
    Compute layer thickness from pressure levels at each (t, lat, lon) using
    Δz = Δp / (ρ̄ g), where ρ̄ is layer-mean density (trapezoidal in 1/ρ).

    Args:
        plev: (P,) pressure levels [Pa], typically decreasing with height
        rho: (T, P, Lat, Lon) density [kg/m^3]
        grav: gravity [m/s^2]

    Returns:
        dz: (T, P-1, Lat, Lon) layer thickness [m] between pressure levels
    """
    # Ensure plev is monotonic from bottom (largest p) to top (smallest p)
    plev = np.asarray(plev)
    if plev.ndim != 1:
        raise ValueError("plev must be 1D of shape (P,).")
    if not np.all(np.diff(plev) < 0):
        # sort descending (bottom→top)
        order = np.argsort(-plev)
        plev = plev[order]
        rho = np.moveaxis(rho, 1, -1)[..., order]
        rho = np.moveaxis(rho, -1, 1)

    T, P, Lat, Lon = rho.shape
    if plev.shape[0] != P:
        raise ValueError("rho second dimension must match plev length.")

    # Layer thickness Δz_k between plev[k] and plev[k+1]
    # Use trapezoid rule on 1/rho:  1/ρ̄ ≈ 0.5*(1/ρ_k + 1/ρ_{k+1})
    inv_rho = 1.0 / rho
    inv_rho_mid = 0.5 * (inv_rho[:, :-1, :, :] + inv_rho[:, 1:, :, :])  # (T, P-1, Lat, Lon)
    dp = (plev[:-1] - plev[1:])[None, :, None, None]                    # (1, P-1, 1, 1), positive if p decreases upward
    dz_layers = (dp * inv_rho_mid) / grav                               # (T, P-1, Lat, Lon)

    return dz_layers


def compute_heights_from_dz(
    dz: np.ndarray,     # (T, P-1, Lat, Lon) layer thickness [m]
    topo: np.ndarray,   # (Lat, Lon) topographic elevation [m]
) -> np.ndarray:
    """
    Compute heights at pressure levels from layer thicknesses and topography.

    Args:
        dz: (T, P-1, Lat, Lon) layer thickness [m] between pressure levels
        topo: (Lat, Lon) topographic elevation [m] above reference

    Returns:
        z: (T, P, Lat, Lon) height [m] at pressure levels, with z[..., 0, :, :] = topo
    """
    T, P_minus_1, Lat, Lon = dz.shape
    P = P_minus_1 + 1

    # Initialize heights with bottom at topographic elevation
    z = np.zeros((T, P, Lat, Lon), dtype=dz.dtype)
    z[:, 0, :, :] = topo[None, :, :]  # Bottom level at topographic elevation

    # Add layer thicknesses cumulatively
    z[:, 1:, :, :] = topo[None, None, :, :] + np.cumsum(dz, axis=1)

    return z


def compute_height_grid(
    rho_tpll: np.ndarray,      # (T, P, Lat, Lon) air density [kg/m^3]
    topo_ll: np.ndarray,       # (Lat, Lon) topographic elevation [m]
    plev: np.ndarray,          # (P,) pressure levels [Pa]
    planet_grav: float,        # gravity constant [m/s^2]
) -> np.ndarray:
    """
    Compute height grid at pressure levels for all grid points.
    
    This function computes the height field once, which can then be reused
    for regridding multiple variables, improving efficiency.
    
    Args:
        rho_tpll: (T, P, Lat, Lon) air density [kg/m^3]
        topo_ll: (Lat, Lon) topographic elevation [m]
        plev: (P,) pressure levels [Pa], typically decreasing with height
        planet_grav: gravity constant [m/s^2]
        
    Returns:
        z_tpll: (T, P, Lat, Lon) height [m] at pressure levels
        
    Example:
        >>> # Compute heights once
        >>> z_tpll = compute_height_grid(rho_tpll, topo_ll, plev, planet_grav)
        >>> # Reuse for multiple variables
        >>> temp_tzyx = regrid_pressure_to_height(temp_tpll, rho_tpll, ..., z_tpll=z_tpll)
        >>> humid_tzyx = regrid_pressure_to_height(humid_tpll, rho_tpll, ..., z_tpll=z_tpll)
    """
    # Step 1: Compute layer thickness from pressure levels
    dz_tpll = compute_dz_from_plev(plev, rho_tpll, planet_grav)  # (T, P-1, Lat, Lon)
    
    # Step 2: Add layer thickness to topographic elevation
    z_tpll = compute_heights_from_dz(dz_tpll, topo_ll)  # (T, P, Lat, Lon)
    
    return z_tpll


def latlon_to_xy(
    lats: np.ndarray,   # (Lat,) latitude [degrees]
    lons: np.ndarray,   # (Lon,) longitude [degrees]
    planet_radius: float,  # planet radius [m]
    lat_center: Optional[float] = None,  # center latitude for projection [degrees]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert latitude/longitude to local Cartesian coordinates (Y, X) in meters.

    Args:
        lats: (Lat,) latitude array [degrees]
        lons: (Lon,) longitude array [degrees]
        planet_radius: radius of the planet [m]
        lat_center: center latitude for projection [degrees]. If None, uses midpoint of lats.

    Returns:
        Y: (Lat,) Y-coordinate in meters (North-South direction)
        X: (Lon,) X-coordinate in meters (East-West direction)
    """
    if lat_center is None:
        lat_center = 0.5 * (lats[0] + lats[-1])

    # Convert to radians
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    lat_center_rad = np.radians(lat_center)

    # Y is simply arc length from center latitude
    Y = (lats_rad - lat_center_rad) * planet_radius

    # X uses the cosine of the center latitude for the projection
    lon_center_rad = np.radians(0.5 * (lons[0] + lons[-1]))
    X = (lons_rad - lon_center_rad) * planet_radius * np.cos(lat_center_rad)

    return Y, X


def _vertical_interp_single_column(args):
    """
    Helper function for parallel vertical interpolation of a single column.
    
    Args:
        args: Tuple of (index, z_col, v_col, z_out)
        
    Returns:
        Tuple of (index, interpolated_values, extrapolation_detected)
    """
    i, zc, vc, z_out = args
    
    # Require strictly increasing z for np.interp
    # If any non-monotonic segments (rare, but can happen), we sort by z
    order = np.argsort(zc)
    z_sorted = zc[order]
    v_sorted = vc[order]
    
    # Remove duplicates in z (if any)
    mask = np.diff(z_sorted, prepend=z_sorted[0]-1) > 0
    z_sorted = z_sorted[mask]
    v_sorted = v_sorted[mask]
    
    extrapolation_detected = False
    if z_sorted.size >= 2:
        # Check for extrapolation
        if np.any(z_out < z_sorted[0]) or np.any(z_out > z_sorted[-1]):
            extrapolation_detected = True
            
        # np.interp requires increasing x; clip outside → returns boundary value,
        # so we post-mask to NaN for true "no extrapolation".
        vals = np.interp(z_out, z_sorted, v_sorted)
        vals[(z_out < z_sorted[0]) | (z_out > z_sorted[-1])] = np.nan
        return (i, vals, extrapolation_detected)
    else:
        return (i, np.full(z_out.shape[0], np.nan, dtype=vc.dtype), False)


def vertical_interp_to_z(
    z_col: np.ndarray,     # (..., P) heights increasing with level index
    v_col: np.ndarray,     # (..., P) values on same levels
    z_out: np.ndarray,     # (Z,) target heights (monotonic increasing)
    bounds_error: bool = True,  # If True, raise error when extrapolation would occur
    n_jobs: Optional[int] = None,  # Number of parallel workers (None = auto, 1 = sequential)
) -> np.ndarray:
    """
    1D vertical interpolation along the last axis (P→Z) using np.interp per column.
    
    Args:
        z_col: (..., P) heights increasing with level index
        v_col: (..., P) values on same levels
        z_out: (Z,) target heights (monotonic increasing)
        bounds_error: If True, raise error when extrapolation would occur
        n_jobs: Number of parallel workers (None=auto, 1=sequential, -1=all CPUs)
        
    Returns:
        out: (..., Z) interpolated values
        
    Raises:
        ValueError: If bounds_error=True and extrapolation would occur
    """
    # Shapes
    *lead, P = z_col.shape
    Z = z_out.shape[0]
    out = np.full((*lead, Z), np.nan, dtype=v_col.dtype)

    # Flatten leading dims to loop neatly
    z_flat = z_col.reshape(-1, P)
    v_flat = v_col.reshape(-1, P)
    out_flat = out.reshape(-1, Z)
    
    n_cols = z_flat.shape[0]
    
    # Determine number of jobs
    if n_jobs is None:
        # Auto: Use parallelization only if we have enough columns to make it worthwhile
        n_jobs = min(cpu_count(), max(1, n_cols // _VERTICAL_INTERP_PARALLEL_THRESHOLD)) if n_cols > _VERTICAL_INTERP_PARALLEL_THRESHOLD else 1
    elif n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    extrapolation_detected = False
    
    if n_jobs == 1:
        # Sequential execution
        for i in range(n_cols):
            zc = z_flat[i]
            vc = v_flat[i]

            # Require strictly increasing z for np.interp
            # If any non-monotonic segments (rare, but can happen), we sort by z
            order = np.argsort(zc)
            z_sorted = zc[order]
            v_sorted = vc[order]

            # Remove duplicates in z (if any)
            mask = np.diff(z_sorted, prepend=z_sorted[0]-1) > 0
            z_sorted = z_sorted[mask]
            v_sorted = v_sorted[mask]

            if z_sorted.size >= 2:
                # Check for extrapolation
                if np.any(z_out < z_sorted[0]) or np.any(z_out > z_sorted[-1]):
                    extrapolation_detected = True
                    
                # np.interp requires increasing x; clip outside → returns boundary value,
                # so we post-mask to NaN for true "no extrapolation".
                vals = np.interp(z_out, z_sorted, v_sorted)
                vals[(z_out < z_sorted[0]) | (z_out > z_sorted[-1])] = np.nan
                out_flat[i] = vals
            else:
                out_flat[i] = np.nan
    else:
        # Parallel execution
        args_list = [(i, z_flat[i], v_flat[i], z_out) for i in range(n_cols)]
        
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_vertical_interp_single_column, args_list)
        
        # Collect results
        for i, vals, extrap_detected in results:
            out_flat[i] = vals
            if extrap_detected:
                extrapolation_detected = True

    if bounds_error and extrapolation_detected:
        raise ValueError(
            f"Vertical interpolation would require extrapolation. "
            f"Target z range [{z_out.min():.2f}, {z_out.max():.2f}] exceeds "
            f"available data range."
        )

    return out


def horizontal_regrid_xy(
    x: np.ndarray, y: np.ndarray, field: np.ndarray, x_out: np.ndarray, y_out: np.ndarray,
    bounds_error: bool = True
) -> np.ndarray:
    """
    Regrid a 2D field f(x,y) defined on 1D grids x,y to a new regular grid x_out,y_out
    using SciPy RegularGridInterpolator (cubic or linear depending on grid size).

    Args:
        x: (X,) input x-coordinate
        y: (Y,) input y-coordinate
        field: (X, Y) on the original grids.
        x_out: (X_out,) output x-coordinate
        y_out: (Y_out,) output y-coordinate
        bounds_error: If True, raise error when interpolation becomes extrapolation

    Returns:
        field_on_out: (X_out, Y_out)
    """
    # Check bounds to prevent extrapolation
    if bounds_error:
        if x_out.min() < x.min() or x_out.max() > x.max():
            raise ValueError(
                f"Output x range [{x_out.min():.2f}, {x_out.max():.2f}] exceeds "
                f"input x range [{x.min():.2f}, {x.max():.2f}]. Extrapolation is not allowed."
            )
        if y_out.min() < y.min() or y_out.max() > y.max():
            raise ValueError(
                f"Output y range [{y_out.min():.2f}, {y_out.max():.2f}] exceeds "
                f"input y range [{y.min():.2f}, {y.max():.2f}]. Extrapolation is not allowed."
            )

    # Use linear interpolation by default for better performance and stability
    method = "linear"
    
    interp = RegularGridInterpolator((x, y),
                                     field,
                                     method=method,
                                     bounds_error=False, 
                                     fill_value=np.nan)

    Xo, Yo = np.meshgrid(x_out, y_out, indexing="ij")  # (X_out, Y_out)
    pts = np.stack([Xo.ravel(), Yo.ravel()], axis=-1)
    Fo = interp(pts).reshape(Xo.shape)
    
    # Check if any NaNs were introduced by extrapolation (not by input NaNs)
    # Only perform this check if bounds_error is True to avoid unnecessary computation
    if bounds_error:
        input_has_nan = np.any(np.isnan(field))
        output_has_nan = np.any(np.isnan(Fo))
        if output_has_nan and not input_has_nan:
            raise ValueError("Interpolation resulted in NaN values, indicating extrapolation occurred.")
    
    return Fo


def _regrid_horizontal_slice(args):
    """
    Helper function for parallel horizontal regridding of a single (time, height) slice.
    
    Args:
        args: Tuple of (ti, zi, slab, X_coord, Y_coord, x3f_shifted, x2f_shifted, bounds_error)
        
    Returns:
        Tuple of (ti, zi, regridded_slab)
    """
    ti, zi, slab, X_coord, Y_coord, x3f_shifted, x2f_shifted, bounds_error = args
    
    # Skip if the slab is all NaNs (e.g., height level above all data)
    if np.all(np.isnan(slab)):
        return (ti, zi, None)
    
    # Note: horizontal_regrid_xy expects (X, Y) order, so we transpose
    slab_t = slab.T  # (Lon, Lat) -> (X, Y) ordering
    result = horizontal_regrid_xy(
        X_coord, Y_coord, slab_t, x3f_shifted, x2f_shifted, bounds_error=bounds_error
    ).T  # Transpose back to (Y, X)
    
    return (ti, zi, result)


def regrid_pressure_to_height(
    var_tpll: np.ndarray,      # (T, P, Lat, Lon) variable on pressure-lat-lon grid
    rho_tpll: np.ndarray,      # (T, P, Lat, Lon) air density [kg/m^3]
    topo_ll: np.ndarray,       # (Lat, Lon) topographic elevation [m]
    plev: np.ndarray,          # (P,) pressure levels [Pa]
    lats: np.ndarray,          # (Lat,) latitude [degrees]
    lons: np.ndarray,          # (Lon,) longitude [degrees]
    x1f: np.ndarray,           # (Z,) vertical height coordinate [m]
    x2f: np.ndarray,           # (Y,) horizontal Y-coordinate [m]
    x3f: np.ndarray,           # (X,) horizontal X-coordinate [m]
    planet_grav: float,        # gravity constant [m/s^2]
    planet_radius: float,      # radius of the planet [m]
    bounds_error: bool = True, # If True, raise error when extrapolation would occur
    z_tpll: Optional[np.ndarray] = None,  # (T, P, Lat, Lon) pre-computed heights [m]
    n_jobs: Optional[int] = None,  # Number of parallel workers (None = auto, 1 = sequential)
) -> np.ndarray:
    """
    Complete regridding pipeline from ECMWF ERA5 pressure-lat-lon data to distance grids.
    
    Steps:
    1. Compute layer thickness from pressure levels and density (or use pre-computed heights)
    2. Add layer thickness to topographic elevation to get heights at pressure levels
    3. Interpolate variables vertically to uniform height grid (parallelized)
    4. Convert lat/lon to local Cartesian coordinates (Y, X)
    5. Interpolate horizontally to desired output grid, matching centers (parallelized)
    
    Args:
        var_tpll: (T, P, Lat, Lon) variable on pressure-lat-lon grid
        rho_tpll: (T, P, Lat, Lon) air density [kg/m^3]
        topo_ll: (Lat, Lon) topographic elevation [m]
        plev: (P,) pressure levels [Pa], typically decreasing with height
        lats: (Lat,) latitude array [degrees]
        lons: (Lon,) longitude array [degrees]
        x1f: (Z,) vertical height coordinate [m]
        x2f: (Y,) horizontal Y-coordinate [m]
        x3f: (X,) horizontal X-coordinate [m]
        planet_grav: gravity constant [m/s^2]
        planet_radius: radius of the planet [m]
        bounds_error: If True, raise error when extrapolation would occur
        z_tpll: Optional pre-computed (T, P, Lat, Lon) heights [m]. If provided,
                skips height computation (Steps 1-2) for efficiency when regridding
                multiple variables. Compute once using compute_height_grid().
        n_jobs: Number of parallel workers (None=auto, 1=sequential, -1=all CPUs)
        
    Returns:
        var_tzyx: (T, Z, Y, X) variable on distance grids
        
    Raises:
        ValueError: If output domain exceeds input domain when bounds_error=True
        
    Example:
        >>> # Efficient regridding of multiple variables with parallelization
        >>> z_tpll = compute_height_grid(rho_tpll, topo_ll, plev, planet_grav)
        >>> temp_tzyx = regrid_pressure_to_height(temp_tpll, ..., z_tpll=z_tpll, n_jobs=-1)
        >>> humid_tzyx = regrid_pressure_to_height(humid_tpll, ..., z_tpll=z_tpll, n_jobs=-1)
    """
    T, P, Lat, Lon = var_tpll.shape
    
    # Validate input shapes
    if rho_tpll.shape != var_tpll.shape:
        raise ValueError("rho_tpll must have the same shape as var_tpll")
    if topo_ll.shape != (Lat, Lon):
        raise ValueError(f"topo_ll shape {topo_ll.shape} must match (Lat={Lat}, Lon={Lon})")
    if plev.shape[0] != P:
        raise ValueError(f"plev length {plev.shape[0]} must match P dimension {P}")
    if lats.shape[0] != Lat:
        raise ValueError(f"lats length {lats.shape[0]} must match Lat dimension {Lat}")
    if lons.shape[0] != Lon:
        raise ValueError(f"lons length {lons.shape[0]} must match Lon dimension {Lon}")
    
    # Step 1-2: Compute or use pre-computed height grid
    if z_tpll is None:
        # Compute layer thickness from pressure levels
        dz_tpll = compute_dz_from_plev(plev, rho_tpll, planet_grav)  # (T, P-1, Lat, Lon)
        # Add layer thickness to topographic elevation
        z_tpll = compute_heights_from_dz(dz_tpll, topo_ll)  # (T, P, Lat, Lon)
    else:
        # Validate pre-computed heights
        if z_tpll.shape != var_tpll.shape:
            raise ValueError(f"z_tpll shape {z_tpll.shape} must match var_tpll shape {var_tpll.shape}")
    
    # Step 3: Vertical interpolation to uniform height grid
    # Reshape to (T, Lat, Lon, P) for vertical_interp_to_z
    var_tllp = np.moveaxis(var_tpll, 1, -1)  # (T, Lat, Lon, P)
    z_tllp = np.moveaxis(z_tpll, 1, -1)      # (T, Lat, Lon, P)
    
    # Interpolate vertically (parallelized)
    var_tllz = vertical_interp_to_z(z_tllp, var_tllp, x1f, bounds_error=bounds_error, n_jobs=n_jobs)  # (T, Lat, Lon, Z)
    
    # Step 4: Convert lat/lon to local Cartesian coordinates
    lat_center = 0.5 * (lats[0] + lats[-1])
    Y_coord, X_coord = latlon_to_xy(lats, lons, planet_radius, lat_center)
    
    # Step 5: Match centers of input and output domains
    # Center of input domain
    Y_center_in = 0.5 * (Y_coord[0] + Y_coord[-1])
    X_center_in = 0.5 * (X_coord[0] + X_coord[-1])
    
    # Center of output domain
    Y_center_out = 0.5 * (x2f[0] + x2f[-1])
    X_center_out = 0.5 * (x3f[0] + x3f[-1])
    
    # Shift output coordinates to match input center
    x2f_shifted = x2f + (Y_center_in - Y_center_out)
    x3f_shifted = x3f + (X_center_in - X_center_out)
    
    # Check if output domain is within input domain when bounds_error is True
    if bounds_error:
        if x2f_shifted.min() < Y_coord.min() or x2f_shifted.max() > Y_coord.max():
            raise ValueError(
                f"Output Y domain [{x2f_shifted.min():.2f}, {x2f_shifted.max():.2f}] m "
                f"exceeds input Y domain [{Y_coord.min():.2f}, {Y_coord.max():.2f}] m. "
                f"Interpolated domain must be embedded within or equal to original domain."
            )
        if x3f_shifted.min() < X_coord.min() or x3f_shifted.max() > X_coord.max():
            raise ValueError(
                f"Output X domain [{x3f_shifted.min():.2f}, {x3f_shifted.max():.2f}] m "
                f"exceeds input X domain [{X_coord.min():.2f}, {X_coord.max():.2f}] m. "
                f"Interpolated domain must be embedded within or equal to original domain."
            )
    
    # Step 6: Horizontal interpolation for each time and height level (parallelized)
    Z = len(x1f)
    Y_out = len(x2f)
    X_out = len(x3f)
    var_tzyx = np.full((T, Z, Y_out, X_out), np.nan, dtype=var_tpll.dtype)
    
    # Determine number of jobs for horizontal regridding
    n_slices = T * Z
    if n_jobs is None:
        # Auto: Use parallelization only if we have enough slices to make it worthwhile
        n_jobs_horiz = min(cpu_count(), max(1, n_slices // _HORIZONTAL_REGRID_PARALLEL_THRESHOLD)) if n_slices > _HORIZONTAL_REGRID_PARALLEL_THRESHOLD else 1
    elif n_jobs == -1:
        n_jobs_horiz = cpu_count()
    elif n_jobs < 1:
        n_jobs_horiz = 1
    else:
        n_jobs_horiz = n_jobs
    
    if n_jobs_horiz == 1:
        # Sequential execution
        for ti in range(T):
            for zi in range(Z):
                slab = var_tllz[ti, :, :, zi]  # (Lat, Lon)
                
                # Skip if the slab is all NaNs (e.g., height level above all data)
                if np.all(np.isnan(slab)):
                    continue
                
                # Note: horizontal_regrid_xy expects (X, Y) order, so we transpose
                slab_t = slab.T  # (Lon, Lat) -> (X, Y) ordering
                var_tzyx[ti, zi, :, :] = horizontal_regrid_xy(
                    X_coord, Y_coord, slab_t, x3f_shifted, x2f_shifted, bounds_error=bounds_error
                ).T  # Transpose back to (Y, X)
    else:
        # Parallel execution
        args_list = [
            (ti, zi, var_tllz[ti, :, :, zi], X_coord, Y_coord, x3f_shifted, x2f_shifted, bounds_error)
            for ti in range(T) for zi in range(Z)
        ]
        
        with Pool(processes=n_jobs_horiz) as pool:
            results = pool.map(_regrid_horizontal_slice, args_list)
        
        # Collect results
        for ti, zi, result in results:
            if result is not None:
                var_tzyx[ti, zi, :, :] = result
    
    return var_tzyx


def _regrid_single_variable(args):
    """
    Helper function for parallel regridding of a single variable.
    
    Args:
        args: Tuple containing all arguments needed for regrid_pressure_to_height
        
    Returns:
        Tuple of (variable_name, regridded_data)
    """
    (var_name, var_tpll, rho_tpll, topo_ll, plev, lats, lons, 
     x1f, x2f, x3f, planet_grav, planet_radius, bounds_error, z_tpll, n_jobs) = args
    
    result = regrid_pressure_to_height(
        var_tpll, rho_tpll, topo_ll, plev, lats, lons,
        x1f, x2f, x3f, planet_grav, planet_radius,
        bounds_error=bounds_error, z_tpll=z_tpll, n_jobs=n_jobs
    )
    
    return (var_name, result)


def regrid_multiple_variables(
    variables: Dict[str, np.ndarray],  # Dictionary of variable_name: (T, P, Lat, Lon) arrays
    rho_tpll: np.ndarray,              # (T, P, Lat, Lon) air density [kg/m^3]
    topo_ll: np.ndarray,               # (Lat, Lon) topographic elevation [m]
    plev: np.ndarray,                  # (P,) pressure levels [Pa]
    lats: np.ndarray,                  # (Lat,) latitude [degrees]
    lons: np.ndarray,                  # (Lon,) longitude [degrees]
    x1f: np.ndarray,                   # (Z,) vertical height coordinate [m]
    x2f: np.ndarray,                   # (Y,) horizontal Y-coordinate [m]
    x3f: np.ndarray,                   # (X,) horizontal X-coordinate [m]
    planet_grav: float,                # gravity constant [m/s^2]
    planet_radius: float,              # radius of the planet [m]
    bounds_error: bool = True,         # If True, raise error when extrapolation would occur
    z_tpll: Optional[np.ndarray] = None,  # (T, P, Lat, Lon) pre-computed heights [m]
    n_jobs: Optional[int] = None,      # Number of parallel workers (prioritizes across variables)
) -> Dict[str, np.ndarray]:
    """
    Regrid multiple atmospheric variables in parallel for improved performance.
    
    This function intelligently distributes parallel workers, prioritizing parallelization
    across variables first, then within each variable if workers are available.
    
    Args:
        variables: Dictionary mapping variable names to (T, P, Lat, Lon) arrays
        rho_tpll: (T, P, Lat, Lon) air density [kg/m^3]
        topo_ll: (Lat, Lon) topographic elevation [m]
        plev: (P,) pressure levels [Pa], typically decreasing with height
        lats: (Lat,) latitude array [degrees]
        lons: (Lon,) longitude array [degrees]
        x1f: (Z,) vertical height coordinate [m]
        x2f: (Y,) horizontal Y-coordinate [m]
        x3f: (X,) horizontal X-coordinate [m]
        planet_grav: gravity constant [m/s^2]
        planet_radius: radius of the planet [m]
        bounds_error: If True, raise error when extrapolation would occur
        z_tpll: Optional pre-computed (T, P, Lat, Lon) heights [m]. If None,
                heights are computed once and reused for all variables.
        n_jobs: Number of parallel workers. None=auto (uses all CPUs), 1=sequential,
                -1=all CPUs. The system prioritizes parallelizing across variables first,
                then within each variable if workers are available.
        
    Returns:
        Dictionary mapping variable names to regridded (T, Z, Y, X) arrays
        
    Raises:
        ValueError: If output domain exceeds input domain when bounds_error=True
        
    Example:
        >>> # Regrid multiple variables efficiently
        >>> variables = {
        ...     'temperature': temp_tpll,
        ...     'humidity': humid_tpll,
        ...     'pressure': press_tpll,
        ... }
        >>> results = regrid_multiple_variables(
        ...     variables, rho_tpll, topo_ll, plev, lats, lons,
        ...     x1f, x2f, x3f, planet_grav, planet_radius,
        ...     n_jobs=-1  # Use all CPUs (prioritizes across variables)
        ... )
        >>> temp_tzyx = results['temperature']
        >>> humid_tzyx = results['humidity']
    """
    # Compute heights once if not provided
    if z_tpll is None:
        z_tpll = compute_height_grid(rho_tpll, topo_ll, plev, planet_grav)
    
    # Determine parallelization strategy: prioritize across variables first
    n_vars = len(variables)
    
    # Determine total available workers
    if n_jobs is None or n_jobs == -1:
        total_workers = cpu_count()
    elif n_jobs < 1:
        total_workers = 1
    else:
        total_workers = n_jobs
    
    # Allocate workers: prioritize across variables
    if n_vars > 1:
        # Use workers for variable parallelization first
        n_jobs_vars = min(total_workers, n_vars)
        # When parallelizing across variables, we cannot nest parallelization
        # within each variable due to multiprocessing daemon process limitations
        n_jobs_per_var = 1
    else:
        # Single variable: use all workers within the variable
        n_jobs_vars = 1
        n_jobs_per_var = total_workers
    
    # If only one variable or sequential processing, use simple loop
    if n_jobs_vars == 1 or n_vars == 1:
        results = {}
        for var_name, var_tpll in variables.items():
            results[var_name] = regrid_pressure_to_height(
                var_tpll, rho_tpll, topo_ll, plev, lats, lons,
                x1f, x2f, x3f, planet_grav, planet_radius,
                bounds_error=bounds_error, z_tpll=z_tpll, n_jobs=n_jobs_per_var
            )
        return results
    
    # Parallel processing across variables
    args_list = [
        (var_name, var_tpll, rho_tpll, topo_ll, plev, lats, lons,
         x1f, x2f, x3f, planet_grav, planet_radius, bounds_error, z_tpll, n_jobs_per_var)
        for var_name, var_tpll in variables.items()
    ]
    
    with Pool(processes=n_jobs_vars) as pool:
        results_list = pool.map(_regrid_single_variable, args_list)
    
    # Convert list of tuples back to dictionary
    results = {var_name: result for var_name, result in results_list}
    
    return results


def regrid_topography(
    topo_ll: np.ndarray,       # (Lat, Lon) topographic elevation [m]
    lats: np.ndarray,          # (Lat,) latitude [degrees]
    lons: np.ndarray,          # (Lon,) longitude [degrees]
    x2f: np.ndarray,           # (Y,) horizontal Y-coordinate [m]
    x3f: np.ndarray,           # (X,) horizontal X-coordinate [m]
    planet_radius: float,      # radius of the planet [m]
    bounds_error: bool = True, # If True, raise error when extrapolation would occur
) -> np.ndarray:
    """
    Regrid topographic elevation from lat/lon to distance grids.
    
    Args:
        topo_ll: (Lat, Lon) topographic elevation [m]
        lats: (Lat,) latitude array [degrees]
        lons: (Lon,) longitude array [degrees]
        x2f: (Y,) horizontal Y-coordinate [m]
        x3f: (X,) horizontal X-coordinate [m]
        planet_radius: radius of the planet [m]
        bounds_error: If True, raise error when extrapolation would occur
        
    Returns:
        topo_yx: (Y, X) topographic elevation on distance grids
        
    Raises:
        ValueError: If output domain exceeds input domain when bounds_error=True
    """
    Lat, Lon = topo_ll.shape
    
    # Validate input shapes
    if lats.shape[0] != Lat:
        raise ValueError(f"lats length {lats.shape[0]} must match Lat dimension {Lat}")
    if lons.shape[0] != Lon:
        raise ValueError(f"lons length {lons.shape[0]} must match Lon dimension {Lon}")
    
    # Convert lat/lon to local Cartesian coordinates
    lat_center = 0.5 * (lats[0] + lats[-1])
    Y_coord, X_coord = latlon_to_xy(lats, lons, planet_radius, lat_center)
    
    # Match centers
    Y_center_in = 0.5 * (Y_coord[0] + Y_coord[-1])
    X_center_in = 0.5 * (X_coord[0] + X_coord[-1])
    Y_center_out = 0.5 * (x2f[0] + x2f[-1])
    X_center_out = 0.5 * (x3f[0] + x3f[-1])
    
    x2f_shifted = x2f + (Y_center_in - Y_center_out)
    x3f_shifted = x3f + (X_center_in - X_center_out)
    
    # Check bounds
    if bounds_error:
        if x2f_shifted.min() < Y_coord.min() or x2f_shifted.max() > Y_coord.max():
            raise ValueError(
                f"Output Y domain [{x2f_shifted.min():.2f}, {x2f_shifted.max():.2f}] m "
                f"exceeds input Y domain [{Y_coord.min():.2f}, {Y_coord.max():.2f}] m."
            )
        if x3f_shifted.min() < X_coord.min() or x3f_shifted.max() > X_coord.max():
            raise ValueError(
                f"Output X domain [{x3f_shifted.min():.2f}, {x3f_shifted.max():.2f}] m "
                f"exceeds input X domain [{X_coord.min():.2f}, {X_coord.max():.2f}] m."
            )
    
    # Horizontal interpolation
    # horizontal_regrid_xy expects (X, Y) order
    topo_t = topo_ll.T  # (Lon, Lat) -> (X, Y) ordering
    topo_yx = horizontal_regrid_xy(
        X_coord, Y_coord, topo_t, x3f_shifted, x2f_shifted, bounds_error=bounds_error
    ).T  # Transpose back to (Y, X)
    
    return topo_yx

def save_regridded_data_to_netcdf(
    filename: str,
    variables: Dict[str, np.ndarray],
    coordinates: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, Any]] = None,
    processing_history: Optional[str] = None,
) -> None:
    """
    Save regridded atmospheric data to NetCDF file with metadata preservation.
    
    Args:
        filename: Output NetCDF file path
        variables: Dictionary of variable arrays, e.g., {'temperature': array, 'density': array}
                  Each array should have shape (T, Z, Y, X) where:
                  - T: time dimension
                  - Z: height dimension (x1f)
                  - Y: Y-coordinate dimension (x2f, North-South)
                  - X: X-coordinate dimension (x3f, East-West)
        coordinates: Dictionary with coordinate arrays:
                    - 'time': (T,) time values
                    - 'x1f': (Z,) height coordinates in meters
                    - 'x2f': (Y,) Y-coordinates in meters
                    - 'x3f': (X,) X-coordinates in meters
        metadata: Optional dictionary with metadata to preserve (e.g., source info, units)
        processing_history: Optional processing history string to add to global attributes
        
    Example:
        >>> variables = {'temperature': temp_tzyx, 'density': rho_tzyx}
        >>> coordinates = {'time': times, 'x1f': x1f, 'x2f': x2f, 'x3f': x3f}
        >>> metadata = {'source': 'ECMWF ERA5', 'region': 'White Sands, NM'}
        >>> save_regridded_data_to_netcdf('output.nc', variables, coordinates, metadata)
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError(
            "netCDF4 package is required for saving data. "
            "Install it with: pip install netCDF4"
        )
    
    with Dataset(filename, "w", format="NETCDF4") as ncfile:
        # Get dimensions from first variable
        first_var = next(iter(variables.values()))
        T, Z, Y, X = first_var.shape
        
        # Create dimensions
        ncfile.createDimension("time", T)
        ncfile.createDimension("x1", Z)  # Height dimension
        ncfile.createDimension("x2", Y)  # Y dimension (North-South)
        ncfile.createDimension("x3", X)  # X dimension (East-West)
        
        # Create coordinate variables
        time_var = ncfile.createVariable("time", "f8", ("time",))
        x1_var = ncfile.createVariable("x1", "f8", ("x1",))
        x2_var = ncfile.createVariable("x2", "f8", ("x2",))
        x3_var = ncfile.createVariable("x3", "f8", ("x3",))
        
        # Set coordinate attributes
        time_var.axis = "T"
        time_var.long_name = "time"
        if 'time' in coordinates:
            time_var[:] = coordinates['time'].astype("f8")
            # Try to infer time units from metadata
            if metadata and 'time_units' in metadata:
                time_var.units = metadata['time_units']
            else:
                time_var.units = "hours since 1900-01-01 00:00:00"
        else:
            time_var[:] = np.arange(T, dtype="f8")
            time_var.units = "timestep"
        
        x1_var.axis = "Z"
        x1_var.long_name = "height"
        x1_var.units = "meters"
        x1_var.positive = "up"
        if 'x1f' in coordinates:
            x1_var[:] = coordinates['x1f'].astype("f8")
        else:
            x1_var[:] = np.arange(Z, dtype="f8")
        
        x2_var.axis = "Y"
        x2_var.long_name = "y_coordinate"
        x2_var.units = "meters"
        x2_var.standard_name = "projection_y_coordinate"
        if 'x2f' in coordinates:
            x2_var[:] = coordinates['x2f'].astype("f8")
        else:
            x2_var[:] = np.arange(Y, dtype="f8")
        
        x3_var.axis = "X"
        x3_var.long_name = "x_coordinate"
        x3_var.units = "meters"
        x3_var.standard_name = "projection_x_coordinate"
        if 'x3f' in coordinates:
            x3_var[:] = coordinates['x3f'].astype("f8")
        else:
            x3_var[:] = np.arange(X, dtype="f8")
        
        # Create data variables
        for var_name, var_data in variables.items():
            if var_data.shape != (T, Z, Y, X):
                raise ValueError(
                    f"Variable '{var_name}' has shape {var_data.shape}, "
                    f"expected ({T}, {Z}, {Y}, {X})"
                )
            
            var = ncfile.createVariable(var_name, "f4", ("time", "x1", "x2", "x3"))
            var[:] = var_data.astype("f4")
            
            # Set variable attributes from metadata
            if metadata and f"{var_name}_units" in metadata:
                var.units = metadata[f"{var_name}_units"]
            if metadata and f"{var_name}_long_name" in metadata:
                var.long_name = metadata[f"{var_name}_long_name"]
            if metadata and f"{var_name}_standard_name" in metadata:
                var.standard_name = metadata[f"{var_name}_standard_name"]
        
        # Set global attributes
        ncfile.title = "Regridded ECMWF ERA5 Data"
        ncfile.institution = "Generated by ECMWF regridding module"
        ncfile.source = metadata.get('source', 'ECMWF ERA5') if metadata else 'ECMWF ERA5'
        ncfile.conventions = "CF-1.8"
        ncfile.creation_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Add coordinate system information
        ncfile.coordinate_system = "Local Cartesian projection"
        ncfile.grid_description = (
            "Distance grid with x1 (height) in meters positive upward, "
            "x2 (Y) in meters positive northward, "
            "x3 (X) in meters positive eastward"
        )
        
        # Add processing history
        if processing_history:
            ncfile.history = processing_history
        else:
            ncfile.history = (
                f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}: "
                f"Regridded from pressure-lat-lon to distance grids using ECMWF regridding module"
            )
        
        # Add any additional metadata
        if metadata:
            for key, value in metadata.items():
                # Skip keys already handled
                if key in ['source', 'time_units'] or '_units' in key or '_long_name' in key or '_standard_name' in key:
                    continue
                # Convert value to string if it's not a basic type
                if isinstance(value, (str, int, float)):
                    setattr(ncfile, key, value)
                else:
                    setattr(ncfile, key, str(value))


def save_topography_to_netcdf(
    filename: str,
    topography: np.ndarray,
    x2f: np.ndarray,
    x3f: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    processing_history: Optional[str] = None,
) -> None:
    """
    Save regridded topographic elevation to NetCDF file.
    
    Args:
        filename: Output NetCDF file path
        topography: (Y, X) topographic elevation array in meters
        x2f: (Y,) Y-coordinates in meters
        x3f: (X,) X-coordinates in meters
        metadata: Optional dictionary with metadata to preserve
        processing_history: Optional processing history string
        
    Example:
        >>> save_topography_to_netcdf('topography.nc', topo_yx, x2f, x3f,
        ...                          metadata={'source': 'USGS DEM'})
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError(
            "netCDF4 package is required for saving data. "
            "Install it with: pip install netCDF4"
        )
    
    Y, X = topography.shape
    
    if x2f.shape[0] != Y:
        raise ValueError(f"x2f length {x2f.shape[0]} must match Y dimension {Y}")
    if x3f.shape[0] != X:
        raise ValueError(f"x3f length {x3f.shape[0]} must match X dimension {X}")
    
    with Dataset(filename, "w", format="NETCDF4") as ncfile:
        # Create dimensions
        ncfile.createDimension("x2", Y)
        ncfile.createDimension("x3", X)
        
        # Create coordinate variables
        x2_var = ncfile.createVariable("x2", "f8", ("x2",))
        x3_var = ncfile.createVariable("x3", "f8", ("x3",))
        
        x2_var.axis = "Y"
        x2_var.long_name = "y_coordinate"
        x2_var.units = "meters"
        x2_var.standard_name = "projection_y_coordinate"
        x2_var[:] = x2f.astype("f8")
        
        x3_var.axis = "X"
        x3_var.long_name = "x_coordinate"
        x3_var.units = "meters"
        x3_var.standard_name = "projection_x_coordinate"
        x3_var[:] = x3f.astype("f8")
        
        # Create topography variable
        topo_var = ncfile.createVariable("topography", "f4", ("x2", "x3"))
        topo_var[:] = topography.astype("f4")
        topo_var.units = "meters"
        topo_var.long_name = "topographic elevation"
        topo_var.standard_name = "surface_altitude"
        topo_var.positive = "up"
        
        # Set global attributes
        ncfile.title = "Regridded Topographic Elevation"
        ncfile.institution = "Generated by ECMWF regridding module"
        ncfile.source = metadata.get('source', 'Unknown') if metadata else 'Unknown'
        ncfile.conventions = "CF-1.8"
        ncfile.creation_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Add coordinate system information
        ncfile.coordinate_system = "Local Cartesian projection"
        ncfile.grid_description = (
            "Distance grid with x2 (Y) in meters positive northward, "
            "x3 (X) in meters positive eastward"
        )
        
        # Add processing history
        if processing_history:
            ncfile.history = processing_history
        else:
            ncfile.history = (
                f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}: "
                f"Regridded from lat-lon to distance grids using ECMWF regridding module"
            )
        
        # Add any additional metadata
        if metadata:
            for key, value in metadata.items():
                if key == 'source':
                    continue
                if isinstance(value, (str, int, float)):
                    setattr(ncfile, key, value)
                else:
                    setattr(ncfile, key, str(value))


# --------------------------
# Example usage:
# --------------------------
# 
# Example: Complete ECMWF ERA5 regridding pipeline
# ---------------------------------------------------
# from ecmwf.regrid import regrid_pressure_to_height, regrid_topography
# 
# # Input data dimensions
# T, P, Lat, Lon = 10, 8, 40, 60
# 
# # Input grids (ECMWF ERA5 format)
# plev = np.array([100000., 92500., 85000., 70000., 50000., 30000., 20000., 10000.])  # Pa
# lats = np.linspace(30.0, 35.0, Lat)   # degrees
# lons = np.linspace(-110.0, -105.0, Lon)  # degrees
# 
# # Input data (T, P, Lat, Lon)
# temp_tpll = ...  # Temperature data
# rho_tpll = ...   # Density data
# topo_ll = ...    # Topographic elevation (Lat, Lon)
# 
# # Output grids (distance in meters)
# x1f = np.linspace(0., 10000., 50)      # Height: 0-10 km
# x2f = np.linspace(-20000., 20000., 80)  # Y-coordinate: ±20 km
# x3f = np.linspace(-30000., 30000., 100) # X-coordinate: ±30 km
# 
# # Constants
# planet_grav = 9.81      # m/s^2
# planet_radius = 6371.e3  # Earth radius in meters
# 
# # Regrid atmospheric variables
# temp_tzyx = regrid_pressure_to_height(
#     temp_tpll, rho_tpll, topo_ll,
#     plev, lats, lons,
#     x1f, x2f, x3f,
#     planet_grav, planet_radius,
#     bounds_error=True  # Raise error if extrapolation would occur
# )
# # Output shape: (T, len(x1f), len(x2f), len(x3f))
# 
# # Regrid topography
# topo_yx = regrid_topography(
#     topo_ll, lats, lons,
#     x2f, x3f, planet_radius,
#     bounds_error=True
# )
# # Output shape: (len(x2f), len(x3f))

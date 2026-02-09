#!/usr/bin/env python3
"""
Plot Lift Condensation Level (LCL) contour map.

This script calculates and plots the Lift Condensation Level (LCL) from
temperature, pressure, and relative humidity data.
"""

import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from topo_utils import load_topography, find_matching_topography, prepare_topography_for_plot


def calculate_lcl(temperature, pressure, rh):
    """
    Calculate the Lift Condensation Level (LCL) height.
    
    Uses the approximation formula based on temperature and dewpoint.
    
    Parameters
    ----------
    temperature : array
        Temperature at surface (K)
    pressure : array
        Pressure at surface (Pa)
    rh : array
        Relative humidity (fraction 0-1)
    
    Returns
    -------
    array
        LCL height in meters
    """
    # Convert temperature to Celsius
    T_c = temperature - 273.15
    
    # Calculate dewpoint using Magnus formula
    # Ensure RH is between 0 and 1
    rh_clipped = np.clip(rh, 0.01, 0.99)
    
    # Magnus formula constants
    a = 17.27
    b = 237.7
    
    # Calculate dewpoint
    alpha = (a * T_c) / (b + T_c) + np.log(rh_clipped)
    dewpoint = (b * alpha) / (a - alpha)
    
    # Calculate LCL height using simplified Espy's formula
    # LCL (m) ≈ 125 * (T - Td) where T and Td are in Celsius
    # Note: This is an approximation valid for typical atmospheric conditions.
    # More rigorous methods (e.g., Bolton 1980) exist but require iterative calculations.
    lcl_height = 125.0 * (T_c - dewpoint)
    
    # Ensure non-negative heights
    lcl_height = np.maximum(lcl_height, 0.0)
    
    return lcl_height


def plot_lcl(input_file_out1, input_file_out2, output_file=None, time_index=0, topo_dir=None, location_prefix=None):
    """
    Create a contour plot of Lift Condensation Level (LCL).
    
    Parameters
    ----------
    input_file_out1 : str
        Path to the NetCDF file containing pressure data (out1.nc)
    input_file_out2 : str
        Path to the NetCDF file containing temperature and RH data (out2.nc)
    output_file : str, optional
        Path to save the output plot. If None, displays interactively.
    time_index : int, optional
        Time index to plot (default: 0)
    topo_dir : str, optional
        Directory containing topography .pt files
    location_prefix : str, optional
        Prefix for topography files (e.g., 'ws-site1')
    """
    # Load the datasets
    ds1 = xr.open_dataset(input_file_out1)
    ds2 = xr.open_dataset(input_file_out2)
    
    # Get surface data (lowest level, index 0 in x1 dimension)
    # Get pressure from out1.nc
    press_surface = ds1['press'].isel(time=time_index, x1=0).values
    
    # Get temperature from out2.nc
    temp_surface = ds2['temp'].isel(time=time_index, x1=0).values
    
    # Get relative humidity from out2.nc
    if 'rh_H2O_l_' in ds2.data_vars:
        rh_surface = ds2['rh_H2O_l_'].isel(time=time_index, x1=0).values
    else:
        # If RH not available, use a default value
        print("Warning: rh_H2O_l_ not found, using default RH=0.7")
        rh_surface = np.ones_like(temp_surface) * 0.7
    
    # Calculate LCL
    lcl = calculate_lcl(temp_surface, press_surface, rh_surface)
    
    # Get coordinates (convert from meters to kilometers)
    x2 = ds1['x2'].values / 1000.0  # X-coordinate in km
    x3 = ds1['x3'].values / 1000.0  # Y-coordinate in km
    
    # Create meshgrid for plotting
    X2, X3 = np.meshgrid(x2, x3)
    
    # Load topography if available
    topo_contours = None
    if topo_dir and location_prefix:
        netcdf_shape = press_surface.shape  # (x3, x2)
        topo_file = find_matching_topography(netcdf_shape, topo_dir, location_prefix)
        if topo_file:
            try:
                topo_data = load_topography(topo_file)
                X2_topo, X3_topo, topo_elevation = prepare_topography_for_plot(topo_data, x2, x3)
                # Convert elevation to km
                topo_contours = (X2_topo, X3_topo, topo_elevation / 1000.0)
            except Exception as e:
                print(f"Warning: Could not load topography: {e}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert LCL to km for plotting
    lcl_km = lcl / 1000.0
    
    # Create contour plot
    contour = ax.contourf(X2, X3, lcl_km, levels=20, cmap='YlOrRd')
    
    # Add contour lines
    contour_lines = ax.contour(X2, X3, lcl_km, levels=10, colors='black', 
                               linewidths=0.5, alpha=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f km')
    
    # Add topography contours if available
    if topo_contours:
        X2_topo, X3_topo, topo_elev_km = topo_contours
        topo_lines = ax.contour(X2_topo, X3_topo, topo_elev_km, levels=8,
                               colors='purple', linewidths=1.0, alpha=0.6, linestyles='solid')
        ax.clabel(topo_lines, inline=True, fontsize=7, fmt='%.1f km')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, label='LCL Height (km)')
    
    # Convert time from seconds to hours
    time_hours = ds1["time"].values[time_index] / 3600.0
    
    # Labels and title
    ax.set_xlabel('X-coordinate (km)')
    ax.set_ylabel('Y-coordinate (km)')
    ax.set_title(f'Lift Condensation Level (LCL) - Time: {time_hours:.2f} hours')
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    ds1.close()
    ds2.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot Lift Condensation Level (LCL) contour map'
    )
    parser.add_argument('input_file_out1', help='Input NetCDF file with pressure (out1.nc)')
    parser.add_argument('input_file_out2', help='Input NetCDF file with temperature (out2.nc)')
    parser.add_argument('-o', '--output', help='Output plot file (PNG)')
    parser.add_argument('-t', '--time', type=int, default=0,
                        help='Time index to plot (default: 0)')
    parser.add_argument('--topo-dir', help='Directory containing topography .pt files')
    parser.add_argument('--location', help='Location prefix for topography files (e.g., ws-site1)')
    
    args = parser.parse_args()
    
    plot_lcl(args.input_file_out1, args.input_file_out2, args.output, args.time, args.topo_dir, args.location)


if __name__ == '__main__':
    main()

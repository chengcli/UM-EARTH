#!/usr/bin/env python3
"""
Plot surface pressure contour map from NetCDF output files.

This script reads pressure data from out1.nc and creates a contour plot
of the surface pressure field.
"""

import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from topo_utils import load_topography, find_matching_topography, prepare_topography_for_plot


def plot_surface_pressure(input_file, output_file=None, time_index=0, topo_dir=None, location_prefix=None):
    """
    Create a contour plot of surface pressure.
    
    Parameters
    ----------
    input_file : str
        Path to the NetCDF file containing pressure data (out1.nc)
    output_file : str, optional
        Path to save the output plot. If None, displays interactively.
    time_index : int, optional
        Time index to plot (default: 0)
    topo_dir : str, optional
        Directory containing topography .pt files
    location_prefix : str, optional
        Prefix for topography files (e.g., 'ws-site1')
    """
    # Load the dataset
    ds = xr.open_dataset(input_file)
    
    # Get surface pressure (lowest level, index 0 in x1 dimension)
    # Data shape is (time, x1, x3, x2)
    press_data = ds['press'].isel(time=time_index, x1=0)
    
    # Get coordinates (convert from meters to kilometers)
    x2 = ds['x2'].values / 1000.0  # X-coordinate in km
    x3 = ds['x3'].values / 1000.0  # Y-coordinate in km
    
    # Create meshgrid for plotting
    X2, X3 = np.meshgrid(x2, x3)
    
    # Load topography if available
    topo_contours = None
    if topo_dir and location_prefix:
        netcdf_shape = press_data.shape  # (x3, x2)
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
    
    # Create contour plot
    # Convert pressure to hPa for easier reading
    press_hPa = press_data.values / 100.0
    
    # Create filled contour plot
    contour = ax.contourf(X2, X3, press_hPa, levels=20, cmap='viridis')
    
    # Add contour lines
    contour_lines = ax.contour(X2, X3, press_hPa, levels=10, colors='black', 
                                linewidths=0.5, alpha=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8)
    
    # Add topography contours if available
    if topo_contours:
        X2_topo, X3_topo, topo_elev_km = topo_contours
        topo_lines = ax.contour(X2_topo, X3_topo, topo_elev_km, levels=8,
                               colors='brown', linewidths=1.0, alpha=0.6, linestyles='solid')
        ax.clabel(topo_lines, inline=True, fontsize=7, fmt='%.1f km')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, label='Pressure (hPa)')
    
    # Convert time from seconds to hours
    time_hours = ds["time"].values[time_index] / 3600.0
    
    # Labels and title
    ax.set_xlabel('X-coordinate (km)')
    ax.set_ylabel('Y-coordinate (km)')
    ax.set_title(f'Surface Pressure - Time: {time_hours:.2f} hours')
    
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
    ds.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot surface pressure contour map from NetCDF output'
    )
    parser.add_argument('input_file', help='Input NetCDF file (out1.nc)')
    parser.add_argument('-o', '--output', help='Output plot file (PNG)')
    parser.add_argument('-t', '--time', type=int, default=0,
                        help='Time index to plot (default: 0)')
    parser.add_argument('--topo-dir', help='Directory containing topography .pt files')
    parser.add_argument('--location', help='Location prefix for topography files (e.g., ws-site1)')
    
    args = parser.parse_args()
    
    plot_surface_pressure(args.input_file, args.output, args.time, args.topo_dir, args.location)


if __name__ == '__main__':
    main()

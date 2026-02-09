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


def plot_surface_pressure(input_file, output_file=None, time_index=0):
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
    """
    # Load the dataset
    ds = xr.open_dataset(input_file)
    
    # Get surface pressure (lowest level, index 0 in x1 dimension)
    # Data shape is (time, x1, x3, x2)
    press_data = ds['press'].isel(time=time_index, x1=0)
    
    # Get coordinates
    x2 = ds['x2'].values  # X-coordinate
    x3 = ds['x3'].values  # Y-coordinate
    
    # Create meshgrid for plotting
    X2, X3 = np.meshgrid(x2, x3)
    
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
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, label='Pressure (hPa)')
    
    # Labels and title
    ax.set_xlabel('X-coordinate (m)')
    ax.set_ylabel('Y-coordinate (m)')
    ax.set_title(f'Surface Pressure - Time: {ds["time"].values[time_index]:.2f} s')
    
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
    
    args = parser.parse_args()
    
    plot_surface_pressure(args.input_file, args.output, args.time)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Plot vertical velocity contour maps at multiple heights.

This script reads vertical velocity data from out1.nc and creates contour plots
at specified heights: surface, 1.6 km, 2 km, 3 km, and 4 km.
"""

import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def find_nearest_level(x1_coords, target_height):
    """
    Find the index of the nearest vertical level to the target height.
    
    Parameters
    ----------
    x1_coords : array
        Vertical coordinate values (in meters)
    target_height : float
        Target height in meters
    
    Returns
    -------
    int
        Index of the nearest level
    """
    idx = np.argmin(np.abs(x1_coords - target_height))
    return idx


def plot_vertical_velocity(input_file, output_file=None, time_index=0):
    """
    Create contour plots of vertical velocity at multiple heights.
    
    Parameters
    ----------
    input_file : str
        Path to the NetCDF file containing velocity data (out1.nc)
    output_file : str, optional
        Path to save the output plot. If None, displays interactively.
    time_index : int, optional
        Time index to plot (default: 0)
    """
    # Load the dataset
    ds = xr.open_dataset(input_file)
    
    # Define target heights in meters
    target_heights = [0, 1600, 2000, 3000, 4000]  # surface, 1.6km, 2km, 3km, 4km
    
    # Get coordinates
    x1 = ds['x1'].values  # Z-coordinate (height)
    x2 = ds['x2'].values  # X-coordinate
    x3 = ds['x3'].values  # Y-coordinate
    
    # Create meshgrid for plotting
    X2, X3 = np.meshgrid(x2, x3)
    
    # Create subplots - 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, target_h in enumerate(target_heights):
        # Find nearest level
        level_idx = find_nearest_level(x1, target_h)
        actual_height = x1[level_idx]
        
        # Get vertical velocity at this level
        # Data shape is (time, x1, x3, x2)
        vel1_data = ds['vel1'].isel(time=time_index, x1=level_idx)
        
        # Create contour plot
        ax = axes[i]
        contour = ax.contourf(X2, X3, vel1_data.values, levels=20, cmap='RdBu_r')
        
        # Add contour lines
        contour_lines = ax.contour(X2, X3, vel1_data.values, levels=10, 
                                   colors='black', linewidths=0.5, alpha=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, label='Vertical Velocity (m/s)')
        
        # Labels and title
        ax.set_xlabel('X-coordinate (m)')
        ax.set_ylabel('Y-coordinate (m)')
        ax.set_title(f'Vertical Velocity at {actual_height:.1f} m')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Hide the last subplot (we only have 5 plots)
    axes[5].axis('off')
    
    # Add overall title
    fig.suptitle(f'Vertical Velocity - Time: {ds["time"].values[time_index]:.2f} s', 
                 fontsize=16, y=0.995)
    
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
        description='Plot vertical velocity contour maps at multiple heights'
    )
    parser.add_argument('input_file', help='Input NetCDF file (out1.nc)')
    parser.add_argument('-o', '--output', help='Output plot file (PNG)')
    parser.add_argument('-t', '--time', type=int, default=0,
                        help='Time index to plot (default: 0)')
    
    args = parser.parse_args()
    
    plot_vertical_velocity(args.input_file, args.output, args.time)


if __name__ == '__main__':
    main()

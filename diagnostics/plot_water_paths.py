#!/usr/bin/env python3
"""
Plot water cloud path and water vapor path contour maps.

This script reads water path data from out2.nc and creates contour plots
for water cloud path and water vapor path.
"""

import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def plot_water_paths(input_file, output_file=None, time_index=0):
    """
    Create contour plots of water cloud path and water vapor path.
    
    Parameters
    ----------
    input_file : str
        Path to the NetCDF file containing water path data (out2.nc)
    output_file : str, optional
        Path to save the output plot. If None, displays interactively.
    time_index : int, optional
        Time index to plot (default: 0)
    """
    # Load the dataset
    ds = xr.open_dataset(input_file)
    
    # Get coordinates
    x2 = ds['x2'].values  # X-coordinate
    x3 = ds['x3'].values  # Y-coordinate
    
    # Create meshgrid for plotting
    X2, X3 = np.meshgrid(x2, x3)
    
    # Create subplots - 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Water Cloud Path
    ax1 = axes[0]
    
    # Get water cloud path data (path_H2O_l_)
    # Data shape is (time, x3, x2)
    if 'path_H2O_l_' in ds.data_vars:
        cloud_path = ds['path_H2O_l_'].isel(time=time_index).values
        
        # Create contour plot
        contour1 = ax1.contourf(X2, X3, cloud_path, levels=20, cmap='Blues')
        
        # Add contour lines
        contour_lines1 = ax1.contour(X2, X3, cloud_path, levels=10, 
                                     colors='black', linewidths=0.5, alpha=0.5)
        ax1.clabel(contour_lines1, inline=True, fontsize=8)
        
        # Add colorbar
        cbar1 = plt.colorbar(contour1, ax=ax1, label='Water Cloud Path (kg/m²)')
        
        # Labels and title
        ax1.set_xlabel('X-coordinate (m)')
        ax1.set_ylabel('Y-coordinate (m)')
        ax1.set_title('Water Cloud Path (Liquid Water)')
        
        # Add grid
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    else:
        ax1.text(0.5, 0.5, 'path_H2O_l_ not found in dataset', 
                ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: Water Vapor Path
    ax2 = axes[1]
    
    # Get water vapor path data (path_H2O)
    if 'path_H2O' in ds.data_vars:
        vapor_path = ds['path_H2O'].isel(time=time_index).values
        
        # Create contour plot
        contour2 = ax2.contourf(X2, X3, vapor_path, levels=20, cmap='Greens')
        
        # Add contour lines
        contour_lines2 = ax2.contour(X2, X3, vapor_path, levels=10, 
                                     colors='black', linewidths=0.5, alpha=0.5)
        ax2.clabel(contour_lines2, inline=True, fontsize=8)
        
        # Add colorbar
        cbar2 = plt.colorbar(contour2, ax=ax2, label='Water Vapor Path (kg/m²)')
        
        # Labels and title
        ax2.set_xlabel('X-coordinate (m)')
        ax2.set_ylabel('Y-coordinate (m)')
        ax2.set_title('Water Vapor Path')
        
        # Add grid
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, 'path_H2O not found in dataset', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # Add overall title
    fig.suptitle(f'Water Paths - Time: {ds["time"].values[time_index]:.2f} s', 
                 fontsize=16, y=1.00)
    
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
        description='Plot water cloud path and water vapor path contour maps'
    )
    parser.add_argument('input_file', help='Input NetCDF file (out2.nc)')
    parser.add_argument('-o', '--output', help='Output plot file (PNG)')
    parser.add_argument('-t', '--time', type=int, default=0,
                        help='Time index to plot (default: 0)')
    
    args = parser.parse_args()
    
    plot_water_paths(args.input_file, args.output, args.time)


if __name__ == '__main__':
    main()

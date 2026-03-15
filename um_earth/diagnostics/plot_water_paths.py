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
from topo_utils import load_topography, find_matching_topography, prepare_topography_for_plot


def plot_water_paths(input_file, output_file=None, time_index=0, topo_dir=None, location_prefix=None):
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
    topo_dir : str, optional
        Directory containing topography .pt files
    location_prefix : str, optional
        Prefix for topography files (e.g., 'ws-site1')
    """
    # Load the dataset
    ds = xr.open_dataset(input_file)
    
    # Get coordinates (convert from meters to kilometers)
    x2 = ds['x2'].values / 1000.0  # X-coordinate in km
    x3 = ds['x3'].values / 1000.0  # Y-coordinate in km
    
    # Create meshgrid for plotting
    X2, X3 = np.meshgrid(x2, x3)
    
    # Load topography if available
    topo_contours = None
    if topo_dir and location_prefix:
        # Get shape from one of the path variables
        if 'path_H2O_l_' in ds.data_vars:
            first_slice = ds['path_H2O_l_'].isel(time=time_index)
            netcdf_shape = first_slice.shape  # (x3, x2)
            topo_file = find_matching_topography(netcdf_shape, topo_dir, location_prefix)
            if topo_file:
                try:
                    topo_data = load_topography(topo_file)
                    X2_topo, X3_topo, topo_elevation = prepare_topography_for_plot(topo_data, x2, x3)
                    # Convert elevation to km
                    topo_contours = (X2_topo, X3_topo, topo_elevation / 1000.0)
                except Exception as e:
                    print(f"Warning: Could not load topography: {e}")
    
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
        
        # Add topography contours if available
        if topo_contours:
            X2_topo, X3_topo, topo_elev_km = topo_contours
            topo_lines = ax1.contour(X2_topo, X3_topo, topo_elev_km, levels=6,
                                    colors='purple', linewidths=0.8, alpha=0.6, linestyles='solid')
            ax1.clabel(topo_lines, inline=True, fontsize=7, fmt='%.1f km')
        
        # Add colorbar
        cbar1 = plt.colorbar(contour1, ax=ax1, label='Water Cloud Path (kg/m²)')
        
        # Labels and title
        ax1.set_xlabel('X-coordinate (km)')
        ax1.set_ylabel('Y-coordinate (km)')
        ax1.set_title('Water Cloud Path (Liquid Water)')
        
        # Set equal aspect ratio
        ax1.set_aspect('equal', adjustable='box')
        
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
        
        # Add topography contours if available
        if topo_contours:
            X2_topo, X3_topo, topo_elev_km = topo_contours
            topo_lines = ax2.contour(X2_topo, X3_topo, topo_elev_km, levels=6,
                                    colors='purple', linewidths=0.8, alpha=0.6, linestyles='solid')
            ax2.clabel(topo_lines, inline=True, fontsize=7, fmt='%.1f km')
        
        # Add colorbar
        cbar2 = plt.colorbar(contour2, ax=ax2, label='Water Vapor Path (kg/m²)')
        
        # Labels and title
        ax2.set_xlabel('X-coordinate (km)')
        ax2.set_ylabel('Y-coordinate (km)')
        ax2.set_title('Water Vapor Path')
        
        # Set equal aspect ratio
        ax2.set_aspect('equal', adjustable='box')
        
        # Add grid
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, 'path_H2O not found in dataset', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # Convert time from seconds to hours
    time_hours = ds["time"].values[time_index] / 3600.0
    
    # Add overall title
    fig.suptitle(f'Water Paths - Time: {time_hours:.2f} hours', 
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
    parser.add_argument('--topo-dir', help='Directory containing topography .pt files')
    parser.add_argument('--location', help='Location prefix for topography files (e.g., ws-site1)')
    
    args = parser.parse_args()
    
    plot_water_paths(args.input_file, args.output, args.time, args.topo_dir, args.location)


if __name__ == '__main__':
    main()

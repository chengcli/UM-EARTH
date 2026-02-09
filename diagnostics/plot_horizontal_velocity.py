#!/usr/bin/env python3
"""
Plot horizontal velocity vector maps at multiple heights.

This script reads horizontal velocity data (vel2, vel3) from out1.nc and creates
vector/quiver plots at specified heights: surface, 1.6 km, 2 km, 3 km, and 4 km.
"""

import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from topo_utils import load_topography, find_matching_topography, prepare_topography_for_plot


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


def plot_horizontal_velocity(input_file, output_file=None, time_index=0, skip=2, topo_dir=None, location_prefix=None):
    """
    Create vector plots of horizontal velocity at multiple heights.
    
    Parameters
    ----------
    input_file : str
        Path to the NetCDF file containing velocity data (out1.nc)
    output_file : str, optional
        Path to save the output plot. If None, displays interactively.
    time_index : int, optional
        Time index to plot (default: 0)
    skip : int, optional
        Skip factor for quiver plot to reduce arrow density (default: 2)
    topo_dir : str, optional
        Directory containing topography .pt files
    location_prefix : str, optional
        Prefix for topography files (e.g., 'ws-site1')
    """
    # Load the dataset
    ds = xr.open_dataset(input_file)
    
    # Define target heights in meters
    target_heights = [0, 1600, 2000, 3000, 4000]  # surface, 1.6km, 2km, 3km, 4km
    
    # Get coordinates (convert from meters to kilometers)
    x1 = ds['x1'].values  # Z-coordinate (height) in meters
    x2 = ds['x2'].values / 1000.0  # X-coordinate in km
    x3 = ds['x3'].values / 1000.0  # Y-coordinate in km
    
    # Create meshgrid for plotting
    X2, X3 = np.meshgrid(x2, x3)
    
    # Load topography if available
    topo_contours = None
    if topo_dir and location_prefix:
        # Use the first data slice to get the shape
        first_slice = ds['vel2'].isel(time=time_index, x1=0)
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
    
    # Create subplots - 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, target_h in enumerate(target_heights):
        # Find nearest level
        level_idx = find_nearest_level(x1, target_h)
        actual_height = x1[level_idx]
        
        # Get horizontal velocities at this level
        # Data shape is (time, x1, x3, x2)
        vel2_data = ds['vel2'].isel(time=time_index, x1=level_idx).values  # U component
        vel3_data = ds['vel3'].isel(time=time_index, x1=level_idx).values  # V component
        
        # Calculate velocity magnitude for color
        vel_mag = np.sqrt(vel2_data**2 + vel3_data**2)
        
        ax = axes[i]
        
        # Create filled contour of velocity magnitude as background
        contour = ax.contourf(X2, X3, vel_mag, levels=20, cmap='viridis', alpha=0.7)
        
        # Add quiver plot (skip every nth point to reduce density)
        quiver = ax.quiver(X2[::skip, ::skip], X3[::skip, ::skip], 
                          vel2_data[::skip, ::skip], vel3_data[::skip, ::skip],
                          vel_mag[::skip, ::skip], cmap='plasma', 
                          scale_units='xy', scale=None, alpha=0.8)
        
        # Add topography contours if available
        if topo_contours:
            X2_topo, X3_topo, topo_elev_km = topo_contours
            topo_lines = ax.contour(X2_topo, X3_topo, topo_elev_km, levels=6,
                                   colors='brown', linewidths=0.8, alpha=0.6, linestyles='solid')
            ax.clabel(topo_lines, inline=True, fontsize=6, fmt='%.1f km')
        
        # Add colorbar for magnitude
        cbar = plt.colorbar(contour, ax=ax, label='Velocity Magnitude (m/s)')
        
        # Labels and title
        ax.set_xlabel('X-coordinate (km)')
        ax.set_ylabel('Y-coordinate (km)')
        ax.set_title(f'Horizontal Velocity at {actual_height/1000.0:.1f} km')
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Hide the last subplot (we only have 5 plots)
    axes[5].axis('off')
    
    # Convert time from seconds to hours
    time_hours = ds["time"].values[time_index] / 3600.0
    
    # Add overall title
    fig.suptitle(f'Horizontal Velocity Vectors - Time: {time_hours:.2f} hours', 
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
        description='Plot horizontal velocity vector maps at multiple heights'
    )
    parser.add_argument('input_file', help='Input NetCDF file (out1.nc)')
    parser.add_argument('-o', '--output', help='Output plot file (PNG)')
    parser.add_argument('-t', '--time', type=int, default=0,
                        help='Time index to plot (default: 0)')
    parser.add_argument('-s', '--skip', type=int, default=2,
                        help='Skip factor for arrows (default: 2)')
    parser.add_argument('--topo-dir', help='Directory containing topography .pt files')
    parser.add_argument('--location', help='Location prefix for topography files (e.g., ws-site1)')
    
    args = parser.parse_args()
    
    plot_horizontal_velocity(args.input_file, args.output, args.time, args.skip, args.topo_dir, args.location)


if __name__ == '__main__':
    main()

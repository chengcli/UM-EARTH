#!/usr/bin/env python3
"""
Utility functions for loading and handling topography data.

This module provides functions to:
1. Load topography data from PyTorch .pt files
2. Match horizontal resolution between NetCDF files and topography data
3. Prepare topography contours for plotting
"""

import os
import numpy as np
import torch


def load_topography(topo_file):
    """
    Load topography data from a PyTorch .pt file.
    
    Parameters
    ----------
    topo_file : str
        Path to the topography .pt file
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'topography': 2D numpy array of elevation values (meters)
        - 'lat_bounds': latitude bounds (if available)
        - 'lon_bounds': longitude bounds (if available)
        - 'shape': shape of the topography array
    """
    if not os.path.exists(topo_file):
        raise FileNotFoundError(f"Topography file not found: {topo_file}")
    
    # Load the PyTorch file
    data = torch.load(topo_file, map_location='cpu')
    
    # Extract topography data
    if isinstance(data, dict):
        topo = data.get('topography', None)
        lat_bounds = data.get('lat_bounds', None)
        lon_bounds = data.get('lon_bounds', None)
    else:
        # If data is a tensor directly
        topo = data
        lat_bounds = None
        lon_bounds = None
    
    if topo is None:
        raise ValueError("No 'topography' key found in the .pt file")
    
    # Convert to numpy array
    if isinstance(topo, torch.Tensor):
        topo_array = topo.numpy()
    else:
        topo_array = np.array(topo)
    
    result = {
        'topography': topo_array,
        'shape': topo_array.shape,
        'lat_bounds': lat_bounds.numpy() if isinstance(lat_bounds, torch.Tensor) else lat_bounds,
        'lon_bounds': lon_bounds.numpy() if isinstance(lon_bounds, torch.Tensor) else lon_bounds
    }
    
    return result


def find_matching_topography(netcdf_shape, topo_dir, location_prefix):
    """
    Find the topography file that matches the horizontal resolution of the NetCDF data.
    
    The function looks for topography files in the format:
    - <location_prefix>_0.pt (60x60)
    - <location_prefix>_1.pt (120x120)
    - <location_prefix>_2.pt (240x240)
    - <location_prefix>_3.pt (480x480)
    
    Parameters
    ----------
    netcdf_shape : tuple
        Shape of the NetCDF horizontal dimensions (x3, x2)
    topo_dir : str
        Directory containing topography files
    location_prefix : str
        Prefix for topography files (e.g., 'ws-site1')
    
    Returns
    -------
    str or None
        Path to the matching topography file, or None if no match found
    """
    # Expected resolutions for each level
    # Based on the issue description:
    # _0.pt: 60x60
    # _1.pt: 120x120
    # _2.pt: 240x240
    # _3.pt: 480x480
    
    if topo_dir is None or not os.path.exists(topo_dir):
        if topo_dir is not None:
            print(f"Warning: Topography directory not found: {topo_dir}")
        return None
    
    # Try to match the resolution
    x3_size, x2_size = netcdf_shape
    
    # Check for matching files
    found_files = []
    for level in range(4):
        topo_file = os.path.join(topo_dir, f"{location_prefix}_{level}.pt")
        if os.path.exists(topo_file):
            found_files.append(topo_file)
            try:
                topo_data = load_topography(topo_file)
                topo_shape = topo_data['shape']
                
                # Check if dimensions match (allowing for transpose)
                if (topo_shape[0] == x3_size and topo_shape[1] == x2_size) or \
                   (topo_shape[0] == x2_size and topo_shape[1] == x3_size):
                    print(f"Using topography file: {topo_file} (shape: {topo_shape})")
                    return topo_file
            except Exception as e:
                print(f"Warning: Could not load {topo_file}: {e}")
                continue
    
    if found_files:
        files_str = '\n  '.join(found_files)
        print(f"Warning: Found topography files but none match NetCDF shape ({x3_size}, {x2_size}):")
        print(f"  {files_str}")
    else:
        print(f"Warning: No topography files found matching pattern: {os.path.join(topo_dir, location_prefix)}_*.pt")
    return None


def prepare_topography_for_plot(topo_data, x2, x3):
    """
    Prepare topography data for plotting as contours.
    
    Parameters
    ----------
    topo_data : dict
        Topography data from load_topography()
    x2 : array
        X-coordinate values from NetCDF (in meters)
    x3 : array
        Y-coordinate values from NetCDF (in meters)
    
    Returns
    -------
    tuple
        (X2_topo, X3_topo, topo_elevation) suitable for contour plotting
    """
    topo_array = topo_data['topography']
    
    # Create coordinate arrays for topography
    # Assume topography is uniformly spaced over the domain
    nx2 = len(x2)
    nx3 = len(x3)
    
    # Check if topography needs to be transposed to match NetCDF convention
    # NetCDF uses (x3, x2) convention for spatial data
    if topo_array.shape[0] != nx3 or topo_array.shape[1] != nx2:
        # Try transpose
        if topo_array.shape[0] == nx2 and topo_array.shape[1] == nx3:
            topo_array = topo_array.T
        else:
            raise ValueError(
                f"Topography shape {topo_array.shape} doesn't match "
                f"NetCDF shape ({nx3}, {nx2})"
            )
    
    # Create meshgrid for topography
    X2_topo, X3_topo = np.meshgrid(x2, x3)
    
    return X2_topo, X3_topo, topo_array

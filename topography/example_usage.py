#!/usr/bin/env python3
"""
Example usage of the topography module

This script demonstrates how to use the topography module to download,
manipulate, and analyze topographic elevation data.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from topography import get_topography, split, merge


def main():
    """Main example function."""
    print("=" * 70)
    print("Surface Topography Module - Example Usage")
    print("=" * 70)
    
    # Example 1: Basic download
    print("\n1. Downloading topography data for a region...")
    topo = get_topography(
        lat_bounds=(35.0, 40.0),
        lon_bounds=(-110.0, -105.0),
        nlat=50,
        nlon=50
    )
    
    print(f"   Downloaded data shape: {topo.data.shape}")
    print(f"   Latitude bounds: {topo.lat_bounds}")
    print(f"   Longitude bounds: {topo.lon_bounds}")
    print(f"   Elevation range: {topo.data.min():.1f} - {topo.data.max():.1f} meters")
    print(f"   Mean elevation: {topo.data.mean():.1f} meters")
    
    # Example 2: Increase resolution with download attempt
    print("\n2. Increasing resolution (try downloading from internet)...")
    topo_downloaded = split(topo, method='linear', try_download=True)
    print(f"   New shape: {topo_downloaded.data.shape}")
    print(f"   Elevation range: {topo_downloaded.data.min():.1f} - {topo_downloaded.data.max():.1f} meters")
    
    # Example 3: Increase resolution with interpolation only
    print("\n3. Increasing resolution (force interpolation, no download)...")
    topo_interp = split(topo, method='cubic', try_download=False)
    print(f"   New shape: {topo_interp.data.shape}")
    print(f"   Elevation range: {topo_interp.data.min():.1f} - {topo_interp.data.max():.1f} meters")
    
    # Example 4: Decrease resolution
    print("\n4. Decreasing resolution by merging...")
    topo_merged = merge(topo_downloaded)
    print(f"   New shape: {topo_merged.data.shape}")
    print(f"   Elevation range: {topo_merged.data.min():.1f} - {topo_merged.data.max():.1f} meters")
    
    # Example 5: Multiple splits
    print("\n5. Progressive refinement with multiple splits...")
    current_topo = topo
    for i in range(3):
        current_topo = split(current_topo, method='linear')
        print(f"   Level {i+1}: {current_topo.data.shape}")
    
    # Example 6: Multiple merges
    print("\n6. Progressive coarsening with multiple merges...")
    current_topo = topo_downloaded
    for i in range(2):
        current_topo = merge(current_topo)
        print(f"   Level {i+1}: {current_topo.data.shape}")
    
    # Example 7: Cache demonstration
    print("\n7. Demonstrating cache usage...")
    print("   First download (will be cached)...")
    topo_cache1 = get_topography(
        lat_bounds=(30.0, 35.0),
        lon_bounds=(-100.0, -95.0),
        nlat=20,
        nlon=20
    )
    
    print("   Second download (from cache, faster)...")
    topo_cache2 = get_topography(
        lat_bounds=(30.0, 35.0),
        lon_bounds=(-100.0, -95.0),
        nlat=20,
        nlon=20
    )
    
    print(f"   Data identical: {(topo_cache1.data == topo_cache2.data).all()}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()

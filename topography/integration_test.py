#!/usr/bin/env python3
"""
Integration test for the topography module

This script performs a comprehensive integration test of the topography module,
testing all major functionality in a realistic scenario.
"""

import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from topography import get_topography, split, merge


def test_integration():
    """Run comprehensive integration test."""
    print("=" * 70)
    print("Topography Module - Integration Test")
    print("=" * 70)
    
    # Create temporary cache directory
    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary cache: {temp_dir}")
    
    try:
        # Test 1: Download and cache
        print("\n[Test 1] Downloading topography data...")
        topo = get_topography(
            lat_bounds=(35.0, 40.0),
            lon_bounds=(-110.0, -105.0),
            nlat=20,
            nlon=20,
            cache_dir=temp_dir
        )
        assert topo.data.shape == (20, 20), "Shape mismatch"
        assert topo.nlat == 20 and topo.nlon == 20, "Dimension mismatch"
        print("✓ Download successful")
        
        # Test 2: Verify caching
        print("\n[Test 2] Verifying cache functionality...")
        topo2 = get_topography(
            lat_bounds=(35.0, 40.0),
            lon_bounds=(-110.0, -105.0),
            nlat=20,
            nlon=20,
            cache_dir=temp_dir
        )
        assert (topo.data == topo2.data).all(), "Cache data mismatch"
        print("✓ Cache working correctly")
        
        # Test 3: Split with linear interpolation
        print("\n[Test 3] Testing split (linear)...")
        topo_split_linear = split(topo, method='linear', cache_dir=temp_dir)
        assert topo_split_linear.data.shape == (40, 40), "Split shape mismatch"
        print("✓ Linear split successful")
        
        # Test 4: Split with cubic interpolation
        print("\n[Test 4] Testing split (cubic)...")
        topo_split_cubic = split(topo, method='cubic', cache_dir=temp_dir)
        assert topo_split_cubic.data.shape == (40, 40), "Split shape mismatch"
        print("✓ Cubic split successful")
        
        # Test 5: Merge
        print("\n[Test 5] Testing merge...")
        topo_merged = merge(topo_split_linear)
        assert topo_merged.data.shape == (20, 20), "Merge shape mismatch"
        print("✓ Merge successful")
        
        # Test 6: Multiple splits
        print("\n[Test 6] Testing multiple splits...")
        current = topo
        expected_sizes = [(40, 40), (80, 80), (160, 160)]
        for i, expected in enumerate(expected_sizes):
            current = split(current, method='linear', cache_dir=temp_dir)
            assert current.data.shape == expected, f"Multi-split {i+1} failed"
        print("✓ Multiple splits successful")
        
        # Test 7: Multiple merges
        print("\n[Test 7] Testing multiple merges...")
        current = topo_split_linear  # Start at 40x40
        expected_sizes = [(20, 20), (10, 10)]
        for i, expected in enumerate(expected_sizes):
            current = merge(current)
            assert current.data.shape == expected, f"Multi-merge {i+1} failed"
        print("✓ Multiple merges successful")
        
        # Test 8: Different regions
        print("\n[Test 8] Testing different geographic regions...")
        regions = [
            ((30.0, 35.0), (-100.0, -95.0)),
            ((40.0, 45.0), (-120.0, -115.0)),
            ((-10.0, 10.0), (0.0, 20.0))
        ]
        for i, (lat_bounds, lon_bounds) in enumerate(regions):
            region_topo = get_topography(
                lat_bounds=lat_bounds,
                lon_bounds=lon_bounds,
                nlat=10,
                nlon=10,
                cache_dir=temp_dir
            )
            assert region_topo.data.shape == (10, 10), f"Region {i+1} failed"
        print("✓ Multiple regions successful")
        
        # Test 9: Elevation properties
        print("\n[Test 9] Verifying elevation properties...")
        assert (topo.data >= 0).all(), "Negative elevation found"
        assert (topo.data < 10000).all(), "Unrealistic elevation found"
        print("✓ Elevation values are valid")
        
        # Test 10: Bounds preservation
        print("\n[Test 10] Verifying bounds preservation...")
        assert topo_split_linear.lat_bounds == topo.lat_bounds, "Lat bounds changed"
        assert topo_split_linear.lon_bounds == topo.lon_bounds, "Lon bounds changed"
        assert topo_merged.lat_bounds == topo.lat_bounds, "Lat bounds changed"
        assert topo_merged.lon_bounds == topo.lon_bounds, "Lon bounds changed"
        print("✓ Bounds preserved correctly")
        
        print("\n" + "=" * 70)
        print("All integration tests passed! ✓")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temporary cache: {temp_dir}")


if __name__ == '__main__':
    success = test_integration()
    sys.exit(0 if success else 1)

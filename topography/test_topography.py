"""
Unit tests for Surface Topography Module

This module contains comprehensive tests for the topography data downloading,
caching, and manipulation functionality.
"""

import unittest
import tempfile
import shutil
import os
import numpy as np

# Add topography module to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from topography import TopographyData, get_topography, split, merge


class TestTopographyData(unittest.TestCase):
    """Test cases for TopographyData class."""
    
    def test_initialization(self):
        """Test TopographyData initialization."""
        data = np.random.rand(10, 20)
        lat_bounds = (30.0, 40.0)
        lon_bounds = (-120.0, -110.0)
        
        topo = TopographyData(data, lat_bounds, lon_bounds, 10, 20)
        
        self.assertEqual(topo.nlat, 10)
        self.assertEqual(topo.nlon, 20)
        self.assertEqual(topo.lat_bounds, lat_bounds)
        self.assertEqual(topo.lon_bounds, lon_bounds)
        np.testing.assert_array_equal(topo.data, data)
    
    def test_repr(self):
        """Test TopographyData string representation."""
        data = np.random.rand(10, 20)
        topo = TopographyData(data, (30.0, 40.0), (-120.0, -110.0), 10, 20)
        
        repr_str = repr(topo)
        self.assertIn("TopographyData", repr_str)
        self.assertIn("shape", repr_str)
        self.assertIn("lat", repr_str)
        self.assertIn("lon", repr_str)


class TestGetTopography(unittest.TestCase):
    """Test cases for get_topography function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_download(self):
        """Test basic topography download."""
        topo = get_topography(
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=10,
            nlon=20,
            cache_dir=self.temp_dir
        )
        
        self.assertIsInstance(topo, TopographyData)
        self.assertEqual(topo.data.shape, (10, 20))
        self.assertEqual(topo.nlat, 10)
        self.assertEqual(topo.nlon, 20)
    
    def test_caching(self):
        """Test that data is properly cached."""
        # First download
        topo1 = get_topography(
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=10,
            nlon=20,
            cache_dir=self.temp_dir
        )
        
        # Second download should use cache
        topo2 = get_topography(
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=10,
            nlon=20,
            cache_dir=self.temp_dir
        )
        
        # Data should be identical
        np.testing.assert_array_equal(topo1.data, topo2.data)
    
    def test_no_cache(self):
        """Test downloading without cache."""
        topo = get_topography(
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=10,
            nlon=20,
            cache_dir=self.temp_dir,
            use_cache=False
        )
        
        self.assertIsInstance(topo, TopographyData)
        self.assertEqual(topo.data.shape, (10, 20))
    
    def test_invalid_lat_bounds(self):
        """Test that invalid latitude bounds raise ValueError."""
        with self.assertRaises(ValueError):
            get_topography(
                lat_bounds=(40.0, 30.0),  # min > max
                lon_bounds=(-120.0, -110.0),
                nlat=10,
                nlon=20,
                cache_dir=self.temp_dir
            )
    
    def test_invalid_lon_bounds(self):
        """Test that invalid longitude bounds raise ValueError."""
        with self.assertRaises(ValueError):
            get_topography(
                lat_bounds=(30.0, 40.0),
                lon_bounds=(-110.0, -120.0),  # min > max
                nlat=10,
                nlon=20,
                cache_dir=self.temp_dir
            )
    
    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            get_topography(
                lat_bounds=(30.0, 40.0),
                lon_bounds=(-120.0, -110.0),
                nlat=0,  # Invalid
                nlon=20,
                cache_dir=self.temp_dir
            )
    
    def test_lat_range_validation(self):
        """Test that latitude range is validated."""
        with self.assertRaises(ValueError):
            get_topography(
                lat_bounds=(-100.0, 40.0),  # < -90
                lon_bounds=(-120.0, -110.0),
                nlat=10,
                nlon=20,
                cache_dir=self.temp_dir
            )
    
    def test_lon_range_validation(self):
        """Test that longitude range is validated."""
        with self.assertRaises(ValueError):
            get_topography(
                lat_bounds=(30.0, 40.0),
                lon_bounds=(-200.0, -110.0),  # < -180
                nlat=10,
                nlon=20,
                cache_dir=self.temp_dir
            )
    
    def test_different_resolutions(self):
        """Test downloading different resolutions."""
        topo1 = get_topography(
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=10,
            nlon=20,
            cache_dir=self.temp_dir
        )
        
        topo2 = get_topography(
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=20,
            nlon=40,
            cache_dir=self.temp_dir
        )
        
        self.assertEqual(topo1.data.shape, (10, 20))
        self.assertEqual(topo2.data.shape, (20, 40))


class TestSplit(unittest.TestCase):
    """Test cases for split function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.topo = get_topography(
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=10,
            nlon=20,
            cache_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_split_linear(self):
        """Test split with linear interpolation."""
        topo_high = split(self.topo, method='linear', cache_dir=self.temp_dir)
        
        self.assertEqual(topo_high.data.shape, (20, 40))
        self.assertEqual(topo_high.nlat, 20)
        self.assertEqual(topo_high.nlon, 40)
        self.assertEqual(topo_high.lat_bounds, self.topo.lat_bounds)
        self.assertEqual(topo_high.lon_bounds, self.topo.lon_bounds)
    
    def test_split_cubic(self):
        """Test split with cubic interpolation."""
        topo_high = split(self.topo, method='cubic', cache_dir=self.temp_dir)
        
        self.assertEqual(topo_high.data.shape, (20, 40))
        self.assertEqual(topo_high.nlat, 20)
        self.assertEqual(topo_high.nlon, 40)
    
    def test_split_invalid_method(self):
        """Test that invalid interpolation method raises ValueError."""
        with self.assertRaises(ValueError):
            split(self.topo, method='invalid', cache_dir=self.temp_dir)
    
    def test_split_preserves_corners(self):
        """Test that split preserves corner values."""
        topo_high = split(self.topo, method='linear', cache_dir=self.temp_dir)
        
        # Corner values should be preserved (or very close due to interpolation)
        self.assertAlmostEqual(topo_high.data[0, 0], self.topo.data[0, 0], places=1)
        self.assertAlmostEqual(topo_high.data[-1, -1], self.topo.data[-1, -1], places=1)
    
    def test_split_caching(self):
        """Test that split results are cached."""
        topo_high1 = split(self.topo, method='linear', cache_dir=self.temp_dir)
        topo_high2 = split(self.topo, method='linear', cache_dir=self.temp_dir)
        
        np.testing.assert_array_equal(topo_high1.data, topo_high2.data)
    
    def test_split_no_cache(self):
        """Test split without caching."""
        topo_high = split(
            self.topo,
            method='linear',
            cache_dir=self.temp_dir,
            use_cache=False
        )
        
        self.assertEqual(topo_high.data.shape, (20, 40))
    
    def test_split_multiple_times(self):
        """Test splitting multiple times."""
        topo1 = split(self.topo, method='linear', cache_dir=self.temp_dir)
        topo2 = split(topo1, method='linear', cache_dir=self.temp_dir)
        
        self.assertEqual(topo1.data.shape, (20, 40))
        self.assertEqual(topo2.data.shape, (40, 80))


class TestMerge(unittest.TestCase):
    """Test cases for merge function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.topo = get_topography(
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=20,
            nlon=40,
            cache_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_merge_basic(self):
        """Test basic merge operation."""
        topo_low = merge(self.topo)
        
        self.assertEqual(topo_low.data.shape, (10, 20))
        self.assertEqual(topo_low.nlat, 10)
        self.assertEqual(topo_low.nlon, 20)
        self.assertEqual(topo_low.lat_bounds, self.topo.lat_bounds)
        self.assertEqual(topo_low.lon_bounds, self.topo.lon_bounds)
    
    def test_merge_invalid_dimensions(self):
        """Test that merge raises ValueError for odd dimensions."""
        topo_odd = get_topography(
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=11,  # Odd
            nlon=21,  # Odd
            cache_dir=self.temp_dir
        )
        
        with self.assertRaises(ValueError):
            merge(topo_odd)
    
    def test_merge_averages_correctly(self):
        """Test that merge correctly averages 2x2 blocks."""
        # Create simple test data with known values
        data = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=float)
        
        topo_test = TopographyData(
            data=data,
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=4,
            nlon=4
        )
        
        topo_merged = merge(topo_test)
        
        # Check that averaging is correct
        expected = np.array([
            [(1+2+5+6)/4, (3+4+7+8)/4],
            [(9+10+13+14)/4, (11+12+15+16)/4]
        ])
        
        np.testing.assert_array_almost_equal(topo_merged.data, expected)
    
    def test_merge_multiple_times(self):
        """Test merging multiple times."""
        topo1 = merge(self.topo)
        topo2 = merge(topo1)
        
        self.assertEqual(topo1.data.shape, (10, 20))
        self.assertEqual(topo2.data.shape, (5, 10))


class TestSplitMergeRoundtrip(unittest.TestCase):
    """Test cases for split-merge roundtrips."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.topo = get_topography(
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=10,
            nlon=20,
            cache_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_split_then_merge(self):
        """Test that split followed by merge returns to original resolution."""
        topo_high = split(self.topo, method='linear', cache_dir=self.temp_dir)
        topo_back = merge(topo_high)
        
        self.assertEqual(topo_back.data.shape, self.topo.data.shape)
        self.assertEqual(topo_back.nlat, self.topo.nlat)
        self.assertEqual(topo_back.nlon, self.topo.nlon)
        
        # Values should be similar but not exactly equal due to interpolation
        # Check that they're reasonably close (within 15% relative tolerance)
        self.assertTrue(np.allclose(topo_back.data, self.topo.data, rtol=0.15))
    
    def test_multiple_splits_and_merges(self):
        """Test multiple split and merge operations."""
        # Split twice
        topo1 = split(self.topo, method='linear', cache_dir=self.temp_dir)
        topo2 = split(topo1, method='linear', cache_dir=self.temp_dir)
        
        # Merge twice
        topo3 = merge(topo2)
        topo4 = merge(topo3)
        
        # Should be back to original resolution
        self.assertEqual(topo4.data.shape, self.topo.data.shape)


class TestDataProperties(unittest.TestCase):
    """Test cases for data properties and characteristics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_elevation_non_negative(self):
        """Test that elevation values are non-negative."""
        topo = get_topography(
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=10,
            nlon=20,
            cache_dir=self.temp_dir
        )
        
        self.assertTrue(np.all(topo.data >= 0))
    
    def test_elevation_reasonable_range(self):
        """Test that elevation values are in a reasonable range."""
        topo = get_topography(
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=10,
            nlon=20,
            cache_dir=self.temp_dir
        )
        
        # Elevation should be reasonable (not exceeding Mount Everest)
        self.assertTrue(np.all(topo.data < 10000))
    
    def test_data_is_float(self):
        """Test that data is in float format."""
        topo = get_topography(
            lat_bounds=(30.0, 40.0),
            lon_bounds=(-120.0, -110.0),
            nlat=10,
            nlon=20,
            cache_dir=self.temp_dir
        )
        
        self.assertTrue(np.issubdtype(topo.data.dtype, np.floating))


if __name__ == '__main__':
    unittest.main()

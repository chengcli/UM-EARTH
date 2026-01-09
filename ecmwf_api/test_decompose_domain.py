"""
Unit tests for decompose_domain module

This module contains tests for the domain decomposition script that breaks
regridded NetCDF files into multiple blocks.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from netCDF4 import Dataset

from decompose_domain import (
    read_netcdf_metadata,
    calculate_block_boundaries,
    extract_block,
    decompose_domain,
)


class TestCalculateBlockBoundaries(unittest.TestCase):
    """Test cases for block boundary calculation."""
    
    def test_simple_even_division(self):
        """Test even division with no remainder."""
        # 100 interior cells, 4 blocks, 3 ghost cells
        # Total grid: 106 cells (3 left ghost + 100 interior + 3 right ghost)
        # Interior region: [3:103] in full grid
        # Each block: 25 interior cells
        boundaries = calculate_block_boundaries(100, 4, 3)
        
        # Should have 4 blocks
        self.assertEqual(len(boundaries), 4)
        
        # First block: [0:31]
        # - Left ghosts: [0:3] (original domain ghosts)
        # - Interior: [3:28] (25 cells)
        # - Right ghosts: [28:31] (3 cells for overlap with next block)
        self.assertEqual(boundaries[0], (0, 31))
        
        # Second block: [25:56]
        # - Left ghosts: [25:28] (overlap from previous block)
        # - Interior: [28:53] (25 cells)
        # - Right ghosts: [53:56] (3 cells for overlap with next block)
        self.assertEqual(boundaries[1], (25, 56))
        
        # Third block: [50:81]
        self.assertEqual(boundaries[2], (50, 81))
        
        # Last block: [75:106]
        # - Left ghosts: [75:78] (overlap from previous block)
        # - Interior: [78:103] (25 cells)
        # - Right ghosts: [103:106] (original domain ghosts)
        self.assertEqual(boundaries[3], (75, 106))
    
    def test_uneven_division(self):
        """Test division with remainder."""
        # 200 interior cells, 4 blocks, 3 ghost cells
        # 200 / 4 = 50 cells per block
        # Total grid: 206 cells (3 + 200 + 3)
        boundaries = calculate_block_boundaries(200, 4, 3)
        
        self.assertEqual(len(boundaries), 4)
        
        # Each block should have 50 interior cells
        # First block: [0:56] (3 left ghost + 50 interior + 3 right ghost)
        self.assertEqual(boundaries[0], (0, 56))
        
        # Last block should end at 206 (200 interior + 6 total ghost)
        self.assertEqual(boundaries[3][1], 206)
    
    def test_single_block(self):
        """Test single block (no decomposition)."""
        # 100 interior cells, 1 block, 3 ghost cells
        boundaries = calculate_block_boundaries(100, 1, 3)
        
        self.assertEqual(len(boundaries), 1)
        # Should cover entire domain
        self.assertEqual(boundaries[0], (0, 106))  # [0:106] (100+6)
    
    def test_small_blocks(self):
        """Test many small blocks."""
        # 50 interior cells, 10 blocks, 2 ghost cells
        boundaries = calculate_block_boundaries(50, 10, 2)
        
        self.assertEqual(len(boundaries), 10)
        
        # First block
        self.assertEqual(boundaries[0][0], 0)
        
        # Last block should end at total size
        self.assertEqual(boundaries[-1][1], 54)  # 50 + 4 ghost
    
    def test_ghost_overlap(self):
        """Test that ghost zones overlap correctly between blocks."""
        # 100 interior cells, 4 blocks, 3 ghost cells
        boundaries = calculate_block_boundaries(100, 4, 3)
        
        # Check overlaps between consecutive blocks
        for i in range(len(boundaries) - 1):
            _, end_i = boundaries[i]
            start_next, _ = boundaries[i + 1]
            
            # The overlap should be 2*nghost cells (3+3=6)
            # end_i and start_next should overlap
            overlap = end_i - start_next
            self.assertGreaterEqual(overlap, 0, 
                f"Blocks {i} and {i+1} should overlap")


class TestCreateTestNetCDF(unittest.TestCase):
    """Helper class to create test NetCDF files."""
    
    @staticmethod
    def create_test_netcdf(
        filename: str,
        n_time: int = 4,
        n_x1: int = 156,
        n_x2: int = 206,
        n_x3: int = 206,
        nghost: int = 3,
        nx1_interior: int = 150,
        nx2_interior: int = 200,
        nx3_interior: int = 200
    ) -> None:
        """
        Create a test NetCDF file with the expected structure.
        
        Args:
            filename: Output filename
            n_time: Number of time steps
            n_x1: Number of vertical cells (including ghost)
            n_x2: Number of Y cells (including ghost)
            n_x3: Number of X cells (including ghost)
            nghost: Number of ghost cells on each side
            nx1_interior: Number of interior vertical cells
            nx2_interior: Number of interior Y cells
            nx3_interior: Number of interior X cells
        """
        with Dataset(filename, 'w', format='NETCDF4') as nc:
            # Create dimensions
            nc.createDimension('time', n_time)
            nc.createDimension('x1', n_x1)
            nc.createDimension('x2', n_x2)
            nc.createDimension('x3', n_x3)
            nc.createDimension('x1f', n_x1 + 1)
            nc.createDimension('x2f', n_x2 + 1)
            nc.createDimension('x3f', n_x3 + 1)
            
            # Create coordinate variables
            time = nc.createVariable('time', 'f8', ('time',))
            time[:] = np.arange(n_time)
            time.axis = 'T'
            time.long_name = 'time'
            
            x1 = nc.createVariable('x1', 'f8', ('x1',))
            x1[:] = np.linspace(0, 15000, n_x1)
            x1.axis = 'Z'
            x1.long_name = 'cell_center_height'
            x1.units = 'meters'
            
            x2 = nc.createVariable('x2', 'f8', ('x2',))
            x2[:] = np.linspace(0, 100000, n_x2)
            x2.axis = 'Y'
            x2.long_name = 'cell_center_y_coordinate'
            x2.units = 'meters'
            
            x3 = nc.createVariable('x3', 'f8', ('x3',))
            x3[:] = np.linspace(0, 100000, n_x3)
            x3.axis = 'X'
            x3.long_name = 'cell_center_x_coordinate'
            x3.units = 'meters'
            
            # Create interface coordinates
            x1f = nc.createVariable('x1f', 'f8', ('x1f',))
            x1f[:] = np.linspace(0, 15000, n_x1 + 1)
            x1f.long_name = 'cell_interface_height'
            x1f.units = 'meters'
            
            x2f = nc.createVariable('x2f', 'f8', ('x2f',))
            x2f[:] = np.linspace(0, 100000, n_x2 + 1)
            x2f.long_name = 'cell_interface_y_coordinate'
            x2f.units = 'meters'
            
            x3f = nc.createVariable('x3f', 'f8', ('x3f',))
            x3f[:] = np.linspace(0, 100000, n_x3 + 1)
            x3f.long_name = 'cell_interface_x_coordinate'
            x3f.units = 'meters'
            
            # Create test data variables on cell centers
            rho = nc.createVariable('rho', 'f4', ('time', 'x1', 'x2', 'x3'),
                                   zlib=True, complevel=4)
            rho[:] = np.random.randn(n_time, n_x1, n_x2, n_x3).astype('f4')
            rho.units = 'kg m-3'
            rho.long_name = 'Air density'
            
            u = nc.createVariable('u', 'f4', ('time', 'x1', 'x2', 'x3'),
                                 zlib=True, complevel=4)
            u[:] = np.random.randn(n_time, n_x1, n_x2, n_x3).astype('f4')
            u.units = 'm s-1'
            u.long_name = 'U component of wind'
            
            # Create pressure on vertical interfaces
            pressure_level = nc.createVariable('pressure_level', 'f4', 
                                             ('time', 'x1f', 'x2', 'x3'),
                                             zlib=True, complevel=4)
            pressure_level[:] = np.random.randn(n_time, n_x1 + 1, n_x2, n_x3).astype('f4')
            pressure_level.units = 'Pa'
            pressure_level.long_name = 'Pressure at cell interfaces'
            
            # Set global attributes
            nc.title = 'Test regridded data'
            nc.nghost = nghost
            nc.nx1_interior = nx1_interior
            nc.nx2_interior = nx2_interior
            nc.nx3_interior = nx3_interior
            nc.center_latitude = 42.3
            nc.center_longitude = -83.7
            nc.history = 'Created for testing'


class TestReadNetCDFMetadata(unittest.TestCase):
    """Test cases for reading NetCDF metadata."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test.nc')
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_read_metadata(self):
        """Test reading metadata from a valid NetCDF file."""
        TestCreateTestNetCDF.create_test_netcdf(self.test_file)
        
        metadata = read_netcdf_metadata(self.test_file)
        
        # Check dimensions
        self.assertEqual(metadata['dims']['time'], 4)
        self.assertEqual(metadata['dims']['x1'], 156)
        self.assertEqual(metadata['dims']['x2'], 206)
        self.assertEqual(metadata['dims']['x3'], 206)
        
        # Check ghost zone
        self.assertEqual(metadata['nghost'], 3)
        
        # Check interior dimensions
        self.assertEqual(metadata['nx2_interior'], 200)
        self.assertEqual(metadata['nx3_interior'], 200)
        
        # Check variables are listed
        self.assertIn('rho', metadata['variables'])
        self.assertIn('u', metadata['variables'])
        self.assertIn('pressure_level', metadata['variables'])
    
    def test_missing_nghost(self):
        """Test error when nghost attribute is missing."""
        # Create a file without nghost
        with Dataset(self.test_file, 'w', format='NETCDF4') as nc:
            nc.createDimension('time', 4)
            nc.createDimension('x1', 156)
            nc.createDimension('x2', 206)
            nc.createDimension('x3', 206)
        
        with self.assertRaises(ValueError) as context:
            read_netcdf_metadata(self.test_file)
        
        self.assertIn('nghost', str(context.exception))


class TestExtractBlock(unittest.TestCase):
    """Test cases for extracting individual blocks."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.temp_dir, 'input.nc')
        self.output_file = os.path.join(self.temp_dir, 'output.nc')
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_extract_single_block(self):
        """Test extracting a single block from the full domain."""
        # Create test file
        TestCreateTestNetCDF.create_test_netcdf(
            self.input_file,
            n_time=2,
            n_x1=10,
            n_x2=26,  # 20 interior + 6 ghost
            n_x3=26,
            nghost=3,
            nx2_interior=20,
            nx3_interior=20
        )
        
        # Read metadata
        metadata = read_netcdf_metadata(self.input_file)
        
        # Extract first block (0, 0) - should be [0:13] in both directions
        # (10 interior cells in first half + 3 right ghost)
        extract_block(
            self.input_file,
            self.output_file,
            0, 0,
            (0, 13),
            (0, 13),
            metadata
        )
        
        # Verify output file
        with Dataset(self.output_file, 'r') as nc:
            self.assertEqual(len(nc.dimensions['time']), 2)
            self.assertEqual(len(nc.dimensions['x1']), 10)
            self.assertEqual(len(nc.dimensions['x2']), 13)
            self.assertEqual(len(nc.dimensions['x3']), 13)
            
            # Check block index attributes
            self.assertEqual(nc.block_index_x2, 0)
            self.assertEqual(nc.block_index_x3, 0)
            self.assertEqual(nc.block_x2_start, 0)
            self.assertEqual(nc.block_x2_end, 13)
            
            # Check that variables were extracted
            self.assertIn('rho', nc.variables)
            self.assertEqual(nc.variables['rho'].shape, (2, 10, 13, 13))
    
    def test_extract_with_pressure_level(self):
        """Test that pressure_level on interfaces is handled correctly."""
        TestCreateTestNetCDF.create_test_netcdf(
            self.input_file,
            n_time=2,
            n_x1=10,
            n_x2=26,
            n_x3=26,
            nghost=3
        )
        
        metadata = read_netcdf_metadata(self.input_file)
        
        extract_block(
            self.input_file,
            self.output_file,
            0, 0,
            (0, 13),
            (0, 13),
            metadata
        )
        
        # Verify pressure_level has correct shape
        with Dataset(self.output_file, 'r') as nc:
            self.assertIn('pressure_level', nc.variables)
            # pressure_level is on (time, x1f, x2, x3)
            self.assertEqual(nc.variables['pressure_level'].shape, (2, 11, 13, 13))


class TestDecomposeDomain(unittest.TestCase):
    """Test cases for full domain decomposition."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.temp_dir, 'regridded_test.nc')
        self.output_dir = os.path.join(self.temp_dir, 'blocks')
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_decompose_2x2(self):
        """Test decomposing into 2x2 blocks."""
        # Create test file with 20x20 interior + 3 ghost = 26x26 total
        TestCreateTestNetCDF.create_test_netcdf(
            self.input_file,
            n_time=2,
            n_x1=10,
            n_x2=26,
            n_x3=26,
            nghost=3,
            nx2_interior=20,
            nx3_interior=20
        )
        
        # Decompose
        output_files = decompose_domain(
            self.input_file,
            2,  # 2 blocks in x2
            2,  # 2 blocks in x3
            self.output_dir
        )
        
        # Should create 4 blocks
        self.assertEqual(len(output_files), 4)
        
        # Check that all files exist
        for f in output_files:
            self.assertTrue(os.path.exists(f))
        
        # Check filenames
        expected_names = [
            'regridded_test_block_0_0.nc',
            'regridded_test_block_0_1.nc',
            'regridded_test_block_1_0.nc',
            'regridded_test_block_1_1.nc',
        ]
        
        for expected in expected_names:
            expected_path = os.path.join(self.output_dir, expected)
            self.assertIn(expected_path, output_files)
    
    def test_decompose_4x4(self):
        """Test decomposing into 4x4 blocks (as in the issue example)."""
        # Create test file with 200x200 interior + 6 ghost = 206x206 total
        TestCreateTestNetCDF.create_test_netcdf(
            self.input_file,
            n_time=4,
            n_x1=156,
            n_x2=206,
            n_x3=206,
            nghost=3,
            nx1_interior=150,
            nx2_interior=200,
            nx3_interior=200
        )
        
        # Decompose
        output_files = decompose_domain(
            self.input_file,
            4,  # 4 blocks in x2
            4,  # 4 blocks in x3
            self.output_dir
        )
        
        # Should create 16 blocks
        self.assertEqual(len(output_files), 16)
        
        # Check dimensions of each block
        # Each block should have approximately (50 + 6) x (50 + 6) horizontal cells
        for f in output_files:
            with Dataset(f, 'r') as nc:
                # No vertical decomposition
                self.assertEqual(len(nc.dimensions['x1']), 156)
                
                # Horizontal dimensions should be approximately 56
                # (50 interior + 6 ghost, but may vary slightly due to remainder)
                x2_size = len(nc.dimensions['x2'])
                x3_size = len(nc.dimensions['x3'])
                
                # Should be in range [53, 59] (50±3 + 6 ghost)
                self.assertGreaterEqual(x2_size, 53)
                self.assertLessEqual(x2_size, 59)
                self.assertGreaterEqual(x3_size, 53)
                self.assertLessEqual(x3_size, 59)
    
    def test_single_block(self):
        """Test decomposition with single block (no actual decomposition)."""
        TestCreateTestNetCDF.create_test_netcdf(
            self.input_file,
            n_time=2,
            n_x1=10,
            n_x2=26,
            n_x3=26,
            nghost=3,
            nx2_interior=20,
            nx3_interior=20
        )
        
        output_files = decompose_domain(
            self.input_file,
            1,  # 1 block in x2
            1,  # 1 block in x3
            self.output_dir
        )
        
        # Should create 1 block
        self.assertEqual(len(output_files), 1)
        
        # Block should have same dimensions as input
        with Dataset(output_files[0], 'r') as nc_out:
            with Dataset(self.input_file, 'r') as nc_in:
                self.assertEqual(
                    len(nc_out.dimensions['x2']),
                    len(nc_in.dimensions['x2'])
                )
                self.assertEqual(
                    len(nc_out.dimensions['x3']),
                    len(nc_in.dimensions['x3'])
                )
    
    def test_preserve_data_values(self):
        """Test that data values are correctly preserved in blocks."""
        # Create small test case for easy verification
        TestCreateTestNetCDF.create_test_netcdf(
            self.input_file,
            n_time=1,
            n_x1=5,
            n_x2=16,  # 10 interior + 6 ghost
            n_x3=16,
            nghost=3,
            nx2_interior=10,
            nx3_interior=10
        )
        
        # Set known values in input file
        with Dataset(self.input_file, 'r+') as nc:
            # Create a simple pattern: value = x2_index + x3_index
            for i2 in range(16):
                for i3 in range(16):
                    nc.variables['rho'][0, :, i2, i3] = i2 + i3
        
        # Decompose into 2x2
        output_files = decompose_domain(
            self.input_file,
            2,
            2,
            self.output_dir
        )
        
        # Verify that values match
        with Dataset(self.input_file, 'r') as nc_in:
            rho_in = nc_in.variables['rho'][0, 0, :, :]
            
            for block_file in output_files:
                with Dataset(block_file, 'r') as nc_block:
                    x2_start = nc_block.block_x2_start
                    x2_end = nc_block.block_x2_end
                    x3_start = nc_block.block_x3_start
                    x3_end = nc_block.block_x3_end
                    
                    rho_block = nc_block.variables['rho'][0, 0, :, :]
                    rho_expected = rho_in[x2_start:x2_end, x3_start:x3_end]
                    
                    np.testing.assert_array_equal(
                        rho_block,
                        rho_expected,
                        err_msg=f"Data mismatch in block {block_file}"
                    )
    
    def test_parallel_processing_with_workers(self):
        """Test parallel processing with explicit number of workers."""
        # Create test file
        TestCreateTestNetCDF.create_test_netcdf(
            self.input_file,
            n_time=2,
            n_x1=10,
            n_x2=26,
            n_x3=26,
            nghost=3,
            nx2_interior=20,
            nx3_interior=20
        )
        
        # Test with 2 workers
        output_files = decompose_domain(
            self.input_file,
            2,
            2,
            self.output_dir,
            num_workers=2
        )
        
        # Should create 4 blocks
        self.assertEqual(len(output_files), 4)
        
        # Verify all files exist
        for f in output_files:
            self.assertTrue(os.path.exists(f))
    
    def test_sequential_processing(self):
        """Test sequential processing with single worker."""
        # Create test file
        TestCreateTestNetCDF.create_test_netcdf(
            self.input_file,
            n_time=2,
            n_x1=10,
            n_x2=26,
            n_x3=26,
            nghost=3,
            nx2_interior=20,
            nx3_interior=20
        )
        
        # Test with 1 worker (sequential)
        output_files = decompose_domain(
            self.input_file,
            2,
            2,
            self.output_dir,
            num_workers=1
        )
        
        # Should create 4 blocks
        self.assertEqual(len(output_files), 4)
        
        # Verify all files exist and contain correct data
        for f in output_files:
            self.assertTrue(os.path.exists(f))
            with Dataset(f, 'r') as nc:
                # Check that variables exist
                self.assertIn('rho', nc.variables)
                self.assertIn('u', nc.variables)


if __name__ == '__main__':
    unittest.main()

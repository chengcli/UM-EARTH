"""
Unit tests for convert_netcdf_to_tensor module

This module contains tests for the NetCDF to PyTorch tensor conversion script
that converts decomposed block files from Step 5 to tensor files for simulation.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

try:
    from netCDF4 import Dataset
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if NETCDF4_AVAILABLE and TORCH_AVAILABLE:
    from convert_netcdf_to_tensor import (
        convert_netcdf_to_tensor,
        convert_directory,
        save_tensors,
    )


def create_test_netcdf_block(
    filename: str,
    n_time: int = 4,
    n_x1: int = 56,
    n_x2: int = 56,
    n_x3: int = 56,
    include_cloud_vars: bool = True,
    pressure_on_interfaces: bool = False
) -> None:
    """
    Create a test NetCDF block file with typical structure from Step 5.
    
    Args:
        filename: Output filename
        n_time: Number of time steps
        n_x1: Number of vertical cells
        n_x2: Number of Y cells
        n_x3: Number of X cells
        include_cloud_vars: Whether to include cloud water variables
        pressure_on_interfaces: Whether pressure is on vertical interfaces
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
        
        x1 = nc.createVariable('x1', 'f8', ('x1',))
        x1[:] = np.linspace(0, 15000, n_x1)
        x1.axis = 'Z'
        x1.units = 'meters'
        
        x2 = nc.createVariable('x2', 'f8', ('x2',))
        x2[:] = np.linspace(0, 28000, n_x2)
        x2.axis = 'Y'
        x2.units = 'meters'
        
        x3 = nc.createVariable('x3', 'f8', ('x3',))
        x3[:] = np.linspace(0, 28000, n_x3)
        x3.axis = 'X'
        x3.units = 'meters'
        
        # Create interface coordinates
        x1f = nc.createVariable('x1f', 'f8', ('x1f',))
        x1f[:] = np.linspace(0, 15000, n_x1 + 1)
        
        x2f = nc.createVariable('x2f', 'f8', ('x2f',))
        x2f[:] = np.linspace(0, 28000, n_x2 + 1)
        
        x3f = nc.createVariable('x3f', 'f8', ('x3f',))
        x3f[:] = np.linspace(0, 28000, n_x3 + 1)
        
        # Create required data variables on cell centers
        rho = nc.createVariable('rho', 'f4', ('time', 'x1', 'x2', 'x3'),
                               zlib=True, complevel=4)
        rho[:] = np.random.randn(n_time, n_x1, n_x2, n_x3).astype('f4') + 1.2
        rho.units = 'kg m-3'
        
        u = nc.createVariable('u', 'f4', ('time', 'x1', 'x2', 'x3'),
                             zlib=True, complevel=4)
        u[:] = np.random.randn(n_time, n_x1, n_x2, n_x3).astype('f4')
        u.units = 'm s-1'
        
        v = nc.createVariable('v', 'f4', ('time', 'x1', 'x2', 'x3'),
                             zlib=True, complevel=4)
        v[:] = np.random.randn(n_time, n_x1, n_x2, n_x3).astype('f4')
        v.units = 'm s-1'
        
        w = nc.createVariable('w', 'f4', ('time', 'x1', 'x2', 'x3'),
                             zlib=True, complevel=4)
        w[:] = np.random.randn(n_time, n_x1, n_x2, n_x3).astype('f4') * 0.01
        w.units = 'Pa s-1'
        
        q = nc.createVariable('q', 'f4', ('time', 'x1', 'x2', 'x3'),
                             zlib=True, complevel=4)
        q[:] = np.abs(np.random.randn(n_time, n_x1, n_x2, n_x3).astype('f4')) * 0.01
        q.units = 'kg kg-1'
        
        # Create pressure variable
        if pressure_on_interfaces:
            # Pressure on vertical interfaces
            pressure_level = nc.createVariable('pressure_level', 'f4', 
                                             ('time', 'x1f', 'x2', 'x3'),
                                             zlib=True, complevel=4)
            pressure_level[:] = (100000.0 - 
                               np.linspace(0, 90000, n_x1 + 1).reshape(1, -1, 1, 1) +
                               np.random.randn(n_time, n_x1 + 1, n_x2, n_x3).astype('f4') * 100)
            pressure_level.units = 'Pa'
        else:
            # Pressure on cell centers
            p = nc.createVariable('p', 'f4', ('time', 'x1', 'x2', 'x3'),
                                 zlib=True, complevel=4)
            p[:] = (100000.0 - 
                   np.linspace(0, 90000, n_x1).reshape(1, -1, 1, 1) +
                   np.random.randn(n_time, n_x1, n_x2, n_x3).astype('f4') * 100)
            p.units = 'Pa'
        
        # Create cloud water variables if requested
        if include_cloud_vars:
            ciwc = nc.createVariable('ciwc', 'f4', ('time', 'x1', 'x2', 'x3'),
                                    zlib=True, complevel=4)
            ciwc[:] = np.abs(np.random.randn(n_time, n_x1, n_x2, n_x3).astype('f4')) * 1e-6
            ciwc.units = 'kg kg-1'
            
            clwc = nc.createVariable('clwc', 'f4', ('time', 'x1', 'x2', 'x3'),
                                    zlib=True, complevel=4)
            clwc[:] = np.abs(np.random.randn(n_time, n_x1, n_x2, n_x3).astype('f4')) * 1e-6
            clwc.units = 'kg kg-1'
            
            cswc = nc.createVariable('cswc', 'f4', ('time', 'x1', 'x2', 'x3'),
                                    zlib=True, complevel=4)
            cswc[:] = np.abs(np.random.randn(n_time, n_x1, n_x2, n_x3).astype('f4')) * 1e-6
            cswc.units = 'kg kg-1'
            
            crwc = nc.createVariable('crwc', 'f4', ('time', 'x1', 'x2', 'x3'),
                                    zlib=True, complevel=4)
            crwc[:] = np.abs(np.random.randn(n_time, n_x1, n_x2, n_x3).astype('f4')) * 1e-6
            crwc.units = 'kg kg-1'
        
        # Set global attributes
        nc.title = 'Test block file'
        nc.nghost = 3
        nc.block_index_x2 = 0
        nc.block_index_x3 = 0


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestSaveTensors(unittest.TestCase):
    """Test cases for save_tensors function."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_single_tensor(self):
        """Test saving a single tensor."""
        tensor = torch.randn(2, 3, 4, 5, 6)
        tensor_map = {'test_tensor': tensor}
        
        output_file = os.path.join(self.temp_dir, 'test.restart')
        save_tensors(tensor_map, output_file)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Load and verify
        loaded = torch.jit.load(output_file)
        self.assertTrue(hasattr(loaded, 'test_tensor'))
        self.assertTrue(torch.allclose(loaded.test_tensor, tensor))
    
    def test_save_multiple_tensors(self):
        """Test saving multiple tensors."""
        tensor1 = torch.randn(2, 3, 4)
        tensor2 = torch.randn(5, 6)
        tensor_map = {
            'tensor1': tensor1,
            'tensor2': tensor2
        }
        
        output_file = os.path.join(self.temp_dir, 'test_multi.restart')
        save_tensors(tensor_map, output_file)
        
        # Load and verify
        loaded = torch.jit.load(output_file)
        self.assertTrue(torch.allclose(loaded.tensor1, tensor1))
        self.assertTrue(torch.allclose(loaded.tensor2, tensor2))


@unittest.skipIf(not (NETCDF4_AVAILABLE and TORCH_AVAILABLE), 
                "netCDF4 and PyTorch not available")
class TestConvertNetCDFToTensor(unittest.TestCase):
    """Test cases for convert_netcdf_to_tensor function."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_conversion(self):
        """Test basic NetCDF to tensor conversion."""
        input_file = os.path.join(self.temp_dir, 'test_block.nc')
        create_test_netcdf_block(input_file, n_time=2, n_x1=10, n_x2=10, n_x3=10)
        
        output_file = convert_netcdf_to_tensor(input_file)
        
        # Verify output file exists
        self.assertTrue(os.path.exists(output_file))
        self.assertTrue(output_file.endswith('.restart'))
        
        # Load and verify tensor
        loaded = torch.jit.load(output_file)
        self.assertTrue(hasattr(loaded, 'hydro_w'))
        
        hydro_w = loaded.hydro_w
        # Expected shape: (time=2, nvar=8, x3=10, x2=10, x1=10)
        self.assertEqual(hydro_w.shape, (2, 8, 10, 10, 10))
    
    def test_output_tensor_shape(self):
        """Test that output tensor has correct shape."""
        input_file = os.path.join(self.temp_dir, 'test_block.nc')
        n_time, n_x1, n_x2, n_x3 = 4, 20, 30, 40
        create_test_netcdf_block(input_file, n_time=n_time, n_x1=n_x1, 
                                n_x2=n_x2, n_x3=n_x3)
        
        output_file = convert_netcdf_to_tensor(input_file)
        loaded = torch.jit.load(output_file)
        hydro_w = loaded.hydro_w
        
        # Shape should be (time, nvar=8, x3, x2, x1)
        expected_shape = (n_time, 8, n_x3, n_x2, n_x1)
        self.assertEqual(hydro_w.shape, expected_shape)
    
    def test_variable_ordering(self):
        """Test that variables are ordered correctly in nvar dimension."""
        input_file = os.path.join(self.temp_dir, 'test_block.nc')
        n_time, n_x1, n_x2, n_x3 = 2, 5, 5, 5
        
        # Create file with known values
        with Dataset(input_file, 'w', format='NETCDF4') as nc:
            nc.createDimension('time', n_time)
            nc.createDimension('x1', n_x1)
            nc.createDimension('x2', n_x2)
            nc.createDimension('x3', n_x3)
            nc.createDimension('x1f', n_x1 + 1)
            nc.createDimension('x2f', n_x2 + 1)
            nc.createDimension('x3f', n_x3 + 1)
            
            # Coordinates
            time = nc.createVariable('time', 'f8', ('time',))
            time[:] = np.arange(n_time)
            x1 = nc.createVariable('x1', 'f8', ('x1',))
            x1[:] = np.arange(n_x1)
            x2 = nc.createVariable('x2', 'f8', ('x2',))
            x2[:] = np.arange(n_x2)
            x3 = nc.createVariable('x3', 'f8', ('x3',))
            x3[:] = np.arange(n_x3)
            
            # Create variables with distinct values
            shape = (n_time, n_x1, n_x2, n_x3)
            
            rho = nc.createVariable('rho', 'f4', ('time', 'x1', 'x2', 'x3'))
            rho[:] = np.full(shape, 1.0, dtype='f4')
            
            w = nc.createVariable('w', 'f4', ('time', 'x1', 'x2', 'x3'))
            w[:] = np.full(shape, 2.0, dtype='f4')
            
            v = nc.createVariable('v', 'f4', ('time', 'x1', 'x2', 'x3'))
            v[:] = np.full(shape, 3.0, dtype='f4')
            
            u = nc.createVariable('u', 'f4', ('time', 'x1', 'x2', 'x3'))
            u[:] = np.full(shape, 4.0, dtype='f4')
            
            p = nc.createVariable('p', 'f4', ('time', 'x1', 'x2', 'x3'))
            p[:] = np.full(shape, 5.0, dtype='f4')
            
            q = nc.createVariable('q', 'f4', ('time', 'x1', 'x2', 'x3'))
            q[:] = np.full(shape, 6.0, dtype='f4')
            
            ciwc = nc.createVariable('ciwc', 'f4', ('time', 'x1', 'x2', 'x3'))
            ciwc[:] = np.full(shape, 7.0, dtype='f4')
            
            clwc = nc.createVariable('clwc', 'f4', ('time', 'x1', 'x2', 'x3'))
            clwc[:] = np.full(shape, 8.0, dtype='f4')
            
            cswc = nc.createVariable('cswc', 'f4', ('time', 'x1', 'x2', 'x3'))
            cswc[:] = np.full(shape, 9.0, dtype='f4')
            
            crwc = nc.createVariable('crwc', 'f4', ('time', 'x1', 'x2', 'x3'))
            crwc[:] = np.full(shape, 10.0, dtype='f4')
            
            nc.nghost = 3
        
        output_file = convert_netcdf_to_tensor(input_file)
        loaded = torch.jit.load(output_file)
        hydro_w = loaded.hydro_w
        
        # Check variable order: rho, w, v, u, p, q, q2, q3
        # q2 = ciwc + clwc = 7 + 8 = 15
        # q3 = cswc + crwc = 9 + 10 = 19
        expected_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 15.0, 19.0]
        
        for i, expected in enumerate(expected_values):
            actual = hydro_w[0, i, 0, 0, 0].item()
            self.assertAlmostEqual(actual, expected, places=5,
                                 msg=f"Variable {i} should be {expected}, got {actual}")
    
    def test_cloud_variable_combination(self):
        """Test that cloud variables are combined correctly."""
        input_file = os.path.join(self.temp_dir, 'test_block.nc')
        create_test_netcdf_block(input_file, n_time=2, n_x1=5, n_x2=5, n_x3=5)
        
        # Read original values
        with Dataset(input_file, 'r') as nc:
            ciwc = nc.variables['ciwc'][:]
            clwc = nc.variables['clwc'][:]
            cswc = nc.variables['cswc'][:]
            crwc = nc.variables['crwc'][:]
        
        q2_expected = ciwc + clwc
        q3_expected = cswc + crwc
        
        output_file = convert_netcdf_to_tensor(input_file)
        loaded = torch.jit.load(output_file)
        hydro_w = loaded.hydro_w
        
        # Extract q2 (index 6) and q3 (index 7) from tensor
        # Remember: shape is (time, nvar, x3, x2, x1)
        q2_tensor = hydro_w[:, 6, :, :, :].numpy()
        q3_tensor = hydro_w[:, 7, :, :, :].numpy()
        
        # Transpose back to (time, x1, x2, x3) for comparison
        q2_tensor = np.transpose(q2_tensor, (0, 3, 2, 1))
        q3_tensor = np.transpose(q3_tensor, (0, 3, 2, 1))
        
        np.testing.assert_allclose(q2_tensor, q2_expected, rtol=1e-5)
        np.testing.assert_allclose(q3_tensor, q3_expected, rtol=1e-5)
    
    def test_axis_reordering(self):
        """Test that axes are reordered correctly."""
        input_file = os.path.join(self.temp_dir, 'test_block.nc')
        n_time, n_x1, n_x2, n_x3 = 2, 3, 4, 5
        
        # Create file with position-dependent values
        with Dataset(input_file, 'w', format='NETCDF4') as nc:
            nc.createDimension('time', n_time)
            nc.createDimension('x1', n_x1)
            nc.createDimension('x2', n_x2)
            nc.createDimension('x3', n_x3)
            nc.createDimension('x1f', n_x1 + 1)
            nc.createDimension('x2f', n_x2 + 1)
            nc.createDimension('x3f', n_x3 + 1)
            
            time = nc.createVariable('time', 'f8', ('time',))
            time[:] = np.arange(n_time)
            x1 = nc.createVariable('x1', 'f8', ('x1',))
            x1[:] = np.arange(n_x1)
            x2 = nc.createVariable('x2', 'f8', ('x2',))
            x2[:] = np.arange(n_x2)
            x3 = nc.createVariable('x3', 'f8', ('x3',))
            x3[:] = np.arange(n_x3)
            
            # Create rho with known pattern
            rho = nc.createVariable('rho', 'f4', ('time', 'x1', 'x2', 'x3'))
            rho_data = np.zeros((n_time, n_x1, n_x2, n_x3), dtype='f4')
            for t in range(n_time):
                for i1 in range(n_x1):
                    for i2 in range(n_x2):
                        for i3 in range(n_x3):
                            rho_data[t, i1, i2, i3] = t*1000 + i1*100 + i2*10 + i3
            rho[:] = rho_data
            
            # Create other required variables
            for var_name in ['w', 'v', 'u', 'p', 'q', 'ciwc', 'clwc', 'cswc', 'crwc']:
                var = nc.createVariable(var_name, 'f4', ('time', 'x1', 'x2', 'x3'))
                var[:] = np.zeros((n_time, n_x1, n_x2, n_x3), dtype='f4')
            
            nc.nghost = 3
        
        output_file = convert_netcdf_to_tensor(input_file)
        loaded = torch.jit.load(output_file)
        hydro_w = loaded.hydro_w
        
        # Extract rho (index 0)
        rho_tensor = hydro_w[:, 0, :, :, :].numpy()
        
        # Verify shape: should be (time, x3, x2, x1)
        self.assertEqual(rho_tensor.shape, (n_time, n_x3, n_x2, n_x1))
        
        # Verify values at specific positions
        for t in range(n_time):
            for i1 in range(n_x1):
                for i2 in range(n_x2):
                    for i3 in range(n_x3):
                        expected = t*1000 + i1*100 + i2*10 + i3
                        actual = rho_tensor[t, i3, i2, i1]
                        self.assertAlmostEqual(actual, expected, places=5)
    
    def test_custom_output_path(self):
        """Test conversion with custom output path."""
        input_file = os.path.join(self.temp_dir, 'test_block.nc')
        output_file = os.path.join(self.temp_dir, 'custom_output.restart')
        
        create_test_netcdf_block(input_file)
        
        result = convert_netcdf_to_tensor(input_file, output_file)
        
        self.assertEqual(result, output_file)
        self.assertTrue(os.path.exists(output_file))
    
    def test_missing_required_variable(self):
        """Test that missing required variables raise error."""
        input_file = os.path.join(self.temp_dir, 'incomplete.nc')
        
        # Create file without 'u' variable
        with Dataset(input_file, 'w', format='NETCDF4') as nc:
            nc.createDimension('time', 2)
            nc.createDimension('x1', 5)
            nc.createDimension('x2', 5)
            nc.createDimension('x3', 5)
            
            time = nc.createVariable('time', 'f8', ('time',))
            time[:] = [0, 1]
            x1 = nc.createVariable('x1', 'f8', ('x1',))
            x1[:] = np.arange(5)
            x2 = nc.createVariable('x2', 'f8', ('x2',))
            x2[:] = np.arange(5)
            x3 = nc.createVariable('x3', 'f8', ('x3',))
            x3[:] = np.arange(5)
            
            # Missing 'u' - only create some variables
            shape = (2, 5, 5, 5)
            for var_name in ['rho', 'w', 'v', 'p', 'q']:
                var = nc.createVariable(var_name, 'f4', ('time', 'x1', 'x2', 'x3'))
                var[:] = np.zeros(shape, dtype='f4')
            
            nc.nghost = 3
        
        with self.assertRaises(ValueError) as cm:
            convert_netcdf_to_tensor(input_file)
        
        self.assertIn('Missing required variables', str(cm.exception))
    
    def test_missing_cloud_variables(self):
        """Test that missing cloud variables are handled gracefully."""
        input_file = os.path.join(self.temp_dir, 'no_clouds.nc')
        create_test_netcdf_block(input_file, include_cloud_vars=False)
        
        # Should not raise error, but use zeros for cloud variables
        output_file = convert_netcdf_to_tensor(input_file)
        
        loaded = torch.jit.load(output_file)
        hydro_w = loaded.hydro_w
        
        # q2 and q3 should be all zeros
        q2 = hydro_w[:, 6, :, :, :]
        q3 = hydro_w[:, 7, :, :, :]
        
        self.assertTrue(torch.allclose(q2, torch.zeros_like(q2)))
        self.assertTrue(torch.allclose(q3, torch.zeros_like(q3)))
    
    def test_pressure_on_interfaces(self):
        """Test handling of pressure on vertical interfaces."""
        input_file = os.path.join(self.temp_dir, 'pressure_interfaces.nc')
        create_test_netcdf_block(input_file, n_time=2, n_x1=10, n_x2=10, n_x3=10,
                                pressure_on_interfaces=True)
        
        # Should average pressure from interfaces to cell centers
        output_file = convert_netcdf_to_tensor(input_file)
        
        loaded = torch.jit.load(output_file)
        hydro_w = loaded.hydro_w
        
        # Should have correct shape
        self.assertEqual(hydro_w.shape, (2, 8, 10, 10, 10))
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            convert_netcdf_to_tensor('/nonexistent/file.nc')


@unittest.skipIf(not (NETCDF4_AVAILABLE and TORCH_AVAILABLE),
                "netCDF4 and PyTorch not available")
class TestConvertDirectory(unittest.TestCase):
    """Test cases for convert_directory function."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_convert_multiple_files(self):
        """Test converting multiple NetCDF files in a directory."""
        # Create multiple test files
        for i in range(3):
            input_file = os.path.join(self.temp_dir, f'block_{i}.nc')
            create_test_netcdf_block(input_file, n_time=2, n_x1=5, n_x2=5, n_x3=5)
        
        output_files = convert_directory(self.temp_dir)
        
        # Should have 3 output files
        self.assertEqual(len(output_files), 3)
        
        # All files should exist and have .restart extension
        for output_file in output_files:
            self.assertTrue(os.path.exists(output_file))
            self.assertTrue(output_file.endswith('.restart'))
    
    def test_custom_output_directory(self):
        """Test conversion with custom output directory."""
        # Create test file
        input_file = os.path.join(self.temp_dir, 'block_0.nc')
        create_test_netcdf_block(input_file)
        
        # Create separate output directory
        output_dir = os.path.join(self.temp_dir, 'output')
        
        output_files = convert_directory(self.temp_dir, output_dir)
        
        self.assertEqual(len(output_files), 1)
        self.assertTrue(output_files[0].startswith(output_dir))
        self.assertTrue(os.path.exists(output_files[0]))
    
    def test_custom_pattern(self):
        """Test conversion with custom file pattern."""
        # Create files with different names
        create_test_netcdf_block(os.path.join(self.temp_dir, 'data_0.nc'))
        create_test_netcdf_block(os.path.join(self.temp_dir, 'data_1.nc'))
        create_test_netcdf_block(os.path.join(self.temp_dir, 'other.nc'))
        
        # Convert only 'data_*.nc' files
        output_files = convert_directory(self.temp_dir, pattern='data_*.nc')
        
        # Should have 2 output files
        self.assertEqual(len(output_files), 2)
    
    def test_empty_directory(self):
        """Test handling of directory with no matching files."""
        output_files = convert_directory(self.temp_dir)
        
        self.assertEqual(len(output_files), 0)
    
    def test_directory_not_found(self):
        """Test that FileNotFoundError is raised for non-existent directory."""
        with self.assertRaises(FileNotFoundError):
            convert_directory('/nonexistent/directory')


if __name__ == '__main__':
    unittest.main()

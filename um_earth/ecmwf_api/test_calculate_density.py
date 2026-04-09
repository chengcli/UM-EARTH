"""
Unit tests for calculate_density.py (ECMWF data curation pipeline - Step 2).

This module contains tests for the density calculation script.
"""

import unittest
import sys
import os
import shutil
from unittest.mock import MagicMock, patch
import warnings
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Mock external dependencies before importing
sys.modules['netCDF4'] = MagicMock()

import calculate_density


class TestPhysicalConstants(unittest.TestCase):
    """Test that physical constants are correctly defined."""
    
    def test_rgas_value(self):
        """Test ideal gas constant value."""
        self.assertAlmostEqual(calculate_density.RGAS, 8.31446, places=5)
    
    def test_molecular_weights(self):
        """Test molecular weight values."""
        self.assertAlmostEqual(calculate_density.M_DRY, 28.96e-3, places=6)
        self.assertAlmostEqual(calculate_density.M_VAPOR, 18.0e-3, places=6)


class TestSolveDensityEquations(unittest.TestCase):
    """Test cases for solve_density_equations function."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            import numpy as np
            self.np = np
        except ImportError:
            self.skipTest("NumPy not available")
    
    def test_simple_case_dry_air_only(self):
        """Test density calculation with dry air only (no humidity, no clouds)."""
        # Standard conditions: T=273.15K, P=101325Pa (1 atm)
        temperature = self.np.array([[[[273.15]]]])
        pressure_pa = self.np.array([[[[101325.0]]]])
        q = self.np.array([[[[0.0]]]])  # No water vapor
        cloud_content = self.np.array([[[[0.0]]]])  # No clouds
        
        rho_total, rho_d, rho_v, rho_c = calculate_density.solve_density_equations(
            temperature, pressure_pa, q, cloud_content
        )
        
        # At STP, dry air density should be approximately 1.29 kg/m³
        # Using ideal gas: rho = P*M/(R*T) = 101325 * 0.02896 / (8.31446 * 273.15)
        expected_rho = 101325.0 * 0.02896 / (8.31446 * 273.15)
        
        self.assertAlmostEqual(rho_total[0,0,0,0], expected_rho, places=2)
        self.assertAlmostEqual(rho_d[0,0,0,0], expected_rho, places=2)
        self.assertAlmostEqual(rho_v[0,0,0,0], 0.0, places=6)
        self.assertAlmostEqual(rho_c[0,0,0,0], 0.0, places=6)
    
    def test_with_humidity(self):
        """Test density calculation with water vapor."""
        # T=300K, P=100000Pa, q=0.01 (1% humidity)
        temperature = self.np.array([[[[300.0]]]])
        pressure_pa = self.np.array([[[[100000.0]]]])
        q = self.np.array([[[[0.01]]]])  # 1% humidity
        cloud_content = self.np.array([[[[0.0]]]])  # No clouds
        
        rho_total, rho_d, rho_v, rho_c = calculate_density.solve_density_equations(
            temperature, pressure_pa, q, cloud_content
        )
        
        # Check that results are reasonable
        self.assertGreater(rho_total[0,0,0,0], 0.0)
        self.assertGreater(rho_d[0,0,0,0], 0.0)
        self.assertGreater(rho_v[0,0,0,0], 0.0)
        self.assertAlmostEqual(rho_c[0,0,0,0], 0.0, places=6)
        
        # Check that rho_v is approximately 1% of total
        self.assertAlmostEqual(rho_v[0,0,0,0] / rho_total[0,0,0,0], 0.01, places=3)
        
        # Check sum
        self.assertAlmostEqual(rho_total[0,0,0,0], 
                              rho_d[0,0,0,0] + rho_v[0,0,0,0] + rho_c[0,0,0,0],
                              places=6)
    
    def test_with_clouds(self):
        """Test density calculation with clouds."""
        temperature = self.np.array([[[[280.0]]]])
        pressure_pa = self.np.array([[[[90000.0]]]])
        q = self.np.array([[[[0.005]]]])  # 0.5% humidity
        cloud_content = self.np.array([[[[0.002]]]])  # 0.2% clouds
        
        rho_total, rho_d, rho_v, rho_c = calculate_density.solve_density_equations(
            temperature, pressure_pa, q, cloud_content
        )
        
        # Check that all components are positive
        self.assertGreater(rho_total[0,0,0,0], 0.0)
        self.assertGreater(rho_d[0,0,0,0], 0.0)
        self.assertGreater(rho_v[0,0,0,0], 0.0)
        self.assertGreater(rho_c[0,0,0,0], 0.0)
        
        # Check that rho_v is approximately 0.5% of total
        self.assertAlmostEqual(rho_v[0,0,0,0] / rho_total[0,0,0,0], 0.005, places=3)
        
        # Check that rho_c is approximately 0.2% of total
        self.assertAlmostEqual(rho_c[0,0,0,0] / rho_total[0,0,0,0], 0.002, places=3)
        
        # Check sum
        self.assertAlmostEqual(rho_total[0,0,0,0], 
                              rho_d[0,0,0,0] + rho_v[0,0,0,0] + rho_c[0,0,0,0],
                              places=6)
    
    def test_multiple_pressure_levels(self):
        """Test density calculation with multiple pressure levels."""
        # Shape: (1 time, 3 levels, 2 lat, 2 lon)
        temperature = self.np.ones((1, 3, 2, 2)) * 280.0
        pressure_pa = self.np.array([100000.0, 85000.0, 70000.0]).reshape(1, 3, 1, 1)
        q = self.np.ones((1, 3, 2, 2)) * 0.01
        cloud_content = self.np.ones((1, 3, 2, 2)) * 0.001
        
        rho_total, rho_d, rho_v, rho_c = calculate_density.solve_density_equations(
            temperature, pressure_pa, q, cloud_content
        )
        
        # Check shape
        self.assertEqual(rho_total.shape, (1, 3, 2, 2))
        
        # Density should decrease with altitude (decreasing pressure)
        self.assertGreater(rho_total[0, 0, 0, 0], rho_total[0, 1, 0, 0])
        self.assertGreater(rho_total[0, 1, 0, 0], rho_total[0, 2, 0, 0])
    
    def test_handles_extreme_values(self):
        """Test that function handles extreme values gracefully."""
        # Very high temperature
        temperature = self.np.array([[[[1000.0]]]])
        pressure_pa = self.np.array([[[[50000.0]]]])
        q = self.np.array([[[[0.0]]]])
        cloud_content = self.np.array([[[[0.0]]]])
        
        rho_total, rho_d, rho_v, rho_c = calculate_density.solve_density_equations(
            temperature, pressure_pa, q, cloud_content
        )
        
        # Should still return valid (non-negative, non-nan, non-inf) values
        self.assertGreater(rho_total[0,0,0,0], 0.0)
        self.assertFalse(self.np.isnan(rho_total[0,0,0,0]))
        self.assertFalse(self.np.isinf(rho_total[0,0,0,0]))
    
    def test_negative_values_become_zero(self):
        """Test that negative intermediate values are clipped to zero."""
        # This shouldn't happen physically, but test robustness
        temperature = self.np.array([[[[300.0]]]])
        pressure_pa = self.np.array([[[[100000.0]]]])
        q = self.np.array([[[[1.5]]]])  # Impossible: >100% humidity
        cloud_content = self.np.array([[[[0.0]]]])
        
        # Should not raise an error, just handle gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho_total, rho_d, rho_v, rho_c = calculate_density.solve_density_equations(
                temperature, pressure_pa, q, cloud_content
            )
        
        # All densities should be non-negative
        self.assertGreaterEqual(rho_total[0,0,0,0], 0.0)
        self.assertGreaterEqual(rho_d[0,0,0,0], 0.0)
        self.assertGreaterEqual(rho_v[0,0,0,0], 0.0)
        self.assertGreaterEqual(rho_c[0,0,0,0], 0.0)


class TestLoadNetCDFData(unittest.TestCase):
    """Test cases for load_netcdf_data function."""
    
    def test_missing_numpy(self):
        """Test that ImportError is raised if NumPy is not available."""
        with patch.dict('sys.modules', {'numpy': None, 'netCDF4': None}):
            with self.assertRaises(ImportError):
                calculate_density.load_netcdf_data('dummy1.nc', 'dummy2.nc')
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        # Skip if numpy is not available
        try:
            import numpy
        except ImportError:
            self.skipTest("NumPy not available")
        
        with self.assertRaises(FileNotFoundError):
            calculate_density.load_netcdf_data('/nonexistent/dynamics.nc', '/nonexistent/densities.nc')
    
    def test_missing_temperature_variable(self):
        """Test that ValueError is raised if temperature variable is missing."""
        # Skip if numpy/netCDF4 are not available
        try:
            import numpy
            import netCDF4
        except ImportError:
            self.skipTest("NumPy or netCDF4 not available")
        
        # Create temporary files
        temp_dir = tempfile.mkdtemp()
        try:
            dynamics_file = os.path.join(temp_dir, 'dynamics.nc')
            densities_file = os.path.join(temp_dir, 'densities.nc')
            
            # Create empty files
            open(dynamics_file, 'w').close()
            open(densities_file, 'w').close()
            
            # Mock netCDF4.Dataset to return dataset without 't' or 'temperature'
            mock_ds = MagicMock()
            mock_ds.variables = {}  # No variables
            mock_ds.ncattrs.return_value = []
            mock_ds.__enter__ = MagicMock(return_value=mock_ds)
            mock_ds.__exit__ = MagicMock(return_value=False)
            
            with patch('netCDF4.Dataset', return_value=mock_ds):
                with self.assertRaises(ValueError) as cm:
                    calculate_density.load_netcdf_data(dynamics_file, densities_file)
                
                self.assertIn('Temperature', str(cm.exception))
        
        finally:
            shutil.rmtree(temp_dir)


class TestCalculateTotalDensity(unittest.TestCase):
    """Test cases for calculate_total_density function."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            import numpy as np
            self.np = np
        except ImportError:
            self.skipTest("NumPy not available")
    
    def test_basic_calculation(self):
        """Test basic density calculation."""
        # Create mock data
        data = {
            'temperature': self.np.ones((1, 2, 3, 4)) * 280.0,
            'q': self.np.ones((1, 2, 3, 4)) * 0.01,
            'ciwc': self.np.ones((1, 2, 3, 4)) * 0.0001,
            'cswc': self.np.ones((1, 2, 3, 4)) * 0.0001,
            'clwc': self.np.ones((1, 2, 3, 4)) * 0.0001,
            'crwc': self.np.ones((1, 2, 3, 4)) * 0.0001,
            'level': self.np.array([1000.0, 850.0])
        }
        
        rho_total, rho_d, rho_v, rho_c, pressure_pa = calculate_density.calculate_total_density(data)
        
        # Check shapes
        self.assertEqual(rho_total.shape, (1, 2, 3, 4))
        self.assertEqual(rho_d.shape, (1, 2, 3, 4))
        self.assertEqual(rho_v.shape, (1, 2, 3, 4))
        self.assertEqual(rho_c.shape, (1, 2, 3, 4))
        
        # Check all values are positive
        self.assertTrue(self.np.all(rho_total > 0))
        self.assertTrue(self.np.all(rho_d >= 0))
        self.assertTrue(self.np.all(rho_v >= 0))
        self.assertTrue(self.np.all(rho_c >= 0))


class TestSaveDensityNetCDF(unittest.TestCase):
    """Test cases for save_density_netcdf function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy not available")
        
        output_file = os.path.join(self.temp_dir, 'subdir', 'density.nc')
        
        # Mock data
        data = {
            'time': np.array([0]),
            'level': np.array([1000.0]),
            'latitude': np.array([30.0]),
            'longitude': np.array([-110.0])
        }
        
        rho_total = np.ones((1, 1, 1, 1))
        rho_d = np.ones((1, 1, 1, 1)) * 0.98
        rho_v = np.ones((1, 1, 1, 1)) * 0.01
        rho_c = np.ones((1, 1, 1, 1)) * 0.01
        
        # Mock netCDF4.Dataset
        with patch('calculate_density.nc.Dataset') as mock_dataset:
            mock_ds = MagicMock()
            mock_dataset.return_value.__enter__.return_value = mock_ds
            
            calculate_density.save_density_netcdf(output_file, data, rho_total, rho_d, rho_v, rho_c)
        
        # Check that directory was created
        self.assertTrue(os.path.exists(os.path.dirname(output_file)))


class TestProcessSingleDate(unittest.TestCase):
    """Test cases for process_single_date function."""
    
    def test_missing_import_error(self):
        """Test that ImportError is properly handled."""
        with patch('calculate_density.load_netcdf_data', side_effect=ImportError("NumPy not found")):
            with self.assertRaises(ImportError):
                calculate_density.process_single_date('dynamics.nc', 'densities.nc', 'output.nc')


class TestProcessDirectory(unittest.TestCase):
    """Test cases for process_directory function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_no_dynamics_files(self):
        """Test that function exits if no dynamics files are found."""
        output_dir = os.path.join(self.temp_dir, 'output')
        
        with self.assertRaises(SystemExit):
            calculate_density.process_directory(self.temp_dir, output_dir)
    
    def test_creates_output_directory(self):
        """Test that output directory is created."""
        # Create dummy dynamics file
        dynamics_file = os.path.join(self.temp_dir, 'era5_hourly_dynamics_20240101.nc')
        open(dynamics_file, 'w').close()
        
        output_dir = os.path.join(self.temp_dir, 'output')
        
        # Mock process_single_date to avoid actual processing
        with patch('calculate_density.process_single_date'):
            try:
                calculate_density.process_directory(self.temp_dir, output_dir)
            except SystemExit:
                pass  # Expected if densities file is missing
        
        # Output directory should be created
        self.assertTrue(os.path.exists(output_dir))


class TestCommandLineInterface(unittest.TestCase):
    """Test cases for command-line interface."""
    
    def test_requires_mode(self):
        """Test that one mode is required."""
        with patch('sys.argv', ['calculate_density.py']):
            with self.assertRaises(SystemExit):
                calculate_density.main()
    
    def test_single_file_mode_requires_all_args(self):
        """Test that single file mode requires all three arguments."""
        with patch('sys.argv', ['calculate_density.py', '--dynamics-file', 'dyn.nc']):
            with self.assertRaises(SystemExit):
                calculate_density.main()
    
    def test_directory_mode_requires_output_dir(self):
        """Test that directory mode requires output-dir."""
        with patch('sys.argv', ['calculate_density.py', '--input-dir', './data']):
            with self.assertRaises(SystemExit):
                calculate_density.main()


if __name__ == '__main__':
    unittest.main()

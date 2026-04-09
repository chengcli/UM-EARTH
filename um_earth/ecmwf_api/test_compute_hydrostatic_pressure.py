"""
Unit tests for compute_hydrostatic_pressure module

This module contains tests for Step 4 of the ECMWF data curation pipeline,
which computes hydrostatically balanced pressure at cell centers.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import yaml

from compute_hydrostatic_pressure import (
    parse_yaml_config,
    extract_gravity,
    load_regridded_data,
    compute_hydrostatic_pressure,
    augment_netcdf_with_pressure,
)


class TestParseYamlConfig(unittest.TestCase):
    """Test cases for YAML configuration parsing."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_parse_valid_config(self):
        """Test parsing a valid YAML configuration."""
        config_data = {
            'forcing': {
                'const-gravity': {
                    'grav1': -9.8
                }
            }
        }
        
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = parse_yaml_config(config_file)
        
        self.assertIn('forcing', config)
        self.assertIn('const-gravity', config['forcing'])
        self.assertIn('grav1', config['forcing']['const-gravity'])
    
    def test_parse_nonexistent_file(self):
        """Test that parsing nonexistent file raises error."""
        with self.assertRaises(FileNotFoundError):
            parse_yaml_config('nonexistent.yaml')
    
    def test_parse_invalid_yaml(self):
        """Test that parsing invalid YAML raises error."""
        config_file = os.path.join(self.temp_dir, 'invalid.yaml')
        with open(config_file, 'w') as f:
            f.write("{\ninvalid yaml: [")
        
        with self.assertRaises(yaml.YAMLError):
            parse_yaml_config(config_file)


class TestExtractGravity(unittest.TestCase):
    """Test cases for extracting gravity from configuration."""
    
    def test_extract_valid_gravity_negative(self):
        """Test extracting valid negative gravity value."""
        config = {
            'forcing': {
                'const-gravity': {
                    'grav1': -9.8
                }
            }
        }
        
        gravity = extract_gravity(config)
        self.assertAlmostEqual(gravity, 9.8)
    
    def test_extract_valid_gravity_positive(self):
        """Test extracting valid positive gravity value."""
        config = {
            'forcing': {
                'const-gravity': {
                    'grav1': 9.8
                }
            }
        }
        
        gravity = extract_gravity(config)
        self.assertAlmostEqual(gravity, 9.8)
    
    def test_extract_missing_forcing(self):
        """Test that missing forcing field returns default."""
        config = {}
        
        # Should return default value without raising error
        gravity = extract_gravity(config)
        self.assertAlmostEqual(gravity, 9.80665)
    
    def test_extract_missing_const_gravity(self):
        """Test that missing const-gravity field returns default."""
        config = {
            'forcing': {}
        }
        
        # Should return default value without raising error
        gravity = extract_gravity(config)
        self.assertAlmostEqual(gravity, 9.80665)
    
    def test_extract_missing_grav1(self):
        """Test that missing grav1 field returns default."""
        config = {
            'forcing': {
                'const-gravity': {}
            }
        }
        
        # Should return default value without raising error
        gravity = extract_gravity(config)
        self.assertAlmostEqual(gravity, 9.80665)
    
    def test_extract_zero_gravity(self):
        """Test that zero gravity returns default."""
        config = {
            'forcing': {
                'const-gravity': {
                    'grav1': 0.0
                }
            }
        }
        
        # Should return default value for invalid (zero) gravity
        gravity = extract_gravity(config)
        self.assertAlmostEqual(gravity, 9.80665)


class TestLoadRegriddedData(unittest.TestCase):
    """Test cases for loading regridded NetCDF data."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        with self.assertRaises(FileNotFoundError):
            load_regridded_data('nonexistent.nc')
    
    def test_load_file_missing_pressure_level(self):
        """Test that file missing pressure_level raises error."""
        try:
            import xarray as xr
            from netCDF4 import Dataset
        except ImportError:
            self.skipTest("xarray or netCDF4 not available")
        
        # Create minimal NetCDF file without pressure_level
        nc_file = os.path.join(self.temp_dir, 'test.nc')
        
        with Dataset(nc_file, 'w') as ncfile:
            ncfile.createDimension('time', 1)
            ncfile.createDimension('x1', 10)
            ncfile.createDimension('x1f', 11)
            ncfile.createDimension('x2', 5)
            ncfile.createDimension('x3', 5)
            
            # Create rho but not pressure_level
            rho_var = ncfile.createVariable('rho', 'f4', ('time', 'x1', 'x2', 'x3'))
            rho_var[:] = np.ones((1, 10, 5, 5))
        
        with self.assertRaises(ValueError) as context:
            load_regridded_data(nc_file)
        
        self.assertIn('pressure_level', str(context.exception))
    
    def test_load_file_missing_rho(self):
        """Test that file missing rho raises error."""
        try:
            import xarray as xr
            from netCDF4 import Dataset
        except ImportError:
            self.skipTest("xarray or netCDF4 not available")
        
        # Create minimal NetCDF file without rho
        nc_file = os.path.join(self.temp_dir, 'test.nc')
        
        with Dataset(nc_file, 'w') as ncfile:
            ncfile.createDimension('time', 1)
            ncfile.createDimension('x1', 10)
            ncfile.createDimension('x1f', 11)
            ncfile.createDimension('x2', 5)
            ncfile.createDimension('x3', 5)
            
            # Create pressure_level but not rho
            plev_var = ncfile.createVariable('pressure_level', 'f4', 
                                            ('time', 'x1f', 'x2', 'x3'))
            plev_var[:] = np.ones((1, 11, 5, 5))
        
        with self.assertRaises(ValueError) as context:
            load_regridded_data(nc_file)
        
        self.assertIn('rho', str(context.exception))


class TestComputeHydrostaticPressure(unittest.TestCase):
    """Test cases for computing hydrostatic pressure."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self):
        """Create a test dataset with known values."""
        try:
            import xarray as xr
            from netCDF4 import Dataset
        except ImportError:
            self.skipTest("xarray or netCDF4 not available")
        
        # Create test NetCDF file
        nc_file = os.path.join(self.temp_dir, 'test_regridded.nc')
        
        # Simple test case: uniform atmosphere
        T, Z, Y, X = 1, 10, 3, 3
        Zf = Z + 1
        
        # Create vertical coordinates (0 to 10 km)
        x1f = np.linspace(0, 10000, Zf)  # 11 interfaces
        x1 = 0.5 * (x1f[:-1] + x1f[1:])  # 10 centers
        
        # Constant density (1.2 kg/m³, typical for sea level)
        rho = np.ones((T, Z, Y, X)) * 1.2
        
        # Initial pressure at interfaces (linearly decreasing with height for test)
        # Start with 100000 Pa at bottom
        pressure_level = np.zeros((T, Zf, Y, X))
        for i in range(Y):
            for j in range(X):
                pressure_level[0, :, i, j] = np.linspace(100000, 10000, Zf)
        
        with Dataset(nc_file, 'w') as ncfile:
            # Create dimensions
            ncfile.createDimension('time', T)
            ncfile.createDimension('x1', Z)
            ncfile.createDimension('x1f', Zf)
            ncfile.createDimension('x2', Y)
            ncfile.createDimension('x3', X)
            
            # Create coordinate variables
            time_var = ncfile.createVariable('time', 'f8', ('time',))
            x1_var = ncfile.createVariable('x1', 'f8', ('x1',))
            x1f_var = ncfile.createVariable('x1f', 'f8', ('x1f',))
            x2_var = ncfile.createVariable('x2', 'f8', ('x2',))
            x3_var = ncfile.createVariable('x3', 'f8', ('x3',))
            
            time_var[:] = [0]
            x1_var[:] = x1
            x1f_var[:] = x1f
            x2_var[:] = np.arange(Y)
            x3_var[:] = np.arange(X)
            
            # Create data variables
            rho_var = ncfile.createVariable('rho', 'f4', ('time', 'x1', 'x2', 'x3'))
            plev_var = ncfile.createVariable('pressure_level', 'f4', 
                                            ('time', 'x1f', 'x2', 'x3'))
            
            rho_var[:] = rho
            plev_var[:] = pressure_level
        
        return xr.open_dataset(nc_file)
    
    def test_compute_pressure_shape(self):
        """Test that computed pressure has correct shape."""
        ds = self.create_test_dataset()
        gravity = 9.8
        
        pressure = compute_hydrostatic_pressure(ds, gravity)
        
        T, Z, Y, X = ds['rho'].shape
        self.assertEqual(pressure.shape, (T, Z, Y, X))
        
        ds.close()
    
    def test_compute_pressure_positive(self):
        """Test that computed pressure is positive."""
        ds = self.create_test_dataset()
        gravity = 9.8
        
        pressure = compute_hydrostatic_pressure(ds, gravity)
        
        # All pressure values should be positive
        self.assertTrue(np.all(pressure > 0))
        
        ds.close()
    
    def test_compute_pressure_decreases_with_height(self):
        """Test that pressure decreases with height."""
        ds = self.create_test_dataset()
        gravity = 9.8
        
        pressure = compute_hydrostatic_pressure(ds, gravity)
        
        # Pressure should decrease with increasing altitude (increasing index)
        # Check for first time step and one horizontal location
        p_column = pressure[0, :, 0, 0]
        
        # Each level should have lower or equal pressure than the one below
        for i in range(len(p_column) - 1):
            self.assertGreaterEqual(p_column[i], p_column[i + 1],
                                   f"Pressure should decrease with height: "
                                   f"p[{i}]={p_column[i]:.1f}, p[{i+1}]={p_column[i+1]:.1f}")
        
        ds.close()
    
    def test_compute_pressure_geometric_mean(self):
        """Test that cell center pressure is geometric mean of interface pressures."""
        ds = self.create_test_dataset()
        gravity = 9.8
        
        # Get the recomputed interface pressures and cell center pressures
        pressure_level = ds['pressure_level'].values
        rho = ds['rho'].values
        x1f = ds['x1f'].values
        
        T, Zf, Y, X = pressure_level.shape
        Z = Zf - 1
        
        # Recompute interface pressures
        dz = np.diff(x1f)
        pf_new = np.zeros((T, Zf, Y, X), dtype=np.float64)
        pf_new[:, -1, :, :] = pressure_level[:, -1, :, :]
        
        for i in range(Z - 1, -1, -1):
            pf_new[:, i, :, :] = pf_new[:, i + 1, :, :] + rho[:, i, :, :] * gravity * dz[i]
        
        # Compute expected geometric mean
        expected_p = np.sqrt(pf_new[:, :-1, :, :] * pf_new[:, 1:, :, :])
        
        # Get actual computed pressure
        pressure = compute_hydrostatic_pressure(ds, gravity)
        
        # They should match
        np.testing.assert_allclose(pressure, expected_p, rtol=1e-5)
        
        ds.close()


class TestAugmentNetcdf(unittest.TestCase):
    """Test cases for augmenting NetCDF file with pressure."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_augment_netcdf(self):
        """Test augmenting NetCDF file with pressure variable."""
        try:
            import xarray as xr
            from netCDF4 import Dataset
        except ImportError:
            self.skipTest("xarray or netCDF4 not available")
        
        # Create test NetCDF file
        nc_file = os.path.join(self.temp_dir, 'test.nc')
        
        T, Z, Y, X = 1, 10, 3, 3
        
        with Dataset(nc_file, 'w') as ncfile:
            ncfile.createDimension('time', T)
            ncfile.createDimension('x1', Z)
            ncfile.createDimension('x2', Y)
            ncfile.createDimension('x3', X)
            
            time_var = ncfile.createVariable('time', 'f8', ('time',))
            time_var[:] = [0]
        
        # Create test pressure array
        pressure = np.random.rand(T, Z, Y, X) * 100000 + 10000
        
        # Augment file
        augment_netcdf_with_pressure(nc_file, pressure)
        
        # Verify
        ds = xr.open_dataset(nc_file)
        
        self.assertIn('p', ds.data_vars)
        self.assertEqual(ds['p'].shape, (T, Z, Y, X))
        
        # Check values match (allowing for float32 precision)
        # Use relative tolerance since we're comparing float32 to float64
        np.testing.assert_allclose(ds['p'].values, pressure, rtol=1e-6, atol=0.01)
        
        # Check attributes
        self.assertEqual(ds['p'].attrs['units'], 'Pa')
        self.assertIn('Hydrostatically balanced', ds['p'].attrs['long_name'])
        
        ds.close()


def run_tests():
    """Run all tests."""
    unittest.main()


if __name__ == '__main__':
    run_tests()

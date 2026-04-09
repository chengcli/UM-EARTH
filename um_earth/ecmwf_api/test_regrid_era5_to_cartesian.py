"""
Unit tests for regrid_era5_to_cartesian module

This module contains tests for Step 3 of the ECMWF data curation pipeline,
which regrids ERA5 data from pressure-lat-lon grids to Cartesian coordinates.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import yaml

from regrid_era5_to_cartesian import (
    parse_yaml_config,
    extract_geometry_info,
    extract_gravity,
    compute_cell_coordinates,
    find_era5_files,
    save_regridded_data_with_interfaces,
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
            'geometry': {
                'type': 'cartesian',
                'bounds': {
                    'x1min': 0.0, 'x1max': 10000.0,
                    'x2min': -5000.0, 'x2max': 5000.0,
                    'x3min': -10000.0, 'x3max': 10000.0
                },
                'cells': {
                    'nx1': 100, 'nx2': 50, 'nx3': 100, 'nghost': 3
                },
                'center_latitude': 30.0,
                'center_longitude': -110.0
            }
        }
        
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = parse_yaml_config(config_file)
        
        self.assertIn('geometry', config)
        self.assertEqual(config['geometry']['type'], 'cartesian')
    
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


class TestExtractGeometryInfo(unittest.TestCase):
    """Test cases for extracting geometry information."""
    
    def test_extract_valid_geometry(self):
        """Test extracting valid geometry information."""
        config = {
            'geometry': {
                'type': 'cartesian',
                'bounds': {
                    'x1min': 0.0, 'x1max': 10000.0,
                    'x2min': -5000.0, 'x2max': 5000.0,
                    'x3min': -10000.0, 'x3max': 10000.0
                },
                'cells': {
                    'nx1': 100, 'nx2': 50, 'nx3': 100, 'nghost': 3
                },
                'center_latitude': 30.0,
                'center_longitude': -110.0
            }
        }
        
        geometry = extract_geometry_info(config)
        
        self.assertIn('bounds', geometry)
        self.assertIn('cells', geometry)
        self.assertEqual(geometry['center_latitude'], 30.0)
        self.assertEqual(geometry['center_longitude'], -110.0)
        self.assertEqual(geometry['bounds']['x1min'], 0.0)
        self.assertEqual(geometry['bounds']['x1max'], 10000.0)
        self.assertEqual(geometry['cells']['nx1'], 100)
        self.assertEqual(geometry['cells']['nghost'], 3)
    
    def test_missing_geometry_field(self):
        """Test that missing geometry field raises error."""
        config = {'other_field': 'value'}
        
        with self.assertRaises(ValueError) as context:
            extract_geometry_info(config)
        
        self.assertIn('geometry', str(context.exception).lower())
    
    def test_invalid_geometry_type(self):
        """Test that invalid geometry type raises error."""
        config = {
            'geometry': {
                'type': 'spherical',
                'bounds': {'x1min': 0.0, 'x1max': 10.0, 
                          'x2min': 0.0, 'x2max': 10.0,
                          'x3min': 0.0, 'x3max': 10.0},
                'cells': {'nx1': 10, 'nx2': 10, 'nx3': 10, 'nghost': 1},
                'center_latitude': 0.0,
                'center_longitude': 0.0
            }
        }
        
        with self.assertRaises(ValueError) as context:
            extract_geometry_info(config)
        
        self.assertIn('cartesian', str(context.exception).lower())
    
    def test_missing_bounds(self):
        """Test that missing bounds raises error."""
        config = {
            'geometry': {
                'type': 'cartesian',
                'cells': {'nx1': 10, 'nx2': 10, 'nx3': 10, 'nghost': 1},
                'center_latitude': 0.0,
                'center_longitude': 0.0
            }
        }
        
        with self.assertRaises(ValueError) as context:
            extract_geometry_info(config)
        
        self.assertIn('bounds', str(context.exception).lower())
    
    def test_missing_cells(self):
        """Test that missing cells raises error."""
        config = {
            'geometry': {
                'type': 'cartesian',
                'bounds': {'x1min': 0.0, 'x1max': 10.0,
                          'x2min': 0.0, 'x2max': 10.0,
                          'x3min': 0.0, 'x3max': 10.0},
                'center_latitude': 0.0,
                'center_longitude': 0.0
            }
        }
        
        with self.assertRaises(ValueError) as context:
            extract_geometry_info(config)
        
        self.assertIn('cells', str(context.exception).lower())
    
    def test_missing_center_coordinates(self):
        """Test that missing center coordinates raises error."""
        config = {
            'geometry': {
                'type': 'cartesian',
                'bounds': {'x1min': 0.0, 'x1max': 10.0,
                          'x2min': 0.0, 'x2max': 10.0,
                          'x3min': 0.0, 'x3max': 10.0},
                'cells': {'nx1': 10, 'nx2': 10, 'nx3': 10, 'nghost': 1}
            }
        }
        
        with self.assertRaises(ValueError) as context:
            extract_geometry_info(config)
        
        self.assertIn('center', str(context.exception).lower())


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
    
    def test_extract_missing_forcing_returns_default(self):
        """Test that missing forcing field returns default."""
        config = {}
        
        # Should return default value without raising error
        gravity = extract_gravity(config)
        self.assertAlmostEqual(gravity, 9.80665)  # Default Earth gravity
    
    def test_extract_missing_const_gravity_returns_default(self):
        """Test that missing const-gravity field returns default."""
        config = {
            'forcing': {}
        }
        
        # Should return default value without raising error
        gravity = extract_gravity(config)
        self.assertAlmostEqual(gravity, 9.80665)
    
    def test_extract_missing_grav1_returns_default(self):
        """Test that missing grav1 field returns default."""
        config = {
            'forcing': {
                'const-gravity': {}
            }
        }
        
        # Should return default value without raising error
        gravity = extract_gravity(config)
        self.assertAlmostEqual(gravity, 9.80665)


class TestComputeCellCoordinates(unittest.TestCase):
    """Test cases for computing cell coordinates."""
    
    def test_basic_cell_coordinates(self):
        """Test basic cell coordinate computation."""
        geometry = {
            'bounds': {
                'x1min': 0.0, 'x1max': 1000.0,
                'x2min': -500.0, 'x2max': 500.0,
                'x3min': -1000.0, 'x3max': 1000.0
            },
            'cells': {
                'nx1': 10, 'nx2': 10, 'nx3': 20, 'nghost': 2
            }
        }
        
        x1, x1f, x2, x2f, x3, x3f = compute_cell_coordinates(geometry)
        
        # Check lengths
        # Total cells = interior + 2*nghost
        nx1_total = 10 + 2*2
        nx2_total = 10 + 2*2
        nx3_total = 20 + 2*2
        
        self.assertEqual(len(x1), nx1_total)
        self.assertEqual(len(x1f), nx1_total + 1)
        self.assertEqual(len(x2), nx2_total)
        self.assertEqual(len(x2f), nx2_total + 1)
        self.assertEqual(len(x3), nx3_total)
        self.assertEqual(len(x3f), nx3_total + 1)
        
        # Check bounds
        np.testing.assert_almost_equal(x1f[0], 0.0)
        np.testing.assert_almost_equal(x1f[-1], 1000.0)
        np.testing.assert_almost_equal(x2f[0], -500.0)
        np.testing.assert_almost_equal(x2f[-1], 500.0)
        np.testing.assert_almost_equal(x3f[0], -1000.0)
        np.testing.assert_almost_equal(x3f[-1], 1000.0)
        
        # Check that centers are midpoints of interfaces
        for i in range(len(x1)):
            expected_center = 0.5 * (x1f[i] + x1f[i+1])
            np.testing.assert_almost_equal(x1[i], expected_center)
    
    def test_uniform_spacing(self):
        """Test that cell spacing is uniform."""
        geometry = {
            'bounds': {
                'x1min': 0.0, 'x1max': 100.0,
                'x2min': 0.0, 'x2max': 200.0,
                'x3min': 0.0, 'x3max': 300.0
            },
            'cells': {
                'nx1': 10, 'nx2': 20, 'nx3': 30, 'nghost': 1
            }
        }
        
        x1, x1f, x2, x2f, x3, x3f = compute_cell_coordinates(geometry)
        
        # Check uniform spacing
        dx1 = np.diff(x1f)
        dx2 = np.diff(x2f)
        dx3 = np.diff(x3f)
        
        # All spacings should be equal (within numerical precision)
        np.testing.assert_array_almost_equal(dx1, dx1[0] * np.ones_like(dx1))
        np.testing.assert_array_almost_equal(dx2, dx2[0] * np.ones_like(dx2))
        np.testing.assert_array_almost_equal(dx3, dx3[0] * np.ones_like(dx3))
    
    def test_ghost_zones_included(self):
        """Test that ghost zones are properly included."""
        geometry = {
            'bounds': {
                'x1min': 0.0, 'x1max': 100.0,
                'x2min': 0.0, 'x2max': 100.0,
                'x3min': 0.0, 'x3max': 100.0
            },
            'cells': {
                'nx1': 8, 'nx2': 8, 'nx3': 8, 'nghost': 2
            }
        }
        
        x1, x1f, x2, x2f, x3, x3f = compute_cell_coordinates(geometry)
        
        # Total cells should include ghost zones
        # 8 interior + 2*2 ghost = 12 total
        self.assertEqual(len(x1), 12)
        self.assertEqual(len(x2), 12)
        self.assertEqual(len(x3), 12)


class TestFindEra5Files(unittest.TestCase):
    """Test cases for finding ERA5 files."""
    
    def setUp(self):
        """Set up temporary directory with mock files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_mock_files(self, dates):
        """Create mock ERA5 files for testing."""
        for date in dates:
            # Create empty files
            dynamics_file = os.path.join(self.temp_dir, f'era5_hourly_dynamics_{date}.nc')
            densities_file = os.path.join(self.temp_dir, f'era5_hourly_densities_{date}.nc')
            density_file = os.path.join(self.temp_dir, f'era5_density_{date}.nc')
            
            for f in [dynamics_file, densities_file, density_file]:
                open(f, 'w').close()
    
    def test_find_complete_files(self):
        """Test finding complete set of ERA5 files."""
        dates = ['20240101', '20240102']
        self.create_mock_files(dates)
        
        file_dict = find_era5_files(self.temp_dir)
        
        self.assertEqual(len(file_dict), 2)
        self.assertIn('20240101', file_dict)
        self.assertIn('20240102', file_dict)
        
        for date in dates:
            self.assertIn('dynamics', file_dict[date])
            self.assertIn('densities', file_dict[date])
            self.assertIn('density', file_dict[date])
    
    def test_find_specific_date(self):
        """Test finding files for a specific date."""
        dates = ['20240101', '20240102']
        self.create_mock_files(dates)
        
        file_dict = find_era5_files(self.temp_dir, '20240101')
        
        self.assertEqual(len(file_dict), 1)
        self.assertIn('20240101', file_dict)
        self.assertNotIn('20240102', file_dict)
    
    def test_missing_densities_file(self):
        """Test that missing densities file raises error."""
        # Create only dynamics and density files
        date = '20240101'
        dynamics_file = os.path.join(self.temp_dir, f'era5_hourly_dynamics_{date}.nc')
        density_file = os.path.join(self.temp_dir, f'era5_density_{date}.nc')
        
        open(dynamics_file, 'w').close()
        open(density_file, 'w').close()
        
        with self.assertRaises(FileNotFoundError) as context:
            find_era5_files(self.temp_dir)
        
        self.assertIn('densities', str(context.exception).lower())
    
    def test_missing_density_file(self):
        """Test that missing density file raises error."""
        # Create only dynamics and densities files
        date = '20240101'
        dynamics_file = os.path.join(self.temp_dir, f'era5_hourly_dynamics_{date}.nc')
        densities_file = os.path.join(self.temp_dir, f'era5_hourly_densities_{date}.nc')
        
        open(dynamics_file, 'w').close()
        open(densities_file, 'w').close()
        
        with self.assertRaises(FileNotFoundError) as context:
            find_era5_files(self.temp_dir)
        
        self.assertIn('density', str(context.exception).lower())
    
    def test_no_files_found(self):
        """Test that no files raises error."""
        with self.assertRaises(FileNotFoundError) as context:
            find_era5_files(self.temp_dir)
        
        self.assertIn('dynamics', str(context.exception).lower())
    
    def test_nonexistent_directory(self):
        """Test that nonexistent directory raises error."""
        with self.assertRaises(FileNotFoundError):
            find_era5_files('/nonexistent/directory')


class TestSaveRegridedDataWithInterfaces(unittest.TestCase):
    """Test cases for saving regridded data with interfaces."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_basic_data(self):
        """Test basic save of regridded data with interfaces."""
        # Create test data
        T, Z, Y, X = 2, 5, 4, 6
        
        variables = {
            'temperature': 280.0 + np.random.randn(T, Z, Y, X) * 10.0,
            'density': 1.0 + np.random.randn(T, Z, Y, X) * 0.1,
        }
        
        x1 = np.linspace(0., 5000., Z)
        x1f = np.linspace(0., 5000., Z + 1)
        x2 = np.linspace(-1000., 1000., Y)
        x2f = np.linspace(-1000., 1000., Y + 1)
        x3 = np.linspace(-2000., 2000., X)
        x3f = np.linspace(-2000., 2000., X + 1)
        
        coordinates = {
            'time': np.arange(T, dtype=float),
            'x1': x1,
            'x1f': x1f,
            'x2': x2,
            'x2f': x2f,
            'x3': x3,
            'x3f': x3f,
        }
        
        metadata = {
            'source': 'Test data',
            'center_latitude': 30.0,
            'center_longitude': -110.0,
            'nx1': Z - 4,  # Assuming 2 ghost cells per side
            'nx2': Y - 4,
            'nx3': X - 4,
            'nghost': 2,
            'temperature_units': 'K',
            'temperature_long_name': 'Air Temperature',
            'density_units': 'kg m-3',
            'density_long_name': 'Air Density',
            'time_units': 'hours since 1900-01-01 00:00:00',
        }
        
        processing_history = "Test regridding"
        
        filename = os.path.join(self.temp_dir, 'test_output.nc')
        
        # Save data
        save_regridded_data_with_interfaces(
            filename,
            variables,
            coordinates,
            metadata,
            processing_history
        )
        
        # Verify file was created
        self.assertTrue(os.path.exists(filename))
        
        # Try to load and verify
        try:
            from netCDF4 import Dataset
            
            with Dataset(filename, 'r') as ncfile:
                # Check dimensions
                self.assertEqual(len(ncfile.dimensions['time']), T)
                self.assertEqual(len(ncfile.dimensions['x1']), Z)
                self.assertEqual(len(ncfile.dimensions['x2']), Y)
                self.assertEqual(len(ncfile.dimensions['x3']), X)
                self.assertEqual(len(ncfile.dimensions['x1f']), Z + 1)
                self.assertEqual(len(ncfile.dimensions['x2f']), Y + 1)
                self.assertEqual(len(ncfile.dimensions['x3f']), X + 1)
                
                # Check variables exist
                self.assertIn('temperature', ncfile.variables)
                self.assertIn('density', ncfile.variables)
                
                # Check coordinate variables
                self.assertIn('x1', ncfile.variables)
                self.assertIn('x1f', ncfile.variables)
                self.assertIn('x2', ncfile.variables)
                self.assertIn('x2f', ncfile.variables)
                self.assertIn('x3', ncfile.variables)
                self.assertIn('x3f', ncfile.variables)
                
                # Check that interfaces have correct length
                self.assertEqual(len(ncfile.variables['x1f'][:]), Z + 1)
                self.assertEqual(len(ncfile.variables['x2f'][:]), Y + 1)
                self.assertEqual(len(ncfile.variables['x3f'][:]), X + 1)
                
                # Check global attributes
                self.assertIn('history', ncfile.ncattrs())
                self.assertEqual(ncfile.source, 'Test data')
                self.assertEqual(ncfile.center_latitude, 30.0)
                self.assertEqual(ncfile.center_longitude, -110.0)
                self.assertEqual(ncfile.nghost, 2)
                
        except ImportError:
            # Skip verification if netCDF4 not available
            pass


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestParseYamlConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestExtractGeometryInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestComputeCellCoordinates))
    suite.addTests(loader.loadTestsFromTestCase(TestFindEra5Files))
    suite.addTests(loader.loadTestsFromTestCase(TestSaveRegridedDataWithInterfaces))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)

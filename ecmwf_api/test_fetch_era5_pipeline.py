"""
Unit tests for ECMWF data fetching pipeline (fetch_era5_pipeline.py).

This module contains tests for the pipeline script that reads YAML configuration,
calculates lat-lon limits, and fetches ERA5 data.
"""

import unittest
import sys
import os
import tempfile
import yaml
import subprocess
from unittest.mock import MagicMock, patch

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Mock external dependencies before importing
sys.modules['cdsapi'] = MagicMock()
sys.modules['xarray'] = MagicMock()
sys.modules['netCDF4'] = MagicMock()

import fetch_era5_pipeline


class TestParseYAMLConfig(unittest.TestCase):
    """Test cases for parse_yaml_config function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parse_valid_yaml(self):
        """Test parsing a valid YAML file."""
        yaml_content = """
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 10.e3, x2min: 0., x2max: 20.e3, x3min: 0., x3max: 10.e3}
  cells: {nx1: 100, nx2: 300, nx3: 150, nghost: 3}
  center_latitude: 30.
  center_longitude: -110.
integration:
  start-date: 2024-01-01
  end-date: 2024-01-02
"""
        yaml_file = os.path.join(self.temp_dir, 'test.yaml')
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        config = fetch_era5_pipeline.parse_yaml_config(yaml_file)
        
        self.assertIn('geometry', config)
        self.assertIn('integration', config)
        self.assertEqual(config['geometry']['type'], 'cartesian')
    
    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            fetch_era5_pipeline.parse_yaml_config('/nonexistent/file.yaml')
    
    def test_parse_invalid_yaml(self):
        """Test parsing invalid YAML content."""
        yaml_file = os.path.join(self.temp_dir, 'invalid.yaml')
        with open(yaml_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with self.assertRaises(yaml.YAMLError):
            fetch_era5_pipeline.parse_yaml_config(yaml_file)


class TestExtractGeometryInfo(unittest.TestCase):
    """Test cases for extract_geometry_info function."""
    
    def test_extract_valid_geometry(self):
        """Test extracting valid geometry information."""
        config = {
            'geometry': {
                'type': 'cartesian',
                'bounds': {
                    'x1min': 0., 'x1max': 10.e3,
                    'x2min': 0., 'x2max': 20.e3,
                    'x3min': 0., 'x3max': 10.e3
                },
                'cells': {
                    'nx1': 100, 'nx2': 300, 'nx3': 150, 'nghost': 3
                },
                'center_latitude': 30.,
                'center_longitude': -110.
            }
        }
        
        geometry = fetch_era5_pipeline.extract_geometry_info(config)
        
        self.assertEqual(geometry['center_latitude'], 30.0)
        self.assertEqual(geometry['center_longitude'], -110.0)
        self.assertEqual(geometry['cells']['nx1'], 100)
        self.assertEqual(geometry['cells']['nghost'], 3)
    
    def test_missing_geometry_field(self):
        """Test with missing geometry field."""
        config = {}
        
        with self.assertRaises(ValueError) as context:
            fetch_era5_pipeline.extract_geometry_info(config)
        
        self.assertIn('geometry', str(context.exception))
    
    def test_invalid_geometry_type(self):
        """Test with invalid geometry type."""
        config = {
            'geometry': {
                'type': 'spherical',
                'bounds': {},
                'cells': {},
                'center_latitude': 0.,
                'center_longitude': 0.
            }
        }
        
        with self.assertRaises(ValueError) as context:
            fetch_era5_pipeline.extract_geometry_info(config)
        
        self.assertIn('cartesian', str(context.exception))
    
    def test_missing_bounds(self):
        """Test with missing bounds field."""
        config = {
            'geometry': {
                'type': 'cartesian',
                'cells': {},
                'center_latitude': 0.,
                'center_longitude': 0.
            }
        }
        
        with self.assertRaises(ValueError) as context:
            fetch_era5_pipeline.extract_geometry_info(config)
        
        self.assertIn('bounds', str(context.exception))
    
    def test_missing_center_coordinates(self):
        """Test with missing center coordinates."""
        config = {
            'geometry': {
                'type': 'cartesian',
                'bounds': {
                    'x1min': 0., 'x1max': 10.e3,
                    'x2min': 0., 'x2max': 20.e3,
                    'x3min': 0., 'x3max': 10.e3
                },
                'cells': {
                    'nx1': 100, 'nx2': 300, 'nx3': 150, 'nghost': 3
                }
            }
        }
        
        with self.assertRaises(ValueError) as context:
            fetch_era5_pipeline.extract_geometry_info(config)
        
        self.assertIn('center_latitude', str(context.exception))


class TestExtractIntegrationInfo(unittest.TestCase):
    """Test cases for extract_integration_info function."""
    
    def test_extract_valid_integration(self):
        """Test extracting valid integration information."""
        config = {
            'integration': {
                'start-date': '2024-01-01',
                'end-date': '2024-01-02'
            }
        }
        
        integration = fetch_era5_pipeline.extract_integration_info(config)
        
        self.assertEqual(integration['start_date'], '2024-01-01')
        self.assertEqual(integration['end_date'], '2024-01-02')
    
    def test_missing_integration_field(self):
        """Test with missing integration field."""
        config = {}
        
        with self.assertRaises(ValueError) as context:
            fetch_era5_pipeline.extract_integration_info(config)
        
        self.assertIn('integration', str(context.exception))
    
    def test_missing_start_date(self):
        """Test with missing start date."""
        config = {
            'integration': {
                'end-date': '2024-01-02'
            }
        }
        
        with self.assertRaises(ValueError) as context:
            fetch_era5_pipeline.extract_integration_info(config)
        
        self.assertIn('start-date', str(context.exception))
    
    def test_invalid_date_format(self):
        """Test with invalid date format."""
        config = {
            'integration': {
                'start-date': 'invalid-date'
            }
        }
        
        with self.assertRaises(ValueError) as context:
            fetch_era5_pipeline.extract_integration_info(config)
        
        self.assertIn('Invalid start-date format', str(context.exception))
    
    def test_default_end_date(self):
        """Test that end-date defaults to start-date if not provided."""
        config = {
            'integration': {
                'start-date': '2024-01-01'
            }
        }
        
        integration = fetch_era5_pipeline.extract_integration_info(config)
        
        self.assertEqual(integration['start_date'], '2024-01-01')
        self.assertEqual(integration['end_date'], '2024-01-01')


class TestCalculateLatLonLimits(unittest.TestCase):
    """Test cases for calculate_latlon_limits function."""
    
    def test_calculate_limits_basic(self):
        """Test basic lat-lon limit calculation."""
        geometry = {
            'bounds': {
                'x1min': 0., 'x1max': 10000.,  # vertical, not used for lat-lon
                'x2min': 0., 'x2max': 20000.,  # Y direction (north-south)
                'x3min': 0., 'x3max': 10000.   # X direction (east-west)
            },
            'center_latitude': 30.0,
            'center_longitude': -110.0
        }
        
        latmin, latmax, lonmin, lonmax = fetch_era5_pipeline.calculate_latlon_limits(geometry)
        
        # Check that results are reasonable
        self.assertIsInstance(latmin, float)
        self.assertIsInstance(latmax, float)
        self.assertIsInstance(lonmin, float)
        self.assertIsInstance(lonmax, float)
        
        # Check ordering
        self.assertLess(latmin, latmax)
        self.assertLess(lonmin, lonmax)
        
        # Check that center is within bounds
        self.assertLess(latmin, 30.0)
        self.assertGreater(latmax, 30.0)
        self.assertLess(lonmin, -110.0)
        self.assertGreater(lonmax, -110.0)
    
    def test_calculate_limits_at_equator(self):
        """Test calculation at the equator."""
        geometry = {
            'bounds': {
                'x1min': 0., 'x1max': 10000.,
                'x2min': 0., 'x2max': 20000.,  # 20 km north-south
                'x3min': 0., 'x3max': 10000.   # 10 km east-west
            },
            'center_latitude': 0.0,
            'center_longitude': 0.0
        }
        
        latmin, latmax, lonmin, lonmax = fetch_era5_pipeline.calculate_latlon_limits(geometry)
        
        # At equator, lat and lon conversions should be similar
        lat_range = latmax - latmin
        lon_range = lonmax - lonmin
        
        # Y range (20 km) should be about 2x X range (10 km)
        # So lat_range should be about 2x lon_range
        self.assertAlmostEqual(lat_range / lon_range, 2.0, places=1)
    
    def test_calculate_limits_high_latitude(self):
        """Test calculation at high latitude."""
        geometry = {
            'bounds': {
                'x1min': 0., 'x1max': 10000.,
                'x2min': 0., 'x2max': 20000.,  # 20 km north-south
                'x3min': 0., 'x3max': 10000.   # 10 km east-west
            },
            'center_latitude': 60.0,  # High latitude
            'center_longitude': 0.0
        }
        
        latmin, latmax, lonmin, lonmax = fetch_era5_pipeline.calculate_latlon_limits(geometry)
        
        # At high latitude, longitude degrees span more distance
        # So lon_range should be larger than at equator for same x3 range
        self.assertIsInstance(lonmin, float)
        self.assertIsInstance(lonmax, float)


class TestValidateDomainSize(unittest.TestCase):
    """Test cases for validate_domain_size function."""
    
    def test_valid_domain_size(self):
        """Test that valid domain size passes validation."""
        # Domain with 10 degrees lat-lon (well above 1 degree minimum)
        latmin, latmax = 30.0, 40.0
        lonmin, lonmax = -110.0, -100.0
        
        # Should not raise exception
        try:
            fetch_era5_pipeline.validate_domain_size(latmin, latmax, lonmin, lonmax)
        except ValueError:
            self.fail("validate_domain_size raised ValueError unexpectedly")
    
    def test_valid_domain_size_at_minimum(self):
        """Test that domain size exactly at minimum passes validation."""
        # Domain with exactly 1 degree in both directions
        latmin, latmax = 30.0, 31.0
        lonmin, lonmax = -110.0, -109.0
        
        # Should not raise exception
        try:
            fetch_era5_pipeline.validate_domain_size(latmin, latmax, lonmin, lonmax)
        except ValueError:
            self.fail("validate_domain_size raised ValueError unexpectedly")
    
    def test_invalid_domain_size_latitude(self):
        """Test that domain too small in latitude direction fails validation."""
        # Domain with 0.5 degrees latitude (below 1 degree minimum)
        latmin, latmax = 30.0, 30.5
        lonmin, lonmax = -110.0, -100.0  # 10 degrees, well above minimum
        
        with self.assertRaises(ValueError) as context:
            fetch_era5_pipeline.validate_domain_size(latmin, latmax, lonmin, lonmax)
        
        self.assertIn("North-South extent", str(context.exception))
        self.assertIn("0.5000°", str(context.exception))
        self.assertIn("minimum required", str(context.exception))
    
    def test_invalid_domain_size_longitude(self):
        """Test that domain too small in longitude direction fails validation."""
        # Domain with 0.3 degrees longitude (below 1 degree minimum)
        latmin, latmax = 30.0, 40.0  # 10 degrees, well above minimum
        lonmin, lonmax = -110.0, -109.7
        
        with self.assertRaises(ValueError) as context:
            fetch_era5_pipeline.validate_domain_size(latmin, latmax, lonmin, lonmax)
        
        self.assertIn("East-West extent", str(context.exception))
        self.assertIn("0.3000°", str(context.exception))
        self.assertIn("minimum required", str(context.exception))
    
    def test_invalid_domain_size_both_directions(self):
        """Test that domain too small in both directions reports both errors."""
        # Domain with 0.5 degrees in both directions (below 1 degree minimum)
        latmin, latmax = 30.0, 30.5
        lonmin, lonmax = -110.0, -109.5
        
        with self.assertRaises(ValueError) as context:
            fetch_era5_pipeline.validate_domain_size(latmin, latmax, lonmin, lonmax)
        
        error_msg = str(context.exception)
        self.assertIn("North-South extent", error_msg)
        self.assertIn("East-West extent", error_msg)
        self.assertIn("0.5000°", error_msg)
    
    def test_domain_size_custom_minimum(self):
        """Test that custom minimum size can be specified."""
        # Domain with 1.5 degrees
        latmin, latmax = 30.0, 31.5
        lonmin, lonmax = -110.0, -108.5
        
        # Should pass with 1 degree minimum
        try:
            fetch_era5_pipeline.validate_domain_size(latmin, latmax, lonmin, lonmax, min_size_degrees=1.0)
        except ValueError:
            self.fail("validate_domain_size raised ValueError unexpectedly")
        
        # Should fail with 2 degree minimum
        with self.assertRaises(ValueError) as context:
            fetch_era5_pipeline.validate_domain_size(latmin, latmax, lonmin, lonmax, min_size_degrees=2.0)
        
        self.assertIn("minimum required", str(context.exception))
    
    def test_error_message_includes_km_conversion(self):
        """Test that error message includes approximate kilometer values."""
        # Domain with 0.5 degrees (approximately 55.66 km)
        latmin, latmax = 30.0, 30.5
        lonmin, lonmax = -110.0, -100.0
        
        with self.assertRaises(ValueError) as context:
            fetch_era5_pipeline.validate_domain_size(latmin, latmax, lonmin, lonmax)
        
        error_msg = str(context.exception)
        # Should mention approximate km values
        self.assertIn("km", error_msg)
        self.assertIn("55.7 km", error_msg)  # 0.5 * 111.32 ≈ 55.66
        self.assertIn("111.3 km", error_msg)  # 1.0 * 111.32 ≈ 111.32


class TestAddBufferZone(unittest.TestCase):
    """Test cases for add_buffer_zone function."""
    
    def test_add_10_percent_buffer(self):
        """Test adding 10% buffer zone."""
        latmin, latmax = 30.0, 40.0
        lonmin, lonmax = -110.0, -100.0
        
        buf_latmin, buf_latmax, buf_lonmin, buf_lonmax = fetch_era5_pipeline.add_buffer_zone(
            latmin, latmax, lonmin, lonmax, buffer_percent=0.10
        )
        
        # Check that buffer increased the region
        self.assertLess(buf_latmin, latmin)
        self.assertGreater(buf_latmax, latmax)
        self.assertLess(buf_lonmin, lonmin)
        self.assertGreater(buf_lonmax, lonmax)
        
        # Check that buffer is approximately 10%
        lat_buffer = (buf_latmax - latmax + latmin - buf_latmin) / 2
        expected_lat_buffer = (latmax - latmin) * 0.10
        self.assertAlmostEqual(lat_buffer, expected_lat_buffer, places=5)
    
    def test_buffer_respects_latitude_bounds(self):
        """Test that buffer respects -90 to 90 latitude bounds."""
        latmin, latmax = -85.0, 85.0
        lonmin, lonmax = -170.0, 170.0
        
        buf_latmin, buf_latmax, buf_lonmin, buf_lonmax = fetch_era5_pipeline.add_buffer_zone(
            latmin, latmax, lonmin, lonmax, buffer_percent=0.20
        )
        
        # Check that latitude stays within bounds
        self.assertGreaterEqual(buf_latmin, -90.0)
        self.assertLessEqual(buf_latmax, 90.0)
    
    def test_buffer_respects_longitude_bounds(self):
        """Test that buffer respects -180 to 180 longitude bounds."""
        latmin, latmax = 0.0, 10.0
        lonmin, lonmax = -175.0, 175.0
        
        buf_latmin, buf_latmax, buf_lonmin, buf_lonmax = fetch_era5_pipeline.add_buffer_zone(
            latmin, latmax, lonmin, lonmax, buffer_percent=0.10
        )
        
        # Check that longitude stays within bounds
        self.assertGreaterEqual(buf_lonmin, -180.0)
        self.assertLessEqual(buf_lonmax, 180.0)


class TestFormatLatLonString(unittest.TestCase):
    """Test cases for format_lat_lon_string function."""
    
    def test_format_north_latitude(self):
        """Test formatting north latitude."""
        result = fetch_era5_pipeline.format_lat_lon_string(30.5, is_latitude=True)
        self.assertEqual(result, '30.50N')
    
    def test_format_south_latitude(self):
        """Test formatting south latitude."""
        result = fetch_era5_pipeline.format_lat_lon_string(-30.5, is_latitude=True)
        self.assertEqual(result, '30.50S')
    
    def test_format_east_longitude(self):
        """Test formatting east longitude."""
        result = fetch_era5_pipeline.format_lat_lon_string(110.25, is_latitude=False)
        self.assertEqual(result, '110.25E')
    
    def test_format_west_longitude(self):
        """Test formatting west longitude."""
        result = fetch_era5_pipeline.format_lat_lon_string(-110.25, is_latitude=False)
        self.assertEqual(result, '110.25W')
    
    def test_format_zero_latitude(self):
        """Test formatting zero latitude."""
        result = fetch_era5_pipeline.format_lat_lon_string(0.0, is_latitude=True)
        self.assertEqual(result, '0.00N')
    
    def test_format_zero_longitude(self):
        """Test formatting zero longitude."""
        result = fetch_era5_pipeline.format_lat_lon_string(0.0, is_latitude=False)
        self.assertEqual(result, '0.00E')


class TestGenerateOutputDirname(unittest.TestCase):
    """Test cases for generate_output_dirname function."""
    
    def test_generate_dirname_north_west(self):
        """Test generating directory name for northern and western hemisphere."""
        dirname = fetch_era5_pipeline.generate_output_dirname(30.0, 40.0, -110.0, -100.0)
        self.assertEqual(dirname, '30.00N_40.00N_110.00W_100.00W')
    
    def test_generate_dirname_south_east(self):
        """Test generating directory name for southern and eastern hemisphere."""
        dirname = fetch_era5_pipeline.generate_output_dirname(-40.0, -30.0, 100.0, 110.0)
        self.assertEqual(dirname, '40.00S_30.00S_100.00E_110.00E')
    
    def test_generate_dirname_mixed(self):
        """Test generating directory name spanning hemispheres."""
        dirname = fetch_era5_pipeline.generate_output_dirname(-10.0, 10.0, -10.0, 10.0)
        self.assertEqual(dirname, '10.00S_10.00N_10.00W_10.00E')


class TestFetchERA5Data(unittest.TestCase):
    """Test cases for fetch_era5_data function."""
    
    @patch('subprocess.run')
    def test_fetch_data_success(self, mock_run):
        """Test successful data fetching."""
        mock_run.return_value = MagicMock(returncode=0)
        
        # Should not raise exception
        fetch_era5_pipeline.fetch_era5_data(
            30.0, 40.0, -110.0, -100.0,
            '2024-01-01', '2024-01-02',
            '/tmp/test_output'
        )
        
        # Check that subprocess.run was called twice (densities and dynamics)
        self.assertEqual(mock_run.call_count, 2)
    
    @patch('subprocess.run')
    def test_fetch_data_densities_failure(self, mock_run):
        """Test handling of densities fetch failure."""
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, 'cmd'),  # First call fails
            MagicMock(returncode=0)  # Second call would succeed
        ]
        
        with self.assertRaises(RuntimeError) as context:
            fetch_era5_pipeline.fetch_era5_data(
                30.0, 40.0, -110.0, -100.0,
                '2024-01-01', '2024-01-02',
                '/tmp/test_output'
            )
        
        self.assertIn('densities', str(context.exception))
    
    @patch('subprocess.run')
    def test_fetch_data_dynamics_failure(self, mock_run):
        """Test handling of dynamics fetch failure."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # First call succeeds
            subprocess.CalledProcessError(1, 'cmd')   # Second call fails
        ]
        
        with self.assertRaises(RuntimeError) as context:
            fetch_era5_pipeline.fetch_era5_data(
                30.0, 40.0, -110.0, -100.0,
                '2024-01-01', '2024-01-02',
                '/tmp/test_output'
            )
        
        self.assertIn('dynamics', str(context.exception))


class TestMainIntegration(unittest.TestCase):
    """Integration tests for main function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_main_with_valid_config(self, mock_run):
        """Test main function with valid configuration."""
        mock_run.return_value = MagicMock(returncode=0)
        
        # Create test YAML file with domain size >= 100 km (1 degree)
        # Using 150 km x 150 km domain to meet minimum requirements
        yaml_content = """
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 10.e3, x2min: 0., x2max: 150.e3, x3min: 0., x3max: 150.e3}
  cells: {nx1: 100, nx2: 300, nx3: 150, nghost: 3}
  center_latitude: 30.
  center_longitude: -110.
integration:
  start-date: 2024-01-01
  end-date: 2024-01-02
"""
        yaml_file = os.path.join(self.temp_dir, 'test.yaml')
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        # Mock sys.argv
        sys.argv = [
            'fetch_era5_pipeline.py',
            yaml_file,
            '--output-base', self.temp_dir
        ]
        
        # Should not raise exception
        try:
            fetch_era5_pipeline.main()
        except SystemExit as e:
            # main() might call sys.exit(0) on success
            self.assertEqual(e.code, 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestParseYAMLConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestExtractGeometryInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestExtractIntegrationInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculateLatLonLimits))
    suite.addTests(loader.loadTestsFromTestCase(TestValidateDomainSize))
    suite.addTests(loader.loadTestsFromTestCase(TestAddBufferZone))
    suite.addTests(loader.loadTestsFromTestCase(TestFormatLatLonString))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateOutputDirname))
    suite.addTests(loader.loadTestsFromTestCase(TestFetchERA5Data))
    suite.addTests(loader.loadTestsFromTestCase(TestMainIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

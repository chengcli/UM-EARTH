"""
Unit tests for ECMWF utilities module.

This module contains tests for utility functions used in ECMWF data fetching scripts.
"""

import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import MagicMock

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Mock cdsapi before importing other modules
sys.modules['cdsapi'] = MagicMock()
sys.modules['xarray'] = MagicMock()
sys.modules['netCDF4'] = MagicMock()

import ecmwf_utils


class TestValidateDateFormat(unittest.TestCase):
    """Test cases for validate_date_format function."""
    
    def test_valid_date_formats(self):
        """Test valid date formats."""
        self.assertTrue(ecmwf_utils.validate_date_format("2024-01-01"))
        self.assertTrue(ecmwf_utils.validate_date_format("2023-12-31"))
        self.assertTrue(ecmwf_utils.validate_date_format("2024-02-29"))  # Leap year
        self.assertTrue(ecmwf_utils.validate_date_format("2020-06-15"))
    
    def test_invalid_date_formats(self):
        """Test invalid date formats."""
        self.assertFalse(ecmwf_utils.validate_date_format("2024/01/01"))
        self.assertFalse(ecmwf_utils.validate_date_format("01-01-2024"))
        self.assertFalse(ecmwf_utils.validate_date_format("2024-13-01"))  # Invalid month
        self.assertFalse(ecmwf_utils.validate_date_format("2024-01-32"))  # Invalid day
        self.assertFalse(ecmwf_utils.validate_date_format("not-a-date"))
        self.assertFalse(ecmwf_utils.validate_date_format(""))


class TestValidateRegionBounds(unittest.TestCase):
    """Test cases for validate_region_bounds function."""
    
    def test_valid_bounds(self):
        """Test valid geographical bounds."""
        # Should not raise exception
        ecmwf_utils.validate_region_bounds(30.0, 40.0, -110.0, -100.0)
        ecmwf_utils.validate_region_bounds(-90.0, 90.0, -180.0, 180.0)  # Full globe
        ecmwf_utils.validate_region_bounds(0.0, 1.0, 0.0, 1.0)
    
    def test_invalid_latitude_range(self):
        """Test invalid latitude values."""
        with self.assertRaises(ValueError) as context:
            ecmwf_utils.validate_region_bounds(-100.0, 40.0, -110.0, -100.0)
        self.assertIn("Latitude must be between -90 and 90", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            ecmwf_utils.validate_region_bounds(30.0, 100.0, -110.0, -100.0)
        self.assertIn("Latitude must be between -90 and 90", str(context.exception))
    
    def test_invalid_longitude_range(self):
        """Test invalid longitude values."""
        with self.assertRaises(ValueError) as context:
            ecmwf_utils.validate_region_bounds(30.0, 40.0, -200.0, -100.0)
        self.assertIn("Longitude must be between -180 and 180", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            ecmwf_utils.validate_region_bounds(30.0, 40.0, -110.0, 200.0)
        self.assertIn("Longitude must be between -180 and 180", str(context.exception))
    
    def test_reversed_latitude(self):
        """Test that latmin must be less than latmax."""
        with self.assertRaises(ValueError) as context:
            ecmwf_utils.validate_region_bounds(40.0, 30.0, -110.0, -100.0)
        self.assertIn("latmin must be less than latmax", str(context.exception))
    
    def test_reversed_longitude(self):
        """Test that lonmin must be less than lonmax."""
        with self.assertRaises(ValueError) as context:
            ecmwf_utils.validate_region_bounds(30.0, 40.0, -100.0, -110.0)
        self.assertIn("lonmin must be less than lonmax", str(context.exception))
    
    def test_equal_bounds(self):
        """Test that equal bounds are invalid."""
        with self.assertRaises(ValueError) as context:
            ecmwf_utils.validate_region_bounds(30.0, 30.0, -110.0, -100.0)
        self.assertIn("latmin must be less than latmax", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            ecmwf_utils.validate_region_bounds(30.0, 40.0, -110.0, -110.0)
        self.assertIn("lonmin must be less than lonmax", str(context.exception))


class TestValidatePressureLevels(unittest.TestCase):
    """Test cases for validate_pressure_levels function."""
    
    def test_valid_pressure_levels(self):
        """Test valid pressure levels."""
        result = ecmwf_utils.validate_pressure_levels([1000, 850, 500])
        self.assertEqual(result, ['1000', '850', '500'])
        
        result = ecmwf_utils.validate_pressure_levels([1, 2, 3, 5])
        self.assertEqual(result, ['1', '2', '3', '5'])
    
    def test_invalid_pressure_level(self):
        """Test invalid pressure level."""
        with self.assertRaises(ValueError) as context:
            ecmwf_utils.validate_pressure_levels([1000, 999, 500])
        self.assertIn("Pressure level 999 hPa is not available", str(context.exception))
    
    def test_all_standard_levels(self):
        """Test all standard pressure levels are valid."""
        result = ecmwf_utils.validate_pressure_levels(ecmwf_utils.STANDARD_PRESSURE_LEVELS)
        self.assertEqual(len(result), len(ecmwf_utils.STANDARD_PRESSURE_LEVELS))
        self.assertEqual(result, [str(level) for level in ecmwf_utils.STANDARD_PRESSURE_LEVELS])
    
    def test_single_level(self):
        """Test single pressure level."""
        result = ecmwf_utils.validate_pressure_levels([850])
        self.assertEqual(result, ['850'])


class TestValidateVariableNames(unittest.TestCase):
    """Test cases for validate_variable_names function."""
    
    def test_valid_variables(self):
        """Test valid variable names."""
        # Should not raise exception
        ecmwf_utils.validate_variable_names(['temperature'])
        ecmwf_utils.validate_variable_names(['temperature', 'u_component_of_wind'])
        ecmwf_utils.validate_variable_names(['specific_humidity', 'relative_humidity'])
    
    def test_invalid_variable(self):
        """Test invalid variable name."""
        with self.assertRaises(ValueError) as context:
            ecmwf_utils.validate_variable_names(['invalid_variable'])
        self.assertIn("Variable 'invalid_variable' is not available", str(context.exception))
    
    def test_all_standard_variables(self):
        """Test all standard variables are valid."""
        # Should not raise exception
        ecmwf_utils.validate_variable_names(ecmwf_utils.STANDARD_VARIABLES)
    
    def test_dynamics_variables(self):
        """Test dynamics variable names."""
        dynamics_vars = [
            'temperature',
            'u_component_of_wind',
            'v_component_of_wind',
            'vertical_velocity',
            'divergence',
            'vorticity',
            'potential_vorticity',
            'geopotential',
        ]
        ecmwf_utils.validate_variable_names(dynamics_vars)
    
    def test_density_variables(self):
        """Test density variable names."""
        density_vars = [
            'specific_cloud_ice_water_content',
            'specific_humidity',
            'specific_snow_water_content',
            'specific_cloud_liquid_water_content',
            'specific_rain_water_content',
            'fraction_of_cloud_cover',
            'relative_humidity'
        ]
        ecmwf_utils.validate_variable_names(density_vars)


class TestAddCommonArguments(unittest.TestCase):
    """Test cases for add_common_arguments function."""
    
    def test_add_arguments(self):
        """Test that common arguments are added to parser."""
        import argparse
        parser = argparse.ArgumentParser()
        parser = ecmwf_utils.add_common_arguments(parser)
        
        # Parse with valid arguments
        args = parser.parse_args([
            '--latmin', '30.0',
            '--latmax', '40.0',
            '--lonmin', '-110.0',
            '--lonmax', '-100.0',
            '--start-date', '2024-01-01',
            '--end-date', '2024-01-02'
        ])
        
        self.assertEqual(args.latmin, 30.0)
        self.assertEqual(args.latmax, 40.0)
        self.assertEqual(args.lonmin, -110.0)
        self.assertEqual(args.lonmax, -100.0)
        self.assertEqual(args.start_date, '2024-01-01')
        self.assertEqual(args.end_date, '2024-01-02')
        self.assertEqual(args.output, '.')  # Default
        self.assertEqual(args.jobs, 4)  # Default
    
    def test_optional_arguments(self):
        """Test optional arguments."""
        import argparse
        parser = argparse.ArgumentParser()
        parser = ecmwf_utils.add_common_arguments(parser)
        
        args = parser.parse_args([
            '--latmin', '30.0',
            '--latmax', '40.0',
            '--lonmin', '-110.0',
            '--lonmax', '-100.0',
            '--start-date', '2024-01-01',
            '--end-date', '2024-01-02',
            '--output', './data',
            '--times', '00:00', '12:00',
            '--jobs', '8',
            '--api-key', 'test-key',
            '--api-url', 'https://test.url'
        ])
        
        self.assertEqual(args.output, './data')
        self.assertEqual(args.times, ['00:00', '12:00'])
        self.assertEqual(args.jobs, 8)
        self.assertEqual(args.api_key, 'test-key')
        self.assertEqual(args.api_url, 'https://test.url')


class TestGenerateDateList(unittest.TestCase):
    """Test cases for generate_date_list function."""
    
    def test_single_day(self):
        """Test generating list for single day."""
        dates = ecmwf_utils.generate_date_list('2024-01-01', '2024-01-01')
        self.assertEqual(dates, ['2024-01-01'])
    
    def test_multiple_days(self):
        """Test generating list for multiple days."""
        dates = ecmwf_utils.generate_date_list('2024-01-01', '2024-01-03')
        self.assertEqual(dates, ['2024-01-01', '2024-01-02', '2024-01-03'])
    
    def test_week_range(self):
        """Test generating list for a week."""
        dates = ecmwf_utils.generate_date_list('2024-01-01', '2024-01-07')
        self.assertEqual(len(dates), 7)
        self.assertEqual(dates[0], '2024-01-01')
        self.assertEqual(dates[-1], '2024-01-07')
    
    def test_month_boundary(self):
        """Test generating list across month boundary."""
        dates = ecmwf_utils.generate_date_list('2024-01-30', '2024-02-02')
        self.assertEqual(dates, ['2024-01-30', '2024-01-31', '2024-02-01', '2024-02-02'])
    
    def test_year_boundary(self):
        """Test generating list across year boundary."""
        dates = ecmwf_utils.generate_date_list('2023-12-30', '2024-01-02')
        self.assertEqual(dates, ['2023-12-30', '2023-12-31', '2024-01-01', '2024-01-02'])
    
    def test_leap_year(self):
        """Test generating list including leap day."""
        dates = ecmwf_utils.generate_date_list('2024-02-28', '2024-03-01')
        self.assertEqual(dates, ['2024-02-28', '2024-02-29', '2024-03-01'])


class TestFetchSingleDay(unittest.TestCase):
    """Test cases for fetch_single_day function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.api = MagicMock()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_successful_fetch(self):
        """Test successful data fetch."""
        # Mock successful API call
        expected_file = os.path.join(self.temp_dir, 'era5_20240101.nc')
        self.api.fetch_weather_data.return_value = expected_file
        
        result = ecmwf_utils.fetch_single_day(
            api=self.api,
            date_str='2024-01-01',
            variables=['temperature'],
            pressure_levels=[1000, 850, 500],
            latmin=30.0,
            latmax=40.0,
            lonmin=-110.0,
            lonmax=-100.0,
            times=['00:00', '12:00'],
            output_dir=self.temp_dir,
            day_idx=0,
            total_days=1,
            prefix='era5'
        )
        
        self.assertEqual(result, expected_file)
        self.api.fetch_weather_data.assert_called_once()
        
        # Check the call arguments
        call_kwargs = self.api.fetch_weather_data.call_args[1]
        self.assertEqual(call_kwargs['latmin'], 30.0)
        self.assertEqual(call_kwargs['latmax'], 40.0)
        self.assertEqual(call_kwargs['start_date'], '2024-01-01')
        self.assertEqual(call_kwargs['end_date'], '2024-01-01')
        self.assertEqual(call_kwargs['variables'], ['temperature'])
        self.assertEqual(call_kwargs['pressure_levels'], [1000, 850, 500])
    
    def test_fetch_with_custom_prefix(self):
        """Test fetch with custom filename prefix."""
        expected_file = os.path.join(self.temp_dir, 'custom_prefix_20240115.nc')
        self.api.fetch_weather_data.return_value = expected_file
        
        result = ecmwf_utils.fetch_single_day(
            api=self.api,
            date_str='2024-01-15',
            variables=['temperature'],
            pressure_levels=[1000],
            latmin=30.0,
            latmax=40.0,
            lonmin=-110.0,
            lonmax=-100.0,
            times=None,
            output_dir=self.temp_dir,
            day_idx=0,
            total_days=1,
            prefix='custom_prefix'
        )
        
        self.assertEqual(result, expected_file)
        
        # Check filename in the API call
        call_kwargs = self.api.fetch_weather_data.call_args[1]
        self.assertIn('custom_prefix_20240115.nc', call_kwargs['output_file'])
    
    def test_fetch_failure(self):
        """Test handling of fetch failure."""
        # Mock API failure
        self.api.fetch_weather_data.side_effect = RuntimeError("API error")
        
        result = ecmwf_utils.fetch_single_day(
            api=self.api,
            date_str='2024-01-01',
            variables=['temperature'],
            pressure_levels=[1000],
            latmin=30.0,
            latmax=40.0,
            lonmin=-110.0,
            lonmax=-100.0,
            times=None,
            output_dir=self.temp_dir,
            day_idx=0,
            total_days=1,
            prefix='era5'
        )
        
        # Should return None on failure
        self.assertIsNone(result)
    
    def test_filename_format(self):
        """Test that filename is correctly formatted."""
        expected_file = os.path.join(self.temp_dir, 'era5_hourly_densities_20240101.nc')
        self.api.fetch_weather_data.return_value = expected_file
        
        ecmwf_utils.fetch_single_day(
            api=self.api,
            date_str='2024-01-01',
            variables=['specific_humidity'],
            pressure_levels=[1000],
            latmin=30.0,
            latmax=40.0,
            lonmin=-110.0,
            lonmax=-100.0,
            times=None,
            output_dir=self.temp_dir,
            day_idx=0,
            total_days=1,
            prefix='era5_hourly_densities'
        )
        
        call_kwargs = self.api.fetch_weather_data.call_args[1]
        output_file = call_kwargs['output_file']
        
        # Check format: prefix_YYYYMMDD.nc
        self.assertTrue(output_file.endswith('era5_hourly_densities_20240101.nc'))
    
    def test_progress_tracking(self):
        """Test that progress information is used correctly."""
        expected_file = os.path.join(self.temp_dir, 'era5_20240101.nc')
        self.api.fetch_weather_data.return_value = expected_file
        
        # Test with different day indices
        for day_idx in range(3):
            result = ecmwf_utils.fetch_single_day(
                api=self.api,
                date_str='2024-01-01',
                variables=['temperature'],
                pressure_levels=[1000],
                latmin=30.0,
                latmax=40.0,
                lonmin=-110.0,
                lonmax=-100.0,
                times=None,
                output_dir=self.temp_dir,
                day_idx=day_idx,
                total_days=3,
                prefix='era5'
            )
            self.assertIsNotNone(result)


class TestConstants(unittest.TestCase):
    """Test cases for module constants."""
    
    def test_standard_pressure_levels_count(self):
        """Test that all 37 standard ERA5 pressure levels are defined."""
        self.assertEqual(len(ecmwf_utils.STANDARD_PRESSURE_LEVELS), 37)
    
    def test_standard_pressure_levels_range(self):
        """Test pressure level range."""
        self.assertEqual(min(ecmwf_utils.STANDARD_PRESSURE_LEVELS), 1)
        self.assertEqual(max(ecmwf_utils.STANDARD_PRESSURE_LEVELS), 1000)
    
    def test_standard_times_count(self):
        """Test that 4 standard times are defined."""
        self.assertEqual(len(ecmwf_utils.STANDARD_TIMES), 4)
    
    def test_standard_times_values(self):
        """Test standard time values."""
        expected_times = ['00:00', '06:00', '12:00', '18:00']
        self.assertEqual(ecmwf_utils.STANDARD_TIMES, expected_times)
    
    def test_standard_variables_coverage(self):
        """Test that both dynamics and density variables are included."""
        # Dynamics variables
        self.assertIn('temperature', ecmwf_utils.STANDARD_VARIABLES)
        self.assertIn('u_component_of_wind', ecmwf_utils.STANDARD_VARIABLES)
        self.assertIn('v_component_of_wind', ecmwf_utils.STANDARD_VARIABLES)
        self.assertIn('geopotential', ecmwf_utils.STANDARD_VARIABLES)
        
        # Density variables
        self.assertIn('specific_humidity', ecmwf_utils.STANDARD_VARIABLES)
        self.assertIn('relative_humidity', ecmwf_utils.STANDARD_VARIABLES)
        self.assertIn('specific_cloud_ice_water_content', ecmwf_utils.STANDARD_VARIABLES)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestValidateDateFormat))
    suite.addTests(loader.loadTestsFromTestCase(TestValidateRegionBounds))
    suite.addTests(loader.loadTestsFromTestCase(TestValidatePressureLevels))
    suite.addTests(loader.loadTestsFromTestCase(TestValidateVariableNames))
    suite.addTests(loader.loadTestsFromTestCase(TestAddCommonArguments))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateDateList))
    suite.addTests(loader.loadTestsFromTestCase(TestFetchSingleDay))
    suite.addTests(loader.loadTestsFromTestCase(TestConstants))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

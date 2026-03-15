"""
Unit tests for regrid module

This module contains tests for the regridding and interpolation functions
that convert ECMWF ERA5 data from pressure-lat-lon grids to distance grids.
"""

import unittest
import os
import tempfile
import numpy as np
from regrid import (
    compute_dz_from_plev,
    compute_heights_from_dz,
    compute_height_grid,
    latlon_to_xy,
    vertical_interp_to_z,
    horizontal_regrid_xy,
    regrid_pressure_to_height,
    regrid_multiple_variables,
    regrid_topography,
    save_regridded_data_to_netcdf,
    save_topography_to_netcdf,
)


class TestComputeDzFromPlev(unittest.TestCase):
    """Test cases for compute_dz_from_plev function."""
    
    def test_basic_computation(self):
        """Test basic layer thickness computation."""
        # Create simple test data
        T, P, Lat, Lon = 2, 3, 4, 5
        plev = np.array([100000., 50000., 10000.])  # Pa, descending
        rho = np.ones((T, P, Lat, Lon)) * 1.2  # kg/m^3
        grav = 10.0  # m/s^2
        
        dz = compute_dz_from_plev(plev, rho, grav)
        
        # Check shape
        self.assertEqual(dz.shape, (T, P-1, Lat, Lon))
        
        # Check that all values are positive (layers go upward)
        self.assertTrue(np.all(dz > 0))
        
    def test_pressure_sorting(self):
        """Test that function handles unsorted pressure levels."""
        T, P, Lat, Lon = 1, 3, 2, 2
        plev = np.array([10000., 50000., 100000.])  # Ascending, should be sorted
        rho = np.ones((T, P, Lat, Lon)) * 1.0
        grav = 10.0
        
        dz = compute_dz_from_plev(plev, rho, grav)
        
        # Should still work and produce positive layer thicknesses
        self.assertEqual(dz.shape, (T, P-1, Lat, Lon))
        self.assertTrue(np.all(dz > 0))
        
    def test_variable_density(self):
        """Test with variable density (decreasing with height)."""
        T, P, Lat, Lon = 1, 3, 2, 2
        plev = np.array([100000., 50000., 10000.])
        # Density decreases with height (lower pressure)
        rho = np.zeros((T, P, Lat, Lon))
        rho[:, 0, :, :] = 1.2  # Surface
        rho[:, 1, :, :] = 0.6  # Mid-level
        rho[:, 2, :, :] = 0.1  # Upper level
        grav = 10.0
        
        dz = compute_dz_from_plev(plev, rho, grav)
        
        # Upper layer (lower density) should be thicker than lower layer
        self.assertTrue(np.all(dz[:, 1, :, :] > dz[:, 0, :, :]))


class TestComputeHeightsFromDz(unittest.TestCase):
    """Test cases for compute_heights_from_dz function."""
    
    def test_basic_height_computation(self):
        """Test basic height computation from layer thicknesses."""
        T, P_minus_1, Lat, Lon = 2, 3, 4, 5
        dz = np.ones((T, P_minus_1, Lat, Lon)) * 100.0  # 100m layers
        topo = np.zeros((Lat, Lon))  # Sea level
        
        z = compute_heights_from_dz(dz, topo)
        
        # Check shape
        self.assertEqual(z.shape, (T, P_minus_1 + 1, Lat, Lon))
        
        # Check bottom is at topographic elevation
        for t in range(T):
            np.testing.assert_array_equal(z[t, 0, :, :], topo)
        
        # Check cumulative heights
        np.testing.assert_array_almost_equal(z[:, 1, :, :], 100.0)
        np.testing.assert_array_almost_equal(z[:, 2, :, :], 200.0)
        np.testing.assert_array_almost_equal(z[:, 3, :, :], 300.0)
        
    def test_with_topography(self):
        """Test height computation with non-zero topography."""
        T, P_minus_1, Lat, Lon = 1, 2, 2, 2
        dz = np.ones((T, P_minus_1, Lat, Lon)) * 50.0
        topo = np.array([[100.0, 200.0], [150.0, 250.0]])
        
        z = compute_heights_from_dz(dz, topo)
        
        # Bottom should be at topographic elevation
        np.testing.assert_array_equal(z[0, 0, :, :], topo)
        
        # Heights should be relative to topography
        np.testing.assert_array_almost_equal(z[0, 1, :, :], topo + 50.0)
        np.testing.assert_array_almost_equal(z[0, 2, :, :], topo + 100.0)


class TestComputeHeightGrid(unittest.TestCase):
    """Test cases for compute_height_grid function."""
    
    def test_basic_height_grid_computation(self):
        """Test basic height grid computation."""
        T, P, Lat, Lon = 2, 4, 5, 6
        plev = np.array([100000., 80000., 60000., 40000.])  # Pa
        rho = np.ones((T, P, Lat, Lon)) * 1.2
        topo = np.random.randn(Lat, Lon) * 100.0
        grav = 9.81
        
        z_tpll = compute_height_grid(rho, topo, plev, grav)
        
        # Check shape
        self.assertEqual(z_tpll.shape, (T, P, Lat, Lon))
        
        # Check that bottom is at topographic elevation
        for t in range(T):
            np.testing.assert_array_almost_equal(z_tpll[t, 0, :, :], topo)
        
        # Check that heights increase with decreasing pressure
        for t in range(T):
            for lat in range(Lat):
                for lon in range(Lon):
                    heights = z_tpll[t, :, lat, lon]
                    self.assertTrue(np.all(np.diff(heights) > 0))


class TestLatlonToXy(unittest.TestCase):
    """Test cases for latlon_to_xy function."""
    
    def test_basic_conversion(self):
        """Test basic lat/lon to X/Y conversion."""
        lats = np.array([30.0, 35.0, 40.0])  # degrees
        lons = np.array([-110.0, -105.0, -100.0])  # degrees
        planet_radius = 6371.e3  # Earth radius in meters
        
        Y, X = latlon_to_xy(lats, lons, planet_radius)
        
        # Check shapes
        self.assertEqual(Y.shape, lats.shape)
        self.assertEqual(X.shape, lons.shape)
        
        # Check that center is near zero
        Y_center = 0.5 * (Y[0] + Y[-1])
        X_center = 0.5 * (X[0] + X[-1])
        self.assertAlmostEqual(Y_center, 0.0, places=5)
        self.assertAlmostEqual(X_center, 0.0, places=5)
        
    def test_monotonicity(self):
        """Test that conversion preserves monotonicity."""
        lats = np.linspace(20.0, 50.0, 10)
        lons = np.linspace(-120.0, -80.0, 10)
        planet_radius = 6371.e3
        
        Y, X = latlon_to_xy(lats, lons, planet_radius)
        
        # Y should be monotonically increasing with latitude
        self.assertTrue(np.all(np.diff(Y) > 0))
        
        # X should be monotonically increasing with longitude
        self.assertTrue(np.all(np.diff(X) > 0))


class TestVerticalInterpToZ(unittest.TestCase):
    """Test cases for vertical_interp_to_z function."""
    
    def test_basic_interpolation(self):
        """Test basic vertical interpolation."""
        # Create simple test data with linear profile
        P = 5
        z_col = np.array([0., 100., 200., 300., 400.])
        v_col = np.array([0., 10., 20., 30., 40.])  # Linear with z
        z_out = np.array([50., 150., 250., 350.])
        
        # Reshape to add dimensions
        z_col = z_col[None, None, :]  # (1, 1, P)
        v_col = v_col[None, None, :]  # (1, 1, P)
        
        result = vertical_interp_to_z(z_col, v_col, z_out, bounds_error=False)
        
        # Check shape
        self.assertEqual(result.shape, (1, 1, len(z_out)))
        
        # Check interpolated values (should be linear)
        expected = np.array([5., 15., 25., 35.])
        np.testing.assert_array_almost_equal(result[0, 0, :], expected, decimal=5)
        
    def test_extrapolation_detection(self):
        """Test that extrapolation raises error when bounds_error=True."""
        z_col = np.array([100., 200., 300.])[None, None, :]
        v_col = np.array([10., 20., 30.])[None, None, :]
        z_out = np.array([50., 150., 350.])  # 50 and 350 are outside range
        
        with self.assertRaises(ValueError) as context:
            vertical_interp_to_z(z_col, v_col, z_out, bounds_error=True)
        
        self.assertIn("extrapolation", str(context.exception).lower())
        
    def test_no_error_when_bounds_ok(self):
        """Test that no error is raised when within bounds."""
        z_col = np.array([100., 200., 300.])[None, None, :]
        v_col = np.array([10., 20., 30.])[None, None, :]
        z_out = np.array([150., 250.])  # Within range
        
        # Should not raise
        result = vertical_interp_to_z(z_col, v_col, z_out, bounds_error=True)
        self.assertEqual(result.shape, (1, 1, 2))


class TestHorizontalRegridXy(unittest.TestCase):
    """Test cases for horizontal_regrid_xy function."""
    
    def test_basic_regridding(self):
        """Test basic horizontal regridding."""
        # Create simple test data
        x = np.array([0., 1., 2.])
        y = np.array([0., 1., 2.])
        field = np.array([[0., 1., 2.],
                         [1., 2., 3.],
                         [2., 3., 4.]])  # (X, Y)
        
        x_out = np.array([0.5, 1.5])
        y_out = np.array([0.5, 1.5])
        
        result = horizontal_regrid_xy(x, y, field, x_out, y_out, bounds_error=False)
        
        # Check shape
        self.assertEqual(result.shape, (len(x_out), len(y_out)))
        
    def test_extrapolation_detection_x(self):
        """Test that X extrapolation raises error when bounds_error=True."""
        x = np.array([0., 1., 2.])
        y = np.array([0., 1., 2.])
        field = np.ones((3, 3))
        
        x_out = np.array([-1., 1., 3.])  # -1 and 3 are outside
        y_out = np.array([0.5, 1.5])
        
        with self.assertRaises(ValueError) as context:
            horizontal_regrid_xy(x, y, field, x_out, y_out, bounds_error=True)
        
        self.assertIn("extrapolation", str(context.exception).lower())
        
    def test_extrapolation_detection_y(self):
        """Test that Y extrapolation raises error when bounds_error=True."""
        x = np.array([0., 1., 2.])
        y = np.array([0., 1., 2.])
        field = np.ones((3, 3))
        
        x_out = np.array([0.5, 1.5])
        y_out = np.array([-1., 1., 3.])  # -1 and 3 are outside
        
        with self.assertRaises(ValueError) as context:
            horizontal_regrid_xy(x, y, field, x_out, y_out, bounds_error=True)
        
        self.assertIn("extrapolation", str(context.exception).lower())


class TestRegridPressureToHeight(unittest.TestCase):
    """Test cases for regrid_pressure_to_height function."""
    
    def test_basic_regridding_pipeline(self):
        """Test complete regridding pipeline."""
        # Create test data
        T, P, Lat, Lon = 2, 4, 6, 8
        plev = np.array([100000., 80000., 60000., 40000.])  # Pa
        lats = np.linspace(30.0, 35.0, Lat)
        lons = np.linspace(-110.0, -105.0, Lon)
        
        # Create simple variable (e.g., temperature)
        var_tpll = 280.0 + np.random.randn(T, P, Lat, Lon) * 10.0
        rho_tpll = 1.0 + 0.1 * np.random.randn(T, P, Lat, Lon)
        rho_tpll = np.maximum(rho_tpll, 0.1)  # Ensure positive
        topo_ll = np.random.randn(Lat, Lon) * 50.0
        
        # Output grids (smaller domain)
        x1f = np.linspace(0., 5000., 20)  # Height
        x2f = np.linspace(-1000., 1000., 10)  # Y
        x3f = np.linspace(-2000., 2000., 15)  # X
        
        planet_grav = 9.81
        planet_radius = 6371.e3
        
        # Run regridding
        result = regrid_pressure_to_height(
            var_tpll, rho_tpll, topo_ll,
            plev, lats, lons,
            x1f, x2f, x3f,
            planet_grav, planet_radius,
            bounds_error=False
        )
        
        # Check output shape
        self.assertEqual(result.shape, (T, len(x1f), len(x2f), len(x3f)))
    
    def test_with_precomputed_heights(self):
        """Test regridding with pre-computed heights for efficiency."""
        # Create test data
        T, P, Lat, Lon = 2, 4, 6, 8
        plev = np.array([100000., 80000., 60000., 40000.])  # Pa
        lats = np.linspace(30.0, 35.0, Lat)
        lons = np.linspace(-110.0, -105.0, Lon)
        
        # Create two variables to regrid
        temp_tpll = 280.0 + np.random.randn(T, P, Lat, Lon) * 10.0
        humid_tpll = 0.5 + np.random.randn(T, P, Lat, Lon) * 0.1
        rho_tpll = 1.0 + 0.1 * np.random.randn(T, P, Lat, Lon)
        rho_tpll = np.maximum(rho_tpll, 0.1)
        topo_ll = np.random.randn(Lat, Lon) * 50.0
        
        # Output grids
        x1f = np.linspace(0., 5000., 20)
        x2f = np.linspace(-1000., 1000., 10)
        x3f = np.linspace(-2000., 2000., 15)
        
        planet_grav = 9.81
        planet_radius = 6371.e3
        
        # Compute heights once
        z_tpll = compute_height_grid(rho_tpll, topo_ll, plev, planet_grav)
        
        # Regrid multiple variables with pre-computed heights
        temp_result = regrid_pressure_to_height(
            temp_tpll, rho_tpll, topo_ll,
            plev, lats, lons,
            x1f, x2f, x3f,
            planet_grav, planet_radius,
            bounds_error=False,
            z_tpll=z_tpll
        )
        
        humid_result = regrid_pressure_to_height(
            humid_tpll, rho_tpll, topo_ll,
            plev, lats, lons,
            x1f, x2f, x3f,
            planet_grav, planet_radius,
            bounds_error=False,
            z_tpll=z_tpll
        )
        
        # Check output shapes
        self.assertEqual(temp_result.shape, (T, len(x1f), len(x2f), len(x3f)))
        self.assertEqual(humid_result.shape, (T, len(x1f), len(x2f), len(x3f)))
        
        # Verify results match when computed without pre-computed heights
        temp_result_no_precomp = regrid_pressure_to_height(
            temp_tpll, rho_tpll, topo_ll,
            plev, lats, lons,
            x1f, x2f, x3f,
            planet_grav, planet_radius,
            bounds_error=False
        )
        
        np.testing.assert_array_almost_equal(temp_result, temp_result_no_precomp)
        
    def test_domain_bounds_error(self):
        """Test that exceeding domain bounds raises error."""
        T, P, Lat, Lon = 1, 3, 5, 5
        plev = np.array([100000., 50000., 10000.])
        lats = np.linspace(32.0, 33.0, Lat)
        lons = np.linspace(-106.0, -105.0, Lon)
        
        var_tpll = np.ones((T, P, Lat, Lon)) * 280.0
        rho_tpll = np.ones((T, P, Lat, Lon)) * 1.0
        topo_ll = np.zeros((Lat, Lon))
        
        # Output grids that exceed input domain
        x1f = np.linspace(0., 5000., 10)
        x2f = np.linspace(-500000., 500000., 10)  # Way too large
        x3f = np.linspace(-500000., 500000., 10)  # Way too large
        
        planet_grav = 9.81
        planet_radius = 6371.e3
        
        with self.assertRaises(ValueError) as context:
            regrid_pressure_to_height(
                var_tpll, rho_tpll, topo_ll,
                plev, lats, lons,
                x1f, x2f, x3f,
                planet_grav, planet_radius,
                bounds_error=True
            )
        
        self.assertIn("exceeds", str(context.exception).lower())


class TestRegridTopography(unittest.TestCase):
    """Test cases for regrid_topography function."""
    
    def test_basic_topography_regridding(self):
        """Test basic topography regridding."""
        Lat, Lon = 10, 12
        lats = np.linspace(30.0, 35.0, Lat)
        lons = np.linspace(-110.0, -105.0, Lon)
        topo_ll = np.random.randn(Lat, Lon) * 100.0
        
        x2f = np.linspace(-1000., 1000., 8)
        x3f = np.linspace(-2000., 2000., 10)
        planet_radius = 6371.e3
        
        result = regrid_topography(
            topo_ll, lats, lons,
            x2f, x3f, planet_radius,
            bounds_error=False
        )
        
        # Check output shape
        self.assertEqual(result.shape, (len(x2f), len(x3f)))


class TestMultipleVariables(unittest.TestCase):
    """Test cases for regrid_multiple_variables function."""
    
    def test_basic_multiple_variables(self):
        """Test regridding multiple variables."""
        # Create test data
        T, P, Lat, Lon = 2, 4, 6, 8
        plev = np.array([100000., 80000., 60000., 40000.])  # Pa
        lats = np.linspace(30.0, 35.0, Lat)
        lons = np.linspace(-110.0, -105.0, Lon)
        
        # Create multiple variables
        temp_tpll = 280.0 + np.random.randn(T, P, Lat, Lon) * 10.0
        humid_tpll = 0.5 + np.random.randn(T, P, Lat, Lon) * 0.1
        press_tpll = 50000.0 + np.random.randn(T, P, Lat, Lon) * 1000.0
        
        variables = {
            'temperature': temp_tpll,
            'humidity': humid_tpll,
            'pressure': press_tpll,
        }
        
        rho_tpll = 1.0 + 0.1 * np.random.randn(T, P, Lat, Lon)
        rho_tpll = np.maximum(rho_tpll, 0.1)
        topo_ll = np.random.randn(Lat, Lon) * 50.0
        
        x1f = np.linspace(0., 5000., 10)
        x2f = np.linspace(-1000., 1000., 6)
        x3f = np.linspace(-2000., 2000., 8)
        
        planet_grav = 9.81
        planet_radius = 6371.e3
        
        # Regrid multiple variables
        results = regrid_multiple_variables(
            variables, rho_tpll, topo_ll,
            plev, lats, lons,
            x1f, x2f, x3f,
            planet_grav, planet_radius,
            bounds_error=False,
            n_jobs=1
        )
        
        # Check that all variables are present
        self.assertEqual(set(results.keys()), {'temperature', 'humidity', 'pressure'})
        
        # Check output shapes
        expected_shape = (T, len(x1f), len(x2f), len(x3f))
        for var_name, result in results.items():
            self.assertEqual(result.shape, expected_shape, 
                           f"Variable {var_name} has incorrect shape")
    
    def test_multiple_variables_parallel_vs_sequential(self):
        """Test that parallel and sequential multi-variable regridding produce same results."""
        import multiprocessing
        
        # Create test data
        T, P, Lat, Lon = 2, 3, 5, 6
        plev = np.array([100000., 70000., 40000.])  # Pa
        lats = np.linspace(30.0, 35.0, Lat)
        lons = np.linspace(-110.0, -105.0, Lon)
        
        temp_tpll = 280.0 + np.random.randn(T, P, Lat, Lon) * 10.0
        humid_tpll = 0.5 + np.random.randn(T, P, Lat, Lon) * 0.1
        
        variables = {
            'temperature': temp_tpll,
            'humidity': humid_tpll,
        }
        
        rho_tpll = 1.0 + 0.1 * np.random.randn(T, P, Lat, Lon)
        rho_tpll = np.maximum(rho_tpll, 0.1)
        topo_ll = np.random.randn(Lat, Lon) * 50.0
        
        x1f = np.linspace(0., 5000., 8)
        x2f = np.linspace(-1000., 1000., 5)
        x3f = np.linspace(-2000., 2000., 6)
        
        planet_grav = 9.81
        planet_radius = 6371.e3
        
        # Sequential variable processing
        results_seq = regrid_multiple_variables(
            variables, rho_tpll, topo_ll,
            plev, lats, lons,
            x1f, x2f, x3f,
            planet_grav, planet_radius,
            bounds_error=False,
            n_jobs=1
        )
        
        # Parallel variable processing
        n_workers = min(2, multiprocessing.cpu_count())
        results_par = regrid_multiple_variables(
            variables, rho_tpll, topo_ll,
            plev, lats, lons,
            x1f, x2f, x3f,
            planet_grav, planet_radius,
            bounds_error=False,
            n_jobs=n_workers
        )
        
        # Results should be identical
        for var_name in variables.keys():
            np.testing.assert_array_almost_equal(
                results_seq[var_name], results_par[var_name], decimal=10,
                err_msg=f"Variable {var_name} differs between sequential and parallel"
            )
    
    def test_multiple_variables_with_precomputed_heights(self):
        """Test that pre-computed heights work with multiple variables."""
        T, P, Lat, Lon = 2, 3, 4, 5
        plev = np.array([100000., 70000., 40000.])
        lats = np.linspace(30.0, 35.0, Lat)
        lons = np.linspace(-110.0, -105.0, Lon)
        
        temp_tpll = 280.0 + np.random.randn(T, P, Lat, Lon) * 10.0
        humid_tpll = 0.5 + np.random.randn(T, P, Lat, Lon) * 0.1
        
        variables = {
            'temperature': temp_tpll,
            'humidity': humid_tpll,
        }
        
        rho_tpll = 1.0 + 0.1 * np.random.randn(T, P, Lat, Lon)
        rho_tpll = np.maximum(rho_tpll, 0.1)
        topo_ll = np.random.randn(Lat, Lon) * 50.0
        
        x1f = np.linspace(0., 5000., 8)
        x2f = np.linspace(-1000., 1000., 5)
        x3f = np.linspace(-2000., 2000., 6)
        
        planet_grav = 9.81
        planet_radius = 6371.e3
        
        # Pre-compute heights
        z_tpll = compute_height_grid(rho_tpll, topo_ll, plev, planet_grav)
        
        # Regrid with pre-computed heights
        results = regrid_multiple_variables(
            variables, rho_tpll, topo_ll,
            plev, lats, lons,
            x1f, x2f, x3f,
            planet_grav, planet_radius,
            bounds_error=False,
            z_tpll=z_tpll,
            n_jobs=1
        )
        
        # Should complete without errors and have correct shapes
        expected_shape = (T, len(x1f), len(x2f), len(x3f))
        for var_name, result in results.items():
            self.assertEqual(result.shape, expected_shape)


class TestParallelProcessing(unittest.TestCase):
    """Test cases for parallel processing functionality."""
    
    def test_vertical_interp_parallel_vs_sequential(self):
        """Test that parallel vertical interpolation produces same results as sequential."""
        import multiprocessing
        
        # Create test data
        T, Lat, Lon, P = 3, 10, 12, 8
        z_tllp = np.random.randn(T, Lat, Lon, P) * 1000.0 + 5000.0
        z_tllp = np.sort(z_tllp, axis=-1)  # Ensure monotonic increasing
        v_tllp = 280.0 + np.random.randn(T, Lat, Lon, P) * 10.0
        z_out = np.linspace(0., 10000., 20)
        
        # Sequential execution
        result_sequential = vertical_interp_to_z(z_tllp, v_tllp, z_out, bounds_error=False, n_jobs=1)
        
        # Parallel execution (use available CPUs, but at least 2 if possible)
        n_workers = min(2, multiprocessing.cpu_count())
        result_parallel = vertical_interp_to_z(z_tllp, v_tllp, z_out, bounds_error=False, n_jobs=n_workers)
        
        # Results should be identical (accounting for floating point precision)
        np.testing.assert_array_almost_equal(result_sequential, result_parallel, decimal=10)
        
    def test_horizontal_regrid_parallel_vs_sequential(self):
        """Test that parallel horizontal regridding produces same results as sequential."""
        import multiprocessing
        
        # Create test data
        T, P, Lat, Lon = 2, 4, 8, 10
        plev = np.array([100000., 80000., 60000., 40000.])  # Pa
        lats = np.linspace(30.0, 35.0, Lat)
        lons = np.linspace(-110.0, -105.0, Lon)
        
        var_tpll = 280.0 + np.random.randn(T, P, Lat, Lon) * 10.0
        rho_tpll = 1.0 + 0.1 * np.random.randn(T, P, Lat, Lon)
        rho_tpll = np.maximum(rho_tpll, 0.1)
        topo_ll = np.random.randn(Lat, Lon) * 50.0
        
        x1f = np.linspace(0., 5000., 10)
        x2f = np.linspace(-1000., 1000., 6)
        x3f = np.linspace(-2000., 2000., 8)
        
        planet_grav = 9.81
        planet_radius = 6371.e3
        
        # Sequential execution
        result_sequential = regrid_pressure_to_height(
            var_tpll, rho_tpll, topo_ll,
            plev, lats, lons,
            x1f, x2f, x3f,
            planet_grav, planet_radius,
            bounds_error=False,
            n_jobs=1
        )
        
        # Parallel execution (use available CPUs, but at least 2 if possible)
        n_workers = min(2, multiprocessing.cpu_count())
        result_parallel = regrid_pressure_to_height(
            var_tpll, rho_tpll, topo_ll,
            plev, lats, lons,
            x1f, x2f, x3f,
            planet_grav, planet_radius,
            bounds_error=False,
            n_jobs=n_workers
        )
        
        # Results should be identical (accounting for floating point precision)
        np.testing.assert_array_almost_equal(result_sequential, result_parallel, decimal=10)
        
    def test_auto_parallelization(self):
        """Test that auto parallelization works correctly."""
        # Small dataset - should use sequential
        T, Lat, Lon, P = 1, 5, 5, 4
        z_tllp = np.random.randn(T, Lat, Lon, P) * 1000.0 + 5000.0
        z_tllp = np.sort(z_tllp, axis=-1)
        v_tllp = 280.0 + np.random.randn(T, Lat, Lon, P) * 10.0
        z_out = np.linspace(0., 10000., 10)
        
        # Auto mode (should choose sequential for small data)
        result_auto = vertical_interp_to_z(z_tllp, v_tllp, z_out, bounds_error=False, n_jobs=None)
        
        # Verify it produces valid results
        self.assertEqual(result_auto.shape, (T, Lat, Lon, len(z_out)))
        
    def test_all_cpus_option(self):
        """Test that n_jobs=-1 uses all available CPUs."""
        T, Lat, Lon, P = 2, 8, 8, 5
        z_tllp = np.random.randn(T, Lat, Lon, P) * 1000.0 + 5000.0
        z_tllp = np.sort(z_tllp, axis=-1)
        v_tllp = 280.0 + np.random.randn(T, Lat, Lon, P) * 10.0
        z_out = np.linspace(0., 10000., 15)
        
        # Sequential execution
        result_sequential = vertical_interp_to_z(z_tllp, v_tllp, z_out, bounds_error=False, n_jobs=1)
        
        # All CPUs
        result_all_cpus = vertical_interp_to_z(z_tllp, v_tllp, z_out, bounds_error=False, n_jobs=-1)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(result_sequential, result_all_cpus, decimal=10)

class TestSaveToNetCDF(unittest.TestCase):
    """Test cases for NetCDF saving functions."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_regridded_data_basic(self):
        """Test basic save of regridded atmospheric data."""
        # Create test data
        T, Z, Y, X = 3, 5, 4, 6
        temp_data = 280.0 + np.random.randn(T, Z, Y, X) * 10.0
        rho_data = 1.0 + np.random.randn(T, Z, Y, X) * 0.1
        
        variables = {
            'temperature': temp_data,
            'density': rho_data,
        }
        
        coordinates = {
            'time': np.arange(T, dtype=float),
            'x1f': np.linspace(0., 5000., Z),
            'x2f': np.linspace(-1000., 1000., Y),
            'x3f': np.linspace(-2000., 2000., X),
        }
        
        metadata = {
            'source': 'Test data',
            'temperature_units': 'K',
            'temperature_long_name': 'Air Temperature',
            'density_units': 'kg/m^3',
            'density_long_name': 'Air Density',
        }
        
        filename = os.path.join(self.temp_dir, 'test_regridded.nc')
        
        # Save data
        save_regridded_data_to_netcdf(filename, variables, coordinates, metadata)
        
        # Verify file was created
        self.assertTrue(os.path.exists(filename))
        
        # Try to load and verify (if netCDF4 is available)
        try:
            from netCDF4 import Dataset
            with Dataset(filename, 'r') as ncfile:
                # Check dimensions
                self.assertEqual(len(ncfile.dimensions['time']), T)
                self.assertEqual(len(ncfile.dimensions['x1']), Z)
                self.assertEqual(len(ncfile.dimensions['x2']), Y)
                self.assertEqual(len(ncfile.dimensions['x3']), X)
                
                # Check variables exist
                self.assertIn('temperature', ncfile.variables)
                self.assertIn('density', ncfile.variables)
                
                # Check coordinate variables
                self.assertIn('x1', ncfile.variables)
                self.assertIn('x2', ncfile.variables)
                self.assertIn('x3', ncfile.variables)
                
                # Check units
                self.assertEqual(ncfile.variables['x1'].units, 'meters')
                self.assertEqual(ncfile.variables['x2'].units, 'meters')
                self.assertEqual(ncfile.variables['x3'].units, 'meters')
                
                # Check global attributes
                self.assertIn('history', ncfile.ncattrs())
                self.assertIn('source', ncfile.ncattrs())
                self.assertEqual(ncfile.source, 'Test data')
        except ImportError:
            # Skip verification if netCDF4 not available
            pass
    
    def test_save_topography_basic(self):
        """Test basic save of topography data."""
        Y, X = 10, 15
        topo_data = 1000.0 + np.random.randn(Y, X) * 100.0
        x2f = np.linspace(-5000., 5000., Y)
        x3f = np.linspace(-7500., 7500., X)
        
        metadata = {
            'source': 'Test topography',
            'region': 'Test region',
        }
        
        filename = os.path.join(self.temp_dir, 'test_topography.nc')
        
        # Save data
        save_topography_to_netcdf(filename, topo_data, x2f, x3f, metadata)
        
        # Verify file was created
        self.assertTrue(os.path.exists(filename))
        
        # Try to load and verify
        try:
            from netCDF4 import Dataset
            with Dataset(filename, 'r') as ncfile:
                # Check dimensions
                self.assertEqual(len(ncfile.dimensions['x2']), Y)
                self.assertEqual(len(ncfile.dimensions['x3']), X)
                
                # Check variables
                self.assertIn('topography', ncfile.variables)
                self.assertEqual(ncfile.variables['topography'].units, 'meters')
                
                # Check global attributes
                self.assertIn('history', ncfile.ncattrs())
                self.assertEqual(ncfile.source, 'Test topography')
        except ImportError:
            pass
    
    def test_save_with_processing_history(self):
        """Test save with custom processing history."""
        T, Z, Y, X = 2, 3, 4, 5
        variables = {'temperature': np.ones((T, Z, Y, X)) * 280.0}
        coordinates = {
            'time': np.arange(T, dtype=float),
            'x1f': np.linspace(0., 1000., Z),
            'x2f': np.linspace(-500., 500., Y),
            'x3f': np.linspace(-750., 750., X),
        }
        
        processing_history = "2024-01-01: Regridded from ERA5 data using custom pipeline"
        
        filename = os.path.join(self.temp_dir, 'test_history.nc')
        save_regridded_data_to_netcdf(
            filename, variables, coordinates, processing_history=processing_history
        )
        
        self.assertTrue(os.path.exists(filename))
        
        try:
            from netCDF4 import Dataset
            with Dataset(filename, 'r') as ncfile:
                self.assertEqual(ncfile.history, processing_history)
        except ImportError:
            pass


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestComputeDzFromPlev))
    suite.addTests(loader.loadTestsFromTestCase(TestComputeHeightsFromDz))
    suite.addTests(loader.loadTestsFromTestCase(TestComputeHeightGrid))
    suite.addTests(loader.loadTestsFromTestCase(TestLatlonToXy))
    suite.addTests(loader.loadTestsFromTestCase(TestVerticalInterpToZ))
    suite.addTests(loader.loadTestsFromTestCase(TestHorizontalRegridXy))
    suite.addTests(loader.loadTestsFromTestCase(TestRegridPressureToHeight))
    suite.addTests(loader.loadTestsFromTestCase(TestRegridTopography))
    suite.addTests(loader.loadTestsFromTestCase(TestMultipleVariables))
    suite.addTests(loader.loadTestsFromTestCase(TestParallelProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestSaveToNetCDF))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)

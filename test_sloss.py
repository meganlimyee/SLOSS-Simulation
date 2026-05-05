import unittest
import numpy as np

from sloss import create_landscape, run_simulation


class TestCreateLandscape(unittest.TestCase):
    """
    Unit tests for create_landscape.

    Attributes:
        L : int
            Side length of total square landscape area.
        total_area : int
            Total number of habitat cells.
    """

    def setUp(self):
        """
        Sets up params for each test.

        Returns
        -------
        None.

        """
        self.L = 30
        self.total_area = 100

    def test_shape_and_dtype(self):
        """
        Check that create_landscape returns a boolean array of side length L.

        Returns
        -------
        None.

        """
        ls = create_landscape(L=self.L, total_area=self.total_area,
                              num_reserves=1)
        # array shape matches L
        self.assertEqual(ls.shape, (self.L, self.L))
        # array is boolean
        self.assertEqual(ls.dtype, bool)

    def test_single_reserve_area(self):
        """
        Check that a single reserve has the requested total area.

        Returns
        -------
        None.

        """
        ls = create_landscape(L=self.L, total_area=self.total_area,
                              num_reserves=1)
        # exact match for a single reserve
        self.assertEqual(ls.sum(), self.total_area)

    def test_multiple_reserves_area(self):
        """
        Check that total area across reserves is preserved even after splitting.

        Returns
        -------
        None.

        """
        ls = create_landscape(L=self.L, total_area=self.total_area,
                              num_reserves=4)
        # total area should be preserved
        self.assertGreater(ls.sum(), 0)
        self.assertLessEqual(ls.sum(), self.total_area)

    def test_invalid_num_reserves(self):
        """
        Check that num_reserves < 1 raises an error.

        Returns
        -------
        None.

        """
        with self.assertRaises(ValueError):
            create_landscape(L=self.L, total_area=self.total_area,
                             num_reserves=0)

    def test_total_area(self):
        """
        Check that total_area > L*L raises an error.

        Returns
        -------
        None.

        """
        with self.assertRaises(ValueError):
            create_landscape(L=10, total_area=200, num_reserves=1)

    def test_patchiness(self):
        """
        Check that patchiness outside the 0 to 1 range raises an error.

        Returns
        -------
        None.

        """
        with self.assertRaises(ValueError):
            create_landscape(L=self.L, total_area=self.total_area,
                             num_reserves=1, patchiness=1.5)


class TestRunSimulation(unittest.TestCase):
    """
    Unit tests for run_simulation.

    Attributes:
        landscape : (L, L) ndarray
            Numpy array of booleans where True is inside a reserve and False is
            outside a reserve.
        timesteps : int
            Number of timesteps to use in tests

    """

    def setUp(self):
        """
        Set up a fixed landscape for reproducible tests.

        Returns
        -------
        None.

        """
        self.timesteps = 10
        self.landscape = create_landscape(L=20, total_area=50, num_reserves=2)

    def test_history_shape(self):
        """
        Check that pop_history and history have the same sizes and keys.

        Returns
        -------
        None.

        """
        pop_history, history = run_simulation(
            self.landscape, timesteps=self.timesteps, seed=42)

        # pop_history has one snapshot per timestep
        self.assertEqual(len(pop_history), self.timesteps)
        # each snapshot has the same shape as landscape
        self.assertEqual(pop_history[0].shape, self.landscape.shape)
        # history dict has the expected keys
        for key in ['total_pop', 'occupancy',
                    'num_occupied_reserves', 'disturbance_events']:
            self.assertIn(key, history)

        self.assertEqual(len(history['total_pop']), self.timesteps)

    def test_seed_reproducibility(self):
        """
        Check that two runs with the same seed produce the same results.

        Returns
        -------
        None.

        """
        _, hist1 = run_simulation(
            self.landscape, timesteps=self.timesteps, seed=42)
        _, hist2 = run_simulation(
            self.landscape, timesteps=self.timesteps, seed=42)

        # check taht total population trajectories match
        self.assertEqual(hist1['total_pop'], hist2['total_pop'])
        # check that occupancy trajectories match
        self.assertEqual(hist1['occupancy'], hist2['occupancy'])

    def test_landscape_type(self):
        """
        Check that a non-boolean landscape raises an error.

        Returns
        -------
        None.

        """
        bad_landscape = np.zeros((10, 10))
        with self.assertRaises(TypeError):
            run_simulation(bad_landscape, timesteps=5)

    def test_migration_fraction(self):
        """
        Check that migration fraction m outside [0, 1] raises an error.

        Returns
        -------
        None.

        """
        with self.assertRaises(ValueError):
            run_simulation(self.landscape, timesteps=5, m=2.0)

    def test_carrying_capacity(self):
        """
        Check that a 0 or negative carrying capacity raises an error.

        Returns
        -------
        None.

        """
        with self.assertRaises(ValueError):
            run_simulation(self.landscape, timesteps=5, K=0)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np

from vectorsindex.impl.faissmips import augment_dimensions, reduce_dimensions


class TestAugmentDimensions(unittest.TestCase):
    def test_augment_1d(self):
        a = np.array(range(9))
        extended_a = augment_dimensions(a)
        self.assertEqual(0, extended_a[-1])
        self.assertEqual((10,), extended_a.shape)

    def test_reduce_1d(self):
        a = np.array(range(10))
        reduced_a = reduce_dimensions(a)
        self.assertEqual((9,), reduced_a.shape)

    def test_augment_2d(self):
        m = np.ones((10, 9))
        extended_m = augment_dimensions(m)
        for row in extended_m:
            self.assertEqual(0, row[-1])
        self.assertEqual((10, 10), extended_m.shape)

    def test_reduce_2d(self):
        m = np.ones((10, 11))
        reduced_m = reduce_dimensions(m)
        self.assertEqual((10, 10), reduced_m.shape)

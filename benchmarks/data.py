import numpy as np


class VectorGenerator:
    def __init__(self):
        self.seed = 0

    def points_on_sphere(self, n: int, d: int):
        """
        Generates np.array[float32] of normally distributed points on a surface of an n-dimensional sphere.
        :param n: Number of points to generate
        :param d: Dimension
        :return: points
        """
        self.seed = self.seed + 1

        deviates = np.random.RandomState(self.seed).normal(loc=0.2, scale=100, size=(n, d))
        radius = np.sqrt((deviates ** 2).sum(axis=0))
        points = deviates / radius
        return points.astype('float32')

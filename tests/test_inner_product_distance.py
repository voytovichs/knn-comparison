import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

from benchmarks.data import VectorGenerator
from vectorsindex.impl import FaissMaximalInnerProductNNIndex


def inner_product(v1, v2):
    return np.dot(v1, v2)


class TestInnerProductDistance(TestCase):
    points = VectorGenerator()

    def test_single_vector_query(self):
        index = FaissMaximalInnerProductNNIndex(10, 512, 1, Path(tempfile.mktemp('vectors123')),
                                                blocking_implementation_replacement=True)
        vector_id = 0
        vector = self.points.points_on_sphere(10, 512)[0]
        index.insert(vector_id, vector)

        query = self.points.points_on_sphere(10, 512)[1]
        result_id, distance = index.query(query)

        self.assertEqual(vector_id, result_id)
        self.assertEqual(inner_product(query, vector), distance)

    def test_batch_query(self):
        index = FaissMaximalInnerProductNNIndex(10, 512, 1, Path(tempfile.mktemp('vectors1234')),
                                                blocking_implementation_replacement=True)
        vector_id = 0
        vector = self.points.points_on_sphere(10, 512)[0]
        index.insert(vector_id, vector)

        batch_query = self.points.points_on_sphere(10, 512)[1:]
        ids, distances = index.batch_query(batch_query)

        for i in range(len(batch_query)):
            self.assertEqual(vector_id, ids[i])
            self.assertEqual(1, len(distances[i]))
            self.assertAlmostEqual(inner_product(batch_query[i], vector), distances[i][0])

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.data import VectorGenerator
from vectorsindex.impl.faiss import FaissMinimalL2DistanceNNIndex

n = 100
d = 512
k = 10


class NearestNeighborsIndexTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(NearestNeighborsIndexTestCase, self).__init__(*args, **kwargs)
        # We want this class to carry test cases without being run
        # by the unit test framework, so the `run' method is overridden to do
        # nothing.  But in order for sub-classes to be able to do something when
        # run is invoked, the constructor will rebind `run' from TestCase.
        if self.__class__ != NearestNeighborsIndexTestCase:
            # Rebind `run' from a child class.
            self.run = unittest.TestCase.run.__get__(self, self.__class__)
        else:
            self.run = lambda self, *args, **kwargs: None

        self.points = VectorGenerator()

    def get_index_instance(self):
        raise NotImplemented

    def test_len(self):
        index = self.get_index_instance()
        ids, vectors = self.__get_vectors_and_ids(5)
        for v_id, vector in zip(ids, vectors):
            index.insert(v_id, vector)
        self.assertEqual(5, len(index))

    def test_contains(self):
        index = self.get_index_instance()
        ids, vectors = self.__get_vectors_and_ids(20)
        for v_id, vector in zip(ids, vectors):
            index.insert(v_id, vector)

        self.assertTrue(ids[0] in index)
        self.assertFalse(ids[-1] + 1 in index)

    def test_nearest_search_on_an_empty_index(self):
        index = self.get_index_instance()
        _, vectors = self.__get_vectors_and_ids(5)
        # Test single query
        q_ids, q_distances = index.query(vectors[0])
        self.assertEqual(0, len(q_ids))
        self.assertEqual(0, len(q_distances))

        # Test batch query
        batch_ids, batch_distances = index.batch_query(vectors)
        self.assertEqual((5, 0), batch_ids.shape)
        self.assertEqual((5, 0), batch_distances.shape)

    def test_neighbors_search(self):
        index = self.get_index_instance()
        ids, vectors = self.__get_vectors_and_ids(20)
        for v_id, vector in zip(ids, vectors):
            index.insert(v_id, vector)

        # Test single query
        q_ids, q_distances = index.query(vectors[0])
        self.assertEqual(k, len(q_ids))
        for q in q_ids:
            isinstance(q, int)
            self.assertTrue(ids[0] <= q <= ids[-1], f'q={q}, id[0]={ids[0]}, id[last]={ids[-1]}')

        # Test batch query
        batch_ids, batch_distances = index.batch_query(np.array([vectors[0], vectors[1]]))
        self.assertEqual((2, k), batch_ids.shape)
        self.assertEqual((2, k), batch_distances.shape)

    def test_neighbors_search_with_only_two_vectors_inserted(self):
        index = self.get_index_instance()
        ids, vectors = self.__get_vectors_and_ids(2)
        for v_id, vector in zip(ids, vectors):
            index.insert(v_id, vector)

        # Test single query
        q_ids, q_distances = index.query(vectors[0])
        self.assertEqual(2, len(q_ids))
        for q in q_ids:
            isinstance(q, int)
            self.assertTrue(ids[0] <= q <= ids[-1], f'q={q}, id[0]={ids[0]}, id[last]={ids[-1]}')

        # Test batch query
        batch_ids, batch_distances = index.batch_query(np.array([vectors[0], vectors[1]]))
        self.assertEqual((2, 2), batch_ids.shape)
        self.assertEqual((2, 2), batch_distances.shape)

    def test_neighbors_search_distances(self):
        index = self.get_index_instance()
        ids, vectors = self.__get_vectors_and_ids(20)
        for v_id, vector in zip(ids, vectors):
            index.insert(v_id, vector)

        query = vectors[0]
        q_ids, q_distances = index.query(query)
        for q_id, actual_dist in zip(q_ids, q_distances):
            expected_dist = self.__calculate_l2_distance(vectors[q_id], query)
            self.assertAlmostEqual(expected_dist, actual_dist, delta=0.001)

    def __calculate_l2_distance(self, v1, v2) -> float:
        dist = [(a - b) ** 2 for a, b in zip(v1, v2)]
        dist = math.sqrt(sum(dist))
        return dist

    def __get_vectors_and_ids(self, size: int):
        vectors = self.points.points_on_sphere(size, d)
        ids = np.arange(0, size, dtype='int64')
        return ids, vectors


class FaissNearestNeighborsIndexTest(NearestNeighborsIndexTestCase):
    def get_index_instance(self):
        return FaissMinimalL2DistanceNNIndex(n, d, k, Path(tempfile.mktemp('vectors')), blocking_implementation_replacement=True)


# todo: Make tests pass, uncomment them later
# class BruteForceNearestNeighborsIndexTest(NearestNeighborsIndexTestCase):
#     def get_index_instance(self):
#         return BruteForceKNN(d, k)

# todo: Make tests pass, uncomment them later
# class AnnoyNearestNeighborsIndexTest(NearestNeighborsIndexTestCase):
#     def get_index_instance(self):
#         return AnnoyNNIndex(d, k)

# todo: Make tests pass, uncomment them later
# class SKLearnNearestNeighborsIndexTest(NearestNeighborsIndexTestCase):
#     def get_index_instance(self):
#         return SKLearnNNIndex(d, k)


if __name__ == '__main__':
    unittest.main()

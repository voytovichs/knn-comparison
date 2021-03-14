import time as t

import numpy as np
import os
import psutil

from pympler import asizeof
from itertools import starmap

from benchmarks.data import VectorGenerator
from benchmarks.format import format_time, format_bytes
from vectorsindex.impl.bruteforce import BruteForceKNN
from vectorsindex.abstract.abstract import NearestNeighborsIndex


class NNBenchmark:
    """
    Stateless benchmark for an arbitrary
    'NearestNeighborsIndex' implementations
    """

    def __init__(self, dimension, k, vectors_size, query_size):
        self.dimension = dimension
        self.k = k
        self.vectors_size = vectors_size
        self.query_size = query_size
        self.points = VectorGenerator()

    def execute(self, index: NearestNeighborsIndex):
        print(f'{index} implementation')
        ids = np.arange(0, self.vectors_size, dtype='int64')

        vectors = self.points.points_on_sphere(self.vectors_size, self.dimension)
        queries = self.points.points_on_sphere(self.query_size, self.dimension)

        # Building index
        self.__build_index(index, ids, vectors)
        vectors_size = asizeof.asizeof(vectors)
        print('Vectors take {}'.format(format_bytes(vectors_size)))

        # Execute nearest neighbors search
        approximate = self.__execute_queries(index, queries)
        exact = self.__execute_exact_nearest_neighbors_search(ids, vectors, queries)

        self.__measure_average_recall(vectors, queries, approximate, exact, 0.005)

    def __build_index(self, index: NearestNeighborsIndex, ids, vectors):
        print('trace: Start indexing vectors')
        mem_before_indexing = self.__get_used_memory()
        indexing_start = t.time_ns()
        for i, v in zip(ids, vectors):
            index.insert(i, v)
        indexing_end = t.time_ns()

        print('Indexing of {0} vectors takes {1}'
              .format(self.vectors_size, format_time(indexing_end - indexing_start)))

        memory = self.__get_used_memory() - mem_before_indexing

        print(f'Index occupies {format_bytes(memory)}')

    def __execute_queries(self, index: NearestNeighborsIndex, queries):
        print('trace: Start running knn queries')

        querying_start = t.time_ns()
        ids, _ = index.batch_query(queries)
        querying_end = t.time_ns()

        print('{0} queries of {1} nearest neighbors takes {2}'
              .format(self.query_size, self.k, format_time(querying_end - querying_start)))

        return ids

    def __execute_exact_nearest_neighbors_search(self, ids, vectors, queries):
        index = BruteForceKNN(self.dimension, self.k)
        index.batch_insert(ids, vectors)
        return index.batch_query(queries)

    def __measure_average_recall(self, vectors, queries, approximate_results, exact_results, eps: float = 0):
        print('trace: measure recall')
        assert approximate_results.shape == exact_results.shape, \
            f'Shapes of approximate and exact nearest neighbors results are different ' \
            f'({approximate_results.shape} vs {exact_results.shape})'

        recall = []

        for q, exact, approximate in zip(queries, exact_results, approximate_results):
            recall.append(self.__calculate_recall(q, exact, approximate, vectors, eps))

        _min = min(recall)
        _median = np.median(recall)
        _max = max(recall)
        print(f'Computed recall: median={_median}, min={_min}, max={_max}')

    def __calculate_recall(self, q, exact, approximate, vectors, eps: float) -> float:
        most_distant_neighbor = max(map(lambda i: self.__l2_squared_distance(vectors[i], q), exact))

        s = sum(1 for i in approximate if self.__l2_squared_distance(vectors[int(i)], q) <=
                most_distant_neighbor * (1 + eps))

        return s / len(exact)

    @staticmethod
    def __l2_squared_distance(x, y) -> float:
        return sum(starmap(lambda a, b: (a - b) * (a - b), zip(x, y)))

    @staticmethod
    def __get_used_memory():
        return psutil.Process(os.getpid()).memory_info().rss

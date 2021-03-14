import numpy as np
from sklearn.neighbors import NearestNeighbors

from vectorsindex.abstract.abstract import NearestNeighborsIndex


class BruteForceKNN(NearestNeighborsIndex):
    def __init__(self, dimensions: int, k: int):
        super().__init__(dimensions, k)
        self.__index = NearestNeighbors(n_neighbors=k, algorithm='brute')

    def __contains__(self, vector_id):
        # todo: use registry
        raise NotImplemented

    def __len__(self) -> int:
        raise NotImplemented

    def insert(self, vector_id: np.int64, vector):
        # todo: use vector buffer
        self.__index.fit(X=np.array([vector]), y=np.array([vector_id]))

    def batch_insert(self, ids, vectors):
        self.__index.fit(X=vectors, y=ids)

    def query(self, vector):
        return self.__index.kneighbors(X=np.array([vector]), n_neighbors=self.k, return_distance=False)

    def batch_query(self, vectors):
        return np.array(self.__index.kneighbors(X=vectors, n_neighbors=self.k, return_distance=False))

    def __repr__(self) -> str:
        return 'brute-force'

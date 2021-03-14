import numpy as np
from sklearn.neighbors import NearestNeighbors

from vectorsindex.abstract.abstract import NearestNeighborsIndex


class SKLearnNNIndex(NearestNeighborsIndex):
    """
    Nearest neighbors index implementation using a kdtree from sklearn library.
    """

    def __init__(self, dimensions: int, k: int):
        super().__init__(dimensions, k)
        self.__neighbors = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')

    def __contains__(self, vector_id):
        # todo: it needs a registry
        raise NotImplemented

    def __len__(self) -> int:
        raise NotImplemented

    def insert(self, vector_id: np.int64, vector):
        # todo: it needs a vector buffer
        self.__neighbors.fit(X=np.array([vector]), y=np.array([vector_id]))

    def query(self, vector):
        raise NotImplemented

    def batch_query(self, vectors):
        return self.__neighbors.kneighbors(X=vectors, n_neighbors=self.k, return_distance=False)

    def __repr__(self) -> str:
        return 'sklearn/kdtree'

from pathlib import Path

import numpy as np

from vectorsindex.impl import FaissMinimalL2DistanceNNIndex


def augment_dimensions(vectors: np.ndarray) -> np.ndarray:
    """
    Modifies vectors originally build for maximum inner product search
    to enable minimum L2 distance searching
    :param vectors: original vectors
    :return: modified vectors with increased dimensionality
    """
    if len(vectors.shape) == 1:
        return np.append(vectors, 0).astype('float32')
    shape = (vectors.shape[0], 1)
    return np.append(vectors, np.zeros(shape), axis=1).astype('float32')


def reduce_dimensions(vectors: np.ndarray) -> np.ndarray:
    """
    Cuts auxiliary item
    :param vectors: original vectors
    :return: modified vectors
    """
    if len(vectors.shape) == 1:
        return vectors[:-1]
    return vectors[:, :-1]


def inner_product(v1, v2):
    return np.dot(v1, v2)


class FaissMaximalInnerProductNNIndex(FaissMinimalL2DistanceNNIndex):
    """
    Turns FaissMinimalL2DistanceNNIndex into a maximum inner product search.
    """

    def __init__(self, vectors_limit: int,
                 dimensions: int,
                 k: int,
                 vectors_metadata: Path,
                 blocking_implementation_replacement: bool = False):
        super().__init__(vectors_limit, dimensions + 1, k, vectors_metadata, blocking_implementation_replacement)

    def insert(self, vector_id, vector):
        super().insert(vector_id, augment_dimensions(vector))

    def query(self, query):
        ids, _ = super().query(augment_dimensions(query))
        distances = []
        for vector_id in ids:
            distances.append(inner_product(query, reduce_dimensions(super().get_vector(vector_id))))
        return ids, np.array(distances, dtype='float32')

    def batch_query(self, query):
        ids, _ = super().batch_query(augment_dimensions(query))
        distances = np.ndarray(shape=ids.shape, dtype='float32')
        for i in range(len(ids)):
            v1 = query[i]
            for j in range(len(ids[i])):
                v2 = reduce_dimensions(super().get_vector(ids[i][j]))
                distances[i][j] = inner_product(v1, v2)
        return ids, distances

    def __repr__(self) -> str:
        return str(super()) + '(MIPS)'

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from vectorsindex.abstract.config import Config


class NearestNeighborsIndex(ABC):
    @abstractmethod
    def __init__(self, dimensions: int, k: int):
        """
        A receiver of parameters at object's creation time
        :param dimensions: vectors dimension
        :param k: number of nearest neighbors to return
        """
        self.dimensions = dimensions
        self.k = k

    def validate_vector(self, vector_id, vector):
        if (len(self) + 1) > Config.stored_vectors_limit:
            raise ValueError('Maximal storage capacity exceeded')

        if not isinstance(vector_id, int) and not isinstance(vector_id, np.int64):
            raise ValueError('vector_id type must be int or np.int64')

        if len(vector) != self.dimensions:
            raise ValueError(f'Unexpected vector length: {len(vector)}')

    @abstractmethod
    def insert(self, vector_id, vector):
        """
        Insert single vector to the index
        :param vector_id: ID_TYPE id of a vector
        :param vector: np.array[np.float32] a vector to insert to the index
        """

    @abstractmethod
    def query(self, vector):
        """
        Return a pair of (n nearest neighbors, distances) for a given vector
        :param vector: np.array[np.float32] self.dim-dimensional vector
        :return: Tuple[np.array[ID_TYPE], np.array[float32]] ids and distances of k nearest neighbors
        """
        ...

    @abstractmethod
    def batch_query(self, vectors):
        """
        Return n nearest neighbors for a batch query
        :param vectors: np.array[np.array[np.float32]] self.dim-dimensional vectors
        :return: [np.array[Tuple[np.array[ID_TYPE], np.array[float32]] n x k ids and distances
                 to nearest neighbors
        """
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """
        Return an identifier of particular implementation
        :return: human readable name
        """
        ...

    @abstractmethod
    def __contains__(self, vector_id) -> bool:
        """
        Check whether a particular vector was indexed.
        :param vector_id: ID_TYPE ID of a vector
        :return: True if a vector with a give ID was indexed. False otherwise.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of indexed vectors
        """
        ...


class VectorsSpace(Enum):
    L2Space = 'L2Space'
    InnerProductSpace = 'InnerProductSpace'


# Errors

class NearestNeighborsIndexException(Exception):
    def __init__(self, message: str):
        self.message = message

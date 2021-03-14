import numpy as np
from annoy import AnnoyIndex

from vectorsindex.abstract.abstract import NearestNeighborsIndex


class AnnoyNNIndex(NearestNeighborsIndex):
    def __init__(self, dimensions: int, k: int):
        super().__init__(dimensions, k)
        self.__index = AnnoyIndex(self.dimensions)

    def __contains__(self, vector_id):
        # use registry
        raise NotImplemented

    def __len__(self) -> int:
        pass

    # todo: use vector buffer
    def insert(self, vector_id: np.int64, vector):
        # trees_numbers = 40
        #
        # for v_id, v in zip(ids, vectors):
        #     self.__index.add_item(v_id, v)
        #
        # self.__index.build(trees_numbers)
        raise NotImplemented

    def query(self, vector):
        return self.__index.get_nns_by_vector(vector=vector, n=self.k, include_distances=False)

    def batch_query(self, vectors):
        result = []
        for v in vectors:
            result.append(self.__index.get_nns_by_vector(vector=v, n=self.k, include_distances=False))
        return np.array(result, dtype='float32')

    @property
    def __repr__(self) -> str:
        return "annoy"

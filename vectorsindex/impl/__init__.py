import atexit

from injector import Binder

from vectorsindex.abstract.abstract import VectorsSpace, NearestNeighborsIndex
from vectorsindex.abstract.config import Config
from vectorsindex.impl.faiss import FaissMinimalL2DistanceNNIndex
from vectorsindex.impl.faissmips import FaissMaximalInnerProductNNIndex
from vectorsindex.util.ids import StringIDMapper, StringIDNearestNeighborsIndex, StatefulIDMapper


def get_vectors_impl():
    if VectorsSpace(Config.vectors_space) == VectorsSpace.L2Space:
        return FaissMinimalL2DistanceNNIndex(Config.stored_vectors_limit, Config.vectors_dimension,
                                             Config.nearest_neighbors_number, Config.vectors_metadata)
    if VectorsSpace(Config.vectors_space) == VectorsSpace.InnerProductSpace:
        return FaissMaximalInnerProductNNIndex(Config.stored_vectors_limit, Config.vectors_dimension,
                                               Config.nearest_neighbors_number, Config.vectors_metadata)
    raise ValueError(f'Unexpected similarity metrics value: {Config.vectors_space}')


def bind_index_implementation(binder: Binder):
    mapper = StatefulIDMapper(Config.vectors_metadata, Config.stored_vectors_limit)
    binder.bind(StringIDMapper, mapper)
    vectors_index = get_vectors_impl()
    atexit.register(vectors_index.shutdown_thread_executor)
    binder.bind(NearestNeighborsIndex, StringIDNearestNeighborsIndex(vectors_index,
                                                                     mapper,
                                                                     Config.vectors_dimension,
                                                                     Config.nearest_neighbors_number))

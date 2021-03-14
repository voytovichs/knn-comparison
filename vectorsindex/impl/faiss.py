from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
from readerwriterlock.rwlock import RWLockRead

from vectorsindex.abstract.abstract import NearestNeighborsIndex
from vectorsindex.util.buffer import VectorsBuffer


class FaissIndexProvider(ABC):
    vectors_limit: int

    @staticmethod
    @abstractmethod
    def new_instance(dimensions: int) -> faiss.Index:
        ...


class FaissFlatIndexProvider(FaissIndexProvider):
    vectors_limit = 300_000

    @staticmethod
    def new_instance(dimensions: int) -> faiss.Index:
        return faiss.IndexIDMap(faiss.IndexFlatL2(dimensions))


class FaissFlatIVFIndexProvider(FaissIndexProvider):
    vectors_limit = 500_000

    @staticmethod
    def new_instance(dimensions: int) -> faiss.Index:
        return faiss.index_factory(dimensions, 'IVF4196,Flat')


class FaissPQIndexProvider(FaissIndexProvider):
    vectors_limit = 2_000_000

    @staticmethod
    def new_instance(dimensions: int) -> faiss.Index:
        return faiss.IndexIDMap(faiss.IndexPQ(dimensions, 8, 8))


class FaissPQIVFIndexProvider(FaissIndexProvider):
    vectors_limit = 30_000_000

    @staticmethod
    def new_instance(dimensions: int) -> faiss.Index:
        return faiss.index_factory(dimensions, 'OPQ16_64,IVF16384_HNSW32,PQ8')


class FaissIndexImplementationProvider:
    def __init__(self, vectors_limit: int, new_instance: '() -> faiss.Index'):
        self.vectors_limit = vectors_limit
        self.new_instance = new_instance


# noinspection PyArgumentList
class FaissIndexImplementationService:
    def __init__(self,
                 dimensions: int,
                 vectors_buffer: VectorsBuffer,
                 replace_implementation_lock,
                 blocking_implementation_replacement: bool):
        self.dim = dimensions
        self.buffer = vectors_buffer

        self.__replace_index_implementation_lock = replace_implementation_lock
        self.__indexing_lock = Lock()
        self.__blocking_implementation_replacement = blocking_implementation_replacement

        provider = min([impl for impl in FaissIndexProvider.__subclasses__()], key=lambda it: it.vectors_limit)
        self.vectors_limit = provider.vectors_limit
        self.index = provider.new_instance(self.dim)

        self.thread_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='faiss_impl_switcher')

    def build_new_index(self, provider: FaissIndexProvider):
        try:
            print(f'FaissIndexImplementationService: Started building a new index for a bigger number of vectors, '
                  f'new capacity is {provider.vectors_limit}')
            new_implementation = provider.new_instance(self.dim)

            new_implementation.train(self.buffer.vectors)
            new_implementation.add_with_ids(self.buffer.vectors, self.buffer.ids)

            with self.__replace_index_implementation_lock:
                old_index = self.index
                self.index = new_implementation
                del old_index
                self.vectors_limit = provider.vectors_limit

            print('FaissIndexImplementationService: Finished building a new index implementation')
        finally:
            self.__indexing_lock.release()

    def maybe_switch_implementation(self):
        capacity_exceeded = self.vectors_limit <= len(self.buffer)
        more_capable_implementation = min([impl for impl in FaissIndexProvider.__subclasses__() if
                                           impl.vectors_limit > len(self.buffer)],
                                          key=lambda it: it.vectors_limit, default=None)

        if capacity_exceeded and more_capable_implementation and self.__indexing_lock.acquire(blocking=False):
            future = self.thread_executor.submit(self.build_new_index, more_capable_implementation)

            if self.__blocking_implementation_replacement:
                future.result()

    def shutdown_thread_executor(self):
        self.thread_executor.shutdown()


# noinspection PyArgumentList
class FaissMinimalL2DistanceNNIndex(NearestNeighborsIndex):
    """
    FAISS vectors index that uses minimizes L2 distance to perform similarity search
    """

    def __init__(self,
                 vectors_limit: int,
                 dimensions: int,
                 k: int,
                 vectors_metadata: Path,
                 blocking_implementation_replacement: bool = False):
        super().__init__(dimensions, k)

        self.__insertion_lock = Lock()  # <- guarantees mutual exclusion in 'insert' method
        self.__query_lock = RWLockRead()  # <- implementation replacement waits all running requests using this lock

        self.buffer = VectorsBuffer(vectors_limit,
                                    dimensions,
                                    vectors_metadata)
        self.index_provider = FaissIndexImplementationService(dimensions,
                                                              self.buffer,
                                                              self.__query_lock.gen_wlock(),
                                                              blocking_implementation_replacement)
        for vector_id, vector in zip(range(len(self.buffer)), self.buffer.persistent_storage):
            self.insert_into_faiss(vector_id, vector)

    def __contains__(self, vector_id: int) -> bool:
        return vector_id < len(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return 'faiss'

    @property
    def index(self):
        return self.index_provider.index

    def insert_into_faiss(self, vector_id, vector):
        self.index.add_with_ids(np.array([vector], np.float32), np.array([vector_id], np.int64))

    def insert(self, vector_id, vector):
        with self.__insertion_lock:
            super().validate_vector(vector_id, vector)
            self.buffer.insert(vector)
            self.insert_into_faiss(vector_id, vector)

        self.index_provider.maybe_switch_implementation()

    def query(self, vector):
        with self.__query_lock.gen_rlock():
            distances, ids = self.index.search(np.array([vector]), k=self.k)

        # Take square root to return true L2 distances
        if len(self.buffer) < self.k:
            return ids[0][0:len(self.buffer)], np.sqrt(distances[0])[0:len(self.buffer)]
        return ids[0], np.sqrt(distances[0])

    def batch_query(self, vectors):
        with self.__query_lock.gen_rlock():
            distances, ids = self.index.search(vectors, k=self.k)

        # Take square root to return true L2 distances
        if len(self.buffer) < self.k:
            return ids[:, :len(self.buffer)], np.sqrt(distances[:, :len(self.buffer)])
        return ids, np.sqrt(distances)

    def get_vector(self, vector_id):
        return self.buffer.persistent_storage[vector_id]

    def shutdown_thread_executor(self):
        self.index_provider.shutdown_thread_executor()

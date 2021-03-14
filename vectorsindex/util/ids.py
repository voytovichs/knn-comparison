import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from vectorsindex.abstract.abstract import NearestNeighborsIndex, VectorsSpace


class StringIDMapper(ABC):
    @abstractmethod
    def add(self, str_id: str) -> int:
        """
        Maps a string ID into an integer.
        :param str_id: string representation of an ID
        :return: integer representation of a given ID
        """
        ...

    @abstractmethod
    def __getitem__(self, int_id: int) -> str:
        """
        Maps an integer ID back into a string.
        :param int_id: integer number representing an ID
        :return: string representation of a given ID
        """
        ...


class AtomicInteger:
    def __init__(self, value: int):
        self.__value = value
        self.__lock = threading.Lock()

    def inc(self) -> int:
        with self.__lock:
            self.__value += 1
            return self.__value

    @property
    def value(self) -> int:
        with self.__lock:
            return self.__value


class StatefulIDMapper(StringIDMapper):
    """
    StatefulIDMapper has 2 purposes:
    1. Maps real string IDs to monotonically increasing integers
    2. Persistently stores read string IDs on disk with an ability
        to restore state from a file
    """

    def __init__(self, storage_path: Path, limit: int):
        storage_path.mkdir(exist_ok=True, parents=True)
        self.storage_path = storage_path

        ids_file = storage_path / 'vector_ids.bin'
        ids_file_exists = ids_file.exists()
        self.int_to_str = np.memmap(
            str(ids_file),
            mode='r+' if ids_file_exists else 'w+',
            dtype='S24',  # <-limits the length, making it 24 symbols at most
            shape=limit
        )

        sentinel_id_value = chr(127) * 24
        if ids_file_exists:
            try:
                initial_size = np.where(self.int_to_str == sentinel_id_value.encode())[0][0]
            except (IndexError, KeyError):
                initial_size = 0
        else:
            self.int_to_str.fill(sentinel_id_value)
            self.int_to_str.flush()
            initial_size = 0

        self.counter = AtomicInteger(initial_size - 1)

    def add(self, str_id: str) -> int:
        int_id = self.counter.inc()
        self.int_to_str[int_id] = str_id
        self.int_to_str.flush()
        return int_id

    def __getitem__(self, int_id: int) -> str:
        return self.int_to_str[int_id].decode('utf-8')

    def __contains__(self, str_id: str) -> bool:
        for index in range(self.counter.value + 1):
            if self[index] == str_id:
                return True
        return False


class StringIDNearestNeighborsIndex(NearestNeighborsIndex):
    """
    This class is a wrapper for NearestNeighborsIndex. It converts
    string external string IDs, maps them to integers, delegate computation
    to NearestNeighborsIndex implementation and maps the integer IDs back.
    """

    def __init__(self, index: NearestNeighborsIndex, mapper: StringIDMapper, dim: int, k: int):
        super().__init__(dim, k)
        self.index = index
        self.mapper = mapper

    def insert(self, vector_id: str, vector):
        int_id = self.mapper.add(vector_id)
        self.index.insert(int_id, vector)

    def query(self, vector):
        ids, distances = self.index.query(vector)
        f = np.frompyfunc(self.mapper.__getitem__, 1, 1)
        return f(ids), distances

    def batch_query(self, vectors):
        ids, distances = self.index.query(vectors)
        f = np.frompyfunc(self.mapper.__getitem__, 1, 1)
        return np.apply_along_axis(f, 1, ids), distances

    def __repr__(self) -> str:
        return str(self.index)

    def __contains__(self, vector_id: str) -> bool:
        return vector_id in self.mapper

    def __len__(self) -> int:
        return len(self.index)

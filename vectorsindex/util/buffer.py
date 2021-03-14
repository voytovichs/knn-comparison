import shelve
from pathlib import Path

import numpy as np


class VectorBufferFullException(Exception):
    ...


class VectorsBuffer:
    def __init__(self, limit: int, vectors_dimensions: int, storage_path: Path):
        self.limit = limit

        storage_path.mkdir(exist_ok=True, parents=True)
        self.storage_path = storage_path

        vector_file = storage_path / 'vectors.bin'
        self.persistent_storage = np.memmap(
            str(vector_file),
            mode='r+' if vector_file.exists() else 'w+',
            dtype='float32',
            shape=(limit, vectors_dimensions)
        )
        self.metadata = shelve.open(str(storage_path / 'metadata.pkl'))

    @property
    def ids(self):
        return np.arange(0, self.size).astype('int64')

    @property
    def vectors(self):
        return self.persistent_storage[:self.size]

    def has_room_for_insert(self) -> bool:
        return self.size < self.limit

    def insert(self, vector: np.ndarray):
        if not self.has_room_for_insert():
            raise VectorBufferFullException('Maximum capacity exceeded')

        self.persistent_storage[self.size] = vector
        self.persistent_storage.flush()
        self.size += 1

    @property
    def size(self):
        return self.metadata.get('size', 0)

    @size.setter
    def size(self, value: int):
        self.metadata['size'] = value

    def __len__(self):
        return self.size

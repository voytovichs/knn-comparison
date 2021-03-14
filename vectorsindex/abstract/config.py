import os
from pathlib import Path
from typing import Optional


def read_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.getenv(name)
    if not value:
        return default
    if value.isdecimal():
        return int(value)
    else:
        raise ValueError(f'Invalid value is given for {name}, expected a decimal integer, but was {value}')


def read_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if not value:
        return default
    return value


class Config:
    # ----- Parameters
    # Constant length of indexed vectors
    vectors_dimension = read_int('VECTORS_DIMENSION', 512)

    # Metric for vectors similarity
    vectors_space = read_str('VECTORS_SPACE', 'InnerProductSpace')

    # The number of returned nearest neighbors
    nearest_neighbors_number = read_int('NEAREST_NEIGBOURS_NUMBER', 50)

    # Files to buffer inserted vectors
    vectors_metadata = Path(read_str('VECTORS_METADATA', 'vectors_metadata'))

    # ----- Implementation details
    # Maximal number of indexed vectors,
    stored_vectors_limit = 1000 if read_str('ENVIRONMENT') == 'TEST' else read_int('STORED_VECTORS_LIMIT', 30_000_000)

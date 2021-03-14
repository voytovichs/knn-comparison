import tempfile
import unittest
from pathlib import Path

import numpy as np

from vectorsindex.util.buffer import VectorsBuffer


class VectorsBufferTest(unittest.TestCase):
    def test_inserting_vector(self):
        buffer = VectorsBuffer(10, 10, Path(tempfile.mkdtemp()))
        buffer.insert(np.array([i for i in range(10)], dtype='float32'))
        self.assertEqual((1,), buffer.ids.shape)
        self.assertEqual((1, 10), buffer.vectors.shape)

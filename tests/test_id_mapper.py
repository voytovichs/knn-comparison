import tempfile
import unittest
from pathlib import Path

from vectorsindex.util.ids import StatefulIDMapper


class StatefulIDMapperTest(unittest.TestCase):
    def test_map_int_to_str(self):
        m = StatefulIDMapper(Path(tempfile.mkdtemp()), 100)
        int_id = m.add('aaaa')
        str_id = m[int_id]
        self.assertEqual('aaaa', str_id)

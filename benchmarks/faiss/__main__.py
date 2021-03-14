from benchmarks.benchmark import NNBenchmark
from benchmarks.constant import nearest_neighbors_number, dimensions, index_size, queries_size
from vectorsindex.abstract.config import Config
from vectorsindex.impl.faiss import FaissMinimalL2DistanceNNIndex

if __name__ == '__main__':
    benchmark = NNBenchmark(dimension=dimensions, k=nearest_neighbors_number, vectors_size=index_size,
                            query_size=queries_size)
    benchmark.execute(FaissMinimalL2DistanceNNIndex(30_000_000, dimensions, Config.vectors_metadata,
                                                    blocking_implementation_replacement=True))

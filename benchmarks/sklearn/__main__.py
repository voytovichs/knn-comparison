from benchmarks.benchmark import NNBenchmark
from benchmarks.constant import dimensions, nearest_neighbors_number, index_size, queries_size
from vectorsindex.impl.sklearn import SKLearnNNIndex

if __name__ == '__main__':
    benchmark = NNBenchmark(dimension=dimensions, k=nearest_neighbors_number, vectors_size=index_size,
                            query_size=queries_size)
    benchmark.execute(SKLearnNNIndex(dimensions, nearest_neighbors_number))

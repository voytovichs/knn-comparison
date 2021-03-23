# Vectors Index

A service for indexing vectors and searching for k nearest neighbors of a given vector.

## `benchmarks`
### Description
Module contains implementations and benchmarks for various solution to knn search problem
### How to run
There are several external implementations of k neighbors search that used in benchmarks.
Those libraries aren't listed `requirements.txt` file due to two reasons:
1) Not all of them could be installed with pip
2) `requirements.txt` is used in production, where only one implementation is needed.
   Consider `requirements-dev.txt` to be able to run benchmarks. Note that `faiss` implementation currently
   (July 2019) can't be installed from pip. Consider reading https://github.com/facebookresearch/faiss/blob/master/INSTALL.md

## `vectorsindex`
A module with service implementation


    


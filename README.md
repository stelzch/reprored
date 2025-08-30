![Logo of ReproRed](./docs/images/logo.svg)



[![DOI](https://zenodo.org/badge/549016550.svg)](https://doi.org/10.5281/zenodo.15004917)


# ReproRed

This repository contains the code for multiple reproducible reduction algorithms.
It is used in [Repro-RAxML-NG](https://github.com/stelzch/repr-raxml-ng)

* `MPI_Allreduce` (not reproducible, used as baseline)
* [ReproBLAS](https://bebop.cs.berkeley.edu/reproblas/)
  achieves reproducibility through pre-rounding input data and using higher-precision accumulators
* $k$-Gather
  reduces k consecutive values in parallel linearily from left to right, gathers all intermediate results on root rank and reduces from left to right
* [Binary Tree Summation](https://doi.org/10.5445/IR/1000170326)
  uses a binary tree to dictate reduction order independent of core count
* ReproRed (formerly Dual Tree Summation)
  decouples the tree used for communication from the tree used to dictate the reduction order, uses theoretically optimal message count.



## Configuration
Selection of the reduction algorithm happens via environment variables

`REPR_REDUCE`
: Chooses the reduction algorithm. Possible values:
  * `ALLREDUCE`
  * `BINARY_TREE`
  * `DUAL_TREE`
  * `KGATHER`
  * `REPROBLAS`

`REPR_REDUCE_K`
: Sets the parameter $k$ for `KGATHER` and `BINARY_TREE`. Must be positive integer. Default 1.

`REPR_REDUCE_M`
: Sets degree of the communication tree for `DUAL_TREE`. See [this visualization](https://ch-st.de/applets/m-ary_trees.html)  on how $m$ affects where PEs send their intermediate results. Must be integer greater than 1. Default 2.

`REPR_REDUCE_TWOPHASE`
: By default, this library will use allreduction algorithms where possible (`ALLREDUCE` and `REPROBLAS`). If the environment variable `REPR_REDUCE_TWOPHASE` is set to any value, it uses a two-phase approach of `MPI_Reduce` followed by an `MPI_Bcast` instead.

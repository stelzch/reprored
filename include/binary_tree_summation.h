#ifndef BINARY_TREE_SUMMATION_H_
#define BINARY_TREE_SUMMATION_H_


/* This header file exposes functions for binary tree summation for use in C programs */

typedef void * ReductionContext;


ReductionContext new_reduction_context(int local_summands, MPI_Comm);
double reproducible_reduce(ReductionContext);
double *get_reduction_buffer(ReductionContext ctx);
void free_reduction_context(ReductionContext);


#endif
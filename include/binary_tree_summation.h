#ifndef BINARY_TREE_SUMMATION_H_
#define BINARY_TREE_SUMMATION_H_

#ifdef __cplusplus
extern "C" {
#endif

/* This header file exposes functions for binary tree summation for use in C programs */

typedef void * ReductionContext;


ReductionContext new_reduction_context(int local_summands);
double reproducible_reduce(ReductionContext);
double *get_reduction_buffer(ReductionContext ctx);
void free_reduction_context(ReductionContext);

#ifdef __cplusplus
}
#endif

#endif

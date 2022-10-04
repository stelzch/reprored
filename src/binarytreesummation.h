#ifndef BINARYTREE_SUMMATION_H_
#define BINARYTREE_SUMMATION_H_

#include <stddef.h>

/**
 * Calculate the reproducible sum across all MPI ranks.
 *
 * N is the global number of elements.
 * The data distribution "even_remainder_at_end" is assumed, that means in a
 * cluster with p processors, the first N - (N mod p) processors have 
 * floor(N / p) elements in their data array and the last N mod p processors 
 * have floor(N / p) + 1 elements in their data array.
 */
double binary_tree_sum(const float *data, const size_t N);

#endif

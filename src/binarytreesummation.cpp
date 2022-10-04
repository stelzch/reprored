#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <vector>
#include "binarytreesummation.h"

extern "C" {
#include <openmpi.h>
#include "/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
}


const uint64_t parent_index(const uint64_t i) {
    assert(i != 0);
    return i & (i - 1);
}

const uint64_t largest_child_index(const uint64_t index) {
    return index | (index - 1);
}

const uint64_t subtree_size(const uint64_t index) {
    assert(index != 0);
    return largest_child_index(index) + 1 - index;
}

/*
 * Return the index of the next rank intersecting summand.
 */
const uint64_t next_rank_intersecting_summand(const uint64_t index) {
    return largest_child_index(index) + 1;
}

/** Constant time algorithm to determine the rank of a certain index using a remainder_at_end distribution
 */
const int rankFromIndex(const uint64_t index, const size_t N, const int p) {
    const ldiv_t elementsPerRank = ldiv(N, p);

    // first rank which has floor(N / p) + 1 elements
    const int remainderRank = p - elementsPerRank.rem;
    const int remainderRankIndex = remainderRank * elementsPerRank.quot;

    if (index < remainderRankIndex) {
        // index is on ranks with floor(N / p) elements, simply divide out.
        return index / elementsPerRank.quot;
    } else {
        // index is on the remainder ranks.
        return remainderRank + ((index - remainderRankIndex) / (elementsPerRank.quot + 1));
    }
}

const uint64_t startIndex(const int rank, const size_t N, const int p) {
    const ldiv_t elementsPerRank = ldiv(N, p);

    // first rank which has floor(N / p) + 1 elements
    const int remainderRank = p - elementsPerRank.rem;
    const int remainderRankIndex = remainderRank * elementsPerRank.quot;

    if (rank < remainderRank) {
        return rank * elementsPerRank.quot;
    } else {
        return remainderRankIndex + (rank - remainderRank) * (elementsPerRank.quot + 1);
    }
}

double accumulate(const uint64_t index, const double *data, const size_t N) {
    const uint64_t begin = startIndex(rank, N, clusterSize);
    const uint64_t end = startIndex(rank + 1, N, clusterSize);

    if (index & 1) {
        return data[idx - startIndex];
    }

    const uint64_t maxX = (index == 0) ? N - 1
        : min(N - 1, index + subtree_size(index) - 1);
    const int maxY = (index == 0) ? ceil(log2(N)) : log2(subtree_size(index));

    const uint64_t largest_local_index = min(maxX, end - 1);
    const uint64_t n_local_elements = largest_local_index + 1 - index;

    uint64_t elementsInBuffer = n_local_elements;
    double *sourceBuffer = data;
    double *destinationBuffer = data;

    for (int y = 1; y <= maxY; y += 3) {
        uint64_t elementsWritten = 0;

        for (uint64_t i = 0; i + 8 <= elementsInBuffer; i += 8) {
            __m256d a = _mm256_loadu_pd(static_cast<double *>(&sourceBuffer[i]));
            __m256d b = _mm256_loadu_pd(static_cast<double *>(&sourceBuffer[i+4]));
            __m256d level1Sum = _mm256_hadd_pd(a, b);

            __m128d c = _mm256_extractf128_pd(level1Sum, 1); // Fetch upper 128bit
            __m128d d = _mm256_castpd256_pd128(level1Sum); // Fetch lower 128bit
            __m128d level2Sum = _mm_add_pd(c, d);

            __m128d level3Sum = _mm_hadd_pd(level2Sum, level2Sum);

            destinationBuffer[elementsWritten++] = _mm_cvtsd_f64(level3Sum);
        }

        // number of remaining elements
        const uint64_t remainder = elementsInBuffer - 8 * elementsWritten;
        assert(0 <= remainder);
        assert(remainder < 8);

        if (remainder > 0) {
            const uint64_t bufferIdx = 8 * elementsWritten;
            const uint64_t indexOfRemainingTree = index + bufferIdx * (1UL << (y - 1));
            const double a = sum_remaining_8tree(indexOfRemainingTree,
                    remainder, y, maxX,
		    &sourceBuffer[0] + bufferIdx,
                    &destinationBuffer[0] + bufferIdx);
            destinationBuffer[elementsWritten++] = a;
        }

        elementsInBuffer = elementsWritten;
    }
        
    assert(elementsInBuffer == 1);

    return destinationBuffer[0];
}

double binary_tree_sum(const double *data, const size_t N) {
    const ldiv_t elementsPerRank = ldiv(N, p);

    const int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int clusterSize;
    MPI_Comm_size(MPI_COMM_WORLD, &clusterSize);

    const uint64_t idx = startIndex(rank, N, clusterSize);
    const uint64_t endIdx = startIndex(rank + 1, N, clusterSize);

    // Iterate over all rank-intersecting summands on ranks > 0
    for (; idx != 0 && idx < endIdx; idx = next_rank_intersecting_summand(idx)) {
        double result = accumulate(idx, data, N);
        
        if (idx != 0) {
            MPI_Send(&result, 1, MPI_DOUBLE, rankFromIndex(parent_index(idx)),
                    0, MPI_COMM_WORLD);
        }
    }

    double result = (rank == 0) ? accumulate(0) : 0.0;
    MPI_Bcast(&result, 1, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

    return result;
}

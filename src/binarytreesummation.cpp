#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <vector>
#include "binarytreesummation.h"
#include <mpi.h>
#include <immintrin.h>


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


inline const double sum_remaining_8tree(const uint64_t bufferStartIndex,
        const uint64_t initialRemainingElements,
        const int y,
        const uint64_t maxX,
        double *srcBuffer,
        double *dstBuffer,
        const uint64_t N,
        int p) {
    uint64_t remainingElements = initialRemainingElements;

    for (int level = 0; level < 3; level++) {
        const int stride = 1 << (y - 1 + level);
        int elementsWritten = 0;
        for (int i = 0; (i + 1) < remainingElements; i += 2) {
            dstBuffer[elementsWritten++] = srcBuffer[i] + srcBuffer[i + 1];
        }


        if (remainingElements % 2 == 1) {
            const uint64_t bufferIndexA = remainingElements - 1;
            const uint64_t bufferIndexB = remainingElements;
            const uint64_t indexB = bufferStartIndex + bufferIndexB * stride;
            const double a = srcBuffer[bufferIndexA];

            if (indexB > maxX) {
                // indexB is the last element because the subtree ends there
                dstBuffer[elementsWritten++] = a;
            } else {
                // indexB must be fetched from another rank
                const int sourceRank = rankFromIndex(indexB, N, p);
                double b;
                MPI_Recv(&b, 1, MPI_DOUBLE, sourceRank, 0, MPI_COMM_WORLD, NULL);
                dstBuffer[elementsWritten++] = a + b;
            }

            remainingElements += 1;
        }

    srcBuffer = dstBuffer;
        remainingElements /= 2;
    }
    assert(remainingElements == 1);

    return dstBuffer[0];
}



double accumulate(const uint64_t index, double *data, const uint64_t N, const int p, const uint64_t begin, const uint64_t end) {

    if (index & 1) {
        return data[index - begin];
    }

    const uint64_t maxX = (index == 0) ? N - 1
        : std::min(N - 1, index + subtree_size(index) - 1);
    const int maxY = (index == 0) ? ceil(log2(N)) : log2(subtree_size(index));

    const uint64_t largest_local_index = std::min(maxX, end - 1);
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
                    &destinationBuffer[0] + bufferIdx, N, p);
            destinationBuffer[elementsWritten++] = a;
        }

        elementsInBuffer = elementsWritten;
        sourceBuffer = destinationBuffer;
    }
        
    assert(elementsInBuffer == 1);

    return destinationBuffer[0];
}

extern double binary_tree_sum(double *data, const size_t N) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int clusterSize;
    MPI_Comm_size(MPI_COMM_WORLD, &clusterSize);

    const ldiv_t elementsPerRank = ldiv(N, clusterSize);

    const uint64_t beginIdx = startIndex(rank, N, clusterSize);
    const uint64_t endIdx = startIndex(rank + 1, N, clusterSize);

    // Iterate over all rank-intersecting summands on ranks > 0
    uint64_t idx;
    for (idx = beginIdx;
            idx != 0 && idx < endIdx;
            idx = next_rank_intersecting_summand(idx)) {
        double result = accumulate(idx, data + idx - beginIdx, N, clusterSize, idx, endIdx);
        
        if (idx != 0) {
            MPI_Send(&result, 1, MPI_DOUBLE, rankFromIndex(parent_index(idx), N, clusterSize),
                    0, MPI_COMM_WORLD);
        }
    }

    double result = (rank == 0) ? accumulate(0, data, N, clusterSize, idx, endIdx) : 0.0;
    MPI_Bcast(&result, 1, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

    return result;
}

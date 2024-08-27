#include "binary_tree_summation.hpp"
#include "k_chunked_array.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <io.hpp>
#include <mpi.h>
#include <numeric>
#include <unistd.h>
#include <util.hpp>

#ifdef AVX
#include <immintrin.h>
#endif

using namespace std;
using namespace std::string_literals;

BinaryTreeSummation::BinaryTreeSummation(uint64_t rank,
                                         const vector<region> regions,
                                         uint64_t k, MPI_Comm comm)
    : comm(comm), rank(rank), k(k), message_buffer(comm),
      chunked_array(rank, regions, k),
      binary_tree(rank, chunked_array.get_k_chunks()), regions(regions),
      k_recv_reqs(chunked_array.get_predecessors().size()),
      accumulation_buffer_offset_pre_k(k - chunked_array.get_left_remainder()),
      accumulation_buffer_offset_post_k(k),
      accumulation_buffer(k + chunked_array.get_local_size() +
                          chunked_array.get_right_remainder()),
      acquisition_duration(std::chrono::duration<double>::zero()),
      acquisition_count(0L), reduction_counter(0UL) {
  int initialized;
  MPI_Initialized(&initialized);
  if (initialized) {
    int c_size;
    MPI_Comm_size(comm, &c_size);
    assert(c_size == regions.size());
  }

#ifdef DEBUG_OUTPUT_TREE
  printf("Rank %lu has %lu summands, starting from index %lu to %lu\n", rank,
         size, begin, end);
  printf("Rank %lu rankIntersectingSummands: ", rank);
  for (int ri : rankIntersectingSummands)
    printf("%u ", ri);
  printf("\n");
#endif
}

BinaryTreeSummation::~BinaryTreeSummation() {
#ifdef ENABLE_INSTRUMENTATION
  cout << "Rank " << rank
       << " avg. acquisition time: " << acquisitionTime() / acquisitionCount
       << "  ns\n";
#endif
}

double *BinaryTreeSummation::getBuffer() {
  return accumulation_buffer.data() + accumulation_buffer_offset_pre_k;
}

void BinaryTreeSummation::storeSummand(uint64_t localIndex, double val) {
  accumulation_buffer[accumulation_buffer_offset_pre_k + localIndex] = val;
}

void BinaryTreeSummation::linear_sum_k() {
  MPI_Request send_req = MPI_REQUEST_NULL;

  if (chunked_array.get_right_remainder() > 0 &&
      !chunked_array.has_left_neighbor_different_successor()) {
    // We do not reduce any summands on our own, we simply pass them to the
    // successor
    assert(chunked_array.get_successor() >= 0);
    assert(chunked_array.get_local_size() ==
           chunked_array.get_right_remainder());

    MPI_Isend(&accumulation_buffer[accumulation_buffer_offset_pre_k],
              chunked_array.get_local_size(), MPI_DOUBLE,
              chunked_array.get_successor(), MESSAGEBUFFER_MPI_TAG, comm,
              &send_req);
    return; // We are done here

  } else if (chunked_array.get_right_remainder() > 0 &&
             !chunked_array.is_last_rank()) {
    // Sum & send right remainder
    assert(chunked_array.get_successor() >= 0);
    double acc = std::accumulate(
        &accumulation_buffer[accumulation_buffer_offset_pre_k +
                             chunked_array.get_local_size() -
                             chunked_array.get_right_remainder()],
        &accumulation_buffer[accumulation_buffer_offset_pre_k +
                             chunked_array.get_local_size()],
        0.0);
    MPI_Isend(&acc, 1, MPI_DOUBLE, chunked_array.get_successor(),
              MESSAGEBUFFER_MPI_TAG, comm, &send_req);
  }

  // Start receive requests for the left remainder
  // TODO: ask someone with more MPI experience if this is really necessary
  // or if the sent values will be cached on the destination machine
  // regardless
  double left_remainder_accumulator;
  uint64_t left_remainder_running_index = 0;

  for (int i = 0U; i < chunked_array.get_predecessors().size(); ++i) {
    const auto other_rank = chunked_array.get_predecessors()[i];
    if (i == 0) {
      // assert((k_regions[other_rank].size > 0) || (other_rank ==
      // start_indices.begin()->second) ||
      // (regions.at(other_rank).globalStartIndex % k == 0));
      MPI_Irecv(&left_remainder_accumulator, 1, MPI_DOUBLE, other_rank,
                MESSAGEBUFFER_MPI_TAG, comm, &k_recv_reqs[i]);
    } else {
      assert(chunked_array.get_k_chunks()[other_rank].size == 0);
      const auto elements_to_receive =
          regions.at(other_rank)
              .size; // We receive all numbers the other rank holds

      MPI_Irecv(&accumulation_buffer[left_remainder_running_index],
                elements_to_receive, MPI_DOUBLE, other_rank,
                MESSAGEBUFFER_MPI_TAG, comm, &k_recv_reqs[i]);
      left_remainder_running_index += elements_to_receive;
      assert(left_remainder_running_index < k);
    }
  }

  // Sum local k-tuples that do not overlap with PE-boundaries
  const bool has_left_remainder = (chunked_array.get_left_remainder() > 0);
  uint64_t target_idx = has_left_remainder ? 1U : 0U;
  for (uint64_t i = chunked_array.get_left_remainder();
       i + k - 1 < chunked_array.get_local_size(); i += k) {
    accumulation_buffer[accumulation_buffer_offset_post_k + target_idx++] =
        std::accumulate(
            &accumulation_buffer[accumulation_buffer_offset_pre_k + i],
            &accumulation_buffer[accumulation_buffer_offset_pre_k + i + k],
            0.0);
  }

  // On the last rank manually sum right remainder since it can not be sent
  // anywhere.
  if (chunked_array.get_right_remainder() > 0 && chunked_array.is_last_rank()) {
    accumulation_buffer[accumulation_buffer_offset_post_k + target_idx++] =
        std::accumulate(
            &accumulation_buffer[accumulation_buffer_offset_pre_k +
                                 chunked_array.get_local_size() -
                                 chunked_array.get_right_remainder()],
            &accumulation_buffer[accumulation_buffer_offset_pre_k +
                                 chunked_array.get_local_size()],
            0.0);
  }

  // Make sure the send request has gone through before waiting on received
  // messages
  if (send_req != MPI_REQUEST_NULL) {
    MPI_Wait(&send_req, nullptr);
  }

  // Sum received values from left remainder
  if (has_left_remainder) {
    left_remainder_running_index = 0;
    MPI_Wait(&k_recv_reqs[0], nullptr);

    // TODO: if possible, join this loop with the MPI_Irecv loop above
    for (int i = 1U; i < chunked_array.get_predecessors().size(); ++i) {
      const auto other_rank = chunked_array.get_predecessors()[i];
      const auto elements_to_sum =
          regions.at(other_rank)
              .size; // We receive all numbers the other rank holds
      MPI_Wait(&k_recv_reqs[i], nullptr);

      left_remainder_accumulator = std::accumulate(
          &accumulation_buffer[left_remainder_running_index],
          &accumulation_buffer[left_remainder_running_index + elements_to_sum],
          left_remainder_accumulator);
      left_remainder_running_index += elements_to_sum;
    }

    // Accumulate local part of the left remainder
    left_remainder_accumulator = std::accumulate(
        &accumulation_buffer[accumulation_buffer_offset_pre_k],
        &accumulation_buffer[accumulation_buffer_offset_pre_k +
                             chunked_array.get_left_remainder()],
        left_remainder_accumulator);

    accumulation_buffer[accumulation_buffer_offset_post_k] =
        left_remainder_accumulator;
  }

  assert(target_idx == binary_tree.get_local_size());
}

/* Sum all numbers. Will return the total sum on rank 0
 */
double BinaryTreeSummation::accumulate(void) {
printf("reducing with bts, k=%lu", k);
  if (k != 1 && chunked_array.get_local_size() > 0) {
    linear_sum_k();
  }

  for (auto summand : binary_tree.get_rank_intersecting_summands()) {
    if (binary_tree.subtree_size(summand) > 16) {
      // If we are about to do some considerable amount of work, make sure
      // the send buffer is empty so noone is waiting for our results
      message_buffer.flush();
    }

    double result = accumulate(summand);

    message_buffer.put(
        binary_tree.rankFromIndexMap(binary_tree.parent(summand)), summand,
        result);
  }
  message_buffer.flush();
  message_buffer.wait();

  double result = 0.0;
  const int root_rank =
      binary_tree.get_global_size() == 0 ? 0 : binary_tree.rankFromIndexMap(0);
  if (rank == root_rank) {
    // Start accumulation on first rank with assigned summands.
    result = accumulate(0);
  }

  MPI_Bcast(&result, 1, MPI_DOUBLE, root_rank, comm);

  ++reduction_counter;

  return result;
}

double BinaryTreeSummation::accumulate(const uint64_t index) {
  if (index & 1) {
    // no accumulation needed
    return accumulation_buffer[accumulation_buffer_offset_post_k + index -
                               binary_tree.get_starting_index()];
  }

  const uint64_t maxX = (index == 0)
                            ? binary_tree.get_global_size() - 1
                            : min(binary_tree.get_global_size() - 1,
                                  index + binary_tree.subtree_size(index) - 1);
  const int maxY = (index == 0) ? ceil(log2(binary_tree.get_global_size()))
                                : log2(binary_tree.subtree_size(index));

  const uint64_t largest_local_index =
      min(maxX, binary_tree.get_end_index() - 1);
  const uint64_t n_local_elements = largest_local_index + 1 - index;

  uint64_t elementsInBuffer = n_local_elements;

  double *destinationBuffer = static_cast<double *>(
      &accumulation_buffer[accumulation_buffer_offset_post_k + index -
                           binary_tree.get_starting_index()]);
  double *sourceBuffer = destinationBuffer;

  for (int y = 1; y <= maxY; y += 3) {
    uint64_t elementsWritten = 0;

    for (uint64_t i = 0; i + 8 <= elementsInBuffer; i += 8) {
      __m256d a = _mm256_loadu_pd(static_cast<double *>(&sourceBuffer[i]));
      __m256d b = _mm256_loadu_pd(static_cast<double *>(&sourceBuffer[i + 4]));
      __m256d level1Sum = _mm256_hadd_pd(a, b);

      __m128d c = _mm256_extractf128_pd(level1Sum, 1); // Fetch upper 128bit
      __m128d d = _mm256_castpd256_pd128(level1Sum);   // Fetch lower 128bit
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
      const uint64_t indexOfRemainingTree =
          index + bufferIdx * (1UL << (y - 1));
      const double a = sum_remaining_8tree(indexOfRemainingTree, remainder, y,
                                           maxX, &sourceBuffer[0] + bufferIdx,
                                           &destinationBuffer[0] + bufferIdx);
      destinationBuffer[elementsWritten++] = a;
    }

    // After first iteration, read only from accumulation buffer
    sourceBuffer = destinationBuffer;

    elementsInBuffer = elementsWritten;
  }

  assert(elementsInBuffer == 1);

  return destinationBuffer[0];
}

const double BinaryTreeSummation::acquisitionTime(void) const {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             acquisition_duration)
      .count();
}

const void BinaryTreeSummation::printStats() const {
  message_buffer.printStats();
}

const int BinaryTreeSummation::get_rank() const { return rank; }

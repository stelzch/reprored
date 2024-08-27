#include "kgather_summation.hpp"
#include "k_chunked_array.hpp"
#include <cassert>
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

KGatherSummation::KGatherSummation(uint64_t rank, const vector<region> regions,
                                   uint64_t k, MPI_Comm comm)
    : comm(comm), rank(rank), k(k), chunked_array(rank, regions, k),
      regions(regions),
      send_counts(calc_send_counts()),
      displs(calc_displs()),
      k_recv_reqs(chunked_array.get_predecessors().size()),
      accumulation_buffer_offset_pre_k(k - chunked_array.get_left_remainder()),
      accumulation_buffer_offset_post_k(k),
      accumulation_buffer(k + chunked_array.get_local_size() +
                          chunked_array.get_right_remainder()),
      root_accumulation_buffer(), reduction_counter(0UL) {
  int initialized;
  MPI_Initialized(&initialized);
  if (initialized) {
    int c_size;
    MPI_Comm_size(comm, &c_size);
    assert(c_size == regions.size());
  }

  if (rank == 0) {
    auto desired_size = 0UL;
    for (const auto &kr : chunked_array.get_k_chunks()) {
      desired_size += kr.size;
    }

    root_accumulation_buffer.resize(desired_size);
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

KGatherSummation::~KGatherSummation() {
#ifdef ENABLE_INSTRUMENTATION
  cout << "Rank " << rank
       << " avg. acquisition time: " << acquisitionTime() / acquisitionCount
       << "  ns\n";
#endif
}

vector<int>
KGatherSummation::calc_send_counts() const {
  vector<int> send_counts;

  for (const auto &kr : chunked_array.get_k_chunks()) {
    send_counts.push_back(kr.size);
  }

  return send_counts;
}
vector<int> KGatherSummation::calc_displs() const {
  vector<int> displs;

  for (const auto &kr : chunked_array.get_k_chunks()) {
    displs.push_back(kr.globalStartIndex);
  }

  return displs;
}

double *KGatherSummation::getBuffer() {
  return accumulation_buffer.data() + accumulation_buffer_offset_pre_k;
}

void KGatherSummation::storeSummand(uint64_t localIndex, double val) {
  accumulation_buffer[accumulation_buffer_offset_pre_k + localIndex] = val;
}

void KGatherSummation::linear_sum_k() {
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

  assert(target_idx == chunked_array.get_k_chunks()[rank].size);
}

/* Sum all numbers. Will return the total sum on rank 0
 */
double KGatherSummation::accumulate(void) {
    printf("reducing with kgather, k=%lu\n", k);
  if (k != 1 && chunked_array.get_local_size() > 0) {
    linear_sum_k();
  }

  MPI_Gatherv(&accumulation_buffer[accumulation_buffer_offset_post_k],
              send_counts[rank], MPI_DOUBLE, root_accumulation_buffer.data(),
              send_counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

  double result;

  if (rank == 0) {
    result = std::accumulate(root_accumulation_buffer.begin(),
                             root_accumulation_buffer.end(), 0.0);
  }

  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, comm);

  ++reduction_counter;

  return result;
}
const int KGatherSummation::get_rank() const { return rank; }

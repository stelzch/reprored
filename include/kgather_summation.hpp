#include "binary_tree.hpp"
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <map>
#include <mpi.h>
#include <vector>

#include <k_chunked_array.hpp>
#include <message_buffer.hpp>
#include <util.hpp>

using std::array;
using std::map;
using std::vector;

class KGatherSummation {
public:
  KGatherSummation(uint64_t rank, const vector<region> regions, uint64_t K = 1,
                   MPI_Comm comm = MPI_COMM_WORLD);

  virtual ~KGatherSummation();

  double *getBuffer();
  void storeSummand(uint64_t localIndex, double val);

  /* Sum all numbers. Will return the total sum on rank 0
   */
  double accumulate(void);

  const int get_rank() const;

protected:
  void linear_sum_k();

private:
  const MPI_Comm comm;
  const int rank;
  const uint64_t k;
  const KChunkedArray chunked_array;
  const vector<region> regions;

  const vector<int> send_counts;
  const vector<int> displs;

  vector<MPI_Request> k_recv_reqs;
  const uint64_t accumulation_buffer_offset_pre_k;
  const uint64_t accumulation_buffer_offset_post_k;

  vector<double, AlignedAllocator<double>> accumulation_buffer;
  vector<double, AlignedAllocator<double>> root_accumulation_buffer;

  uint64_t reduction_counter;

  vector<int> calc_send_counts(const vector<region> &regions) const;
  vector<int> calc_displs(const vector<region> &regions) const;
};

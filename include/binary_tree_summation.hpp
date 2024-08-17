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

class BinaryTreeSummation {
public:
  /** Instantiate new binary tree accumulator.
   * For a reproducible result, the order of numbers must remain the same
   * over different runs. This order is represented by startIndices, which
   * contains the start index for each processor in the cluster, and
   * globalSize, which is the total number of summands.
   *
   */
  BinaryTreeSummation(uint64_t rank, const vector<region> regions,
                      uint64_t K = 1, MPI_Comm comm = MPI_COMM_WORLD);

  virtual ~BinaryTreeSummation();

  double *getBuffer();
  void storeSummand(uint64_t localIndex, double val);

  /* Sum all numbers. Will return the total sum on rank 0
   */
  double accumulate(void);

  /* Return the average number of nanoseconds spend in total on waiting for
   * intermediary results from other hosts
   */
  const double acquisitionTime(void) const;

  double accumulate(uint64_t index);

  const void printStats(void) const;

  const int get_rank() const;

protected:
  void linear_sum_k();

  const double accumulate_local_8subtree(const uint64_t startIndex) const;

  inline const double sum_remaining_8tree(
      const uint64_t bufferStartIndex, const uint64_t initialRemainingElements,
      const int y, const uint64_t maxX, double *srcBuffer, double *dstBuffer) {
    uint64_t remainingElements = initialRemainingElements;

    for (int level = 0; level < 3; level++) {
      const int stride = 1 << (y - 1 + level);
      int elementsWritten = 0;
      for (uint64_t i = 0; (i + 1) < remainingElements; i += 2) {
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
          const double b =
              message_buffer.get(binary_tree.rankFromIndexMap(indexB), indexB);
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

private:
  const MPI_Comm comm;
  const int rank;
  const uint64_t k;
  MessageBuffer message_buffer;
  const KChunkedArray chunked_array;
  const BinaryTree binary_tree;
  const vector<region> regions;

  vector<MPI_Request> k_recv_reqs;
  const uint64_t accumulation_buffer_offset_pre_k;
  const uint64_t accumulation_buffer_offset_post_k;

  vector<double, AlignedAllocator<double>> accumulation_buffer;
  std::chrono::duration<double> acquisition_duration;
  long int acquisition_count;

  uint64_t reduction_counter;
};

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <util.hpp>
#include <vector>

using std::map;
using std::vector;

/**
 * This class captures the topology of a distributed array consisting of chunks
 * with constant size K. An example array would look like this:
 *
 *      ▼        ▼        ▼        ▼
 *     ┌────────┬──┬──┬──────────────┐
 *     │  p2    │p0│p3│     p1       │
 *     └────────┴──┴──┴──────────────┘
 *      0  1  2  3  4  5  6  7  8  9
 *
 * Rank 2 holds the first 3 elements, the fourth element is on rank 0 and so
 * fort. The chunks always start at indices which are divisible by k. In this
 * case, K=3.
 *
 * This class provides methods that are useful when reducing each k-chunk
 * individually by e.g. summing them from left to right. For this to happen,
 * the ranks must exchange elements that belong to a chunk that started on a
 * different rank. After the reduction, the array from above will look like
 * this:
 *
 *  ┌──┬───────┐
 *  │p2│  p1   │
 *  └──┴───────┘
 *   0  1  2  3
 *
 * The first three elements are located on p2 so this chunk is reduced locally
 * into the so called k-chunk 0.  p0 and p3 send their elements to p1 (their
 * successor), because p1 holds the last element of k-chunk 1.  k-chunk 2 is
 * also locally reduced on p1 and k-chunk 3 is cut short due to the global
 * array length and consists of only one elmement, also resident on p1.
 *
 * Other terminology includes the left and right remainder. These occur if the
 * rank boundary is not perfectly aligned with the rank boundary:
 *
 *
 *               left remainder                   right remainder
 *                   ┌───┐                              ┌─┐
 *                   │   │                              │ │
 * index mod k: ▼         ▼         ▼         ▼         ▼         ▼
 *                   ┌────────────────────────────────────┐
 *                   │             p2                     │
 *                   └────────────────────────────────────┘
 */
class KChunkedArray {
public:
  KChunkedArray(uint64_t rank, vector<region> regions, uint64_t K = 1);
  virtual ~KChunkedArray();

  const vector<int> &get_predecessors() const;
  const int get_successor() const;
  const bool has_no_k_intercept() const;
  const vector<region> &get_k_chunks() const;
  const bool has_left_neighbor_different_successor() const;

  const uint64_t get_left_remainder() const;
  const uint64_t get_right_remainder() const;

  const uint64_t get_local_size() const;

  const bool is_last_rank() const;

private:
  const vector<region> calculate_k_regions(const vector<region> &regions) const;
  const vector<int> calculate_k_predecessors() const;
  const int calculate_k_successor() const;

  const map<int, region>
  calculate_regions_map(const vector<region> &regions) const;
  const map<uint64_t, int>
  calculate_start_indices(const vector<region> &regions) const;

  const uint64_t begin, end;
  const uint64_t k;

  const int rank, clusterSize;
  const uint64_t size;
  const map<int, region>
      regions; // maps rank to region. does not contain zero regions
  const map<uint64_t, int>
      start_indices; // maps global array start index to rank
  const vector<region> k_chunks;

  const bool is_last_rank_flag;

  const bool
      no_k_intercept; // if true no number in [begin, end) is divisible by k
  const uint64_t left_remainder;
  const uint64_t right_remainder;
  const bool
      left_neighbor_has_different_successor; // If true the left (index-order)
                                             // neighbor has a different
                                             // successor than the current rank,
                                             // which means we can reduce our
                                             // right remainder before sending
                                             // it

  const vector<int>
      predecessor_ranks; // ranks we receive from during linear sum.
                         // In non-degenerate case this is the next lower rank
  const int
      successor_rank; // ranks we send to during linear sum.
                      // In non-degenerate case this is the next higher rank.
};

inline KChunkedArray::KChunkedArray(uint64_t rank, vector<region> regions, uint64_t K)
    : begin(regions[rank].globalStartIndex), end(begin + regions[rank].size),
      k(K), rank(rank), clusterSize(regions.size()), size(regions[rank].size),
      regions(calculate_regions_map(regions)),
      start_indices(calculate_start_indices(regions)),
      k_chunks(calculate_k_regions(regions)),
      is_last_rank_flag(rank == this->start_indices.rbegin()->second),
      no_k_intercept(begin % k != 0 && begin / k == end / k),
      left_remainder(k_chunks[rank].size == 0
                         ? 0
                         : std::min(round_up_to_multiple(begin, k), end) -
                               begin),
      right_remainder(
          (is_last_rank_flag && no_k_intercept)
              ? 0
              : end - std::max(round_down_to_multiple(end, k), begin)),
      left_neighbor_has_different_successor(
          start_indices.begin()->second == rank || k_chunks[rank].size > 0 ||
          begin % k == 0),
      predecessor_ranks(calculate_k_predecessors()),
      successor_rank(calculate_k_successor()) {
  assert(k > 0);
  assert(left_remainder < k);
  assert(right_remainder < k);
  assert(implicates(no_k_intercept, size < k));
}

inline KChunkedArray::~KChunkedArray() {}

inline const map<int, region>
KChunkedArray::calculate_regions_map(const vector<region> &regions) const {
  std::map<int, region> region_map;

  for (int p = 0; p < clusterSize; ++p) {
    if (regions[p].size == 0)
      continue;
    region_map[p] = regions[p];
  }

  return region_map;
}

inline const map<uint64_t, int>
KChunkedArray::calculate_start_indices(const vector<region> &regions) const {
  std::map<uint64_t, int> start_indices;

  /* Initialize start indices map */
  for (int p = 0; p < clusterSize; ++p) {
    if (regions[p].size == 0)
      continue;
    start_indices[regions[p].globalStartIndex] = p;
  }

  return start_indices;
}

inline const vector<region>
KChunkedArray::calculate_k_regions(const vector<region> &regions) const {
  const auto last_rank = this->start_indices.rbegin()->second;
  const auto last_region = regions.begin() + last_rank;

  vector<region> k_regions;
  k_regions.reserve(regions.size());

  for (auto it = regions.begin(); it < regions.end(); ++it) {
    const region &r = *it;

    if (r.size == 0) {
      k_regions.emplace_back(0, 0);
    } else {
      const auto start = round_down_to_multiple(r.globalStartIndex, k) / k;
      auto end = round_down_to_multiple(r.globalStartIndex + r.size, k) / k;

      // If the last region has a right remainder (that is, its end index
      // is not divisible by k) we assign it an additional, truncated
      // k-region.  Normally, the remainder would be sent to the
      // successor, but because it is the last region the rank has to
      // take care of it on its own.
      if (it == last_region && (r.globalStartIndex + r.size) % k != 0) {
        end += 1;
      }

      k_regions.emplace_back(start, end - start);
    }
  }

  return k_regions;
}

inline const vector<int> KChunkedArray::calculate_k_predecessors() const {
  vector<int> predecessors;
  if (left_remainder == 0 || size == 0) {
    // There is no-one we receive from
    return predecessors;
  }

  auto it = start_indices.find(begin);

  while (it != start_indices.begin()) {
    --it;
    const auto other_rank = it->second;
    const auto &other_region = regions.at(other_rank);

    if (other_region.size > 0 &&
        (other_region.globalStartIndex + other_region.size) % k == 0) {
      // This rank won't have to send us a remainder because the
      // PE-border coincides with the k-region border
      break;
    }

    predecessors.push_back(other_rank);

    if (k_chunks[other_rank].size >= 1) {
      // The other_rank has a k-region assigned so any ranks lower than i
      // will send their remainder to the other_rank instead.
      break;
    }
  }

  // We build the list of predecessors from right to left (global indices
  // descending) but during traversal we want the indices to ascend since we
  // compute our sum left-to-right
  std::reverse(predecessors.begin(), predecessors.end());

  return predecessors;
}

inline const int KChunkedArray::calculate_k_successor() const {
  // No successor on last rank & on ranks that do not participate in the
  // reduction
  if (is_last_rank_flag || size == 0) {
    return -1;
  }

  auto it = start_indices.find(begin);
  assert(it != start_indices.end());

  // Increase i past current rank until we encounter either a rank that has a
  // k_region assigned or we reach the end of the map.
  do {
    it++;
  } while (it != start_indices.end() && k_chunks[it->second].size == 0);

  assert(it != start_indices.end());

  return it->second;
}

inline const vector<int> &KChunkedArray::get_predecessors() const {
  return predecessor_ranks;
}
inline const int KChunkedArray::get_successor() const { return successor_rank; }
inline const bool KChunkedArray::has_no_k_intercept() const { return no_k_intercept; }
inline const vector<region> &KChunkedArray::get_k_chunks() const { return k_chunks; }
inline const bool KChunkedArray::has_left_neighbor_different_successor() const {
  return left_neighbor_has_different_successor;
}

inline const uint64_t KChunkedArray::get_left_remainder() const {
  return left_remainder;
}
inline const uint64_t KChunkedArray::get_right_remainder() const {
  return right_remainder;
}
inline const uint64_t KChunkedArray::get_local_size() const { return size; }
inline const bool KChunkedArray::is_last_rank() const { return is_last_rank_flag; }

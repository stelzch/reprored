#pragma once

#include "k_chunked_array.hpp"
#include <cstdint>
#include <cassert>
#include <numeric>

class BinaryTree {
public:
  BinaryTree(uint64_t rank, vector<region> regions);
  virtual ~BinaryTree();

  static const uint64_t parent(const uint64_t i);

  bool isLocal(uint64_t index) const;

  /** Determine which rank has the number with a given index */
  uint64_t rankFromIndexMap(const uint64_t index) const;

  const uint64_t largest_child_index(const uint64_t index) const;
  const uint64_t subtree_size(const uint64_t index) const;

  /** Figure out if the parts that make up a certain index are all local and
   * form a subtree of a specifc size */
  const bool is_local_subtree_of_size(const uint64_t expectedSubtreeSize,
                                      const uint64_t i) const;
  const vector<uint64_t> &get_rank_intersecting_summands(void) const;

  const uint64_t get_starting_index() const;
  const uint64_t get_end_index() const;
  const uint64_t get_global_size() const;
  const uint64_t get_local_size() const;

private:
  const uint64_t rank;
  const uint64_t clusterSize;
  const uint64_t globalSize;
  const vector<region> regions;
  const uint64_t size, begin, end;
  const map<uint64_t, int> start_indices;
  const vector<uint64_t> rank_intersecting_summands;

protected:
  /* Calculate all rank-intersecting summands that must be sent out because
   * their parent is non-local and located on another rank
   */
  vector<uint64_t> calculateRankIntersectingSummands(void) const;
  const map<uint64_t, int> calculate_k_start_indices() const;
};

inline BinaryTree::BinaryTree(uint64_t rank, vector<region> regions)
    : rank(rank), clusterSize(regions.size()),
      globalSize(std::accumulate(
          regions.begin(), regions.end(), 0UL,
          [](uint64_t acc, const region &r) { return acc + r.size; })),
      regions(regions), size(regions[rank].size),
      begin(regions[rank].globalStartIndex), end(begin + size),
      start_indices(calculate_k_start_indices()),
      rank_intersecting_summands(calculateRankIntersectingSummands()) {
  assert(globalSize > 0);

  // Verify that the regions are actually correct.
  // This is given if the difference to the next start index is equal to the
  // region size
  for (auto it = start_indices.begin(); it != start_indices.end(); ++it) {
    auto next = std::next(it);
    if (next == start_indices.end())
      break;

    assert(it->first + regions[it->second].size == next->first);
  }
}

inline BinaryTree::~BinaryTree() {}

inline const uint64_t BinaryTree::parent(const uint64_t i) {
  assert(i != 0);

  // clear least significand set bit
  return i & (i - 1);
}

inline bool BinaryTree::isLocal(uint64_t index) const {
  return (index >= begin && index < end);
}

inline uint64_t BinaryTree::rankFromIndexMap(const uint64_t index) const {
  // Get an iterator to the start index that is greater than index
  auto it = start_indices.upper_bound(index);
  assert(it != start_indices.begin());
  --it;

  return it->second;
}

/* Calculate all rank-intersecting summands that must be sent out because
 * their parent is non-local and located on another rank
 */
inline vector<uint64_t> BinaryTree::calculateRankIntersectingSummands(void) const {
  vector<uint64_t> result;

  if (begin == 0 || size == 0) {
    return result;
  }

  assert(begin != 0);

  uint64_t index = begin;
  while (index < end) {
    assert(parent(index) < begin);
    result.push_back(index);

    index = index + subtree_size(index);
  }

  return result;
}

inline const map<uint64_t, int> BinaryTree::calculate_k_start_indices() const {
  std::map<uint64_t, int> start_indices;

  /* Initialize start indices map */
  for (int p = 0; p < clusterSize; ++p) {
    if (regions[p].size == 0)
      continue;
    start_indices[regions[p].globalStartIndex] = p;
  }

  // guardian element
  start_indices[globalSize] = clusterSize;

  return start_indices;
}

inline const uint64_t BinaryTree::largest_child_index(const uint64_t index) const {
  return index | (index - 1);
}

inline const uint64_t BinaryTree::subtree_size(const uint64_t index) const {
  assert(index != 0);
  return largest_child_index(index) + 1 - index;
}

inline const vector<uint64_t> &BinaryTree::get_rank_intersecting_summands(void) const {
  return rank_intersecting_summands;
}

inline const uint64_t BinaryTree::get_starting_index() const { return begin; }
inline const uint64_t BinaryTree::get_end_index() const { return end; }
inline const uint64_t BinaryTree::get_global_size() const { return globalSize; }

inline const uint64_t BinaryTree::get_local_size() const { return size; }

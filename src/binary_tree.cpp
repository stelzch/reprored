#include <binary_tree.hpp>
#include <cassert>
#include <numeric>

BinaryTree::BinaryTree(uint64_t rank, vector<region> regions)
    :
      rank(rank),
      clusterSize(regions.size()),
      globalSize(std::accumulate(regions.begin(), regions.end(), 0UL,
                 [](uint64_t acc, const region& r) {
                  return acc + r.size;
      })),
      regions(regions),
      size(regions[rank].size),
      begin(regions[rank].globalStartIndex),
      end(begin + size),
      start_indices(calculate_k_start_indices()),
      rank_intersecting_summands(calculateRankIntersectingSummands())
{
    assert(globalSize > 0);

    // Verify that the regions are actually correct.
    // This is given if the difference to the next start index is equal to the region size
    for (auto it = start_indices.begin(); it != start_indices.end(); ++it) {
        auto next = std::next(it);
        if (next == start_indices.end()) break;

        assert(it->first + regions[it->second].size == next->first);
    }
}

BinaryTree::~BinaryTree() {}

const uint64_t BinaryTree::parent(const uint64_t i) {
    assert(i != 0);

    // clear least significand set bit
    return i & (i - 1);
}

bool BinaryTree::isLocal(uint64_t index) const {
    return (index >= begin && index < end);
}

uint64_t BinaryTree::rankFromIndexMap(const uint64_t index) const {
    // Get an iterator to the start index that is greater than index
    auto it = start_indices.upper_bound(index);
    assert(it != start_indices.begin());
    --it;

    return it->second;
}

/* Calculate all rank-intersecting summands that must be sent out because
    * their parent is non-local and located on another rank
    */
vector<uint64_t> BinaryTree::calculateRankIntersectingSummands(void) const {
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

const map<uint64_t, int> BinaryTree::calculate_k_start_indices() const {
    std::map<uint64_t, int> start_indices;

    /* Initialize start indices map */
    for (int p = 0; p < clusterSize; ++p) {
        if (regions[p].size == 0) continue;
        start_indices[regions[p].globalStartIndex] = p;
    }

    // guardian element
    start_indices[globalSize] = clusterSize;

    return start_indices;
}

const uint64_t BinaryTree::largest_child_index(const uint64_t index) const {
    return index | (index - 1);
}

const uint64_t BinaryTree::subtree_size(const uint64_t index) const {
    assert(index != 0);
    return largest_child_index(index) + 1 - index;
}

const vector<uint64_t>& BinaryTree::get_rank_intersecting_summands(void) const {
    return rank_intersecting_summands;
}

const uint64_t BinaryTree::get_starting_index() const {
    return begin;
}
const uint64_t BinaryTree::get_end_index() const {
    return end;
}
const uint64_t BinaryTree::get_global_size() const {
    return globalSize;
}

const uint64_t BinaryTree::get_local_size() const {
    return size;
}

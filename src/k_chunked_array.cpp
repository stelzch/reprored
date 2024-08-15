#include <algorithm>
#include <cassert>
#include <numeric>
#include <util.hpp>
#include <k_chunked_array.hpp>

KChunkedArray::KChunkedArray(uint64_t rank, vector<region> regions, uint64_t K) 
    :
      k(K),
      rank(rank),
      clusterSize(regions.size()),
      regions(regions),
      index_order_permutation(calculate_rank_order_permutation()),
      index(calculate_index()),
      size(regions[rank].size),
      begin(regions[rank].globalStartIndex),
      end(begin + size),
      is_last_rank(index == clusterSize - 1), // TODO: if there are multiple trailing PEs with zero-assignment this might be wrong
      no_k_intercept(begin % k != 0 && begin / k == end / k),
      k_regions(calculate_k_regions(regions)),
      k_size(k_regions[rank].size),
      k_begin(k_regions[rank].globalStartIndex),
      k_end(k_begin + k_size),
      k_left_remainder(k_regions[rank].size == 0 ? 0 : std::min(round_up_to_multiple(begin, k),end) - begin),
      k_right_remainder((is_last_rank && no_k_intercept) ? 0 : end - std::max(round_down_to_multiple(end, k), begin)),
      left_neighbor_has_different_successor(index == 0 || k_regions[rank].size > 0),
      globalSize(std::accumulate(k_regions.begin(), k_regions.end(), 0UL,
                 [](uint64_t acc, const region& r) {
                  return acc + r.size;
      })),
      start_indices(calculate_start_indices()),
      k_predecessor_ranks(calculate_k_predecessors()),
      k_successor_rank(calculate_k_successor()),
      rank_intersecting_summands(calculateRankIntersectingSummands())
{

}

KChunkedArray::~KChunkedArray() {
}

const uint64_t KChunkedArray::parent(const uint64_t i) {
    assert(i != 0);

    // clear least significand set bit
    return i & (i - 1);
}

bool KChunkedArray::isLocal(uint64_t index) const {
    return (index >= k_begin && index < k_end);
}

uint64_t KChunkedArray::rankFromIndexMap(const uint64_t index) const {
    // Get an iterator to the start index that is greater than index
    auto it = start_indices.upper_bound(index);
    assert(it != start_indices.begin());
    --it;

    return it->second;
}

/* Calculate all rank-intersecting summands that must be sent out because
    * their parent is non-local and located on another rank
    */
vector<uint64_t> KChunkedArray::calculateRankIntersectingSummands(void) const {
    vector<uint64_t> result;

    if (k_begin == 0 || k_size == 0) {
        return result;
    }

    assert(k_begin != 0);

    uint64_t index = k_begin;
    while (index < k_end) {
        assert(parent(index) < k_begin);
        result.push_back(index);

        index = index + subtree_size(index);
    }

    return result;
}

const map<uint64_t, int> KChunkedArray::calculate_start_indices() const {
    std::map<uint64_t, int> start_indices;

    /* Initialize start indices map */
    for (int p = 0; p < clusterSize; ++p) {
        if (k_regions[p].size == 0) continue;
        start_indices[k_regions[p].globalStartIndex] = p;
    }
    // guardian element
    start_indices[globalSize] = clusterSize;

    return start_indices;
}

const vector<region> KChunkedArray::calculate_k_regions(const vector<region> regions) const {
    const auto last_region = std::max_element(regions.begin(), regions.end(), [] (const auto& a, const auto& b) {
            return a.globalStartIndex < b.globalStartIndex;
    });

    vector<region> k_regions;
    k_regions.reserve(regions.size());

    for (auto it = regions.begin(); it < regions.end(); ++it) {
        const region& r = *it;
        const auto start = round_down_to_multiple(r.globalStartIndex, k) / k;
        auto end = round_down_to_multiple(r.globalStartIndex + r.size, k) / k;

        // Additional element at the end
        if (it == last_region && (r.globalStartIndex + r.size) % k != 0) {
            end += 1;
        }

        k_regions.emplace_back(start, end - start);
    }

    return k_regions;
}
const vector<size_t> KChunkedArray::calculate_rank_order_permutation() const {
    vector<size_t> permutation(regions.size());
    std::iota(permutation.begin(), permutation.end(), 0);

    std::sort(permutation.begin(), permutation.end(), [this] (const auto a, const auto b) {
            return regions[a].globalStartIndex < regions[b].globalStartIndex;
    });

    return permutation;
}

const size_t KChunkedArray::calculate_index() const {
    // Move i to current rank
    auto i = 0U;
    while (index_order_permutation.at(i) != rank) {
        i++;
    }
    assert(i < index_order_permutation.size());

    return i;
}

const vector<int> KChunkedArray::calculate_k_predecessors() const {
    vector<int> predecessors;

    auto i = index;

    if (k_left_remainder == 0 || i == 0) {
        // There is no-one we receive from
        return predecessors;
    }

    do {
        --i;
        const auto other_rank = index_order_permutation.at(i);

        if ((regions[other_rank].globalStartIndex + regions[other_rank].size) % k == 0) {
            // This rank won't have to send us a remainder because the
            // PE-border coincides with the k-region border
            break;
        }

        if (regions[other_rank].size > 0) {
            predecessors.push_back(other_rank);
        }

        if (k_regions[other_rank].size >= 1) {
            // The other_rank has a k-region assigned so any ranks lower than i
            // will send their remainder to the other_rank instead.
            break;
        }
    } while (i > 0);

    // We build the list of predecessors from right to left (global indices
    // descending) but during traversal we want the indices to ascend since we
    // compute our sum left-to-right
    std::reverse(predecessors.begin(), predecessors.end());

    return predecessors;
}

const int KChunkedArray::calculate_k_successor() const {
    // No successor on last rank
    if (is_last_rank) {
        return -1;
    }

    auto i = rank;

    // Increase i past current rank
    do {
        i++;
    } while(k_regions[index_order_permutation[i]].size == 0);

    assert(i < clusterSize);

    return index_order_permutation[i];
}

const uint64_t KChunkedArray::largest_child_index(const uint64_t index) const {
    return index | (index - 1);
}

const uint64_t KChunkedArray::subtree_size(const uint64_t index) const {
    assert(index != 0);
    return largest_child_index(index) + 1 - index;
}

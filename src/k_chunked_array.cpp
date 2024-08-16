#include <algorithm>
#include <cassert>
#include <numeric>
#include <util.hpp>
#include <k_chunked_array.hpp>

KChunkedArray::KChunkedArray(uint64_t rank, vector<region> regions, uint64_t K) 
    :
      begin(regions[rank].globalStartIndex),
      end(begin + regions[rank].size),
      k(K),
      rank(rank),
      clusterSize(regions.size()),
      size(regions[rank].size),
      regions(calculate_regions_map(regions)),
      start_indices(calculate_start_indices(regions)),
      k_chunks(calculate_k_regions(regions)),
      is_last_rank_flag(rank == this->start_indices.rbegin()->second),
      no_k_intercept(begin % k != 0 && begin / k == end / k),
      left_remainder(k_chunks[rank].size == 0 ? 0 : std::min(round_up_to_multiple(begin, k),end) - begin),
      right_remainder((is_last_rank_flag && no_k_intercept) ? 0 : end - std::max(round_down_to_multiple(end, k), begin)),
      left_neighbor_has_different_successor(start_indices.begin()->second == rank || k_chunks[rank].size > 0 || begin % k == 0), 
      predecessor_ranks(calculate_k_predecessors()),
      successor_rank(calculate_k_successor())
{
    assert(k > 0);
    assert(left_remainder < k);
    assert(right_remainder < k);
    assert(implicates(no_k_intercept, size < k));
}

KChunkedArray::~KChunkedArray() {
}


const map<int, region> KChunkedArray::calculate_regions_map(const vector<region>& regions) const {
    std::map<int, region> region_map;

    for (int p = 0; p < clusterSize; ++p) {
        if (regions[p].size == 0) continue;
        region_map[p] = regions[p];
    }

    return region_map;
}

const map<uint64_t, int> KChunkedArray::calculate_start_indices(const vector<region>& regions) const {
    std::map<uint64_t, int> start_indices;

    /* Initialize start indices map */
    for (int p = 0; p < clusterSize; ++p) {
        if (regions[p].size == 0) continue;
        start_indices[regions[p].globalStartIndex] = p;
    }

    return start_indices;
}


const vector<region> KChunkedArray::calculate_k_regions(const vector<region>& regions) const {
    const auto last_rank = this->start_indices.rbegin()->second;
    const auto last_region = regions.begin() + last_rank;

    vector<region> k_regions;
    k_regions.reserve(regions.size());

    for (auto it = regions.begin(); it < regions.end(); ++it) {
        const region& r = *it;

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

const vector<int> KChunkedArray::calculate_k_predecessors() const {
    vector<int> predecessors;
    if (left_remainder == 0 || size == 0) {
        // There is no-one we receive from
        return predecessors;
    }

    auto it = start_indices.find(begin);


    while (it != start_indices.begin()) {
        --it;
        const auto other_rank = it->second;
        const auto& other_region = regions.at(other_rank);

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

const int KChunkedArray::calculate_k_successor() const {
    // No successor on last rank & on ranks that do not participate in the reduction
    if (is_last_rank_flag || size == 0) {
        return -1;
    }

    auto it = start_indices.find(begin);
    assert(it != start_indices.end());


    // Increase i past current rank until we encounter either a rank that has a k_region assigned
    // or we reach the end of the map.
    do {
        it++;
    } while(it != start_indices.end() && k_chunks[it->second].size == 0);

    assert(it != start_indices.end());

    return it->second;
}

const vector<int>& KChunkedArray::get_predecessors() const {
    return predecessor_ranks;
}
const int KChunkedArray::get_successor() const {
    return successor_rank;
}
const bool KChunkedArray::has_no_k_intercept() const {
    return no_k_intercept;
}
const vector<region>& KChunkedArray::get_k_chunks() const {
    return k_chunks;
}
const bool KChunkedArray::has_left_neighbor_different_successor() const {
    return left_neighbor_has_different_successor;
}

const uint64_t KChunkedArray::get_left_remainder() const {
    return left_remainder;
}
const uint64_t KChunkedArray::get_right_remainder() const {
    return right_remainder;
}
const uint64_t KChunkedArray::get_local_size() const {
    return size;
}
const bool KChunkedArray::is_last_rank() const {
    return is_last_rank_flag;
}

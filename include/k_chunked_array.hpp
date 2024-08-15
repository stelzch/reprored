#pragma once

#include <vector>
#include <map>
#include <cstdint>

using std::vector;
using std::map;

typedef struct {
    uint64_t globalStartIndex;
    uint64_t size;
} region;




class KChunkedArray {
    public:
        KChunkedArray(uint64_t rank, vector<region> regions, uint64_t K = 1);
        virtual ~KChunkedArray();


        static const uint64_t parent(const uint64_t i);

        bool isLocal(uint64_t index) const;

        /** Determine which rank has the number with a given index */
        uint64_t rankFromIndexMap(const uint64_t index) const;

        const uint64_t largest_child_index(const uint64_t index) const;
        const uint64_t subtree_size(const uint64_t index) const;

        /** Figure out if the parts that make up a certain index are all local and form a subtree
        * of a specifc size */
        const bool is_local_subtree_of_size(const uint64_t expectedSubtreeSize, const uint64_t i) const;
    protected:

        /* Calculate all rank-intersecting summands that must be sent out because
        * their parent is non-local and located on another rank
        */
        vector<uint64_t> calculateRankIntersectingSummands(void) const;
        const vector<region> calculate_k_regions(const vector<region> regions) const;
        const vector<int> calculate_k_predecessors() const;
        const int calculate_k_successor() const;


        const map<int, region> calculate_regions_map(const vector<region>& regions) const;
        const map<uint64_t, int> calculate_start_indices(const vector<region>& regions) const;
        const map<uint64_t, int> calculate_k_start_indices() const;




        const uint64_t k;
        const int rank, clusterSize;

        const map<int, region> regions; // maps rank to region. does not contain zero regions
        const map<uint64_t, int> start_indices; // maps global array start index to rank

        const uint64_t size,  begin, end;
        const bool is_last_rank;

        const bool no_k_intercept; // if true no number in [begin, end) is divisible by k
        const vector<region> k_regions;
        const uint64_t k_size,  k_begin, k_end;
        const uint64_t k_left_remainder;
        const uint64_t k_right_remainder;
        const bool left_neighbor_has_different_successor; // If true the left (index-order) neighbor has a different successor than the current rank,
                                                      // which means we can reduce our right remainder before sending it


        const uint64_t globalSize;
        const map<uint64_t, int> k_start_indices;

        const vector<int> k_predecessor_ranks; // ranks we receive from during linear sum.
                                        // In non-degenerate case this is the next lower rank
        const int k_successor_rank; // ranks we send to during linear sum.
                            // In non-degenerate case this is the next higher rank.

        vector<uint64_t> rank_intersecting_summands;
};

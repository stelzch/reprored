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

/**
 * This class captures the topology of a distributed array consisting of chunks with constant size K.
 * An example array would look like this:
 *                                  
 *      ▼        ▼        ▼        ▼   
 *     ┌────────┬──┬──┬──────────────┐ 
 *     │  p2    │p0│p3│     p1       │ 
 *     └────────┴──┴──┴──────────────┘ 
 *      0  1  2  3  4  5  6  7  8  9
 *
 * Rank 2 holds the first 3 elements, the fourth element is on rank 0 and so fort.
 * The chunks always start at indices which are divisible by k. In this case, K=3.
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

        const vector<int>& get_predecessors() const;
        const int get_successor() const;
        const bool has_no_k_intercept() const;
        const vector<region>& get_k_chunks() const;
        const bool has_left_neighbor_different_successor() const;

        const uint64_t get_left_remainder() const;
        const uint64_t get_right_remainder() const;

        const uint64_t get_local_size() const;

        const bool is_last_rank() const;


    private:
        const vector<region> calculate_k_regions(const vector<region>& regions) const;
        const vector<int> calculate_k_predecessors() const;
        const int calculate_k_successor() const;


        const map<int, region> calculate_regions_map(const vector<region>& regions) const;
        const map<uint64_t, int> calculate_start_indices(const vector<region>& regions) const;


        const uint64_t begin, end;
        const uint64_t k;

        const int rank, clusterSize;
        const uint64_t size;
        const map<int, region> regions; // maps rank to region. does not contain zero regions
        const map<uint64_t, int> start_indices; // maps global array start index to rank
        const vector<region> k_chunks;


        const bool is_last_rank_flag;

        const bool no_k_intercept; // if true no number in [begin, end) is divisible by k
        const uint64_t left_remainder;
        const uint64_t right_remainder;
        const bool left_neighbor_has_different_successor; // If true the left (index-order) neighbor has a different successor than the current rank,
                                                      // which means we can reduce our right remainder before sending it



        const vector<int> predecessor_ranks; // ranks we receive from during linear sum.
                                        // In non-degenerate case this is the next lower rank
        const int successor_rank; // ranks we send to during linear sum.
                            // In non-degenerate case this is the next higher rank.

};

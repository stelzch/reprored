#pragma once

#include "k_chunked_array.hpp"
#include <cstdint>

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

        /** Figure out if the parts that make up a certain index are all local and form a subtree
        * of a specifc size */
        const bool is_local_subtree_of_size(const uint64_t expectedSubtreeSize, const uint64_t i) const;
        const vector<uint64_t>& get_rank_intersecting_summands(void) const;

        const uint64_t get_starting_index() const;
        const uint64_t get_end_index() const;
        const uint64_t get_global_size() const;
        const uint64_t get_local_size() const;
        
    private:
        const uint64_t rank;
        const uint64_t clusterSize;
        const uint64_t globalSize;
        const vector<region> regions;
        const uint64_t size,  begin, end;
        const map<uint64_t, int> start_indices;
        const vector<uint64_t> rank_intersecting_summands;
    protected:

        /* Calculate all rank-intersecting summands that must be sent out because
        * their parent is non-local and located on another rank
        */
        vector<uint64_t> calculateRankIntersectingSummands(void) const;
        const map<uint64_t, int> calculate_k_start_indices() const;

};

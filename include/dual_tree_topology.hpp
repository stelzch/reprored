#pragma once
#include <cassert>
#include <cmath>
#include <string>
#include <util.hpp>

#include <bit>
#include <set>
#include <utility>
#include <vector>

#include "MAryTree.hpp"

using std::pair;
using std::vector;

using TreeCoordinates = pair<uint64_t, uint32_t>; // x and y coordinate


typedef bool operation;
constexpr auto OPERATION_PUSH = true; /// Consume value from available intermediate results and push onto working stack
constexpr auto OPERATION_REDUCE = false; /// Reduce & consume two top-most values from working stack and push result

/**
 * Defines operations to be performed on a given rank.
 */
struct operation_result {
    vector<operation> ops; /// Steps to perform on local results and in values from inbox
    vector<TreeCoordinates> local_coords; /// Coordinates that must be computed from PE-local array data
};

template<>
struct std::hash<TreeCoordinates> {
    std::size_t operator()(const TreeCoordinates &k) const noexcept {
        return k.first ^ static_cast<uint64_t>(k.second);
    }
};

class DualTreeTopology {
public:
    /**
     * Construct class representing a dual tree topology where
     *  (a) one binary tree spanning the array elements defines the reduction order (reduction tree) and
     *  (b) another m-ary tree spanning the list of processing elements (PEs) defines the communication order (comm
     * tree)
     *
     *  We require that the regions are allocated in ascending order, i.e. the first few elements must lie on rank 0,
     * the next on rank 1 and so on.
     * @param rank Rank of the calling process
     * @param regions List of regions
     * @param m construct comm tree as m-ary tree
     */
    DualTreeTopology(const int rank, const vector<region> &regions, const unsigned int m = 2) :
        rank{rank},
        cluster_size{regions.size()},
        is_last_rank(compute_is_last_rank(rank, regions)),
        local_start_index(regions.at(rank).globalStartIndex),
        local_end_index(local_start_index + regions[rank].size),
        global_size{compute_global_size(regions)},
        comm_tree{cluster_size, m},
        comm_children(comm_tree.subtree_children(rank)),
        largest_comm_child{comm_tree.largest_child_index(rank)},
        comm_end_index(compute_global_comm_end_index(rank, regions)),
        outgoing{compute_outgoing(regions)} {

        assert(!regions.empty());
        for (auto i = 0U; i < regions.size() - 1; ++i) {
            assert(regions.at(i).globalStartIndex <= regions[i + 1].globalStartIndex);
        }
        assert(local_end_index <= comm_end_index);
    };


    /// Get coordinates of intermediate results that are computed on this rank and sent out accordingly.
    const vector<TreeCoordinates> &get_outgoing() const { return outgoing; }

    const vector<uint64_t> &get_comm_children() const { return comm_children; }

    // Helper functions
    /**
     * Get largest index inside subtree rooted at given argument.
     * Does not do boundary checks at the end of the tree.
     */
    static uint64_t largest_child_index(const uint64_t index) { return index | (index - 1); }

    static uint64_t subtree_size_untrunc(const uint64_t index) {
        assert(index != 0);
        return largest_child_index(index) + 1 - index;
    }

    /**
     * Get the number of elements part of the subtree rooted at the given argument.
     * Index must be non-zero.
     */
    uint64_t subtree_size(const uint64_t index) const {
        assert(index != 0);
        return std::min(global_size, largest_child_index(index) + 1) - index;
    }

    /**
     * Get index of parent element.
     * Index must be non-zero.
     */
    static uint64_t parent(const uint64_t i) {
        assert(i != 0);

        // clear least significand set bit
        return i & (i - 1);
    }

    /**
     * Get the maximum level of the subtree rooted at given index.
     * Does not perform boundary checks.
     */
    static inline uint16_t max_y_untrunc(const uint32_t index, const uint64_t global_size) {
        return std::countr_zero(index);
    }
    /**
     * Get the maximum level of the subtree rooted at given index.
     * Does perform boundary checks at the end of the tree in case the global number of elements is not a power of 2.
     */
    static inline uint16_t max_y(const uint32_t index, const uint64_t global_size) {
        if (index != 0 && largest_child_index(index) < global_size) {
            // Normal case, i.e. non-zero index without truncated subtree
            return max_y_untrunc(index, global_size);
        } else {
            const auto end_index = index == 0 ? global_size : std::min(global_size, largest_child_index(index) + 1);
            return std::ceil(std::log2(end_index - index));
        }
    }

    static constexpr int32_t pow2(uint32_t x) { return 1 << x; }
    static uint64_t compute_global_size(const vector<region> &regions) {
        uint64_t global_size{0};

        for (const auto &region: regions) {
            global_size += region.size;
        }

        return global_size;
    }

    bool is_subtree_local(const uint64_t x, const int32_t y) const {
        if (y > 0) {
            const auto largest_child_index = std::min(x + pow2(y) - 1, global_size - 1);
            return largest_child_index >= local_start_index && largest_child_index < local_end_index;
        } else {
            assert(y >= 0);
            return x >= local_start_index && x < local_end_index;
        }
    }
    bool is_subtree_local(const TreeCoordinates &coords) const { return is_subtree_local(coords.first, coords.second); }

    bool is_subtree_comm_local(const uint64_t x, const int32_t y) const {
        if (y > 0) {
            const auto largest_child_index = std::min(x + pow2(y) - 1, global_size - 1);
            return largest_child_index >= local_start_index && largest_child_index < comm_end_index;
        } else {
            assert(y >= 0);
            return x >= local_start_index && x < comm_end_index;
        }
    }

    operation_result compute_operations(const std::set<TreeCoordinates> &incoming) const {
        operation_result result;

        for (const auto &[x, y]: outgoing) {
            compute_operations_rec(incoming, result, x, y);
        }

        return result;
    }

    uint64_t get_local_size() const { return local_end_index - local_start_index; }
    uint64_t get_local_start_index() const { return local_start_index; }
    uint64_t get_local_end_index() const { return local_end_index; }
    uint64_t get_global_size() const { return global_size; }

    uint64_t get_comm_parent() const { return comm_tree.parent(rank); }

private:
    /**
     * Determine the set of operations \p result that transform an inbox of \p incoming coordinates into an outgoing
     * coordinate \p x, \p y.
     */
    void compute_operations_rec(const std::set<TreeCoordinates> &incoming, operation_result &result, uint64_t x,
                                uint32_t y) const {
        if (incoming.contains(TreeCoordinates(x, y))) {
            result.ops.push_back(OPERATION_PUSH);
            return;
        }

        if (y == 0 || is_subtree_local(x, y)) {
            result.local_coords.push_back(TreeCoordinates(x, y));
            result.ops.push_back(OPERATION_PUSH);
            return;
        }

        const TreeCoordinates left_child = {x, y - 1};
        const TreeCoordinates right_child = {x + pow2(y - 1), y - 1};


        compute_operations_rec(incoming, result, left_child.first, left_child.second);
        if (right_child.first < get_global_size()) {
            compute_operations_rec(incoming, result, right_child.first, right_child.second);
            result.ops.push_back(OPERATION_REDUCE);
        }
    }

    // Constructor-related functions
    vector<TreeCoordinates> compute_outgoing(const vector<region> &regions) const {
        vector<TreeCoordinates> outgoing;

        if (local_start_index == local_end_index) {
            return outgoing;
        }

        uint64_t x = local_start_index;

        while (x < comm_end_index) {
            for (int32_t y = max_y(x, global_size); y >= 0; --y) {
                if (is_subtree_local(x, y)) {
                    outgoing.emplace_back(x, y);
                    x += pow2(y);
                    break; // continue with new x index
                }

                if (is_subtree_comm_local(x, y)) {
                    // add to outbox
                    outgoing.emplace_back(x, y);

                    x += pow2(y);
                    break;
                }
            }
        }


        return outgoing;
    }

    uint64_t compute_global_comm_end_index(uint64_t rank, const vector<region> &regions) const {
        if (is_last_rank || largest_comm_child >= cluster_size - 1) {
            return global_size;
        }

        const auto successor_region =
                std::find_if(regions.begin() + largest_comm_child + 1, regions.end(), region_not_empty);
        if (successor_region == regions.end()) {
            return global_size;
        }

        return successor_region->globalStartIndex;
    }

    bool compute_is_last_rank(const int rank, const vector<region> &regions) {
        if (rank == cluster_size - 1) {
            return true;
        }

        const auto next_pe_with_elements = std::find_if(regions.begin() + rank + 1, regions.end(), region_not_empty);

        return next_pe_with_elements == regions.end();
    }


    // Member variables
    const int rank;
    const uint64_t cluster_size;
    const bool is_last_rank;

    /** Global index of the first element this PE holds locally. */
    const uint64_t local_start_index;


    /** Global index of first element that is no longer located on this rank. */
    const uint64_t local_end_index;

    /** Total number of elements */
    const uint64_t global_size;


    const MAryTree comm_tree;
    const vector<uint64_t> comm_children;

    /** Largest rank that sends their result to us, possibly over an intermediary. */
    const uint64_t largest_comm_child;

    /** The global index of the first element that no longer located on a rank that sends their intermediate
     * results to us. I.e. the global start index of the first PE with a higher rank than ours who sends their results
     * to a rank lower than us. This is important because we do not have to wait on intermediate results with index
     * higher or equal to this field.
     */
    const uint64_t comm_end_index;

    /// TreeCoordinates that are sent outwards for processing on the parent rank
    const vector<TreeCoordinates> outgoing;
};

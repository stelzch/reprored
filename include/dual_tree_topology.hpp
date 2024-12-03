#pragma once
#include <cassert>
#include <util.hpp>

#include <utility>
#include <vector>

using std::pair;
using std::vector;

using TreeCoordinates = pair<uint64_t, uint32_t>; // x and y coordinate

class DualTreeTopology {
public:
    /**
     * Construct class representing a dual tree topology where
     *  (a) one binary tree spanning the array elements defines the reduction order (reduction tree) and
     *  (b) another binary tree spanning the list of processing elements (PEs) defines the communication order (comm
     * tree)
     *
     *  We require that the regions are allocated in ascending order, i.e. the first few elements must lie on rank 0,
     * the next on rank 1 and so on.
     * @param rank Rank of the calling process
     * @param regions List
     */
    DualTreeTopology(int rank, const vector<region> &regions) :
        clusterSize{regions.size()}, outgoing{compute_outgoing(regions)} {
        assert(regions.size() > 0);
        for (auto i = 0U; i < regions.size() - 1; ++i) {
            assert(regions[i].globalStartIndex <= regions[i + 1].globalStartIndex);
        }
    };


    const vector<TreeCoordinates> &get_outgoing() const { return outgoing; }
    const vector<TreeCoordinates> &get_incoming() const { return incoming; }

private:
    const size_t clusterSize;
    const vector<TreeCoordinates> outgoing;


private:
    const vector<TreeCoordinates> incoming;


    vector<TreeCoordinates> compute_outgoing(const vector<region> &regions) { return {}; }
};

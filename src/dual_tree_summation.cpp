#include "dual_tree_summation.hpp"

#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <set>

#include "binary_tree_summation.h"

#ifdef SCOREP
#include <scorep/SCOREP_User.h>
#endif


DualTreeSummation::DualTreeSummation(uint64_t rank, const vector<region> regions_, MPI_Comm comm) :
    comm{comm},
    comm_size(regions_.size()),
    rank{rank},
    regions{compute_normalized_regions(regions_)},
    rank_order{compute_rank_order(regions)},
    inverse_rank_order{compute_inverse_rank_order(rank_order)},
    topology{rank_to_array_order(static_cast<int>(rank)), compute_permuted_regions(regions)},
    outgoing(topology.get_outgoing()),
    rank_of_comm_parent(
            rank_to_array_order(rank) == 0 ? -1 : array_to_rank_order(topology.parent(rank_to_array_order(rank)))),
    is_root(rank_to_array_order(DualTreeSummation::rank) == 0) {
    accumulation_buffer.resize(topology.get_local_size());

    assert(rank_to_array_order(rank) != 0 || is_root);
    assert(regions[array_to_rank_order(0)].size > 0); // Need elements on first rank

    incoming_element_count.reserve(topology.get_comm_children().size());

    // Comm child ranks must be sorted from low to high because that implies that the indices of their elements are also
    // sorted low to high. This is important in order to obtain correct ordering inside the inbox array.
    assert(std::is_sorted(topology.get_comm_children().begin(), topology.get_comm_children().end()));

    // Determine incoming + outgoing values
    // 1. Receive incoming values from children
    std::set<TreeCoordinates> incoming_coordinates;
    for (auto permuted_child_rank: topology.get_comm_children()) {
        const auto child_rank = array_to_rank_order(permuted_child_rank);
        uint64_t count;
        MPI_Recv(&count, 1, MPI_UINT64_T, child_rank, OUTGOING_SIZE_MSG_TAG, comm, MPI_STATUS_IGNORE);

        incoming_element_count.push_back(count);

        vector<TreeCoordinates> incoming_from_child(count);
        MPI_Recv(incoming_from_child.data(), count * sizeof(TreeCoordinates), MPI_BYTE, child_rank, OUTGOING_MSG_TAG,
                 comm, MPI_STATUS_IGNORE);

        for (auto tc: incoming_from_child) {
            incoming_coordinates.insert(tc);
        }
    }

    // 2. Send out our outgoing values to parent in comm tree
    if (!is_root) {
        uint64_t count = outgoing.size();
        MPI_Send(&count, 1, MPI_UINT64_T, rank_of_comm_parent, OUTGOING_SIZE_MSG_TAG, comm);
        MPI_Send(outgoing.data(), count * sizeof(TreeCoordinates), MPI_BYTE, rank_of_comm_parent, OUTGOING_MSG_TAG,
                 comm);
    }

    for (auto &[x, y]: outgoing) {
        compute_operations(incoming_coordinates, operations, local_compute_coords, x, y);
    }

    // Compute maximum size of local accumulation stack
    auto maximum_stack_size = 0UL;
    {
        auto stack_size = 0UL;

        for (const auto op: operations) {
            if (op == OPERATION_REDUCE) {
                assert(stack_size >= 2);
                --stack_size;
            } else {
                ++stack_size;
            }
            maximum_stack_size = std::max(maximum_stack_size, stack_size);
        }

        stack.reserve(maximum_stack_size);
    }


    inbox.reserve(local_compute_coords.size() + incoming_coordinates.size());

#ifdef DEBUG_TRACE
    printf("rank %lu (permuted %i) region %zu-%zu max_stack_size %zu inbox capacity %zu incoming ", rank,
           rank_to_array_order(rank), regions[rank].globalStartIndex,
           regions[rank].globalStartIndex + regions[rank].size, maximum_stack_size, inbox.capacity());

    for (const auto &e: incoming_coordinates) {

        printf("(%lu, %i) ", e.first, e.second);
    }
    printf(" outgoing ");
    for (const auto &v: outgoing) {
        printf("(%zu, %u)", v.first, v.second);
    }

    printf("\n");
#endif
}

DualTreeSummation::~DualTreeSummation() {}


double *DualTreeSummation::getBuffer() { return accumulation_buffer.data(); }
uint64_t DualTreeSummation::getBufferSize() { return accumulation_buffer.size(); }

void DualTreeSummation::storeSummand(uint64_t localIndex, double val) { accumulation_buffer[localIndex] = val; }

void DualTreeSummation::receive_values_into_inbox() {
    for (auto i = 0U; i < topology.get_comm_children().size(); ++i) {
        const uint64_t count = incoming_element_count[i];

        const auto permuted_child_rank = topology.get_comm_children()[i];
        const auto child_rank = array_to_rank_order(permuted_child_rank);

        // Insert elements at the end of the inbox
        const auto buf_start_index = inbox.size();
        inbox.resize(inbox.size() + count);

        MPI_Recv(&inbox[buf_start_index], count, MPI_DOUBLE, child_rank, TRANSFER_MSG_TAG, comm, MPI_STATUS_IGNORE);
    }
}
double DualTreeSummation::broadcast_result() {
    double result;
    if (is_root) {
        assert(stack.size() == 1);
        result = stack.at(0);
    }

    MPI_Bcast(&result, 1, MPI_DOUBLE, array_to_rank_order(0), comm);
    return result;
}
void DualTreeSummation::send_outgoing_values() {
    assert(stack.size() == outgoing.size());
    MPI_Send(stack.data(), outgoing.size(), MPI_DOUBLE, rank_of_comm_parent, TRANSFER_MSG_TAG, comm);
}
double DualTreeSummation::accumulate(void) {
    inbox.clear();

    // 1. Reduce all fully local subtrees
    local_accumulate_into_inbox();

    // 2. Receive all values from child nodes
    receive_values_into_inbox();

    // 3. Compute values
    execute_operations();

    // 4. Send out computed values
    if (!is_root) {
        send_outgoing_values();
    }

    // 4. Broadcast global value
    return broadcast_result();
}

/** Special case where the subtree under (x,y) is fully local, we do not need to perform any boundary checks */
double DualTreeSummation::local_accumulate(uint64_t x, uint32_t maxY) {
    if (maxY == 0) {
        return accumulation_buffer.at(x - topology.get_local_start_index());
    }

    // Iterative approach
    const auto end_index = std::min(x + topology.pow2(maxY), topology.get_global_size());
    uint64_t elementsInBuffer = end_index - x;

    double *buffer = &accumulation_buffer.at(x - topology.get_local_start_index());


    constexpr auto stride = 8;
    for (int y = 1; y <= maxY; y += 3) {
        uint64_t elementsWritten = 0;

        for (uint64_t i = 0; i + stride <= elementsInBuffer; i += stride) {
            __m256d a = _mm256_loadu_pd(&buffer[i]);
            __m256d b = _mm256_loadu_pd(&buffer[i + 4]);
            __m256d level1Sum = _mm256_hadd_pd(a, b);

            __m128d c = _mm256_extractf128_pd(level1Sum, 1); // Fetch upper 128bit
            __m128d d = _mm256_castpd256_pd128(level1Sum); // Fetch lower 128bit
            __m128d level2Sum = _mm_add_pd(c, d);

            __m128d level3Sum = _mm_hadd_pd(level2Sum, level2Sum);

            buffer[elementsWritten++] = _mm_cvtsd_f64(level3Sum);
        }

        const auto remainder = elementsInBuffer - stride * elementsWritten;
        if (remainder) {
            const double a = sum_remaining_8tree(remainder, y, &buffer[stride * elementsWritten]);
            buffer[elementsWritten++] = a;
        }
        elementsInBuffer = elementsWritten;
    }

    assert(elementsInBuffer == 1);

    return buffer[0];
}

void DualTreeSummation::execute_operations() {
    stack.clear();

    auto it = inbox.begin();

    for (const auto op: operations) {
        if (op == OPERATION_PUSH) {
            stack.push_back(*it);
            ++it;
        } else {
            assert(op == OPERATION_REDUCE);
            assert(stack.size() >= 2);

            const double b = stack.back();
            stack.pop_back();
            const double a = stack.back();
            stack.pop_back();

            stack.push_back(a + b);
        }
    }

    assert(stack.size() == outgoing.size());
}

void DualTreeSummation::local_accumulate_into_inbox() {
    for (auto &[x, y]: local_compute_coords) {
        inbox.push_back(local_accumulate(x, y));
    }
}

void DualTreeSummation::compute_operations(const std::set<TreeCoordinates> &incoming, vector<operation> &ops,
                                           vector<TreeCoordinates> &local_coords, uint64_t x, uint32_t y) {
    if (incoming.contains(TreeCoordinates(x, y))) {
        ops.push_back(OPERATION_PUSH);
        return;
    }

    if (y == 0 || topology.is_subtree_local(x, y)) {
        local_coords.push_back(TreeCoordinates(x, y));
        ops.push_back(OPERATION_PUSH);
        return;
    }

    const TreeCoordinates left_child = {x, y - 1};
    const TreeCoordinates right_child = {x + DualTreeTopology::pow2(y - 1), y - 1};


    compute_operations(incoming, ops, local_coords, left_child.first, left_child.second);
    if (right_child.first < topology.get_global_size()) {
        compute_operations(incoming, ops, local_coords, right_child.first, right_child.second);
        ops.push_back(OPERATION_REDUCE);
    }
}

vector<region> DualTreeSummation::compute_normalized_regions(const vector<region> &regions) const {
    const auto global_size = total_region_size(regions.begin(), regions.end());
    vector<region> result;

    for (const auto &r: regions) {
        if (r.size == 0) {
            result.emplace_back(global_size, 0);
        } else {
            result.emplace_back(r.globalStartIndex, r.size);
        }
    }

    return result;
}

const vector<int> DualTreeSummation::compute_rank_order(const vector<region> &regions) const {
    vector<int> rank_order(comm_size);

    std::iota(rank_order.begin(), rank_order.end(), 0);
    std::sort(rank_order.begin(), rank_order.end(), [&regions](const int a, const int b) {
        const auto &region_a = regions.at(a);
        const auto &region_b = regions.at(b);
        return region_a.globalStartIndex < region_b.globalStartIndex;
    });

    const bool no_elements_on_first_pe = regions.at(rank_order[0]).size == 0;
    if (no_elements_on_first_pe) {

        // We require that there are elements on the (logical, i.e. permuted) first rank
        // If the distribution assigns zero elements to the first element, find the first rank that has
        // elements assigned and bring it to the front
        auto first_rank_with_elements = std::find_if(rank_order.begin(), rank_order.end(), [&regions](const auto i) {
            return region_not_empty(regions.at(i));
        });

        assert(first_rank_with_elements != rank_order.end());
        assert(regions.at(*first_rank_with_elements).globalStartIndex == 0);
        std::iter_swap(first_rank_with_elements, rank_order.begin());
    }

    return rank_order;
}

const vector<int> DualTreeSummation::compute_inverse_rank_order(const vector<int> &rank_order) const {
    vector<int> inverse(rank_order.size());

    for (auto i = 0U; i < rank_order.size(); ++i) {
        inverse[rank_order[i]] = i;
    }

    return inverse;
}

const vector<region> DualTreeSummation::compute_permuted_regions(const vector<region> &regions) const {
    vector<region> result(regions.size());

    for (auto i = 0U; i < regions.size(); ++i) {
        result[i] = regions[array_to_rank_order(i)];
    }

    return result;
}

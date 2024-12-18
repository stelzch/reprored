#include "dual_tree_summation.hpp"

#include <cmath>
#include <immintrin.h>
#include <iostream>


#ifdef DEBUG_TRACE
#include <format>
auto print_tuple(const TreeCoordinates &rhs) { return std::format("({}, {})", rhs.first, rhs.second); }
#endif

DualTreeSummation::DualTreeSummation(uint64_t rank, const vector<region> &regions, MPI_Comm comm) :
    comm{comm},
    comm_size(regions.size()),
    rank{rank},
    rank_order{compute_rank_order(regions)},
    inverse_rank_order{compute_inverse_rank_order(rank_order)},
    regions{regions},
    topology{rank_to_array_order(static_cast<int>(rank)), compute_permuted_regions(regions)},
    outgoing(topology.get_locally_computed()),
    rank_of_comm_parent(
            rank_to_array_order(rank) == 0 ? -1 : array_to_rank_order(topology.parent(rank_to_array_order(rank)))),
    is_root(rank_to_array_order(DualTreeSummation::rank) == 0) {
    accumulation_buffer.resize(topology.get_local_size());

    assert(regions[array_to_rank_order(0)].size > 0); // Need elements on first rank

    // Determine incoming + outgoing values
    // 1. Receive incoming values from children
    incoming.clear();
    for (auto permuted_child_rank: topology.get_comm_children()) {
        const auto child_rank = array_to_rank_order(permuted_child_rank);
        uint64_t count;
        MPI_Recv(&count, 1, MPI_UINT64_T, child_rank, OUTGOING_SIZE_MSG_TAG, comm, MPI_STATUS_IGNORE);

        vector<TreeCoordinates> incoming_from_child(count);
        MPI_Recv(incoming_from_child.data(), count * sizeof(TreeCoordinates), MPI_BYTE, child_rank, OUTGOING_MSG_TAG,
                 comm, MPI_STATUS_IGNORE);

        incoming[child_rank] = incoming_from_child;

        for (const auto &coords: incoming_from_child) {
            // If a certain value is not consumed in the local computations, simply pass it along the communication tree
            if (rank_to_array_order(rank) != 0 && topology.is_passthrough_element(coords.first, coords.second)) {
                passthrough_elements.push_back(coords);
                outgoing.push_back(coords);
            }
        }
    }

    // 2. Send out our outgoing values to parent in comm tree
    if (!is_root) {
        uint64_t count = outgoing.size();
        MPI_Send(&count, 1, MPI_UINT64_T, rank_of_comm_parent, OUTGOING_SIZE_MSG_TAG, comm);
        MPI_Send(outgoing.data(), count * sizeof(TreeCoordinates), MPI_BYTE, rank_of_comm_parent, OUTGOING_MSG_TAG,
                 comm);
    }

    size_t max_received_elements = 0;
    for (const auto &[k, v]: incoming) {
        max_received_elements = std::max(max_received_elements, v.size());
    }
    comm_buffer.resize(std::max(max_received_elements, outgoing.size()));


#ifdef DEBUG_TRACE
    std::cout << std::format("rank {} (permuted {}) region {}-{} incoming ", rank, rank_to_array_order(rank),
                             regions[rank].globalStartIndex, regions[rank].globalStartIndex + regions[rank].size);

    for (const auto &e: incoming) {

        std::cout << std::format("({} -> ", e.first);
        for (const auto &v: e.second) {
            std::cout << print_tuple(v) << " ";
        }
        std::cout << ") ";
    }
    std::cout << " outgoing ";
    for (const auto &v: outgoing) {
        std::cout << print_tuple(v) << " ";
    }
    std::cout << std::endl;

    assert(rank_to_array_order(rank) != 0 || is_root);
#endif
}

DualTreeSummation::~DualTreeSummation() {}


double *DualTreeSummation::getBuffer() { return accumulation_buffer.data(); }
uint64_t DualTreeSummation::getBufferSize() { return accumulation_buffer.size(); }

void DualTreeSummation::storeSummand(uint64_t localIndex, double val) { accumulation_buffer[localIndex] = val; }

double DualTreeSummation::accumulate(void) {
    outbox.clear();
    inbox.clear();

    // 1. Receive all values from child nodes
    for (auto permuted_child_rank: topology.get_comm_children()) {
        const auto child_rank = array_to_rank_order(permuted_child_rank);
        uint64_t count = incoming[child_rank].size();
        comm_buffer.resize(count);

#ifdef DEBUG_TRACE
        std::cout << std::format("rank {} receiving {} elements to rank {}\n", rank, count, child_rank);
#endif
        MPI_Recv(comm_buffer.data(), count, MPI_DOUBLE, child_rank, TRANSFER_MSG_TAG, comm, MPI_STATUS_IGNORE);

        // Insert values from temporary buffer into inbox map
        for (auto i = 0UL; i < count; ++i) {
            const auto key = incoming[array_to_rank_order(permuted_child_rank)][i];
            const auto value = comm_buffer[i];

            if (topology.is_passthrough_element(key.first, key.second)) {
                outbox[key] = value;
            } else {
                inbox[key] = value;
            }
        }
    }

#ifdef DEBUG_TRACE
    for (const auto &[other_rank, coords]: incoming) {
        for (const auto &coord: coords) {
            if (!(inbox.contains(coord) || outbox.contains(coord))) {
                fprintf(stderr, "rank %i expected to receive (%lu, %lu) from rank %i, but it was not delivered\n", rank,
                        coord.first, coord.second, other_rank);
            }
        }
    }
    printf("rank %i computing ", rank);
#endif


    // 2. Compute local values
    for (const auto &coords: topology.get_locally_computed()) {
        outbox[coords] = accumulate(coords.first, coords.second);
#ifdef DEBUG_TRACE
        printf(" (%lu, %u) = %f  ", coords.first, coords.second, outbox[coords]);

#endif
    }
#ifdef DEBUG_TRACE
    printf("\n");
#endif

    assert(rank_to_array_order(rank) != 0 || is_root);

    // 3. Send out computed values
    if (!is_root) {
        comm_buffer.resize(outgoing.size());
        for (auto i = 0UL; i < outgoing.size(); ++i) {
            const auto key = outgoing[i];
            comm_buffer[i] = outbox.at(key);
        }
#ifdef DEBUG_TRACE
        std::cout << std::format("rank {} sending {} elements to rank {}\n", rank, outgoing.size(),
                                 rank_of_comm_parent);
#endif
        MPI_Send(comm_buffer.data(), outgoing.size(), MPI_DOUBLE, rank_of_comm_parent, TRANSFER_MSG_TAG, comm);
    }

    // 4. Broadcast global value
    double result;
    if (is_root) {
        const TreeCoordinates root_coord = {0, topology.max_y(0, topology.get_global_size())};
        result = outbox.at(root_coord);
    }

    MPI_Bcast(&result, 1, MPI_DOUBLE, array_to_rank_order(0), comm);

    return result;
}


double DualTreeSummation::fetch_or_accumulate(uint64_t x, uint32_t y) {
    const auto &e = inbox.find(TreeCoordinates(x, y));

    if (e == inbox.end()) {
        return accumulate(x, y);
    } else {
        return e->second;
    }
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

double DualTreeSummation::accumulate(uint64_t x, uint32_t y) {
#ifdef DEBUG_VERBOSE
    std::cout << std::format("rank {} reducing ({}, {})\n", rank, x, y);
#endif


    if (topology.is_subtree_local(x, y)) {
        return local_accumulate(x, y);
    } else if (y == 0) {
        try {
            return accumulation_buffer.at(x - topology.get_local_start_index());
        } catch (...) {
            fprintf(stderr, "rank %lu could not read value %lu from local buffer starting at %lu\n", rank, x,
                    topology.get_local_start_index());
            assert(0);
        }
    }

    const TreeCoordinates left_child = {x, y - 1};
    const TreeCoordinates right_child = {x + DualTreeTopology::pow2(y - 1), y - 1};


    double left_child_val = fetch_or_accumulate(left_child.first, left_child.second);
    if (right_child.first < topology.get_global_size()) {
        double right_child_val = fetch_or_accumulate(right_child.first, right_child.second);
        return left_child_val + right_child_val;
    } else {
        return left_child_val;
    }
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

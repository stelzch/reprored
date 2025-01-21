#include <unordered_map>
#include <vector>
#include "binary_tree.hpp"
#include "dual_tree_topology.hpp"
#include "summation.hpp"

#include <set>

using std::unordered_map;

// https://xkcd.com/221
constexpr int OUTGOING_SIZE_MSG_TAG = 20232;
constexpr int OUTGOING_MSG_TAG = 20233;
constexpr int TRANSFER_MSG_TAG = 20234;


/**
 * Reproducible summation using two binary trees: a reduction tree that defines the order of computation and a
 * communication tree that defines the direction of communication. Each rank receives intermediate results from
 * its child nodes in the communication tree, reduces them as much as possible together with the "local" array
 * values and sends them out to the communication tree parent.
 *
 * Figuring out which of the received values can be used with local computations and which must be passed on is
 * non-trivial. To avoid large overhead during the reduction, we use a precomputed set of operations on a stack.
 * The #inbox vector stores the reduction results of the local elements alongside the intermediate results
 * received from other ranks. Because these are stored in the same order as they are consumed, we do not need to
 * keep track of the exact coordinates. The operations then encode when to push a value from the inbox onto the
 * stack and when to reduce the two topmost values.
 */
class DualTreeSummation : public Summation {
public:
    DualTreeSummation(uint64_t rank, const vector<region> regions, MPI_Comm comm = MPI_COMM_WORLD,
                      const unsigned int m = 2);


    virtual ~DualTreeSummation();

    double *getBuffer() override;
    uint64_t getBufferSize() const;
    void storeSummand(uint64_t localIndex, double val);

    /* Sum all numbers. Will return the total sum on rank 0
     */
    double accumulate(void) override;


private:
    std::set<TreeCoordinates> exchange_coordinates(MPI_Comm comm);
    void receive_incoming_coordinates(MPI_Comm comm, std::set<std::pair<uint64_t, uint32_t>> &incoming_coordinates);
    void send_outgoing_coordinates(MPI_Comm comm) const;
    unsigned long compute_maximum_stack_size() const;
    double local_accumulate(uint64_t x, uint32_t y);
    void local_accumulate_into_inbox();
    void execute_operations();
    void trigger_receive_requests();
    double broadcast_result() const;
    void send_outgoing_values() const;
    inline auto array_to_rank_order(const int rank) const { return rank_order[rank]; }
    inline auto rank_to_array_order(const int rank) const { return inverse_rank_order[rank]; }
    double accumulate(uint64_t x, uint32_t y);
    void flush();

    static vector<region> compute_normalized_regions(const vector<region> &regions);
    vector<int> compute_rank_order(const vector<region> &regions) const;
    vector<int> compute_inverse_rank_order(const vector<int> &rank_order) const;
    vector<region> compute_permuted_regions(const vector<region> &regions) const;

    double sum_remaining_8tree(const uint64_t initialRemainingElements, const int y, double *buffer) const {
        uint64_t remainingElements = initialRemainingElements;

        for (int level = 0; level < 3; level++) {
            int elementsWritten = 0;
            for (uint64_t i = 0; (i + 1) < remainingElements; i += 2) {
                buffer[elementsWritten++] = buffer[i] + buffer[i + 1];
            }

            if (remainingElements % 2 == 1) {
                // indexB is the last element because the subtree ends there
                const uint64_t bufferIndexA = remainingElements - 1;
                buffer[elementsWritten++] = buffer[remainingElements - 1];

                remainingElements += 1;
            }

            remainingElements /= 2;
        }
        assert(remainingElements == 1);

        return buffer[0];
    }


    const MPI_Comm comm;
    const uint64_t comm_size;
    const uint64_t rank;

    const vector<region> regions;

    /**
     * DualTreeTopology assumes that ranks are ordered in ascending order of the array indices (i.e. the first elements
     * are on rank 0, the next on rank 1 and so on) This might not necessarily be true (e.g. RAxML-NG can assign the
     * last elements to rank 0). To keep track, we keep this permutation which maps our MPI rank to the ordering of the
     * global array.
     *
     *                              ┌─────────────┐┌──────┐┌────────────┐
     *              ordered by rank │     PE0     ││  PE1 ││    PE2     │
     *                              └──────┬──────┘└───┬──┘└──────┬─────┘
     *               displacement   12     │       25  │    0     │
     *                                     │           │          │
     *                                    ┌┼───────────┼──────────┘
     *                                    ││           └────────────┐
     *                                    │└────────────┐           │
     *                                    │             │           │
     *                              ┌─────▼──────┐┌─────▼───────┐┌──▼───┐
     *       ordered by array index │     PE2    ││      PE0    ││ PE1  │
     *                              └────────────┘└─────────────┘└──────┘
     *                              0             12             25
     *
     * In this example, rank_order = (2, 0, 1) and inverse_rank_order = (1, 2, 0)
     * rank_order maps from array order to PE rank
     * inverse_rank_order maps from PE rank to array order
     */
    const vector<int> rank_order; ///< maps array order -> PE rank
    const vector<int> inverse_rank_order; ///< maps PE rank -> array order

    const DualTreeTopology topology;
    vector<uint64_t> incoming_element_count; ///< Number of elements received from each child rank.

    operation_result operations;
    vector<double> stack; ///< Stack for intermediate values during computations

    vector<double, AlignedAllocator<double>> accumulation_buffer;
    vector<double> inbox; ///< Storage of incoming elements

    const int rank_of_comm_parent;
    const bool is_root;


    vector<MPI_Request> requests;
};

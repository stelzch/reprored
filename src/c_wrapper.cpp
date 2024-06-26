#include <vector>
#include <stdint.h>
#include <mpi.h>
#include <binary_tree_summation.h>
#include <numeric>
#include "binary_tree.hpp"

/* TODO: Remove global state. This is a crude hack. */
MPI_Comm default_communicator = MPI_COMM_WORLD;

void set_default_reduction_context_communicator(void *communicator) {
    MPI_Comm comm = static_cast<MPI_Comm>(communicator);
    default_communicator = comm;
}

ReductionContext new_reduction_context(int global_start_idx, int local_summands) {
    return new_reduction_context_comm(global_start_idx, local_summands, static_cast<void *>(default_communicator));
}

ReductionContext new_reduction_context_comm(int global_start_idx, int local_summands, void *communicator) {
    MPI_Comm comm = static_cast<MPI_Comm>(communicator);

    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    region r;
    r.globalStartIndex = global_start_idx;
    r.size = local_summands;

    std::vector<region> regions(size, r);

    MPI_Allgather(&r, sizeof(region), MPI_BYTE,
                  &regions[0], sizeof(region), MPI_BYTE,
                  comm);

    return new BinaryTreeSummation(rank, std::move(regions), comm);
}

double * get_reduction_buffer(ReductionContext ctx) {
    auto *ptr = static_cast<BinaryTreeSummation *>(ctx);

    return ptr->getBuffer();
}


union num {
    double val;
    unsigned char bytes[8];
};

uint64_t reduction_counter = 0;

double reproducible_reduce(ReductionContext ctx) {
    auto *ptr = static_cast<BinaryTreeSummation *>(ctx);

    double result = ptr->accumulate();
#ifdef TRACE
    if (ptr->get_rank() == 0) {
        union num n;
        n.val = result;

        printf("reproducible_reduce call %lu = %f (0x",
                reduction_counter,
                result);
        for (int i=0; i<8; i++) {
            printf("%02x", n.bytes[i]);
        }
        printf(")\n");
    }
#endif


    ++reduction_counter;
    return result;
}

void free_reduction_context(ReductionContext ctx) {
    auto *ptr = static_cast<BinaryTreeSummation *>(ctx);

    delete ptr;
}
void store_summand(ReductionContext ctx, uint64_t local_idx, double val) {
    auto *ptr = static_cast<BinaryTreeSummation *>(ctx);

    ptr->storeSummand(local_idx, val);
}

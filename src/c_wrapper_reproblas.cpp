#include "binary_tree_summation.h"
#include <stdint.h>
#include <mpi.h>
#include <reproblas_summation.hpp>

extern "C" {
#include <binned.h>
#include <binnedBLAS.h>
#include <reproBLAS.h>
#include <binnedMPI.h>
}

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
    ReductionContext *ctx = reinterpret_cast<ReductionContext *>(new ReproblasSummation(comm, local_summands));

    return ctx;
}

double * get_reduction_buffer(ReductionContext ctx) {
    auto *ptr = static_cast<ReproblasSummation *>(ctx);

    return ptr->getBuffer();
}


union num {
    double val;
    unsigned char bytes[8];
};

uint64_t reduction_counter = 0;

double reproducible_reduce(ReductionContext ctx) {
    auto *ptr = static_cast<ReproblasSummation *>(ctx);
    ++reduction_counter;

    return ptr->accumulate();
}

void free_reduction_context(ReductionContext ctx) {
    auto *ptr = static_cast<ReproblasSummation *>(ctx);
    delete ptr;
}
void store_summand(ReductionContext ctx, uint64_t local_idx, double val) {
    auto *ptr = static_cast<ReproblasSummation *>(ctx);

    ptr->getBuffer()[local_idx] =  val;
}

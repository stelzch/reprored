#include <stdint.h>
#include <mpi.h>
#include <binary_tree_summation.h>

extern "C" {
#include <binned.h>
#include <binnedBLAS.h>
#include <reproBLAS.h>
#include <binnedMPI.h>
}

/* TODO: Remove global state. This is a crude hack. */
MPI_Comm default_communicator = MPI_COMM_WORLD;

struct MReductionContext {
    double *buffer;
    size_t local_summands;
    int rank;
    MPI_Comm comm;
};

void set_default_reduction_context_communicator(void *communicator) {
    MPI_Comm comm = static_cast<MPI_Comm>(communicator);
    default_communicator = comm;
}

ReductionContext new_reduction_context(int global_start_idx, int local_summands) {
    return new_reduction_context_comm(global_start_idx, local_summands, static_cast<void *>(default_communicator));
}

ReductionContext new_reduction_context_comm(int global_start_idx, int local_summands, void *communicator) {

    MReductionContext *ctx = new MReductionContext();
    ctx->buffer = new double[local_summands];
    ctx->local_summands = local_summands;
    ctx->comm = static_cast<MPI_Comm>(communicator);
    MPI_Comm_rank(ctx->comm, &ctx->rank);

    return ctx;
}

double * get_reduction_buffer(ReductionContext ctx) {
    auto *ptr = static_cast<MReductionContext *>(ctx);

    return ptr->buffer;
}


union num {
    double val;
    unsigned char bytes[8];
};

uint64_t reduction_counter = 0;

double reproducible_reduce(ReductionContext ctx) {
    auto *ptr = static_cast<MReductionContext *>(ctx);

    /* Adopted from the ReproBLAS MPI_sum_sine.c example, line 105 onwards */
    double_binned *isum = NULL;
    double_binned *local_isum = binned_dballoc(3);

    binned_dbsetzero(3, local_isum);

    if (ptr->rank == 0) {
        isum = binned_dballoc(3);
        binned_dbsetzero(3, isum);
    }

    // Local summation
    binnedBLAS_dbdsum(3, ptr->local_summands, ptr->buffer, 1, local_isum);

    // Reducation across communicator
    MPI_Reduce(local_isum, isum, 1, binnedMPI_DOUBLE_BINNED(3),
            binnedMPI_DBDBADD(3), 0, ptr->comm);

    double sum;
    if (ptr->rank == 0) {
        sum = binned_ddbconv(3, isum);
        free(isum);
    }

    MPI_Bcast(&sum, 1, MPI_DOUBLE, 0, ptr->comm);

    free(local_isum);
    ++reduction_counter;
    return sum;
}

void free_reduction_context(ReductionContext ctx) {
    auto *ptr = static_cast<MReductionContext *>(ctx);

    delete[] ptr->buffer;
    delete ptr;
}
void store_summand(ReductionContext ctx, uint64_t local_idx, double val) {
    auto *ptr = static_cast<MReductionContext *>(ctx);

    ptr->buffer[local_idx] =  val;
}

#include <allreduce_summation.hpp>
#include <binary_tree_summation.h>
#include <binary_tree_summation.hpp>
#include <cstring>
#include <kgather_summation.hpp>
#include <mpi.h>
#include <reproblas_summation.hpp>
#include <stdint.h>
#include <string>
#include <vector>
#include "allreduce_summation.hpp"

/* TODO: Remove global state. This is a crude hack. */
MPI_Comm default_communicator = MPI_COMM_WORLD;

enum ReductionMode { ALLREDUCE, REPROBLAS, BINARY_TREE, KGATHER };
ReductionMode env2mode();
uint64_t env2k();

// Load parameters from enviornment variables
ReductionMode global_reduction_mode = env2mode();
uint64_t global_k = env2k();

void set_default_reduction_context_communicator(intptr_t communicator) {
    MPI_Comm comm = (MPI_Comm) (communicator);
    default_communicator = comm;
}

ReductionMode env2mode() {
    const char *mode_env = getenv("REPR_REDUCE");

    if (mode_env == nullptr) {
        return ALLREDUCE;
    }

    const auto mode = std::string(mode_env);

    if (mode == "REPROBLAS") {
        return REPROBLAS;
    } else if (mode == "BINARY_TREE") {
        return BINARY_TREE;
    } else if (mode == "KGATHER") {
        return KGATHER;
    } else if (mode == "ALLREDUCE") {
        return ALLREDUCE;
    } else {
        throw std::runtime_error("invalid reduction mode given in environment variable REPR_REDUCE");
    }
}

uint64_t env2k() {
    const char *k_env = getenv("REPR_REDUCE_K");
    if (k_env == nullptr) {
        return 1;
    }

    const auto k = std::stoul(std::string(k_env));

    if (k < 1) {
        throw std::runtime_error("Invalid choice of k in environment variable REPR_REDUCE_K");
    }

    return k;
}


ReductionContext new_reduction_context_comm_k(int global_start_idx, int local_summands, intptr_t communicator, int k) {
    MPI_Comm comm = (MPI_Comm) (communicator);

    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    switch (global_reduction_mode) {
        case BINARY_TREE: {
            region r;
            r.globalStartIndex = global_start_idx;
            r.size = local_summands;

            std::vector<region> regions(size, r);

            MPI_Allgather(&r, sizeof(region), MPI_BYTE, &regions[0], sizeof(region), MPI_BYTE, comm);

            return new BinaryTreeSummation(rank, std::move(regions), k, comm);
        }
        case KGATHER: {
            region r;
            r.globalStartIndex = global_start_idx;
            r.size = local_summands;

            std::vector<region> regions(size, r);

            MPI_Allgather(&r, sizeof(region), MPI_BYTE, &regions[0], sizeof(region), MPI_BYTE, comm);

            return new KGatherSummation(rank, std::move(regions), k, comm);
        }
        case REPROBLAS: {
            return new ReproblasSummation(comm, local_summands);
        }
        case ALLREDUCE:
        default: {
            return new AllreduceSummation(comm, local_summands);
        }
    }
}

ReductionContext new_reduction_context_comm(int global_start_idx, int local_summands, intptr_t communicator) {
    return new_reduction_context_comm_k(global_start_idx, local_summands, communicator, global_k);
}

ReductionContext new_reduction_context(int global_start_idx, int local_summands) {
    return new_reduction_context_comm(global_start_idx, local_summands, (intptr_t) (default_communicator));
}


double *get_reduction_buffer(ReductionContext ctx) {
    auto *ptr = static_cast<Summation *>(ctx);

    return ptr->getBuffer();
}

union num {
    double val;
    unsigned char bytes[8];
};

uint64_t reduction_counter = 0;

double reproducible_reduce(ReductionContext ctx) {
    auto *ptr = static_cast<Summation *>(ctx);

    double result = ptr->accumulate();
#ifdef TRACE
    union num n;
    n.val = result;

    printf("reproducible_reduce call %lu = %f (0x", reduction_counter, result);
    for (int i = 0; i < 8; i++) {
        printf("%02x", n.bytes[i]);
    }
    printf(")\n");
#endif

    ++reduction_counter;
    return result;
}

void free_reduction_context(ReductionContext ctx) {
    auto *ptr = static_cast<Summation *>(ctx);

    delete ptr;
}
void store_summand(ReductionContext ctx, uint64_t local_idx, double val) {
    auto *ptr = static_cast<Summation *>(ctx);

    ptr->getBuffer()[local_idx] = val;
}

void __attribute__((optimize("O0"))) attach_debugger(bool condition) {
    if (!condition)
        return;
    volatile bool attached = false;

    // also write PID to a file
    std::ofstream os("/tmp/mpi_debug.pid");
    os << getpid() << std::endl;
    os.close();

    std::cout << "Waiting for debugger to be attached, PID: " << getpid() << std::endl;
    while (!attached)
        sleep(1);
}

void attach_debugger_env() {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char *debug_rank_str = std::getenv("DEBUG_MPI_RANK");
    if (debug_rank_str != nullptr) {
        bool debug_this_rank = false;
        try {
            unsigned int debug_rank = std::stoul(debug_rank_str);
            debug_this_rank = (debug_rank == rank);
        } catch (std::invalid_argument) {
        }

        if (debug_this_rank) {
            printf("Debugging rank %i\n", rank);
        }
        attach_debugger(debug_this_rank);
    }
}

#include <fstream>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "mpi.h"
#include "binary_tree_summation.h"
#include "io.hpp"
#include <cmath>

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;

void __attribute__((optimize("O0"))) attach_debugger(bool condition) {
    if (!condition) return;
    volatile bool attached = false;

    // also write PID to a file
    std::ofstream os("/tmp/mpi_debug.pid");
    os << getpid() << endl;
    os.close();

    cout << "Waiting for debugger to be attached, PID: "
        << getpid() << endl;
    while (!attached) sleep(1);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << "file.binpsllh|file.psllh" << endl;
        return -1;
    }

    string filename(argv[1]);

    std::vector<double> data;

    if (filename.ends_with(".psllh")) {
        data = IO::read_psllh(filename);
    } else if (filename.ends_with(".binpsllh")) {
        data = IO::read_binpsllh(filename);
    } else {
        cerr << "File must end with .psllh or .binpsllh" << endl;
        return -2;
    }

    const uint64_t N = data.size();
    cout << "Summing " << N << " summands" << endl;

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    char *debug_rank_str = std::getenv("DEBUG_MPI_RANK");
    if(debug_rank_str != nullptr) {
        bool debug_this_rank = false;
        if (strlen(debug_rank_str) == comm_size) {
            debug_this_rank = (debug_rank_str[rank] == '1');
        } else {
            try {
                unsigned int debug_rank = std::stoul(debug_rank_str);
                debug_this_rank = (debug_rank == rank);
            } catch (std::invalid_argument) {}
        }

        if (debug_this_rank)
        printf("Debugging rank %i\n", rank);
        attach_debugger(debug_this_rank);
    }


    // Create remainder on last distribution
    uint64_t perRank = floor(N / comm_size);
    uint64_t remainder = N % comm_size;

    uint64_t index = 0;
    vector<int> startIndices, nSummands;
    startIndices.resize(comm_size);
    nSummands.resize(comm_size);
    for (uint64_t i = 0; i < comm_size; i++) {
        startIndices[i] = index;

        uint64_t n = (i >=  comm_size - remainder) ? (perRank + 1) : perRank;
        nSummands[i] = n;
        index += n;
    }
    nSummands.push_back(index);

    cout << "Cluster size: " << comm_size << endl;

    uint64_t start = startIndices[rank];
    int length = nSummands[rank];

    ReductionContext ctx = new_reduction_context(length, MPI_COMM_WORLD);

    // Copy data into accumulation buffer
    for (size_t i = 0; i < length; i++) {
        get_reduction_buffer(ctx)[i] = data[start + i];
    }

    double result = reproducible_reduce(ctx);

    printf("%.32f\n", result);
    free_reduction_context(ctx);

    MPI_Finalize();
}

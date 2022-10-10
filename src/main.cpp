#include <fstream>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "mpi.h"
#include "binarytreesummation.h"
#include "io.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;

#pragma GCC push_options
#pragma GCC optimize("O0")
void attach_debugger(bool condition) {
    if (!condition) return;
    bool attached = false;

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

    cout << "Summing " << data.size() << " summands" << endl;

    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    char *debug_rank_str = std::getenv("MPI_DEBUG_RANK");
    if(debug_rank_str != NULL) {
        int debug_rank = std::atoi(debug_rank_str);
        printf("Debugging rank %i\n", debug_rank);
        attach_debugger(rank == debug_rank);
    }


    uint64_t start_index = startIndex(rank, data.size(), comm_size);
    cout << "Start index of rank " << rank  << ": " << start_index << endl;

    double result = binary_tree_sum(&data[start_index], data.size());

    printf("%.32f\n", result);
    MPI_Finalize();
}

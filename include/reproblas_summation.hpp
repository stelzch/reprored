#pragma once

#include "util.hpp"
#include <mpi.h>
#include <vector>

extern "C" {
#include <binned.h>
#include <binnedBLAS.h>
#include <reproBLAS.h>
#include <binnedMPI.h>
}

using std::vector;


class ReproblasSummation {
    public:
        ReproblasSummation(MPI_Comm comm, size_t local_summands);
        ~ReproblasSummation();

        double *getBuffer();
        double accumulate();

    private:
        const size_t local_summands;
        int rank;
        MPI_Comm comm;
        vector<double, AlignedAllocator<double>> buffer;
        double_binned *isum;
        double_binned *local_isum;
};

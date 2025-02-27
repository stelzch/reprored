#pragma once

#include <mpi.h>
#include <vector>
#include "summation.hpp"
#include "util.hpp"

using std::vector;


enum class AllreduceType { REDUCE, REDUCE_AND_BCAST, ALLREDUCE, VECTORIZED_ALLREDUCE };

class AllreduceSummation : public Summation {
public:
    AllreduceSummation(MPI_Comm comm, size_t local_summands, AllreduceType type = AllreduceType::REDUCE_AND_BCAST);
    virtual ~AllreduceSummation();

    double *getBuffer() override;
    double accumulate() override;

private:
    const size_t local_summands;
    int rank;
    MPI_Comm comm;
    vector<double, AlignedAllocator<double>> buffer;
    AllreduceType type;
};

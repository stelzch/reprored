#pragma once

#include "summation.hpp"
#include "util.hpp"
#include <mpi.h>
#include <vector>

extern "C" {
#include <binned.h>
#include <binnedBLAS.h>
#include <binnedMPI.h>
#include <reproBLAS.h>
}

using std::vector;


class ReproblasSummation : public Summation {
public:
  ReproblasSummation(MPI_Comm comm, size_t local_summands, ReduceType type = ReduceType::ALLREDUCE);
  virtual ~ReproblasSummation();

  double *getBuffer() override;
  double accumulate() override;

private:
  ReduceType type;
  const size_t local_summands;
  int rank;
  MPI_Comm comm;
  vector<double, AlignedAllocator<double>> buffer;
  double_binned *isum;
  double_binned *local_isum;
};

#pragma once

#include "util.hpp"
#include <mpi.h>
#include <vector>

using std::vector;

class AllreduceSummation {
public:
  AllreduceSummation(MPI_Comm comm, size_t local_summands);
  ~AllreduceSummation();

  double *getBuffer();
  double accumulate();

private:
  const size_t local_summands;
  int rank;
  MPI_Comm comm;
  vector<double, AlignedAllocator<double>> buffer;
};

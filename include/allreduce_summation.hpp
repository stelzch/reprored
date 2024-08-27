#pragma once

#include "summation.hpp"
#include "util.hpp"
#include <mpi.h>
#include <vector>

using std::vector;

class AllreduceSummation : public Summation {
public:
  AllreduceSummation(MPI_Comm comm, size_t local_summands);
  virtual ~AllreduceSummation();

  double *getBuffer() override;
  double accumulate() override;

private:
  const size_t local_summands;
  int rank;
  MPI_Comm comm;
  vector<double, AlignedAllocator<double>> buffer;
};

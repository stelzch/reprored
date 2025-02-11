#include "reproblas_summation.hpp"

ReproblasSummation::ReproblasSummation(MPI_Comm comm, size_t local_summands)
    : local_summands(local_summands), comm(comm), buffer(local_summands) {

  MPI_Comm_rank(comm, &rank);
  local_isum = binned_dballoc(3);
  isum = binned_dballoc(3);
}

ReproblasSummation::~ReproblasSummation() {
  free(local_isum);
  free(isum);
}

double *ReproblasSummation::getBuffer() { return buffer.data(); }

double ReproblasSummation::accumulate() {
  /* Adopted from the ReproBLAS MPI_sum_sine.c example, line 105 onwards */

  binned_dbsetzero(3, local_isum);

binned_dbsetzero(3, isum);

  // Local summation
  binnedBLAS_dbdsum(3, local_summands, buffer.data(), 1, local_isum);

  // Reducation across communicator
  MPI_Allreduce(local_isum, isum, 1, binnedMPI_DOUBLE_BINNED(3),
             binnedMPI_DBDBADD(3), comm);

  double sum;
  sum = binned_ddbconv(3, isum);

  return sum;
}

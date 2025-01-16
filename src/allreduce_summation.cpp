#include "allreduce_summation.hpp"
#include <execution>
#include <numeric>


AllreduceSummation::AllreduceSummation(MPI_Comm comm, size_t local_summands, AllreduceType type) :
    local_summands(local_summands), comm(comm), buffer(local_summands), type(type) {

    MPI_Comm_rank(comm, &rank);
}

AllreduceSummation::~AllreduceSummation() {}

double *AllreduceSummation::getBuffer() { return buffer.data(); }

double AllreduceSummation::accumulate() {
    double sum = 0, local_sum = 0;
    switch (type) {
        case AllreduceType::REDUCE_AND_BCAST:
            local_sum = std::reduce(buffer.begin(), buffer.end(), 0.0, std::plus<double>());

            MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
            MPI_Bcast(&sum, 1, MPI_DOUBLE, 0, comm);
            break;
        case AllreduceType::ALLREDUCE:
            local_sum = std::reduce(buffer.begin(), buffer.end(), 0.0, std::plus<double>());

            MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);
            break;
        case AllreduceType::VECTORIZED_ALLREDUCE:
            local_sum = std::reduce(std::execution::par_unseq, buffer.begin(), buffer.end(), 0.0, std::plus<double>());
            MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);
            break;
    }

    return sum;
}

#include <gtest/gtest.h>
#include <mpi.h>
#include "gtest-mpi-listener.hpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);
    int init_flag;
    MPI_Initialized(&init_flag);
    if (!init_flag) {
        throw std::runtime_error("MPI not initialized");
    }

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    ::testing::TestEventListener* l = listeners.Release(listeners.default_result_printer());
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));

    auto result = RUN_ALL_TESTS();

    return result;
}

#include <gtest/gtest.h>
#include <mpi.h>

#include "binary_tree_summation.h"
#include "gtest-mpi-listener.hpp"


class DebugAttachListener : public testing::EmptyTestEventListener {
    void OnTestPartResult(const testing::TestPartResult &test_part_result) override {
        if (test_part_result.type() == testing::TestPartResult::kFatalFailure) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);


            fprintf(stderr, "[ERROR] Assertion on line %s:%i failed on rank %i. ", test_part_result.file_name(),
                    test_part_result.line_number(), rank);
            attach_debugger(true);
        }
    }
};

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);
    int init_flag;
    MPI_Initialized(&init_flag);
    if (!init_flag) {
        throw std::runtime_error("MPI not initialized");
    }

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    ::testing::TestEventListener *l = listeners.Release(listeners.default_result_printer());
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));
    // listeners.Append(new DebugAttachListener());

    attach_debugger_env();
    auto result = RUN_ALL_TESTS();

    return result;
}

#include <binary_tree_summation.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kassert/kassert.hpp>
#include <kgather_summation.hpp>
#include <mpi.h>

// Test adapted from KaMPI-NG.
// See
// https://github.com/kamping-site/kamping/blob/a3c61e335537291bf1af52258cc19bad445a7cd7/tests/plugins/reproducible_reduce.cpp


#include <chrono>
#include <cmath>
#include <random>
#include <util.hpp>
#include <vector>


// Test generators
TEST(ReproducibleReduceTest, DistributionGeneration) {
    Distribution distr1 = distribute_evenly(9, 4);
    EXPECT_THAT(distr1.send_counts, testing::ElementsAre(3, 2, 2, 2));
    EXPECT_THAT(distr1.displs, testing::ElementsAre(0, 3, 5, 7));

    Distribution distr2 = distribute_evenly(2, 5);
    EXPECT_THAT(distr2.send_counts, testing::ElementsAre(1, 1, 0, 0, 0));
    EXPECT_THAT(distr2.displs, testing::ElementsAre(0, 1, 2, 2, 2));

    Distribution distr3 = distribute_randomly(30, 4, 42);
    EXPECT_EQ(distr3.send_counts.size(), 4);
    EXPECT_THAT(std::accumulate(distr3.send_counts.begin(), distr3.send_counts.end(), 0), 30);
}


constexpr double const epsilon = std::numeric_limits<double>::epsilon();
TEST(ReproducibleReduceTest, SimpleSum) {
    constexpr int comm_size = 2;
    int actual_comm_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &actual_comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ASSERT_GE(actual_comm_size, comm_size) << "Comm is of insufficient size";


    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < comm_size, 0, &comm);

    if (rank >= comm_size)
        return;

    std::vector const a{1e3, epsilon, epsilon / 2, epsilon / 2};
    EXPECT_EQ(std::accumulate(a.begin(), a.end(), 0.0), 1e3 + epsilon);

    Distribution distr({2, 2}, {0, 2});

    auto local_a = scatter_array(comm, a, distr);

    ASSERT_EQ(2, distr.send_counts.size());
    ASSERT_EQ(2, distr.displs.size());
    BinaryTreeSummation bts(rank, regions_from_distribution(distr), 1, comm);

    memcpy(bts.getBuffer(), local_a.data(), local_a.size() * sizeof(double));

    double sum = bts.accumulate();
    EXPECT_EQ(sum, (1e3 + epsilon) + (epsilon / 2 + epsilon / 2));

    MPI_Comm_free(&comm);
}


TEST(ReproducibleReduceTest, WorksWithNonzeroRoot) {

    auto full_comm = MPI_COMM_WORLD;
    std::vector<double> array{1.0, 2.0, 3.0, 4.0};
    Distribution distribution({2, 2}, {2, 0});

    with_comm_size_n(full_comm, 2, [&distribution, &array](auto comm, auto rank, auto _) {
        BinaryTreeSummation bts(rank, regions_from_distribution(distribution), 8, comm);


        auto local_array = scatter_array(comm, array, distribution);
        memcpy(bts.getBuffer(), local_array.data(), local_array.size() * sizeof(double));

        double result = bts.accumulate();

        EXPECT_EQ(result, (1.0 + 2.0) + (3.0 + 4.0));
    });
}

TEST(ReproducibleReduceTest, OtherExample) {
    /*
     *           ▼        ▼        ▼
     * ┌────────┬──┬──┬──────────────┐
     * │  p2    │p0│p3│     p1       │
     * └────────┴──┴──┴──────────────┘
     *  0  1  2  3  4  5  6  7  8  9
     *
     */
    auto full_comm = MPI_COMM_WORLD;
    Distribution distribution({1, 5, 3, 1}, {3, 5, 0, 4});
    const auto array = generate_test_vector(10, 42);
    const auto K = 3;

    with_comm_size_n(full_comm, 4, [&distribution, &array, &K](auto comm, auto rank, auto _) {
        BinaryTreeSummation bts(rank, regions_from_distribution(distribution), K);

        auto local_array = scatter_array(comm, array, distribution);
        memcpy(bts.getBuffer(), local_array.data(), local_array.size() * sizeof(double));

        double result = bts.accumulate();

        EXPECT_NEAR(result, std::accumulate(array.begin(), array.end(), 0.0), 1e-9);
    });
}

TEST(ReproducibleReduceTest, Fuzzing) {
    int full_comm_size;
    int full_comm_rank;
    auto comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &full_comm_size);
    MPI_Comm_rank(comm, &full_comm_rank);

    MPI_Barrier(comm);

    ASSERT_GT(full_comm_size, 1) << "Fuzzing with only one rank is useless";

    constexpr auto NUM_ARRAYS = 20000; // 15;
    constexpr auto NUM_KS = 20;
    constexpr auto NUM_DISTRIBUTIONS = 300; // 5000;

    // Seed random number generator with same seed across all ranks for consistent number generation
    std::random_device rd;
    unsigned long seed;

    if (full_comm_rank == 0) {
        seed = rd();
    }
    MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG, 0, comm);

    std::uniform_int_distribution<size_t> array_length_distribution(1, 20);
    std::uniform_int_distribution<size_t> rank_distribution(1, full_comm_size);
    std::uniform_int_distribution<size_t> k_distribution(1, 24);
    std::mt19937 rng(seed); // RNG for distribution & rank number
    std::mt19937 rng_root(rng()); // RNG for data generation (out-of-sync with other ranks)

    auto checks = 0UL;

    for (auto i = 0U; i < NUM_ARRAYS; ++i) {
        std::vector<double> data_array;
        size_t const data_array_size = array_length_distribution(rng);
        if (full_comm_rank == 0) {
            data_array = generate_test_vector(data_array_size, rng_root());
        }

        for (auto ik = 0U; ik < NUM_KS; ++ik) {
            const auto k = k_distribution(rng);
            double reference_result = 0;

            // Calculate reference result
            with_comm_size_n(comm, 1, [&reference_result, &data_array, &comm, &k](auto comm_, auto rank, auto _) {
                const auto distribution = distribute_evenly(data_array.size(), 1);
                BinaryTreeSummation bts(rank, regions_from_distribution(distribution), k, comm_);
                memcpy(bts.getBuffer(), data_array.data(), data_array.size() * sizeof(double));

                reference_result = bts.accumulate();

                // Sanity check
                ASSERT_NEAR(reference_result, std::accumulate(data_array.begin(), data_array.end(), 0.0), 1e-9);
            });

            MPI_Bcast(&reference_result, 1, MPI_DOUBLE, 0, comm);

            for (auto j = 0U; j < NUM_DISTRIBUTIONS; ++j) {
                auto const ranks = rank_distribution(rng);
                auto const distribution = distribute_randomly(data_array_size, static_cast<size_t>(ranks), rng());

                if (full_comm_rank == 0) {
                    printf("n=%zu, p=%zu, k=%zu, distribution=", data_array_size, ranks, k);
                    for (auto i = 0; i < ranks; ++i) {
                        printf("(%i, %i) ", distribution.displs[i], distribution.send_counts[i]);
                    }
                    printf("\n");
                }

                with_comm_size_n(comm, ranks,
                                 [&distribution, &data_array, &reference_result, &checks, &ranks, &comm,
                                  &k](auto comm_, auto rank, auto size) {
                                     MPI_Barrier(comm_);
                                     int comm_size;
                                     MPI_Comm_size(comm_, &comm_size);
                                     ASSERT_EQ(static_cast<int>(ranks), comm_size);
                                     ASSERT_EQ(distribution.displs.size(), comm_size);
                                     ASSERT_EQ(distribution.send_counts.size(), comm_size);

                                     BinaryTreeSummation bts(rank, regions_from_distribution(distribution), k, comm_);

                                     std::vector<double> local_arr = scatter_array(comm_, data_array, distribution);
                                     memcpy(bts.getBuffer(), local_arr.data(), local_arr.size() * sizeof(double));

                                     double computed_result = bts.accumulate();

                                     EXPECT_EQ(computed_result, reference_result);
                                     ++checks;
                                 });
            }
        }
    }

    if (full_comm_rank == 0) {
        printf("Performed %zu checks\n", checks);
    }
}

TEST(ReproducibleReduceTest, FuzzingKGather) {
    int full_comm_size;
    int full_comm_rank;
    auto comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &full_comm_size);
    MPI_Comm_rank(comm, &full_comm_rank);

    MPI_Barrier(comm);

    ASSERT_GT(full_comm_size, 1) << "Fuzzing with only one rank is useless";

    constexpr auto NUM_ARRAYS = 20000; // 15;
    constexpr auto NUM_KS = 20;
    constexpr auto NUM_DISTRIBUTIONS = 300; // 5000;

    // Seed random number generator with same seed across all ranks for consistent number generation
    std::random_device rd;
    unsigned long seed;

    if (full_comm_rank == 0) {
        seed = rd();
    }
    MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG, 0, comm);

    std::uniform_int_distribution<size_t> array_length_distribution(1, 20);
    std::uniform_int_distribution<size_t> rank_distribution(1, full_comm_size);
    std::uniform_int_distribution<size_t> k_distribution(1, 24);
    std::mt19937 rng(seed); // RNG for distribution & rank number
    std::mt19937 rng_root(rng()); // RNG for data generation (out-of-sync with other ranks)

    auto checks = 0UL;

    for (auto i = 0U; i < NUM_ARRAYS; ++i) {
        std::vector<double> data_array;
        size_t const data_array_size = array_length_distribution(rng);
        if (full_comm_rank == 0) {
            data_array = generate_test_vector(data_array_size, rng_root());
        }

        for (auto ik = 0U; ik < NUM_KS; ++ik) {
            const auto k = k_distribution(rng);
            double reference_result = 0;

            // Calculate reference result
            with_comm_size_n(comm, 1, [&reference_result, &data_array, &comm, &k](auto comm_, auto rank, auto _) {
                const auto distribution = distribute_evenly(data_array.size(), 1);
                KGatherSummation kgs(rank, regions_from_distribution(distribution), k, comm_);
                memcpy(kgs.getBuffer(), data_array.data(), data_array.size() * sizeof(double));

                reference_result = kgs.accumulate();

                // Sanity check
                ASSERT_NEAR(reference_result, std::accumulate(data_array.begin(), data_array.end(), 0.0), 1e-9);
            });

            MPI_Bcast(&reference_result, 1, MPI_DOUBLE, 0, comm);

            for (auto j = 0U; j < NUM_DISTRIBUTIONS; ++j) {
                auto const ranks = rank_distribution(rng);
                auto const distribution = distribute_randomly(data_array_size, static_cast<size_t>(ranks), rng());

                if (full_comm_rank == 0) {
                    printf("n=%zu, p=%zu, k=%zu, distribution=", data_array_size, ranks, k);
                    for (auto i = 0; i < ranks; ++i) {
                        printf("(%i, %i) ", distribution.displs[i], distribution.send_counts[i]);
                    }
                    printf("\n");
                }

                with_comm_size_n(comm, ranks,
                                 [&distribution, &data_array, &reference_result, &checks, &ranks, &comm,
                                  &k](auto comm_, auto rank, auto size) {
                                     MPI_Barrier(comm_);
                                     int comm_size;
                                     MPI_Comm_size(comm_, &comm_size);
                                     ASSERT_EQ(static_cast<int>(ranks), comm_size);
                                     ASSERT_EQ(distribution.displs.size(), comm_size);
                                     ASSERT_EQ(distribution.send_counts.size(), comm_size);

                                     KGatherSummation kgs(rank, regions_from_distribution(distribution), k, comm_);

                                     std::vector<double> local_arr = scatter_array(comm_, data_array, distribution);
                                     memcpy(kgs.getBuffer(), local_arr.data(), local_arr.size() * sizeof(double));

                                     double computed_result = kgs.accumulate();

                                     EXPECT_EQ(computed_result, reference_result);
                                     ++checks;
                                 });
            }
        }
    }

    if (full_comm_rank == 0) {
        printf("Performed %zu checks\n", checks);
    }
}

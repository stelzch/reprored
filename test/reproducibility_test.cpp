#include <mpi.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <kassert/kassert.hpp>
#include <binary_tree_summation.hpp>
#include <kgather_summation.hpp>

// Test adapted from KaMPI-NG.
// See https://github.com/kamping-site/kamping/blob/a3c61e335537291bf1af52258cc19bad445a7cd7/tests/plugins/reproducible_reduce.cpp


#include <chrono>
#include <cmath>
#include <random>
#include <vector>

using Distribution = struct Distribution {
    std::vector<int> send_counts;
    std::vector<int> displs;

    Distribution(std::vector<int> _send_counts, std::vector<int> recv_displs)
        : send_counts(_send_counts),
          displs(recv_displs) {}
};

auto regions_from_distribution(const Distribution& d) {
    vector<region> regions;
    regions.reserve(d.displs.size());

    for (auto i = 0U; i < d.displs.size(); ++i) {
        regions.emplace_back(d.displs[i], d.send_counts[i]);
    }

    return regions;
}

template <typename C, typename T>
std::vector<T> scatter_array(C comm, std::vector<T> const& global_array, Distribution const d) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::vector<T> result(d.send_counts[rank]);

    MPI_Scatterv(global_array.data(),
            d.send_counts.data(), d.displs.data(),
            MPI_DOUBLE, result.data(), result.size(), MPI_DOUBLE, 0, comm);

    return result;
}

auto displacement_from_sendcounts(std::vector<int>& send_counts) {
    std::vector<int> displacement;
    displacement.reserve(send_counts.size());

    int start_index = 0;
    for (auto const& send_count: send_counts) {
        displacement.push_back(start_index);
        start_index += send_count;
    }

    return displacement;
}

auto distribute_evenly(size_t const collection_size, size_t const comm_size) {
    auto const elements_per_rank = collection_size / comm_size;
    auto const remainder         = collection_size % comm_size;

    std::vector<int> send_counts(comm_size, elements_per_rank);
    std::for_each_n(send_counts.begin(), remainder, [](auto& n) { n += 1; });

    return Distribution(send_counts, displacement_from_sendcounts(send_counts));
}

auto distribute_randomly(size_t const collection_size, size_t const comm_size, size_t const seed) {
    std::mt19937                    rng(seed);
    std::uniform_int_distribution<> dist(0, collection_size);

    // See https://stackoverflow.com/a/48205426 for details
    std::vector<int> points(comm_size, 0UL);
    points.push_back(collection_size);
    std::generate(points.begin() + 1, points.end() - 1, [&dist, &rng]() { return dist(rng); });
    std::sort(points.begin(), points.end());

    std::vector<int> send_counts(comm_size);
    for (size_t i = 0; i < send_counts.size(); ++i) {
        send_counts[i] = points[i + 1] - points[i];
    }

    // Shuffle to generate distributions where start indices are not monotonically increasing
    std::vector<size_t> indices(send_counts.size(), 0);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    auto displacement = displacement_from_sendcounts(send_counts);
    EXPECT_EQ(send_counts.size(), displacement.size());

    decltype(send_counts)  shuffled_send_counts(send_counts.size(), 0);
    decltype(displacement) shuffled_displacement(displacement.size(), 0);
    for (auto i = 0UL; i < send_counts.size(); ++i) {
        shuffled_send_counts[i]  = send_counts[indices[i]];
        shuffled_displacement[i] = displacement[indices[i]];
    }

    EXPECT_EQ(
        collection_size,
        std::reduce(shuffled_send_counts.begin(), shuffled_send_counts.end(), 0UL, std::plus<>())
    );

    return Distribution(shuffled_send_counts, shuffled_displacement);
}
auto generate_test_vector(size_t length, size_t seed) {
    std::mt19937                   rng(seed);
    std::uniform_real_distribution distr;
    std::vector<double>            result(length);
    std::generate(result.begin(), result.end(), [&distr, &rng]() { return distr(rng); });

    return result;
}

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
    constexpr int                                                                 comm_size = 2;
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

template <typename F>
void with_comm_size_n(
    MPI_Comm& comm, size_t comm_size, F f
) {
    int full_comm_size, full_comm_rank;
    MPI_Comm_size(comm, &full_comm_size);
    MPI_Comm_rank(comm, &full_comm_rank);
    assert(full_comm_size >= comm_size);

    int  rank_active = full_comm_rank < comm_size;
    MPI_Comm new_comm;
    MPI_Comm_split(comm, rank_active, 0, &new_comm);

    int new_rank;
    int new_comm_size;
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_comm_size);

    if (rank_active) {
        f(new_comm, new_rank, new_comm_size);
    }

    MPI_Comm_free(&new_comm);
}

TEST(ReproducibleReduceTest, WorksWithNonzeroRoot) {

    auto full_comm = MPI_COMM_WORLD;
    std::vector<double> array{1.0, 2.0, 3.0, 4.0};
    Distribution        distribution({2, 2}, {2, 0});

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
    Distribution        distribution({1, 5, 3, 1}, {3, 5, 0, 4});
    const auto array = generate_test_vector(10, 42);
    const auto K = 3;

    with_comm_size_n(full_comm, 4, [&distribution, &array, &K] (auto comm, auto rank, auto _) {
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

    constexpr auto NUM_ARRAYS        = 20000; // 15;
    constexpr auto NUM_KS            = 20;
    constexpr auto NUM_DISTRIBUTIONS = 300; // 5000;

    // Seed random number generator with same seed across all ranks for consistent number generation
    std::random_device rd;
    unsigned long      seed;

    if (full_comm_rank == 0) {
        seed = rd();
    }
    MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG, 0, comm);

    std::uniform_int_distribution<size_t> array_length_distribution(1, 20);
    std::uniform_int_distribution<size_t> rank_distribution(1, full_comm_size);
    std::uniform_int_distribution<size_t> k_distribution(1, 24);
    std::mt19937                          rng(seed);       // RNG for distribution & rank number
    std::mt19937                          rng_root(rng()); // RNG for data generation (out-of-sync with other ranks)

    auto checks = 0UL;

    for (auto i = 0U; i < NUM_ARRAYS; ++i) {
        std::vector<double> data_array;
        size_t const        data_array_size = array_length_distribution(rng);
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
                auto const ranks        = rank_distribution(rng);
                auto const distribution = distribute_randomly(data_array_size, static_cast<size_t>(ranks), rng());

                if (full_comm_rank == 0) {
                    printf("n=%zu, p=%zu, k=%zu, distribution=", data_array_size, ranks, k);
                    for (auto i = 0; i < ranks; ++i) {
                        printf("(%i, %i) ", distribution.displs[i], distribution.send_counts[i]);
                    }
                    printf("\n");
                }

                with_comm_size_n(comm, ranks, [&distribution, &data_array, &reference_result, &checks, &ranks, &comm, &k](auto comm_, auto rank, auto size) {
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

    constexpr auto NUM_ARRAYS        = 20000; // 15;
    constexpr auto NUM_KS            = 20;
    constexpr auto NUM_DISTRIBUTIONS = 300; // 5000;

    // Seed random number generator with same seed across all ranks for consistent number generation
    std::random_device rd;
    unsigned long      seed;

    if (full_comm_rank == 0) {
        seed = rd();
    }
    MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG, 0, comm);

    std::uniform_int_distribution<size_t> array_length_distribution(1, 20);
    std::uniform_int_distribution<size_t> rank_distribution(1, full_comm_size);
    std::uniform_int_distribution<size_t> k_distribution(1, 24);
    std::mt19937                          rng(seed);       // RNG for distribution & rank number
    std::mt19937                          rng_root(rng()); // RNG for data generation (out-of-sync with other ranks)

    auto checks = 0UL;

    for (auto i = 0U; i < NUM_ARRAYS; ++i) {
        std::vector<double> data_array;
        size_t const        data_array_size = array_length_distribution(rng);
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
                auto const ranks        = rank_distribution(rng);
                auto const distribution = distribute_randomly(data_array_size, static_cast<size_t>(ranks), rng());

                if (full_comm_rank == 0) {
                    printf("n=%zu, p=%zu, k=%zu, distribution=", data_array_size, ranks, k);
                    for (auto i = 0; i < ranks; ++i) {
                        printf("(%i, %i) ", distribution.displs[i], distribution.send_counts[i]);
                    }
                    printf("\n");
                }

                with_comm_size_n(comm, ranks, [&distribution, &data_array, &reference_result, &checks, &ranks, &comm, &k](auto comm_, auto rank, auto size) {
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

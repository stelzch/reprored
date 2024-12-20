#include <dual_tree_summation.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <k_chunked_array.hpp>
#include <util.hpp>

#include <random>

#include "binary_tree_summation.h"
#include "dual_tree_topology.hpp"

using std::vector;
using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::IsEmpty;

using TC = TreeCoordinates;
// Create one KChunkedArray for each rank
vector<DualTreeTopology> instantiate_all_ranks(const vector<region> &regions) {
    vector<DualTreeTopology> result;
    result.reserve(regions.size());

    for (auto i = 0U; i < regions.size(); ++i) {
        result.emplace_back(i, regions);
    }

    return result;
}


/*
 *   ├───────────────────────────────┐
 *   ├───────────────┐               │
 *   ├───────┐       ├───────┐       ├───────┐
 *   ├───┐   ├───┐   ├───┐   ├───┐   ├───┐   │
 *   0   1   2   3   4   5   6   7   8   9  10
 */
TEST(DualTree, BinaryTreePrimitives) {
    const auto global_size = 11;
    const vector<region> exampleA{{0, global_size}};

    DualTreeTopology topology(0, exampleA);

    EXPECT_EQ(topology.get_reduction_partner(0, 2), TC(4, 2));
    EXPECT_EQ(topology.get_reduction_partner(4, 0), TC(5, 0));
    EXPECT_EQ(topology.get_reduction_partner(4, 1), TC(6, 1));
    EXPECT_EQ(topology.get_reduction_partner(9, 0), TC(8, 0));
    EXPECT_EQ(topology.get_reduction_partner(10, 0), TC(8, 1));
    EXPECT_EQ(topology.get_reduction_partner(10, 1), TC(8, 1));


    EXPECT_EQ(topology.max_y(0, global_size), 4);
    EXPECT_EQ(topology.max_y(1, global_size), 0);
    EXPECT_EQ(topology.max_y(2, global_size), 1);
    EXPECT_EQ(topology.max_y(4, global_size), 2);
    EXPECT_EQ(topology.max_y(8, global_size), 2);
    EXPECT_EQ(topology.max_y(9, global_size), 0);
    EXPECT_EQ(topology.max_y(10, global_size), 0);

    EXPECT_EQ(topology.parent(9), 8);
    EXPECT_EQ(topology.parent(6), 4);
    EXPECT_EQ(topology.parent(2), 0);

    EXPECT_EQ(topology.largest_child_index(4), 7);
    EXPECT_EQ(topology.largest_child_index(8), 15);
    EXPECT_EQ(topology.largest_child_index(9), 9);

    EXPECT_EQ(topology.subtree_size(1), 1);
    EXPECT_EQ(topology.subtree_size(4), 4);
    EXPECT_EQ(topology.subtree_size(6), 2);
    EXPECT_EQ(topology.subtree_size(8), 3);

    EXPECT_TRUE(topology.is_subtree_local(8, 3));

    EXPECT_THAT(topology.get_comm_children(), IsEmpty());
}

/**
 *
 *           Reduction Tree
 *   │
 *   ├───────────────────────────────┐
 *   ├───────────────┐               │
 *   ├───────┐       ├───────┐       ├───────┐
 *   ├───┐   ├───┐   ├───┐   ├───┐   ├───┐   │
 *   │   │   │   │   │   │   │   │   │   │   │
 *   0   1   2 │ 3   4   5   6 │ 7 │ 8   9 │10
 *             │               │   │       │
 *       PE0           PE1      PE2   PE3   PE4
 *        │             │        │     │     │
 *        │3,0 4,1 6,0  │        │ 8,1 │     │
 *        │◄────────────┘        │◄────┘     │
 *        │ 7,0 8,1              │           │
 *        │◄─────────────────────┘           │
 *        │ 10,0                             │
 *        │◄─────────────────────────────────┘
 *        │    Communication Tree
 *        │
 *
 */
TEST(DualTree, ExampleA) {
    const vector<region> exampleA{{0, 3}, {3, 4}, {7, 1}, {8, 2}, {10, 1}};

    const auto t = instantiate_all_ranks(exampleA);

    EXPECT_THAT(t[0].get_comm_children(), ElementsAre(1, 2, 4));
    EXPECT_THAT(t[1].get_comm_children(), IsEmpty());
    EXPECT_THAT(t[2].get_comm_children(), ElementsAre(3));
    EXPECT_THAT(t[3].get_comm_children(), IsEmpty());
    EXPECT_THAT(t[4].get_comm_children(), IsEmpty());

    // start from the back at PE4
    EXPECT_THAT(t[4].get_outgoing(), ElementsAre(TC(10, 0)));

    EXPECT_THAT(t[3].get_outgoing(), ElementsAre(TC(8, 1)));

    EXPECT_THAT(t[2].get_outgoing(), ElementsAre(TC(7, 0), TC(8, 1)));

    EXPECT_FALSE(t[1].is_subtree_comm_local(4, 2));
    EXPECT_TRUE(t[1].is_subtree_comm_local(4, 1));
    EXPECT_TRUE(t[1].is_subtree_local(4, 1));
    EXPECT_THAT(t[1].get_outgoing(), ElementsAre(TC(3, 0), TC(4, 1), TC(6, 0)));

    EXPECT_THAT(t[0].get_outgoing(), ElementsAre(TC(0, 4)));
}

/**
 *    │      Reduction Tree
 *    ├───────────────────────────────┐
 *    ├───────────────┐               │
 *    ├───────┐       ├───────┐       ├───────┐
 *    ├───┐   ├───┐   ├───┐   ├───┐   ├───┐   │
 *    │   │   │   │   │   │   │   │   │   │   │
 *    0 │ 1   2 │ 3   4   5   6   7 │ 8   9  10
 *      │       │                   │
 *   PE0   PE1           PE2              PE3
 *    │     │             │                │
 *    │ 1,0 │2,0          │ 8,2            │
 *    │◄────┘             │◄───────────────┘
 *    │3,0 4,2 8,1 10,0   │
 *    │◄──────────────────┘
 *    │        Communication Tree
 */
TEST(DualTree, ExampleB) {
    const vector<region> exampleB{{0, 1}, {1, 2}, {3, 5}, {8, 3}};

    const auto t = instantiate_all_ranks(exampleB);

    EXPECT_THAT(t[0].get_comm_children(), ElementsAre(1, 2));
    EXPECT_THAT(t[1].get_comm_children(), IsEmpty());
    EXPECT_THAT(t[2].get_comm_children(), ElementsAre(3));
    EXPECT_THAT(t[3].get_comm_children(), IsEmpty());


    const vector<TC> t3_out{{8, 2}};
    EXPECT_THAT(t[3].get_outgoing(), ElementsAreArray(t3_out));

    const vector<TC> t2_out{{3, 0}, {4, 2}, {8, 2}};
    EXPECT_THAT(t[2].get_outgoing(), ElementsAreArray(t2_out));

    const vector<TC> t1_out{{1, 0}, {2, 0}};
    EXPECT_THAT(t[1].get_outgoing(), ElementsAreArray(t1_out));

    EXPECT_THAT(t[0].get_outgoing(), ElementsAre(TC(0, 4)));
}


/**
 *
 *  │
 *  ├───────────────────────────────┐
 *  ├───────────────┐               ├───────────────┐
 *  ├───────┐       ├───────┐       ├───────┐       ├───────┐
 *  ├───┐   ├───┐   ├───┐   ├───┐   ├───┐   ├───┐   ├───┐   ├───┐
 *  │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
 *  0   1   2   3 │ 4   5 │ 6   7   8 │ 9  10 │11  12 │13 │14  15
 *                │       │           │       │       │   │
 *       PE0         PE1        PE2      PE3     PE4   PE5   PE6
 *        │           │          │        │       │     │     │
 *        │  4,1      │          │9,0 10,0│       │ 13,0│     │
 *        │◄──────────┘          │◄───────┘       │◄────┘     │
 *        │  6,1 8,1 10,0        │                │     14,1  │
 *        │◄─────────────────────┘                │◄──────────┘
 *        │                             11,0 12,2 │
 *        │◄──────────────────────────────────────┘
 *        │
 *
 */
TEST(DualTree, ExampleC) {
    const vector<region> exampleC{{0, 4}, {4, 2}, {6, 3}, {9, 2}, {11, 2}, {13, 1}, {14, 2}};
    const auto t = instantiate_all_ranks(exampleC);

    EXPECT_THAT(t[0].get_comm_children(), ElementsAre(1, 2, 4));
    EXPECT_THAT(t[1].get_comm_children(), IsEmpty());
    EXPECT_THAT(t[2].get_comm_children(), ElementsAre(3));
    EXPECT_THAT(t[3].get_comm_children(), IsEmpty());
    EXPECT_THAT(t[4].get_comm_children(), ElementsAre(5, 6));
    EXPECT_THAT(t[5].get_comm_children(), IsEmpty());
    EXPECT_THAT(t[6].get_comm_children(), IsEmpty());

    const vector<TC> t6_out{{14, 1}};
    const vector<TC> t5_out{{13, 0}};
    const vector<TC> t4_out{{11, 0}, {12, 2}};
    const vector<TC> t3_out{{9, 0}, {10, 0}};
    const vector<TC> t2_out{{6, 1}, {8, 1}, {10, 0}};
    const vector<TC> t1_out{{4, 1}};

    EXPECT_THAT(t[6].get_outgoing(), ElementsAreArray(t6_out));
    EXPECT_THAT(t[5].get_outgoing(), ElementsAreArray(t5_out));
    EXPECT_THAT(t[4].get_outgoing(), ElementsAreArray(t4_out));
    EXPECT_THAT(t[3].get_outgoing(), ElementsAreArray(t3_out));
    EXPECT_THAT(t[2].get_outgoing(), ElementsAreArray(t2_out));
    EXPECT_THAT(t[1].get_outgoing(), ElementsAreArray(t1_out));
    EXPECT_THAT(t[0].get_outgoing(), ElementsAre(TC(0, 4)));
}


TEST(DualTree, SingleRankReduction) {
    const auto v = generate_test_vector(11, 42);
    const auto distribution = distribute_evenly(v.size(), 1);
    const auto regions = regions_from_distribution(distribution);

    DualTreeSummation dts(0, regions, MPI_COMM_WORLD);

    memcpy(dts.getBuffer(), v.data(), v.size() * sizeof(double));

    const auto computed = dts.accumulate();
    const auto reference = std::accumulate(v.begin(), v.end(), 0.0);
    EXPECT_NEAR(reference, computed, 1e-9);
}

TEST(DualTree, Fuzzer) {
    int full_comm_size;
    int full_comm_rank;
    auto comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &full_comm_size);
    MPI_Comm_rank(comm, &full_comm_rank);

    MPI_Barrier(comm);

    // ASSERT_GT(full_comm_size, 1) << "Fuzzing with only one rank is useless";

    constexpr auto NUM_ARRAYS = 20000; // 15;
    constexpr auto NUM_DISTRIBUTIONS = 5000;

    // Seed random number generator with same seed across all ranks for consistent number generation
    std::random_device rd;
    unsigned long seed;

    if (full_comm_rank == 0) {
        seed = rd();
    }
    MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG, 0, comm);

    std::uniform_int_distribution<size_t> array_length_distribution(1, 200);
    std::uniform_int_distribution<size_t> rank_distribution(1, full_comm_size);
    std::mt19937 rng(seed); // RNG for distribution & rank number
    std::mt19937 rng_root(rng()); // RNG for data generation (out-of-sync with other ranks)

    auto checks = 0UL;

    for (auto i = 0U; i < NUM_ARRAYS; ++i) {
        std::vector<double> data_array;
        size_t const data_array_size = array_length_distribution(rng);
        if (full_comm_rank == 0) {
            data_array = generate_test_vector(data_array_size, rng_root());
        }

        double reference_result = 0;

        // Calculate reference result
        {
            MPI_Comm new_comm;
            MPI_Comm_split(comm, full_comm_rank == 0 ? 0 : 1, full_comm_rank, &new_comm);
            if (full_comm_rank == 0) {
                const auto distribution = distribute_evenly(data_array.size(), 1);
                DualTreeSummation dts(full_comm_rank, regions_from_distribution(distribution), new_comm);
                memcpy(dts.getBuffer(), data_array.data(), data_array.size() * sizeof(double));

                reference_result = dts.accumulate();
                const auto std_accumulate_result = std::accumulate(data_array.begin(), data_array.end(), 0.0);

                // Sanity check
                ASSERT_NEAR(reference_result, std_accumulate_result, 1e-9);
            }
            MPI_Comm_free(&new_comm);
        }

        MPI_Bcast(&reference_result, 1, MPI_DOUBLE, 0, comm);


        for (auto j = 0U; j < NUM_DISTRIBUTIONS; ++j) {
            auto const ranks = rank_distribution(rng);
            auto const distribution = distribute_randomly(data_array_size, static_cast<size_t>(ranks), rng());

            if (full_comm_rank == 0) {
                printf("n=%zu, p=%zu, distribution={", data_array_size, ranks);
                for (auto i = 0; i < ranks; ++i) {
                    printf("{%i, %i}", distribution.displs[i], distribution.send_counts[i]);
                    const bool last_element = i == ranks - 1;
                    if (!last_element) {
                        printf(", ");
                    }
                }
                printf("}\n");
            }

            {
                MPI_Comm new_comm;
                MPI_Comm_split(comm, full_comm_rank < distribution.displs.size() ? 0 : 1, full_comm_rank, &new_comm);


                if (full_comm_rank < distribution.displs.size()) {
                    int comm_size;
                    MPI_Comm_size(new_comm, &comm_size);
                    ASSERT_EQ(static_cast<int>(ranks), comm_size);
                    ASSERT_EQ(distribution.displs.size(), comm_size);
                    ASSERT_EQ(distribution.send_counts.size(), comm_size);

                    DualTreeSummation dts(full_comm_rank, regions_from_distribution(distribution), new_comm);

                    std::vector<double> local_arr = scatter_array(new_comm, data_array, distribution);
                    if (local_arr.size() > dts.getBufferSize()) {
                        attach_debugger(true);
                    }
                    EXPECT_LE(local_arr.size(), dts.getBufferSize());
                    memcpy(dts.getBuffer(), local_arr.data(), local_arr.size() * sizeof(double));

                    double result = dts.accumulate();
                    EXPECT_EQ(result, reference_result)
                            << "Expected result to be the same as with p=1 " << reference_result
                            << " but new_result is " << result << ", difference is " << (reference_result - result)
                            << " on rank " << full_comm_rank;
                }

                MPI_Comm_free(&new_comm);
            }
            ++checks;
        }
    }

    if (full_comm_rank == 0) {
        printf("Performed %zu checks\n", checks);
    }
}


TEST(DualTree, DifficultDistributions) {
    MPI_Comm full_comm = MPI_COMM_WORLD;
    int comm_size;
    int rank;
    MPI_Comm_rank(full_comm, &rank);
    MPI_Comm_size(full_comm, &comm_size);


    vector<vector<region>> test_distributions{
            {{0, 63}, {81, 13}, {63, 15}, {80, 1}, {78, 2}, {94, 13}},
            {{16, 17}, {13, 2}, {15, 1}, {4, 1}, {5, 8}, {0, 4}, {33, 3}, {36, 2}},
            {{99, 73}, {0, 42}, {172, 16}, {42, 57}},
            {{86, 15}, {45, 14}, {0, 43}, {101, 72}, {43, 2}, {59, 27}},
            {{0, 51}, {119, 11}, {68, 7}, {51, 16}, {67, 1}, {75, 44}},
            {{7, 4}, {11, 2}, {0, 7}, {13, 6}, {25, 2}, {19, 6}},
            {{91, 45}, {0, 42}, {47, 44}, {42, 5}},
            {{4, 8}, {12, 2}, {0, 4}, {4, 0}},
            {{54, 46}, {100, 36}, {0, 15}, {53, 1}, {15, 38}},
    };

    for (const auto &regions: test_distributions) {
        MPI_Barrier(MPI_COMM_WORLD);
        ASSERT_GE(comm_size, regions.size());
        uint64_t global_array_length = 0;
        for (const auto &[i, size]: regions) {
            global_array_length += size;
        }

        ASSERT_GE(comm_size, regions.size());

        vector<double> v = generate_test_vector(global_array_length, 4);

        if (rank == 0) {
            printf("rank %i v = ", rank);
            for (const auto &val: v) {
                printf("%f ", val);
            }
            printf("\n");
        }

        double reference = std::accumulate(v.begin(), v.end(), 0.0);

        double reference_xor = 1;

        MPI_Allreduce(&reference, &reference_xor, 8, MPI_BYTE, MPI_BXOR, MPI_COMM_WORLD);
        ASSERT_EQ(reference_xor, 0);

        double single_rank_result = 0.0;

        {
            MPI_Comm new_comm;
            MPI_Comm_split(MPI_COMM_WORLD, rank == 0 ? 0 : 1, rank, &new_comm);

            if (rank == 0) {
                Distribution single_d(1);
                single_d.displs[0] = 0;
                single_d.send_counts[0] = global_array_length;

                const auto single_region_ii = regions_from_distribution(single_d);
                DualTreeSummation dts(0, single_region_ii, new_comm);
                memcpy(dts.getBuffer(), v.data(), v.size() * sizeof(double));

                single_rank_result = dts.accumulate();
                printf("rank 0 single rank result=%f\n", single_rank_result);
            }

            MPI_Comm_free(&new_comm);
        }

        MPI_Bcast(&single_rank_result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        {
            MPI_Comm new_comm;
            MPI_Comm_split(MPI_COMM_WORLD, rank < regions.size() ? 0 : 1, rank, &new_comm);

            if (rank < regions.size()) {

                ASSERT_NEAR(single_rank_result, reference, 1e-3);
                Distribution d(regions.size());

                // Convert regions to distribution
                for (auto i = 0U; i < regions.size(); ++i) {
                    if (i < regions.size()) {
                        d.displs[i] = static_cast<int>(regions.at(i).globalStartIndex);
                        d.send_counts[i] = static_cast<int>(regions.at(i).size);
                    } else {
                        d.displs[i] = global_array_length + 10;
                        d.send_counts[i] = 0;
                    }
                }

                auto local_arr = scatter_array(new_comm, v, d);
                EXPECT_EQ(local_arr.size(), d.send_counts.at(rank));

                for (auto i = 0U; i < d.send_counts[rank]; ++i) {
                    EXPECT_EQ(local_arr.at(i), v.at(i + d.displs.at(rank)));
                }

                DualTreeSummation dts(rank, regions, new_comm);
                memcpy(dts.getBuffer(), local_arr.data(), local_arr.size() * sizeof(double));


                {
                    printf("rank %i local_arr = ", rank);
                    for (const auto &v : local_arr) {
                        printf("%f ", v);
                    }
                    printf("\n");

                    printf("rank %i regions = {", rank);
                    for (auto i = 0; i < regions.size(); ++i) {
                        printf("{%lu, %lu}, ", regions[i].globalStartIndex, regions[i].size);
                    }
                    printf("}");
                }

                double result = dts.accumulate();

                EXPECT_EQ(single_rank_result, result);
                EXPECT_NEAR(reference, result, 1e-9);
                printf("rank %i done\n", rank);
            }
            MPI_Comm_free(&new_comm);
        }
    }
}


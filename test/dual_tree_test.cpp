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
TEST(DualTreeTest, BinaryTreePrimitives) {
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
TEST(DualTreeTest, ExampleA) {
    const vector<region> exampleA{{0, 3}, {3, 4}, {7, 1}, {8, 2}, {10, 1}};

    const auto t = instantiate_all_ranks(exampleA);

    EXPECT_THAT(t[0].get_comm_children(), ElementsAre(1, 2, 4));
    EXPECT_THAT(t[1].get_comm_children(), IsEmpty());
    EXPECT_THAT(t[2].get_comm_children(), ElementsAre(3));
    EXPECT_THAT(t[3].get_comm_children(), IsEmpty());
    EXPECT_THAT(t[4].get_comm_children(), IsEmpty());

    // start from the back at PE4
    EXPECT_THAT(t[4].get_locally_computed(), ElementsAre(TC(10, 0)));

    EXPECT_THAT(t[3].get_locally_computed(), ElementsAre(TC(8, 1)));

    EXPECT_THAT(t[2].get_locally_computed(), ElementsAre(TC(7, 0))); // TC(8, 1) indirectly

    EXPECT_FALSE(t[1].is_subtree_comm_local(4, 2));
    EXPECT_TRUE(t[1].is_subtree_comm_local(4, 1));
    EXPECT_TRUE(t[1].is_subtree_local(4, 1));
    EXPECT_THAT(t[1].get_locally_computed(), ElementsAre(TC(3, 0), TC(4, 1), TC(6, 0)));

    EXPECT_THAT(t[0].get_locally_computed(), ElementsAre(TC(0, 4)));
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
 *    │ 1,0 │2,0          │ 8,1            │
 *    │◄────┘             │◄───────────────┘
 *    │3,0 4,2 8,1 10,0   │
 *    │◄──────────────────┘
 *    │        Communication Tree
 */
TEST(DualTreeTest, ExampleB) {
    const vector<region> exampleB{{0, 1}, {1, 2}, {3, 5}, {8, 3}};

    const auto t = instantiate_all_ranks(exampleB);

    EXPECT_THAT(t[0].get_comm_children(), ElementsAre(1, 2));
    EXPECT_THAT(t[1].get_comm_children(), IsEmpty());
    EXPECT_THAT(t[2].get_comm_children(), ElementsAre(3));
    EXPECT_THAT(t[3].get_comm_children(), IsEmpty());


    const vector<TC> t3_out{{8, 2}};
    EXPECT_THAT(t[3].get_locally_computed(), ElementsAreArray(t3_out));

    const vector<TC> t2_out{{3, 0}, {4, 2}}; // {8, 1}, {10, 0} indirectly
    EXPECT_THAT(t[2].get_locally_computed(), ElementsAreArray(t2_out));

    const vector<TC> t1_out{{1, 0}, {2, 0}};
    EXPECT_THAT(t[1].get_locally_computed(), ElementsAreArray(t1_out));

    EXPECT_THAT(t[0].get_locally_computed(), ElementsAre(TC(0, 4)));
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
TEST(DualTreeTest, ExampleC) {
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
    const vector<TC> t2_out{{6, 1}, {8, 1}};
    const vector<TC> t1_out{{4, 1}};

    EXPECT_THAT(t[6].get_locally_computed(), ElementsAreArray(t6_out));
    EXPECT_THAT(t[5].get_locally_computed(), ElementsAreArray(t5_out));
    EXPECT_THAT(t[4].get_locally_computed(), ElementsAreArray(t4_out));
    EXPECT_THAT(t[3].get_locally_computed(), ElementsAreArray(t3_out));
    EXPECT_THAT(t[2].get_locally_computed(), ElementsAreArray(t2_out));
    EXPECT_THAT(t[1].get_locally_computed(), ElementsAreArray(t1_out));
    EXPECT_THAT(t[0].get_locally_computed(), ElementsAre(TC(0, 4)));
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
        with_comm_size_n(comm, 1, [&reference_result, &data_array, &comm](auto comm_, auto rank, auto comm_size_) {
            ASSERT_EQ(rank, 0);
            ASSERT_EQ(comm_size_, 1);
            const auto distribution = distribute_evenly(data_array.size(), 1);
            DualTreeSummation dts(rank, regions_from_distribution(distribution), comm_);
            memcpy(dts.getBuffer(), data_array.data(), data_array.size() * sizeof(double));

            reference_result = dts.accumulate();
            const auto std_accumulate_result = std::accumulate(data_array.begin(), data_array.end(), 0.0);

            // Sanity check
            ASSERT_NEAR(reference_result, std_accumulate_result, 1e-9);
        });

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

            with_comm_size_n(comm, ranks,
                             [&distribution, &data_array, &ranks, &reference_result](auto comm_, auto rank, auto size) {
                                 MPI_Barrier(comm_);
                                 int comm_size;
                                 MPI_Comm_size(comm_, &comm_size);
                                 ASSERT_EQ(static_cast<int>(ranks), comm_size);
                                 ASSERT_EQ(distribution.displs.size(), comm_size);
                                 ASSERT_EQ(distribution.send_counts.size(), comm_size);

                                 DualTreeSummation dts(rank, regions_from_distribution(distribution), comm_);

                                 std::vector<double> local_arr = scatter_array(comm_, data_array, distribution);
                                 if (local_arr.size() > dts.getBufferSize()) {
                                     attach_debugger(true);
                                 }
                                 EXPECT_LE(local_arr.size(), dts.getBufferSize());
                                 memcpy(dts.getBuffer(), local_arr.data(), local_arr.size() * sizeof(double));

                                 double result = dts.accumulate();
                                 EXPECT_EQ(result, reference_result);
                             });
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


    vector<vector<region>> test_distributions {
        {{7, 4}, {11, 2}, {0, 7}, {13, 6}, {25, 2}, {19, 6}},
        {{91, 45}, {0, 42}, {47, 44}, {42, 5}},
       {{4, 8}, {12, 2}, {0, 4}, {4, 0}},
        {{54, 46}, {100, 36}, {0, 15}, {53, 1}, {15, 38}},
    };

    for (const auto& regions : test_distributions) {
        ASSERT_GE(comm_size, regions.size());
        uint64_t global_array_length = 0;
        for (const auto &[i, size] : regions) {
            global_array_length += size;
        }

        ASSERT_GE(comm_size, regions.size());


        with_comm_size_n(full_comm, regions.size(), [&,  &regions](auto comm, auto rank, auto size) {
            Distribution d(size);
            vector<double> v = generate_test_vector(global_array_length, 4);
            double reference = std::accumulate(v.begin(), v.end(), 0.0);

            // Convert regions to distribution
            for (auto i = 0U; i < size; ++i) {
                d.displs[i] = static_cast<int>(regions[i].globalStartIndex);
                d.send_counts[i] = static_cast<int>(regions[i].size);
            }

            auto local_arr = scatter_array(comm, v, d);
            DualTreeSummation dts(rank, regions, comm);
            memcpy(dts.getBuffer(), local_arr.data(), local_arr.size() * sizeof(double));

            double result = dts.accumulate();
            EXPECT_NEAR(reference, result, 1e-9);
        });
    }
}


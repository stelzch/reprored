#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <k_chunked_array.hpp>
#include <util.hpp>

#include <random>
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

    EXPECT_THAT(t[0].get_locally_computed(), IsEmpty());
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

    EXPECT_THAT(t[0].get_locally_computed(), IsEmpty());
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
    EXPECT_THAT(t[0].get_locally_computed(), IsEmpty());
}


auto distribute_randomly(std::mt19937 rng, size_t const collection_size, size_t const comm_size) {
    // Compare to
    // https://github.com/kamping-site/kamping/blob/8e1f3955345ad669c90658181edf5b6c2c77ea48/tests/plugins/reproducible_reduce.cpp#L67-L104
    std::uniform_int_distribution<> dist(0, collection_size);

    // See https://stackoverflow.com/a/48205426 for details
    std::vector<int> points(comm_size, 0UL);
    points.push_back(collection_size);
    std::generate(points.begin() + 1, points.end() - 1, [&dist, &rng]() { return dist(rng); });
    std::sort(points.begin(), points.end());

    std::vector<int> region_lengths(comm_size);
    for (size_t i = 0; i < region_lengths.size(); ++i) {
        region_lengths[i] = points[i + 1] - points[i];
    }

    vector<region> regions;
    regions.reserve(comm_size);

    auto index = 0UL;
    for (const auto length: region_lengths) {
        regions.emplace_back(index, length);
        index += length;
    }

    return regions;
}

TEST(DualTree, Fuzzer) {
    constexpr auto NUM_TESTS = 30000;

    std::random_device rd;
    unsigned long seed = rd();

    std::uniform_int_distribution<size_t> array_length_distribution(0, 50);
    std::uniform_int_distribution<size_t> rank_distribution(2, 20);
    std::mt19937 rng(seed);

    for (auto i = 0U; i < NUM_TESTS; i++) {
        auto array_length = array_length_distribution(rng);
        const auto regions = distribute_randomly(rng, array_length, rank_distribution(rng));
        const auto topologies = instantiate_all_ranks(regions);

        std::map<uint64_t, short> subtree_size; // maps x -> y
        for (const auto &t : topologies) {
            for (const auto [x, y] : t.get_locally_computed()) {
                auto it = subtree_size.find(x);

                if (it == subtree_size.end()) {
                    subtree_size[x] = y;
                } else {
                    const short existing = it->second;
                    const short new_value = y;
                    it->second = std::max(existing, new_value);
                }
            }
        }

        {
            // No need to perform tests on the first rank, since it does not send the values anywhere.
            auto x = regions[1].globalStartIndex;

            // Try to see if the whole array is covered through the reduction by iterating over the subtrees
            // Very incomplete check that does not guarantee correctness.
            for (const auto [subtree_x, subtree_y] : subtree_size) {
                if (x > subtree_x) {
                    continue;
                }
                ASSERT_EQ(x, subtree_x);

                ASSERT_GE(subtree_y, 0);
                const auto step = DualTreeTopology::pow2(subtree_y);
                ASSERT_GT(step, 0);

                x += step;
            }
            ASSERT_GE(x, array_length);
        }

    }
}
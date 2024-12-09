#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <k_chunked_array.hpp>
#include <util.hpp>

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
    const vector<region> exampleA{{0, 11}};

    DualTreeTopology topology(0, exampleA);
    EXPECT_EQ(topology.max_y(0), 4);
    EXPECT_EQ(topology.max_y(1), 0);
    EXPECT_EQ(topology.max_y(2), 1);
    EXPECT_EQ(topology.max_y(4), 2);
    EXPECT_EQ(topology.max_y(8), 2);
    EXPECT_EQ(topology.max_y(9), 0);
    EXPECT_EQ(topology.max_y(10), 0);

    EXPECT_EQ(topology.parent(9), 8);
    EXPECT_EQ(topology.parent(6), 4);
    EXPECT_EQ(topology.parent(2), 0);

    EXPECT_EQ(topology.largest_child_index(4), 7);
    EXPECT_EQ(topology.largest_child_index(8), 15);
    EXPECT_EQ(topology.largest_child_index(9), 9);
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

    // start from the back at PE4
    EXPECT_THAT(t[4].get_incoming(), IsEmpty());
    EXPECT_THAT(t[4].get_outgoing(), ElementsAre(TC(10, 0)));

    EXPECT_THAT(t[3].get_incoming(), IsEmpty());
    EXPECT_THAT(t[3].get_outgoing(), ElementsAre(TC(8, 1)));

    EXPECT_THAT(t[2].get_incoming(), ElementsAre(TC(8, 1)));
    EXPECT_THAT(t[2].get_outgoing(), ElementsAre(TC(7, 0), TC(8, 1)));

    EXPECT_THAT(t[1].get_incoming(), IsEmpty());
    EXPECT_THAT(t[1].get_outgoing(), ElementsAre(TC(3, 0), TC(4, 1), TC(6, 0)));

    EXPECT_THAT(t[0].get_incoming(), ElementsAre(TC(3, 0), TC(4, 1), TC(6, 0), TC(7, 0), TC(8, 1), TC(10, 0)));
    EXPECT_THAT(t[0].get_outgoing(), IsEmpty());
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
 *    │ 1,0 │2,0          │ 8,1 10,0       │
 *    │◄────┘             │◄───────────────┘
 *    │3,0 4,2 8,1 10,0   │
 *    │◄──────────────────┘
 *    │        Communication Tree
 */
TEST(DualTreeTest, ExampleB) {
    const vector<region> exampleB{{0, 1}, {1, 2}, {3, 5}, {8, 3}};

    const auto t = instantiate_all_ranks(exampleB);
    const vector<TC> t3_out{{8, 1}, {10, 0}};
    EXPECT_THAT(t[3].get_incoming(), IsEmpty());
    EXPECT_THAT(t[3].get_outgoing(), ElementsAreArray(t3_out));

    const vector<TC> t2_out{{3, 0}, {4, 2}, {8, 1}, {10, 0}};
    EXPECT_THAT(t[2].get_incoming(), ElementsAreArray(t3_out));
    EXPECT_THAT(t[2].get_outgoing(), ElementsAreArray(t2_out));

    const vector<TC> t1_out{{1, 0}, {2, 0}};
    EXPECT_THAT(t[1].get_incoming(), ElementsAreArray(t2_out));
    EXPECT_THAT(t[1].get_outgoing(), ElementsAreArray(t1_out));

    const vector<TC> t0_in{{1, 0}, {2, 0}, {3, 0}, {4, 2}, {8, 1}, {10, 0}};
    EXPECT_THAT(t[0].get_incoming(), ElementsAreArray(t0_in));
    EXPECT_THAT(t[0].get_outgoing(), IsEmpty());
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
    const vector<region> exampleC{{0,4}, {4,2}, {6,3}, {9,2}, {11, 2}, {13,1}, {14,2}};

    const auto t = instantiate_all_ranks(exampleC);
    const vector<TC> t6_out {{14,1}};
    const vector<TC> t5_out {{13,0}};
    const vector<TC> t4_out {{11,0}, {12,2}};
    const vector<TC> t3_out {{9,0}, {10,0}};
    const vector<TC> t2_out {{6,1}, {8,1}, {10,0}};
    const vector<TC> t1_out {{4,1}};

    EXPECT_THAT(t[6].get_outgoing(), ElementsAreArray(t6_out));
    EXPECT_THAT(t[5].get_outgoing(), ElementsAreArray(t5_out));
    EXPECT_THAT(t[4].get_outgoing(), ElementsAreArray(t4_out));
    EXPECT_THAT(t[3].get_outgoing(), ElementsAreArray(t3_out));
    EXPECT_THAT(t[2].get_outgoing(), ElementsAreArray(t2_out));
    EXPECT_THAT(t[1].get_outgoing(), ElementsAreArray(t1_out));
    EXPECT_THAT(t[0].get_outgoing(), IsEmpty());

    const vector<TC> t4_in {{13,0}, {14,1}};
    const vector<TC> t0_in {{4,1}, {6,1}, {8,1}, {10,0}, {11,0}, {12,2}};
    EXPECT_THAT(t[6].get_incoming(), IsEmpty());
    EXPECT_THAT(t[5].get_incoming(), IsEmpty());
    EXPECT_THAT(t[4].get_incoming(), ElementsAreArray(t4_in));
    EXPECT_THAT(t[3].get_incoming(), IsEmpty());
    EXPECT_THAT(t[2].get_incoming(), ElementsAreArray(t3_out));
    EXPECT_THAT(t[1].get_incoming(), IsEmpty());
    EXPECT_THAT(t[0].get_incoming(), ElementsAreArray(t0_in));
}
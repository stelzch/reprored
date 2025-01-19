#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "MAryTree.hpp"

using testing::ElementsAre;
using testing::IsEmpty;

/**
 * Test case with n=12 and m=3
 * ![](../docs/images/tree_n=12_m=3.svg)
 */
TEST(MAryTree, TreeA) {
    const MAryTree tree(12, 3);

    EXPECT_EQ(tree.tree_height(), 3);
    EXPECT_EQ(tree.max_y(3), 1);
    EXPECT_EQ(tree.max_y(7), 0);
    EXPECT_EQ(tree.largest_child_index(6), 8);
    EXPECT_EQ(tree.largest_child_index(7), 7);
    EXPECT_EQ(tree.largest_child_index(0), 11);

    EXPECT_EQ(tree.parent(11), 9);
    EXPECT_EQ(tree.parent(9), 0);

    const std::vector<std::pair<uint64_t, uint64_t>> parent_result{
            {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 3}, {5, 3}, {6, 0}, {7, 6}, {8, 6}, {9, 0}, {10, 9}, {11, 9}}};
    for (const auto &[x, y]: parent_result) {
        EXPECT_EQ(tree.parent(x), y);
    }

    EXPECT_THAT(tree.subtree_children(3), ElementsAre(4, 5));
    EXPECT_THAT(tree.subtree_children(0), ElementsAre(1, 2, 3, 6, 9));
}

/**
 * Test case with n=47 and m=5
 * ![](../docs/images/tree_n=47_m=5.svg)
 */
TEST(MAryTree, TreeB) {
    const MAryTree tree(47, 5);

    EXPECT_EQ(tree.tree_height(), 3);
    EXPECT_EQ(tree.max_y(0), 3);
    EXPECT_EQ(tree.max_y(20), 1);
    EXPECT_EQ(tree.max_y(25), 2);
    EXPECT_EQ(tree.max_y(45), 1);
    EXPECT_EQ(tree.max_y(46), 0);

    EXPECT_EQ(tree.largest_child_index(0), 46);
    EXPECT_EQ(tree.largest_child_index(5), 9);
    EXPECT_EQ(tree.largest_child_index(13), 13);
    EXPECT_EQ(tree.largest_child_index(35), 39);
    EXPECT_EQ(tree.largest_child_index(25), 46);

    EXPECT_EQ(tree.parent(25), 0);
    EXPECT_EQ(tree.parent(4), 0);
    EXPECT_EQ(tree.parent(8), 5);
    EXPECT_EQ(tree.parent(20), 0);
    EXPECT_EQ(tree.parent(24), 20);
    EXPECT_EQ(tree.parent(45), 25);
    EXPECT_EQ(tree.parent(46), 45);

    EXPECT_THAT(tree.subtree_children(0), ElementsAre(1, 2, 3, 4, 5, 10, 15, 20, 25));
    EXPECT_THAT(tree.subtree_children(6), IsEmpty());
    EXPECT_THAT(tree.subtree_children(15), ElementsAre(16, 17, 18, 19));
    EXPECT_THAT(tree.subtree_children(25), ElementsAre(26, 27, 28, 29, 30, 35, 40, 45));
    EXPECT_THAT(tree.subtree_children(45), ElementsAre(46));
}

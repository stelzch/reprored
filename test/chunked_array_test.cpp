#include "k_chunked_array.hpp"
#include "gmock/gmock.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using testing::ElementsAre;
using testing::IsEmpty;

// Class that exposes private attributes for us to test
class KChunkedArrayTestAdapter : public KChunkedArray {
    public:
        KChunkedArrayTestAdapter(uint64_t rank, vector<region> regions, uint64_t K) :
            KChunkedArray(rank, regions, K)
        {
        }

        int get_successor() {
            return k_successor_rank;
        }

        vector<int> get_predecessors() {
            return k_predecessor_ranks;
        }

        uint64_t get_left_remainder() {
            return k_left_remainder;
        }

        uint64_t get_right_remainder() {
            return k_right_remainder;
        }

        bool get_is_last_rank() {
            return is_last_rank;
        }

};

// Create one KChunkedArray for each rank
vector<KChunkedArrayTestAdapter> instantiate_all_ranks(const vector<region>& regions, uint64_t K) {
    vector<KChunkedArrayTestAdapter> result;
    result.reserve(regions.size());

    for (auto i = 0U; i < regions.size(); ++i) {
        result.emplace_back(i, regions, K);
    }

    return result;
}


TEST(KChunkedArrayTest, SimpleConsecutive) {
    /*
    *  Test array with length 15 and K=4.
    *  Boxes denote PE boundaries and are labeled with MPI rank.
    *  "▼" denotes K-boundaries
    *
    *    ▼           ▼           ▼           ▼           ▼
    *   ┌────────┬───────────┬───────────────────────┐    
    *   │   p0   │    p1     │           p2          │    
    *   └────────┴───────────┴───────────────────────┘    
    *    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14      
    */
                                                     
    const vector<region> regions {{0, 3}, {3, 4}, {7, 8}};
    ASSERT_EQ(regions.size(), 3);

    auto a = instantiate_all_ranks(regions, 4);

    EXPECT_EQ(a[0].get_left_remainder(), 0);
    EXPECT_EQ(a[0].get_right_remainder(), 3);
    EXPECT_THAT(a[0].get_predecessors(), IsEmpty());
    EXPECT_EQ(a[0].get_successor(), 1);
    EXPECT_FALSE(a[0].get_is_last_rank());

    EXPECT_EQ(a[1].get_left_remainder(), 1);
    EXPECT_EQ(a[1].get_right_remainder(), 3);
    EXPECT_THAT(a[1].get_predecessors(), ElementsAre(0));
    EXPECT_EQ(a[1].get_successor(), 2);
    EXPECT_FALSE(a[1].get_is_last_rank());

    EXPECT_EQ(a[2].get_left_remainder(), 1);
    EXPECT_EQ(a[2].get_right_remainder(), 3);
    EXPECT_THAT(a[2].get_predecessors(), ElementsAre(1));
    EXPECT_LT(a[2].get_successor(), 0); // Last rank does not have a successor
    EXPECT_TRUE(a[2].get_is_last_rank());
}

TEST(KChunkedArrayTest, ContrivedExample) {
    /*
    * This test shows a more complicated possible use of a K-chunked, distributed
    * array.  Several ranks (1,4,5,6) have zero numbers assigned.
    *
    * All ranks have less numbers assigned than parameter K=8, causing more
    * complicated sends.
    *
    *                                                           
    *  p=4,6,5                         ▼                 p=1   ▼
    *      ┌┬┬┬─────┬─────┬─────┬──────────────┬───────────┬┐   
    *      ││││  p2 │ p8  │ p7  │      p0      │    p3     ││   
    *      └┴┴┴─────┴─────┴─────┴──────────────┴───────────┴┘   
    *          0  1  2  3  4  5  6  7  8  9 10 11 12 13 14      0
    */ 

    const vector<region> regions {{6, 5}, {15, 0}, {0, 2}, {11, 4}, {0, 0}, {0, 0}, {0, 0}, {4, 2}, {2, 2}};

    auto a = instantiate_all_ranks(regions, 4);

    //const vector<int> empty_ranks {1,4,5,6};
    EXPECT_EQ(a[2].get_successor(), 0);
    EXPECT_EQ(a[8].get_successor(), 0);
    EXPECT_EQ(a[7].get_successor(), 0);
    EXPECT_THAT(a[0].get_predecessors(), ElementsAre(2, 8, 7));

    EXPECT_EQ(a[0].get_left_remainder(), 2);
    EXPECT_EQ(a[0].get_right_remainder(), 3);

    EXPECT_EQ(a[3].get_left_remainder(), 4);
    EXPECT_EQ(a[3].get_right_remainder(), 0);
    EXPECT_LT(a[3].get_successor(), 0); // last rank has no successor
    EXPECT_THAT(a[3].get_predecessors(), ElementsAre(0));
}

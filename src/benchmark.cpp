#include <cassert>
#include <random>
#include <limits>
#include <vector>
#include <numeric>
#include <algorithm>
#include <benchmark/benchmark.h>
#include "binarytreesummation.h"

static void BM_randomAdditionSingleRank(benchmark::State& state) {
    std::mt19937 gen;
    std::vector<double> arr;
    for (size_t i = 0; i < 8192; i++) {
        float num = gen() / (std::numeric_limits<unsigned int>::max() + 1UL);
        assert(0.0f <= num);
        assert(num < 1.0f);
        arr.push_back(num);

    }
    

    double result = -1.0f;
    for (auto _ : state) {
        std::vector<double> copy_arr(arr);

        double result2 = binary_tree_sum(copy_arr.data(), arr.size());

        if (result == -1.0f) {
            result = result2;
        } else {
            assert(result == result2);
        }
    }
}
BENCHMARK(BM_randomAdditionSingleRank);

BENCHMARK_MAIN();

#include "allreduce_summation.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <util.hpp>
#include <vector>
#include <string>
#include <utility>
#include <regex>

#include <binary_tree_summation.hpp>
#include <reproblas_summation.hpp>
#include <kgather_summation.hpp>

using std::vector;
using std::string;
using std::pair;


void print_usage(char *program_name) {
    fprintf(stderr, "Usage: %s n,p,k,r n,p,k,r,...\n\twhere\n\t\tn = length of array\n\t\tp = cluster size\n\t\tk = linear sum parameter\n\t\tr = number of repetitions\n", program_name);
}

struct TestConfig {
    uint64_t n;
    uint64_t p;
    uint64_t k;
    uint64_t r;
};

struct BenchmarkIteration {
    Timer::duration duration;
    double result;
    uint64_t iteration;
};

void print_result(const TestConfig& config, const BenchmarkIteration& it, const char *variant) {
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(it.duration);
    printf("%zu,%zu,%zu,%s,%zu,%zu,%.50f\n", config.n, config.p, config.k, variant, it.iteration, duration.count(), it.result);
}
    

template <typename F>
vector<BenchmarkIteration> measure(F f, uint64_t repetitions) {
    vector<BenchmarkIteration> results(repetitions);
    for (auto i = 0U; i < repetitions; ++i) {
        Timer t;
        double result = f();
        const auto duration = t.stop();

        results[i].duration = duration;
        results[i].result = result;
        results[i].iteration = i;
    }

    return results;

}


vector<TestConfig> collect_arguments(int argc, char **argv) {
    vector<TestConfig> configs;
    std::regex pattern(R"((\d+),(\d+),(\d+),(\d+))");
    std::cmatch match;

    for (int i = 0; i < argc; ++i) {
        bool ret = std::regex_match(argv[i], match, pattern);
        if (!ret) continue;

        const auto n = std::stoul(match[1].str());
        const auto p = std::stoul(match[2].str());
        const auto k = std::stoul(match[3].str());
        const auto r = std::stoul(match[4].str());

        configs.emplace_back(n, p, k, r);
    }


    return configs;
}



int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int cluster_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);

    if (argc == 1) {
        print_usage(argv[0]);
        return -1;
    }

    const auto configs = collect_arguments(argc - 1, &argv[1]);


    const auto seed = 42;


    printf("n,p,k,variant,i,time,result\n");
    for (auto config : configs) {
        MPI_Comm comm;
        MPI_Comm_split(MPI_COMM_WORLD, rank < config.p, 0, &comm);

        const auto array = generate_test_vector(config.n, seed);
        const auto distribution = distribute_evenly(config.n, config.p);
        const auto regions = regions_from_distribution(distribution);
        const auto local_array = scatter_array(comm, array, distribution);

        {
            BinaryTreeSummation bts(rank, regions, config.k, comm);

            memcpy(bts.getBuffer(), local_array.data(), distribution.send_counts[rank] * sizeof(double));

            const auto results = measure([&bts] () {
                    return bts.accumulate();
            }, config.r);

            for (const auto &result : results) {
                print_result(config, result, "bts");
            }
        }

        {
            ReproblasSummation reproblas(comm, regions[rank].size);

            memcpy(reproblas.getBuffer(), local_array.data(), distribution.send_counts[rank] * sizeof(double));

            const auto results = measure([&reproblas] () {
                    return reproblas.accumulate();
            }, config.r);

            for (const auto &result : results) {
                print_result(config, result, "reproblas");
            }
        }

        {
            KGatherSummation kgs(rank, regions, config.k, comm);

            memcpy(kgs.getBuffer(), local_array.data(), distribution.send_counts[rank] * sizeof(double));

            const auto results = measure([&kgs] () {
                    return kgs.accumulate();
            }, config.r);

            for (const auto &result : results) {
                print_result(config, result, "kgather");
            }
        }

        {
        {
            AllreduceSummation ars(comm, regions[rank].size);

            memcpy(ars.getBuffer(), local_array.data(), distribution.send_counts[rank] * sizeof(double));

            const auto results = measure([&ars] () {
                    return ars.accumulate();
            }, config.r);

            for (const auto &result : results) {
                print_result(config, result, "allreduce");
            }
        }

        }
    }


    MPI_Finalize();
}

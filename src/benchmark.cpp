#include <chrono>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <regex>
#include <string>
#include <util.hpp>
#include <utility>
#include <vector>
#include "allreduce_summation.hpp"
#include "binary_tree_summation.h"
#include "dual_tree_summation.hpp"

#ifdef SCOREP
#include <scorep/SCOREP_User.h>
#endif

#include <binary_tree_summation.hpp>
#include <kgather_summation.hpp>
#include <reproblas_summation.hpp>

using std::pair;
using std::string;
using std::vector;

constexpr auto MAX_MESSAGE_SIZE_BYTES = 4UL * 1024 * 1024 * 1024;
constexpr auto MAX_MESSAGE_SIZE_DOUBLES = MAX_MESSAGE_SIZE_BYTES / 8;

void print_usage(char *program_name) {
    fprintf(stderr,
            "Usage: %s n,p,k,r n,p,k,r,...\n\twhere\n\t\tn = length of "
            "array\n\t\tp = cluster size\n\t\tk = linear sum parameter\n\t\tr = "
            "number of repetitions\n",
            program_name);
}

struct TestConfig {
    uint64_t n;
    uint64_t p;
    uint64_t k;
    uint64_t r;

    TestConfig(uint64_t n, uint64_t p, uint64_t k, uint64_t r) :
        n{n}, p{p}, k{k}, r{r} {}
};

struct BenchmarkIteration {
    Timer::duration duration;
    double result;
    uint64_t iteration;
};

void print_result(const TestConfig &config, const BenchmarkIteration &it, const char *variant) {
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(it.duration);
    printf("%zu,%zu,%zu,%s,%zu,%zu,%.50f\n", config.n, config.p, config.k, variant, it.iteration, duration.count(),
           it.result);
}

template<typename F, typename G>
vector<BenchmarkIteration> measure(F prepare, G run, uint64_t repetitions) {
    vector<BenchmarkIteration> results(repetitions);
    for (auto i = 0U; i < repetitions; ++i) {
        prepare();
        Timer t;
        double result = run();
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
        if (!ret)
            continue;

        const auto n = std::stoul(match[1].str());
        const auto p = std::stoul(match[2].str());
        const auto k = std::stoul(match[3].str());
        const auto r = std::stoul(match[4].str());

        configs.emplace_back(n, p, k, r);
    }

    return configs;
}

int main(int argc, char **argv) {
#ifdef SCOREP
    SCOREP_USER_REGION_DEFINE(region_benchmark_loop);
    SCOREP_USER_REGION_DEFINE(region_bts);
    SCOREP_USER_REGION_DEFINE(region_dts);
    SCOREP_USER_REGION_DEFINE(region_kgather);
    SCOREP_USER_REGION_DEFINE(region_allreduce);
    SCOREP_USER_REGION_DEFINE(region_reproblas);
#endif
    MPI_Init(&argc, &argv);

    int rank;
    int cluster_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);

    const auto debug_str = std::getenv("DEBUG_MPI_RANK");
    if (debug_str != nullptr) {
        const auto debug_rank = std::stoi(debug_str);
        attach_debugger(debug_rank == rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (argc == 1) {
        print_usage(argv[0]);
        return -1;
    }

    const auto configs = collect_arguments(argc - 1, &argv[1]);

    const auto seed = 42;

    if (rank == 0) {
        printf("n,p,k,variant,i,time,result\n");
    }

    for (auto config: configs) {
        const bool participating = rank < config.p;
        MPI_Comm comm;
        MPI_Comm_split(MPI_COMM_WORLD, participating, 0, &comm);

        if (!participating)
            continue;

        int new_rank, new_size;
        MPI_Comm_rank(comm, &new_rank);
        MPI_Comm_size(comm, &new_size);
        assert(config.p == new_size);
        assert(new_rank == rank);

        vector<double> array;
        if (rank == 0) {
            array = generate_test_vector(config.n, seed);
        }

        const auto distribution = distribute_evenly(config.n, config.p);
        const auto regions = regions_from_distribution(distribution);
        const auto local_array = scatter_array(comm, array, distribution);
#ifdef SCOREP
        SCOREP_USER_REGION_BEGIN(region_benchmark_loop, "benchmark", SCOREP_USER_REGION_TYPE_LOOP);
        SCOREP_USER_PARAMETER_UINT64("n", config.n);
        SCOREP_USER_PARAMETER_UINT64("p", config.p);
        SCOREP_USER_PARAMETER_UINT64("k", config.k);
#endif

        {
            BinaryTreeSummation bts(rank, regions, config.k, comm);

#ifdef SCOREP
            SCOREP_USER_REGION_BEGIN(region_bts, "binarytreesummation", SCOREP_USER_REGION_TYPE_COMMON);
#endif
            const auto results = measure(
                    [&bts, &local_array, &distribution, &rank]() {
                        memcpy(bts.getBuffer(), local_array.data(), distribution.send_counts[rank] * sizeof(double));
                    },
                    [&bts]() { return bts.accumulate(); }, config.r);
#ifdef SCOREP
            SCOREP_USER_REGION_END(region_bts);
#endif
            if (rank == 0) {
                for (const auto &result: results) {
                    print_result(config, result, "bts");
                }
            }
        }
        {
            ReproblasSummation reproblas(comm, regions[rank].size);

            memcpy(reproblas.getBuffer(), local_array.data(), distribution.send_counts[rank] * sizeof(double));
#ifdef SCOREP
            SCOREP_USER_REGION_BEGIN(region_reproblas, "reproblas", SCOREP_USER_REGION_TYPE_COMMON);
#endif

            const auto results = measure([]() {}, [&reproblas]() { return reproblas.accumulate(); }, config.r);

#ifdef SCOREP
            SCOREP_USER_REGION_END(region_reproblas);
#endif
            if (rank == 0) {
                for (const auto &result: results) {
                    print_result(config, result, "reproblas");
                }
            }
        }

        {
#ifdef SCOREP
            SCOREP_USER_REGION_BEGIN(region_dts, "dualtreesummation", SCOREP_USER_REGION_TYPE_COMMON);
#endif
            DualTreeSummation dual_tree_summation(rank, regions, comm);

            const auto results = measure(
                    [&dual_tree_summation, &local_array, &distribution, &rank]() {
                        memcpy(dual_tree_summation.getBuffer(), local_array.data(),
                               distribution.send_counts[rank] * sizeof(double));
                    },
                    [&dual_tree_summation]() { return dual_tree_summation.accumulate(); }, config.r);

            if (rank == 0) {
                for (const auto &result: results) {
                    print_result(config, result, "dts");
                }
            }


#ifdef SCOREP
            SCOREP_USER_REGION_END(region_dts);
#endif
        }

        if (config.k > 1 && config.n / config.k < MAX_MESSAGE_SIZE_DOUBLES) {
            KGatherSummation kgs(rank, regions, config.k, comm);

            memcpy(kgs.getBuffer(), local_array.data(), distribution.send_counts[rank] * sizeof(double));

#ifdef SCOREP
            SCOREP_USER_REGION_BEGIN(region_kgather, "kgather", SCOREP_USER_REGION_TYPE_COMMON);
#endif
            const auto results = measure(
                    [&kgs, &local_array, &distribution, &rank]() {
                        memcpy(kgs.getBuffer(), local_array.data(), distribution.send_counts[rank] * sizeof(double));
                    },
                    [&kgs]() { return kgs.accumulate(); }, config.r);
#if SCOREP
            SCOREP_USER_REGION_END(region_kgather);
#endif

            if (rank == 0) {
                for (const auto &result: results) {
                    print_result(config, result, "kgather");
                }
            }
        }

        {
            AllreduceSummation ars(comm, regions[rank].size, AllreduceType::REDUCE_AND_BCAST);

            memcpy(ars.getBuffer(), local_array.data(), distribution.send_counts[rank] * sizeof(double));
#ifdef SCOREP
            SCOREP_USER_REGION_BEGIN(region_allreduce, "allreduce", SCOREP_USER_REGION_TYPE_COMMON);
#endif

            const auto results = measure([]() {}, [&ars]() { return ars.accumulate(); }, config.r);

#ifdef SCOREP
            SCOREP_USER_REGION_END(region_allreduce);
#endif
            if (rank == 0) {
                for (const auto &result: results) {
                    print_result(config, result, "reduce_bcast");
                }
            }
        }
        {
            AllreduceSummation ars(comm, regions[rank].size, AllreduceType::ALLREDUCE);

            memcpy(ars.getBuffer(), local_array.data(), distribution.send_counts[rank] * sizeof(double));

            const auto results = measure([]() {}, [&ars]() { return ars.accumulate(); }, config.r);

            if (rank == 0) {
                for (const auto &result: results) {
                    print_result(config, result, "allreduce");
                }
            }
        }
        {
            AllreduceSummation ars(comm, regions[rank].size, AllreduceType::VECTORIZED_ALLREDUCE);

            memcpy(ars.getBuffer(), local_array.data(), distribution.send_counts[rank] * sizeof(double));

            const auto results = measure([]() {}, [&ars]() { return ars.accumulate(); }, config.r);

            if (rank == 0) {
                for (const auto &result: results) {
                    print_result(config, result, "vectorized_allreduce");
                }
            }
        }
#ifdef SCOREP
        SCOREP_USER_REGION_END(region_benchmark_loop);
#endif
        MPI_Comm_free(&comm);
    }

    MPI_Finalize();
}

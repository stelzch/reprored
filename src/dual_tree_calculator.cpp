#include <dual_tree_topology.hpp>
#include <vector>

pair<vector<operation_result>, vector<DualTreeTopology>> compute_operations_per_rank(const uint64_t n, const uint64_t p,
                                                                                     const unsigned int m) {
    vector<operation_result> result;
    result.reserve(p);


    vector<DualTreeTopology> topologies;
    topologies.reserve(p);

    // Create topologies in first iteration
    const auto regions = regions_from_distribution(distribute_evenly(n, p));
    for (auto rank = 0U; rank < p; ++rank) {
        topologies.emplace_back(rank, regions, m);
    }

    // Compute operations in second iteration, because only then are all outgoing coords known
    for (auto rank = 0U; rank < p; ++rank) {
        std::set<TreeCoordinates> incoming_coords;

        for (const uint64_t &child_rank: topologies[rank].get_comm_children()) {
            const auto &outgoing = topologies[child_rank].get_outgoing();
            incoming_coords.insert(outgoing.begin(), outgoing.end());
        }

        result.emplace_back(topologies[rank].compute_operations(incoming_coords));
    }

    return std::make_pair(result, topologies);
}


int main(int argc, char **argv) {
    while (!std::cin.eof()) {
        uint64_t n, p, m;
        std::cin >> n;
        if (std::cin.fail())
            return 2;

        std::cin >> p;
        if (std::cin.fail())
            return 2;

        std::cin >> m;
        if (std::cin.fail())
            return 2;

        auto [result, topologies] = compute_operations_per_rank(n, p, m);

        printf("[\n");
        for (auto rank = 0U; rank < p; ++rank) {
            auto local_elements = 0UL;
            for (const auto &[x, y]: result[rank].local_coords) {
                local_elements += DualTreeTopology::pow2(y);
            }


            printf("{\"rank\":%u, \"comm_parent\": %zu, \"n_outgoing\": %zu, \"n_ops\":%zu, \"n_local_elements\":%zu, "
                   "\"local_elements\": [",
                   rank, topologies[rank].get_comm_parent(), topologies[rank].get_outgoing().size(),
                   result[rank].ops.size(), local_elements);

            auto i = 0UL;
            for (const auto &[x, y]: result[rank].local_coords) {
                bool is_last_element = i++ == result[rank].local_coords.size() - 1;
                printf("[%zu, %u]%s", x, y, is_last_element ? "" : ", ");
            }

            printf("], \"ops\":\"");
            for (const auto &op: result[rank].ops) {
                printf("%s", op == OPERATION_PUSH ? "p" : "r");
            }

            printf("\"}%s\n", rank < p - 1 ? ", " : "");
        }
        printf("]\n\n");
    }
}

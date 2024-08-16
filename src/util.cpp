#include <util.hpp>

#include "k_chunked_array.hpp"
#include <algorithm>
#include <cassert>
#include <random>

vector<region> regions_from_distribution(const Distribution& d) {
    vector<region> regions;
    regions.reserve(d.displs.size());

    for (auto i = 0U; i < d.displs.size(); ++i) {
        regions.emplace_back(d.displs[i], d.send_counts[i]);
    }

    return regions;
}


vector<int> displacement_from_sendcounts(std::vector<int>& send_counts) {
    std::vector<int> displacement;
    displacement.reserve(send_counts.size());

    int start_index = 0;
    for (auto const& send_count: send_counts) {
        displacement.push_back(start_index);
        start_index += send_count;
    }

    return displacement;
}

Distribution distribute_evenly(size_t const collection_size, size_t const comm_size) {
    auto const elements_per_rank = collection_size / comm_size;
    auto const remainder         = collection_size % comm_size;

    std::vector<int> send_counts(comm_size, elements_per_rank);
    std::for_each_n(send_counts.begin(), remainder, [](auto& n) { n += 1; });

    return Distribution(send_counts, displacement_from_sendcounts(send_counts));
}

Distribution distribute_randomly(size_t const collection_size, size_t const comm_size, size_t const seed) {
    std::mt19937                    rng(seed);
    std::uniform_int_distribution<> dist(0, collection_size);

    // See https://stackoverflow.com/a/48205426 for details
    std::vector<int> points(comm_size, 0UL);
    points.push_back(collection_size);
    std::generate(points.begin() + 1, points.end() - 1, [&dist, &rng]() { return dist(rng); });
    std::sort(points.begin(), points.end());

    std::vector<int> send_counts(comm_size);
    for (size_t i = 0; i < send_counts.size(); ++i) {
        send_counts[i] = points[i + 1] - points[i];
    }

    // Shuffle to generate distributions where start indices are not monotonically increasing
    std::vector<size_t> indices(send_counts.size(), 0);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    auto displacement = displacement_from_sendcounts(send_counts);
    assert(send_counts.size() == displacement.size());

    decltype(send_counts)  shuffled_send_counts(send_counts.size(), 0);
    decltype(displacement) shuffled_displacement(displacement.size(), 0);
    for (auto i = 0UL; i < send_counts.size(); ++i) {
        shuffled_send_counts[i]  = send_counts[indices[i]];
        shuffled_displacement[i] = displacement[indices[i]];
    }

    assert( collection_size == std::reduce(shuffled_send_counts.begin(), shuffled_send_counts.end(), 0UL, std::plus<>()));

    return Distribution(shuffled_send_counts, shuffled_displacement);
}
vector<double> generate_test_vector(size_t length, size_t seed) {
    std::mt19937                   rng(seed);
    std::uniform_real_distribution distr;
    std::vector<double>            result(length);
    std::generate(result.begin(), result.end(), [&distr, &rng]() { return distr(rng); });

    return result;
}

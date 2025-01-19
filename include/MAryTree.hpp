#pragma once
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <valarray>


/**
 * M-ary tree over consecutive array indices.
 */
class MAryTree {
public:
    /**
     * Construct helper class for a full m-ary tree over consecutive array indices.
     * Parameter \p n denotes the array length and each inner node of the tree has at most \p m children.
     */
    MAryTree(const uint64_t n, const unsigned int m) :
        n{n}, m{m} {
        assert(m > 0);
    }

    /// Logarithm to base m \f$\log_m(x)\f$
    double log_m(const double x) const { return std::log(x) / std::log(m); }

    /// Number of levels in the tree
    uint64_t tree_height() const { return std::ceil(log_m(n)); }

    /** @brief Maximum levels of subtree rooted at \p x
     *
     * For x = 0, this is equivalent to \ref tree_height()
     */
    uint64_t max_y(const uint64_t x) const {
        if (x == 0)
            return tree_height();

        uint64_t divisor = m;
        for (auto i = 0U; i < sizeof(decltype(x)) * 8; ++i) {
            if (x % divisor != 0) {
                return i;
            }

            divisor *= m;
        }

        return 0;
    }

    uint64_t largest_child_index(const uint64_t x) const {
        return std::min(n - 1, static_cast<uint64_t>(x + std::pow(m, max_y(x)) - 1));
    }

    uint64_t parent(const uint64_t x) const {
        if (x == 0) {
            return 0;
        }

        // The parent operation is equivalent to clearing the least significant digit of x when represented in base m.
        // We achieve this by dividing the number by m to the power of the least significant digit without rest and
        // multiplying it back to its previous place.
        const auto place_of_least_significand_digit = 1 + max_y(x);
        const auto divisor = static_cast<uint64_t>(std::pow(m, place_of_least_significand_digit));

        return divisor * (x / divisor);
    }


    std::vector<uint64_t> subtree_children(const uint64_t x) const {
        std::vector<uint64_t> children;
        children.reserve(max_y(x) * (m - 1));


        for (auto y = 1U; y <= max_y(x); ++y) {
            const auto stride = static_cast<uint64_t>(std::pow(m, y - 1));

            for (auto i = 1U; i < m; ++i) {
                const auto child_x = x + i * stride;

                if (child_x >= n) {
                    break;
                }

                children.push_back(child_x);
            }
        }

        return children;
    }


private:
    const uint64_t n;
    const unsigned int m;
};

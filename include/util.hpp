#include <cstddef>

template<typename T>
inline T round_up_to_multiple(T x, T n) {
    return (x % n == 0) ? x : x + n - (x % n);
}

template<typename T>
inline T round_down_to_multiple(T x, T n) {
    return x - (x % n);
}

inline bool index_in_bounds(size_t idx, size_t size) {
    return idx < size;
}

inline bool implicates(bool a, bool b) {
    return !a || b;
}

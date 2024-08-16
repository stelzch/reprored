#pragma once

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>

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

template<class T>
struct AlignedAllocator
{
    typedef T value_type;

    // default constructor
    AlignedAllocator () =default;

    // copy constructor
    template <class U> constexpr AlignedAllocator (const AlignedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();

        if (auto p = static_cast<T*>(std::aligned_alloc(32, n * sizeof(T)))) {
            return p;
        }

        throw std::bad_alloc();
    }

    void deallocate(T* p, std::size_t n) noexcept {
        std::free(p);
    }
};

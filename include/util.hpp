#pragma once

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <mpi.h>
#include <new>
#include <vector>

struct region {
  uint64_t globalStartIndex;
  uint64_t size;

  region() : globalStartIndex(0), size(0) {}
  region(uint64_t globalStartIndex, uint64_t size)
      : globalStartIndex(globalStartIndex), size(size) {}
};

template <typename T> inline T round_up_to_multiple(T x, T n) {
  return (x % n == 0) ? x : x + n - (x % n);
}

template <typename T> inline T round_down_to_multiple(T x, T n) {
  return x - (x % n);
}

inline bool index_in_bounds(size_t idx, size_t size) { return idx < size; }

inline bool implicates(bool a, bool b) { return !a || b; }

template <class T> struct AlignedAllocator {
  typedef T value_type;

  // default constructor
  AlignedAllocator() = default;

  // copy constructor
  template <class U>
  constexpr AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

  T *allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_array_new_length();

    if (auto p = static_cast<T *>(std::aligned_alloc(32, n * sizeof(T)))) {
      return p;
    }

    throw std::bad_alloc();
  }

  void deallocate(T *p, std::size_t n) noexcept { std::free(p); }
};

using Distribution = struct Distribution {
  std::vector<int> send_counts;
  std::vector<int> displs;

  Distribution(std::vector<int> _send_counts, std::vector<int> recv_displs)
      : send_counts(_send_counts), displs(recv_displs) {}
};

template <typename C, typename T>
std::vector<T> scatter_array(C comm, std::vector<T> const &global_array,
                             Distribution const d) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  std::vector<T> result(d.send_counts[rank]);

  MPI_Scatterv(global_array.data(), d.send_counts.data(), d.displs.data(),
               MPI_DOUBLE, result.data(), result.size(), MPI_DOUBLE, 0, comm);

  return result;
}

std::vector<region> regions_from_distribution(const Distribution &d);
std::vector<int> displacement_from_sendcounts(std::vector<int> &send_counts);
Distribution distribute_evenly(size_t const collection_size,
                               size_t const comm_size);
Distribution distribute_randomly(size_t const collection_size,
                                 size_t const comm_size, size_t const seed);
std::vector<double> generate_test_vector(size_t length, size_t seed);

class Timer {
public:
  using duration = decltype(std::chrono::steady_clock::now() -
                            std::chrono::steady_clock::now());

  Timer() { start(); }

  duration stop() {
    asm("" ::: "memory");
    auto end = std::chrono::steady_clock::now();
    asm("" ::: "memory");

    return end - _start;
  }

  void start() {
    asm("" ::: "memory"); // prevent compiler reordering
    _start = std::chrono::steady_clock::now();
    asm("" ::: "memory");
  }

  template <class Func> static duration time_func(Func func) {
    Timer timer;
    func();
    return timer.stop();
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> _start;
};

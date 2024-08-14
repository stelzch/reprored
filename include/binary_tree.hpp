#include <cassert>
#include <cstdint>
#include <vector>
#include <chrono>
#include <array>
#include <map>
#include <utility>
#include <mpi.h>

using std::vector;
using std::array;
using std::map;

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

const uint8_t MAX_MESSAGE_LENGTH = 4;

struct MessageBufferEntry {
    uint64_t index;
    double value;
};

class MessageBuffer {

public:
    MessageBuffer(MPI_Comm comm);
    const void receive(const int sourceRank);
    void flush(void);
    void wait(void);

    void put(const int targetRank, const uint64_t index, const double value);
    const double get(const int sourceRank, const uint64_t index);

    const void printStats(void) const;

protected:

    array<MessageBufferEntry, MAX_MESSAGE_LENGTH> entries;
    map<uint64_t, double> inbox;
    int targetRank;
    vector<MessageBufferEntry> outbox;
    vector<MessageBufferEntry> buffer;
    vector<MPI_Request> reqs;
    size_t awaitedNumbers;
    size_t sentMessages;
    size_t sentSummands;
    bool sendBufferClear;
    MPI_Comm comm;
};

typedef struct {
    uint64_t globalStartIndex;
    uint64_t size;
} region;

class BinaryTreeSummation {
public:
    /** Instantiate new binary tree accumulator.
     * For a reproducible result, the order of numbers must remain the same
     * over different runs. This order is represented by startIndices, which
     * contains the start index for each processor in the cluster, and
     * globalSize, which is the total number of summands. 
     *
     */
    BinaryTreeSummation(uint64_t rank, const vector<region> regions,
            uint64_t K = 1, MPI_Comm comm = MPI_COMM_WORLD);

    virtual ~BinaryTreeSummation();

    static const uint64_t parent(const uint64_t i);

    bool isLocal(uint64_t index) const;

    /** Determine which rank has the number with a given index */
    uint64_t rankFromIndexMap(const uint64_t index) const;

    double *getBuffer();
    void storeSummand(uint64_t localIndex, double val);


    /* Sum all numbers. Will return the total sum on rank 0
     */
    double accumulate(void);

    /* Calculate all rank-intersecting summands that must be sent out because
     * their parent is non-local and located on another rank
     */
    vector<uint64_t> calculateRankIntersectingSummands(void) const;

    /* Return the average number of nanoseconds spend in total on waiting for intermediary
     * results from other hosts
     */
    const double acquisitionTime(void) const;

    double accumulate(uint64_t index);

    const void printStats(void) const;

    const int get_rank() const;
protected:
    void linear_sum_k();

    const vector<region> calculate_k_regions(const vector<region> regions) const;
    const vector<int> calculate_k_predecessors() const;
    const int calculate_k_successor() const;
    const uint64_t largest_child_index(const uint64_t index) const;
    const uint64_t subtree_size(const uint64_t index) const;

    /** Figure out if the parts that make up a certain index are all local and form a subtree
     * of a specifc size */
    const bool is_local_subtree_of_size(const uint64_t expectedSubtreeSize, const uint64_t i) const;
    const double accumulate_local_8subtree(const uint64_t startIndex) const;

    inline const double sum_remaining_8tree(const uint64_t bufferStartIndex,
            const uint64_t initialRemainingElements,
            const int y,
            const uint64_t maxX,
            double *srcBuffer,
            double *dstBuffer) {
        uint64_t remainingElements = initialRemainingElements;

        for (int level = 0; level < 3; level++) {
            const int stride = 1 << (y - 1 + level);
            int elementsWritten = 0;
            for (uint64_t i = 0; (i + 1) < remainingElements; i += 2) {
                dstBuffer[elementsWritten++] = srcBuffer[i] + srcBuffer[i + 1];
            }


            if (remainingElements % 2 == 1) {
                const uint64_t bufferIndexA = remainingElements - 1;
                const uint64_t bufferIndexB = remainingElements;
                const uint64_t indexB = bufferStartIndex + bufferIndexB * stride;
                const double a = srcBuffer[bufferIndexA];

                if (indexB > maxX) {
                    // indexB is the last element because the subtree ends there
                    dstBuffer[elementsWritten++] = a;
                } else {
                    // indexB must be fetched from another rank
                    const double b = message_buffer.get(rankFromIndexMap(indexB), indexB);
                    dstBuffer[elementsWritten++] = a + b;
                }

                remainingElements += 1;
            }

	    srcBuffer = dstBuffer;
            remainingElements /= 2;
        }
        assert(remainingElements == 1);

        return dstBuffer[0];
    }


private:
    const uint64_t k;
    const int rank, clusterSize;
    const bool is_last_rank;
    const vector<region> regions;
    const MPI_Comm comm;
    const uint64_t size,  begin, end;

    const bool no_k_intercept; // if true no number in [begin, end) is divisible by k
    const vector<region> k_regions;
    const uint64_t k_size,  k_begin, k_end;
    const uint64_t k_left_remainder;
    const uint64_t k_right_remainder;

    const vector<int> k_predecessor_ranks; // ranks we receive from during linear sum.
                                      // In non-degenerate case this is the next lower rank
    const int k_successor_rank; // ranks we send to during linear sum.
                           // In non-degenerate case this is the next higher rank.
    vector<MPI_Request> k_recv_reqs;

    const uint64_t globalSize;
    const uint64_t accumulation_buffer_offset_pre_k;
    const uint64_t accumulation_buffer_offset_post_k;


    vector<double, AlignedAllocator<double>> accumulation_buffer;
    std::chrono::duration<double> acquisition_duration;
    std::map<uint64_t, int> start_indices;
    long int acquisition_count;
    vector<uint64_t> rank_intersecting_summands;

    uint64_t reduction_counter;


    MessageBuffer message_buffer;
};

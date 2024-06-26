#include <mpi.h>
#include <iostream>
#include <fstream>
#include <exception>
#include <vector>
#include <numeric>
#include <cstring>
#include <cassert>
#include <cmath>
#include <unistd.h>
#include <memory>
#include <functional>
#include <chrono>
#include <io.hpp>
#include "binary_tree.hpp"

#ifdef AVX
#include <immintrin.h>
#endif

using namespace std;
using namespace std::string_literals;


const int MESSAGEBUFFER_MPI_TAG = 1;

MessageBuffer::MessageBuffer(MPI_Comm comm) :
    inbox(),
    targetRank(-1),
    awaitedNumbers(0),
    sentMessages(0),
    sentSummands(0),
    sendBufferClear(true),
    comm(comm)
    {
    outbox.reserve(MAX_MESSAGE_LENGTH + 1);
    buffer.resize(MAX_MESSAGE_LENGTH);
    reqs.reserve(16);
}

void MessageBuffer::wait() {
    for (MPI_Request &r : reqs) {
        MPI_Wait(&r, MPI_STATUS_IGNORE);
    }

    reqs.clear();
    sendBufferClear = true;
}


void MessageBuffer::flush() {
    if(targetRank == -1 || outbox.size() == 0) return;
    MPI_Request r;
    reqs.push_back(r);

    const int messageByteSize = sizeof(MessageBufferEntry) * outbox.size();

    assert(targetRank >= 0);
    MPI_Isend(static_cast<void *>(&outbox[0]), messageByteSize, MPI_BYTE, targetRank,
            MESSAGEBUFFER_MPI_TAG, comm, &reqs.back());
    sentMessages++;

    targetRank = -1;
    outbox.clear();
    sendBufferClear = false;
}

const void MessageBuffer::receive(const int sourceRank) {
    MPI_Status status;

    MPI_Recv(static_cast<void *>(&buffer[0]), sizeof(MessageBufferEntry) * MAX_MESSAGE_LENGTH, MPI_BYTE,
            sourceRank, MESSAGEBUFFER_MPI_TAG, comm, &status);
    awaitedNumbers++;

    const int receivedEntries = status._ucount / sizeof(MessageBufferEntry);

    for (int i = 0; i < receivedEntries; i++) {
        MessageBufferEntry entry = buffer[i];
        inbox[entry.index] = entry.value;
    }
}

void MessageBuffer::put(const int targetRank, const uint64_t index, const double value) {
    if (outbox.size() >= MAX_MESSAGE_LENGTH || this->targetRank != targetRank) {
        flush();
    }

    /* Since we send asynchronously, we must check whether the buffer can currently be written to */
    if(!sendBufferClear) {
        wait();
    }

    if (this->targetRank == -1) {
        this->targetRank = targetRank;
    }

    MessageBufferEntry e;
    e.index = index;
    e.value = value;
    outbox.push_back(e);

    if (outbox.size() == MAX_MESSAGE_LENGTH) flush();

    sentSummands++;
}

const double MessageBuffer::get(const int sourceRank, const uint64_t index) {
    // If we have the number in our inbox, directly return it
    if (inbox.contains(index)) {
        double result = inbox[index];
        inbox.erase(index);
        return result;
    }

    // If not, we will wait for a message, but make sure no one is waiting for our results.
    flush();
    wait();
    receive(sourceRank);

    //cout << "Waiting for rank " << sourceRank << ", index " << index ;

    // Our computation order should guarantee that the number is contained within
    // the next package
    assert(inbox.contains(index));

    //cout << " [RECEIVED]" << endl;
    double result = inbox[index];
    inbox.erase(index);
    return result;
}

const void MessageBuffer::printStats() const {
    int rank;
    MPI_Comm_rank(comm, &rank);

    size_t globalStats[] {0, 0, 0};
    size_t localStats[] {sentMessages, sentMessages, sentSummands};

    MPI_Reduce(localStats, globalStats, 3, MPI_LONG, MPI_SUM,
            0, comm);

    if (rank == 0) {
        printf("sentMessages=%li\naverageSummandsPerMessage=%f\n",
                globalStats[0],
                globalStats[2] / static_cast<double>(globalStats[0]));

    }

}

BinaryTreeSummation::BinaryTreeSummation(uint64_t rank, vector<region> regions, MPI_Comm comm)
    :
      rank(rank),
      clusterSize(regions.size()),
      globalSize(std::accumulate(regions.begin(), regions.end(), 0,
                 [](uint64_t acc, const region& r) {
                  return acc + r.size;
        })),
      comm(comm),
      size(regions[rank].size),
      begin(regions[rank].globalStartIndex),
      end(begin + size),
      accumulationBuffer(1024),
      acquisitionDuration(std::chrono::duration<double>::zero()),
      acquisitionCount(0L),
      rankIntersectingSummands(calculateRankIntersectingSummands()),
      reduction_counter(0UL),
      messageBuffer(comm)
{

    /* Initialize start indices map */
    for (int p = 0; p < clusterSize; ++p) {
        if (regions[p].size == 0) continue;
        this->startIndices[regions[p].globalStartIndex] = p;
    }
    // guardian element
    this->startIndices[globalSize] = clusterSize;

    // Verify that the regions are actually correct.
    // This is given if the difference to the next start index is equal to the region size
    for (auto it = startIndices.begin(); it != startIndices.end(); ++it) {
        auto next = std::next(it);
        if (next == startIndices.end()) break;

        assert(it->first + regions[it->second].size == next->first);
    }
    
    if (accumulationBuffer.size() < (size  - 8)) {
        accumulationBuffer.resize(size);
    }

    int initialized;
    MPI_Initialized(&initialized);
    if (initialized) {
        int c_size;
        MPI_Comm_size(comm, &c_size);
        assert(c_size == regions.size());
    }



#ifdef DEBUG_OUTPUT_TREE
    printf("Rank %lu has %lu summands, starting from index %lu to %lu\n", rank, size, begin, end);
    printf("Rank %lu rankIntersectingSummands: ", rank);
    for (int ri : rankIntersectingSummands)
        printf("%u ", ri);
    printf("\n");
#endif
}


BinaryTreeSummation::~BinaryTreeSummation() {
#ifdef ENABLE_INSTRUMENTATION
    cout << "Rank " << rank << " avg. acquisition time: "
        << acquisitionTime() / acquisitionCount << "  ns\n";
#endif
}

double *BinaryTreeSummation::getBuffer() {
    return accumulationBuffer.data();
}
void BinaryTreeSummation::storeSummand(uint64_t localIndex, double val) {
    accumulationBuffer[localIndex] = val;
}

const uint64_t BinaryTreeSummation::parent(const uint64_t i) {
    assert(i != 0);

    // clear least significand set bit
    return i & (i - 1);
}

bool BinaryTreeSummation::isLocal(uint64_t index) const {
    return (index >= begin && index < end);
}

uint64_t BinaryTreeSummation::rankFromIndexMap(const uint64_t index) const {
    // Get an iterator to the start index that is greater than index
    auto it = startIndices.upper_bound(index);
    assert(it != startIndices.begin());
    --it;

    return it->second;
}

/* Calculate all rank-intersecting summands that must be sent out because
    * their parent is non-local and located on another rank
    */
vector<uint64_t> BinaryTreeSummation::calculateRankIntersectingSummands(void) const {
    vector<uint64_t> result;

    if (begin == 0 || size == 0) {
        return result;
    }

    assert(begin != 0);

    uint64_t index = begin;
    while (index < end) {
        assert(parent(index) < begin);
        result.push_back(index);

        index = index + subtree_size(index);
    }

    return result;
}

/* Sum all numbers. Will return the total sum on rank 0
    */
double BinaryTreeSummation::accumulate(void) {
    for (auto summand : rankIntersectingSummands) {
        if (subtree_size(summand) > 16) {
            // If we are about to do some considerable amount of work, make sure
            // the send buffer is empty so noone is waiting for our results
            messageBuffer.flush();
        }

        double result = accumulate(summand);

        messageBuffer.put(rankFromIndexMap(parent(summand)), summand, result);
    }
    messageBuffer.flush();
    messageBuffer.wait();

    double result = 0.0;
    const int root_rank = globalSize == 0 ? 0 : rankFromIndexMap(0);
    if (rank == root_rank) {
        // Start accumulation on first rank with assigned summands.
        result = accumulate(0);
    }


    MPI_Bcast(&result, 1, MPI_DOUBLE,
              root_rank, comm);

    ++reduction_counter;

    return result;
}


double BinaryTreeSummation::accumulate(const uint64_t index) {
    if (index & 1) {
        // no accumulation needed
        return accumulationBuffer[index - begin];
    }

    const uint64_t maxX = (index == 0) ? globalSize - 1
        : min(globalSize - 1, index + subtree_size(index) - 1);
    const int maxY = (index == 0) ? ceil(log2(globalSize)) : log2(subtree_size(index));

    const uint64_t largest_local_index = min(maxX, end - 1);
    const uint64_t n_local_elements = largest_local_index + 1 - index;

    uint64_t elementsInBuffer = n_local_elements;

    double *destinationBuffer = static_cast<double *>(&accumulationBuffer[index - begin]);
    double *sourceBuffer = destinationBuffer;


    for (int y = 1; y <= maxY; y += 3) {
        uint64_t elementsWritten = 0;

        for (uint64_t i = 0; i + 8 <= elementsInBuffer; i += 8) {
            __m256d a = _mm256_loadu_pd(static_cast<double *>(&sourceBuffer[i]));
            __m256d b = _mm256_loadu_pd(static_cast<double *>(&sourceBuffer[i+4]));
            __m256d level1Sum = _mm256_hadd_pd(a, b);

            __m128d c = _mm256_extractf128_pd(level1Sum, 1); // Fetch upper 128bit
            __m128d d = _mm256_castpd256_pd128(level1Sum); // Fetch lower 128bit
            __m128d level2Sum = _mm_add_pd(c, d);

            __m128d level3Sum = _mm_hadd_pd(level2Sum, level2Sum);

            destinationBuffer[elementsWritten++] = _mm_cvtsd_f64(level3Sum);
        }

        // number of remaining elements
        const uint64_t remainder = elementsInBuffer - 8 * elementsWritten;
        assert(0 <= remainder);
        assert(remainder < 8);

        if (remainder > 0) {
            const uint64_t bufferIdx = 8 * elementsWritten;
            const uint64_t indexOfRemainingTree = index + bufferIdx * (1UL << (y - 1));
            const double a = sum_remaining_8tree(indexOfRemainingTree,
                    remainder, y, maxX,
		    &sourceBuffer[0] + bufferIdx,
                    &destinationBuffer[0] + bufferIdx);
            destinationBuffer[elementsWritten++] = a;
        }

	// After first iteration, read only from accumulation buffer
	sourceBuffer = destinationBuffer;

        elementsInBuffer = elementsWritten;
    }

    assert(elementsInBuffer == 1);

    return destinationBuffer[0];
}

const double BinaryTreeSummation::acquisitionTime(void) const {
    return std::chrono::duration_cast<std::chrono::nanoseconds> (acquisitionDuration).count();
}

const uint64_t BinaryTreeSummation::largest_child_index(const uint64_t index) const {
    return index | (index - 1);
}

const uint64_t BinaryTreeSummation::subtree_size(const uint64_t index) const {
    assert(index != 0);
    return largest_child_index(index) + 1 - index;
}

const void BinaryTreeSummation::printStats() const {
    messageBuffer.printStats();
}

const int BinaryTreeSummation::get_rank() const {
    return rank;
}

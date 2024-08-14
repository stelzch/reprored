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

BinaryTreeSummation::BinaryTreeSummation(uint64_t rank, const vector<region> regions, uint64_t k, MPI_Comm comm)
    :
      k(k),
      rank(rank),
      clusterSize(regions.size()),
      is_last_rank(rank == clusterSize  - 1),
      regions(regions),
      comm(comm),
      size(regions[rank].size),
      begin(regions[rank].globalStartIndex),
      end(begin + size),
      no_k_intercept(begin % k != 0 && begin / k == end / k),
      k_regions(calculate_k_regions(regions)),
      k_size(k_regions[rank].size),
      k_begin(k_regions[rank].globalStartIndex),
      k_end(k_begin + k_size),
      k_left_remainder(k_regions[rank].size == 0 ? 0 : min(round_up_to_multiple(begin, k),end) - begin),
      k_right_remainder((is_last_rank && no_k_intercept) ? 0 : end - max(round_down_to_multiple(end, k), begin)),
      k_predecessor_ranks(calculate_k_predecessors()),
      k_successor_rank(calculate_k_successor()),
      k_recv_reqs(k_predecessor_ranks.size()),
      globalSize(std::accumulate(k_regions.begin(), k_regions.end(), 0,
                 [](uint64_t acc, const region& r) {
                  return acc + r.size;
        })),
      accumulation_buffer_offset_pre_k(k - k_left_remainder),
      accumulation_buffer_offset_post_k(k),
      accumulation_buffer(k + size + k_right_remainder), // TODO: this memory allocation is non-optimal on ranks
                                                                       // with no left remainder. Reduce it to an appropriate value.
      acquisition_duration(std::chrono::duration<double>::zero()),
      acquisition_count(0L),
      rank_intersecting_summands(calculateRankIntersectingSummands()),
      reduction_counter(0UL),
      message_buffer(comm)
{
    printf("rank %i left remainder %zu, right remainder %zu, size %zu, no_k_intercept %i, is_last_rank %i\n", rank, k_left_remainder, k_right_remainder, size, no_k_intercept, is_last_rank);

    assert(k > 0);
    assert(k_left_remainder < k);
    assert(k_right_remainder < k);
    assert(implicates(no_k_intercept, size < k));

    /* Initialize start indices map */
    for (int p = 0; p < clusterSize; ++p) {
        if (k_regions[p].size == 0) continue;
        this->start_indices[k_regions[p].globalStartIndex] = p;
    }
    // guardian element
    this->start_indices[globalSize] = clusterSize;

    // Verify that the regions are actually correct.
    // This is given if the difference to the next start index is equal to the region size
    for (auto it = start_indices.begin(); it != start_indices.end(); ++it) {
        auto next = std::next(it);
        if (next == start_indices.end()) break;

        assert(it->first + k_regions[it->second].size == next->first);
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
    return accumulation_buffer.data() + accumulation_buffer_offset_pre_k;
}
void BinaryTreeSummation::storeSummand(uint64_t localIndex, double val) {
    accumulation_buffer[accumulation_buffer_offset_pre_k + localIndex] = val;
}

const uint64_t BinaryTreeSummation::parent(const uint64_t i) {
    assert(i != 0);

    // clear least significand set bit
    return i & (i - 1);
}

bool BinaryTreeSummation::isLocal(uint64_t index) const {
    return (index >= k_begin && index < k_end);
}

uint64_t BinaryTreeSummation::rankFromIndexMap(const uint64_t index) const {
    // Get an iterator to the start index that is greater than index
    auto it = start_indices.upper_bound(index);
    assert(it != start_indices.begin());
    --it;

    return it->second;
}

/* Calculate all rank-intersecting summands that must be sent out because
    * their parent is non-local and located on another rank
    */
vector<uint64_t> BinaryTreeSummation::calculateRankIntersectingSummands(void) const {
    vector<uint64_t> result;

    if (k_begin == 0 || k_size == 0) {
        return result;
    }

    assert(k_begin != 0);

    uint64_t index = k_begin;
    while (index < k_end) {
        assert(parent(index) < k_begin);
        result.push_back(index);

        index = index + subtree_size(index);
    }

    return result;
}

const vector<region> BinaryTreeSummation::calculate_k_regions(const vector<region> regions) const {
    const auto last_region = std::max_element(regions.begin(), regions.end(), [] (const auto& a, const auto& b) {
            return a.globalStartIndex < b.globalStartIndex;
    });

    vector<region> k_regions;
    k_regions.reserve(regions.size());

    for (auto it = regions.begin(); it < regions.end(); ++it) {
        const region& r = *it;
        const auto start = round_down_to_multiple(r.globalStartIndex, k) / k;
        auto end = round_down_to_multiple(r.globalStartIndex + r.size, k) / k;

        // Additional element at the end
        if (it == last_region && (r.globalStartIndex + r.size) % k != 0) {
            end += 1;
        }

        k_regions.emplace_back(start, end - start);
    }

    return k_regions;
}

const vector<int> BinaryTreeSummation::calculate_k_predecessors() const {
    vector<int> predecessors;

    if (rank == 0 || k_left_remainder == 0) {
        // There is no-one we receive from
        return predecessors;
    }

    // Move left 
    for (int i = rank - 1; i >= 0; --i) {
        if ((regions[i].globalStartIndex + regions[i].size) % k == 0) {
            // This rank won't have to send us a remainder because the
            // PE-border coincides with the k-region border
            break;
        }

        // TODO: handle ranks that have zero numbers assigned
        predecessors.push_back(i);

        if (k_regions[i].size >= 1) {
            // The rank i has a k-region assigned so any ranks lower than i
            // will send their remainder to to i instead.
            break;
        }
    }

    // We build the list of predecessors from right to left (ranks descending)
    // but during traversal we want the ranks to ascend.
    std::reverse(predecessors.begin(), predecessors.end());


    return predecessors;
}

const int BinaryTreeSummation::calculate_k_successor() const {
    // No successor on last rank
    if (rank == clusterSize - 1) {
        return -1;
    }

    for (int i = rank + 1; i < clusterSize; ++i) {
        if (k_regions[i].size > 0) {
            return i;
        }
    }

    assert(0); // We should never reach this statement, unless the global size is 0.
    return -2;
}

void BinaryTreeSummation::linear_sum_k() {
    MPI_Request send_req;

    // Sum & send right remainder
    // TODO: support unreduced sends
    const bool is_last_rank = rank == clusterSize - 1;
    if (k_size == 0 && rank != 0) {
        // We do not reduce any summands on our own, we simply pass them to the successor
        assert(k_successor_rank > 0);
        assert(size == k_right_remainder);

        printf("rank %i sending all data to successor %i\n", k_successor_rank);

        MPI_Send(&accumulation_buffer[accumulation_buffer_offset_pre_k], size, MPI_DOUBLE, k_successor_rank, MESSAGEBUFFER_MPI_TAG, comm);
        return; // We are done here

    } else if (k_right_remainder > 0 && !is_last_rank) {
        assert(k_successor_rank > 0);
        double acc = std::accumulate(&accumulation_buffer[accumulation_buffer_offset_pre_k + size - k_right_remainder], &accumulation_buffer[accumulation_buffer_offset_pre_k + size], 0.0);
        printf("rank %i successor is %i, sending accumulated %zu numbers (result %f)\n", rank, k_successor_rank, k_right_remainder, acc);
        MPI_Isend(&acc, 1, MPI_DOUBLE, k_successor_rank, MESSAGEBUFFER_MPI_TAG, comm, &send_req);
    }

    // Start receive requests for the left remainder
    // TODO: ask someone with more MPI experience if this is really necessary
    // or if the sent values will be cached on the destination machine
    // regardless
    double left_remainder_accumulator;
    uint64_t left_remainder_running_index = 0;

    printf("rank %i receiving from ", rank);
    for (int i = 0U; i < k_predecessor_ranks.size(); ++i) {
        const auto other_rank = k_predecessor_ranks[i];
        if (i == 0) {
            printf("%i (1 element) ", other_rank);
            assert(k_regions[other_rank].size > 0 || other_rank == 0);
            MPI_Irecv(&left_remainder_accumulator, 1, MPI_DOUBLE, other_rank, MESSAGEBUFFER_MPI_TAG, comm, &k_recv_reqs[i]);
        } else {
            assert(k_regions[other_rank].size == 0);
            const auto elements_to_receive = regions[other_rank].size; // We receive all numbers the other rank holds
            printf("%i (%zu elements)  ", other_rank, elements_to_receive);

            MPI_Irecv(&accumulation_buffer[left_remainder_running_index], elements_to_receive, MPI_DOUBLE, other_rank, MESSAGEBUFFER_MPI_TAG, comm, &k_recv_reqs[i]);
            left_remainder_running_index += elements_to_receive;
            assert(left_remainder_running_index < k);
        }
    }
    printf("\n");

    // Sum local k-tuples that do not overlap with PE-boundaries
    const bool has_left_remainder = (k_left_remainder > 0);
    uint64_t target_idx = has_left_remainder ? 1U : 0U;
    for (uint64_t i = k_left_remainder; i + k - 1 < size; i += k) {
        printf("rank %i summing local k tuple %zu-%zu\n", rank, i, i+k);
        accumulation_buffer[accumulation_buffer_offset_post_k + target_idx++] = std::accumulate(&accumulation_buffer[accumulation_buffer_offset_pre_k + i], &accumulation_buffer[accumulation_buffer_offset_pre_k + i + k], 0.0);
    }

    // On the last rank manually sum right remainder since it can not be sent anywhere.
    if (k_right_remainder > 0 && is_last_rank) {
        printf("rank %i summing last buffer entries %zu-%zu into buffer index %zu\n", rank, accumulation_buffer_offset_pre_k + size - k_right_remainder, accumulation_buffer_offset_pre_k + size, accumulation_buffer_offset_post_k + target_idx);
        accumulation_buffer[accumulation_buffer_offset_post_k + target_idx++] = std::accumulate(
                                                            &accumulation_buffer[accumulation_buffer_offset_pre_k + size - k_right_remainder],
                                                            &accumulation_buffer[accumulation_buffer_offset_pre_k + size],
                                                            0.0);
    }

    // Sum received values from left remainder
    if (has_left_remainder) {
        left_remainder_running_index = 0;
        MPI_Wait(&k_recv_reqs[0], nullptr);

        // TODO: if possible, join this loop with the MPI_Irecv loop above
        for (int i = 1U; i < k_predecessor_ranks.size(); ++i) {
            const auto other_rank = k_predecessor_ranks[i];
            const auto elements_to_sum = regions[other_rank].size; // We receive all numbers the other rank holds
            MPI_Wait(&k_recv_reqs[i], nullptr);

            left_remainder_accumulator = std::accumulate(&accumulation_buffer[left_remainder_running_index],
                                                        &accumulation_buffer[left_remainder_running_index + elements_to_sum],
                                                        left_remainder_accumulator);
            left_remainder_running_index += elements_to_sum;
        }

        // Accumulate local part of the left remainder
        left_remainder_accumulator = std::accumulate(&accumulation_buffer[accumulation_buffer_offset_pre_k],
                                                     &accumulation_buffer[accumulation_buffer_offset_pre_k + k_left_remainder],
                                                     left_remainder_accumulator);



        accumulation_buffer[accumulation_buffer_offset_post_k] = left_remainder_accumulator;
    }



    printf("rank %i asserting target_idx = %zu == %zu = k_size\n", rank, target_idx, k_size);
    assert(target_idx == k_size);

    printf("rank %i has k_reduced elements ", rank);
    for (auto i = 0U; i < k_size; ++i) {
        printf("%f ", accumulation_buffer[accumulation_buffer_offset_post_k + i]);
    }
    printf("\n");
}

/* Sum all numbers. Will return the total sum on rank 0
    */
double BinaryTreeSummation::accumulate(void) {
    linear_sum_k();
    for (auto summand : rank_intersecting_summands) {
        if (subtree_size(summand) > 16) {
            // If we are about to do some considerable amount of work, make sure
            // the send buffer is empty so noone is waiting for our results
            message_buffer.flush();
        }

        double result = accumulate(summand);

        message_buffer.put(rankFromIndexMap(parent(summand)), summand, result);
    }
    message_buffer.flush();
    message_buffer.wait();

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
        return accumulation_buffer[accumulation_buffer_offset_post_k + index - k_begin];
    }

    const uint64_t maxX = (index == 0) ? globalSize - 1
        : min(globalSize - 1, index + subtree_size(index) - 1);
    const int maxY = (index == 0) ? ceil(log2(globalSize)) : log2(subtree_size(index));

    const uint64_t largest_local_index = min(maxX, k_end - 1);
    const uint64_t n_local_elements = largest_local_index + 1 - index;

    uint64_t elementsInBuffer = n_local_elements;

    double *destinationBuffer = static_cast<double *>(&accumulation_buffer[accumulation_buffer_offset_post_k + index - k_begin]);
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
    return std::chrono::duration_cast<std::chrono::nanoseconds> (acquisition_duration).count();
}

const uint64_t BinaryTreeSummation::largest_child_index(const uint64_t index) const {
    return index | (index - 1);
}

const uint64_t BinaryTreeSummation::subtree_size(const uint64_t index) const {
    assert(index != 0);
    return largest_child_index(index) + 1 - index;
}

const void BinaryTreeSummation::printStats() const {
    message_buffer.printStats();
}

const int BinaryTreeSummation::get_rank() const {
    return rank;
}

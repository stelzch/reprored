#include <mpi.h>
#include <numeric>
#include <cstring>
#include <cassert>
#include <cmath>
#include <unistd.h>
#include <chrono>
#include <io.hpp>
#include "binary_tree_summation.hpp"
#include "k_chunked_array.hpp"
#include <util.hpp>

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
      KChunkedArray(rank, regions, k),
      comm(comm),
      k_recv_reqs(k_predecessor_ranks.size()),
      accumulation_buffer_offset_pre_k(k - k_left_remainder),
      accumulation_buffer_offset_post_k(k),
      accumulation_buffer(k + size + k_right_remainder), // TODO: this memory allocation is non-optimal on ranks
                                                         // with no left remainder. Reduce it to an appropriate value.
      acquisition_duration(std::chrono::duration<double>::zero()),
      acquisition_count(0L),
      reduction_counter(0UL),
      message_buffer(comm)
{
    assert(globalSize > 0);
    assert(k > 0);
    assert(k_left_remainder < k);
    assert(k_right_remainder < k);
    assert(implicates(no_k_intercept, size < k));


    // Verify that the regions are actually correct.
    // This is given if the difference to the next start index is equal to the region size
    for (auto it = k_start_indices.begin(); it != k_start_indices.end(); ++it) {
        auto next = std::next(it);
        if (next == k_start_indices.end()) break;

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

void BinaryTreeSummation::linear_sum_k() {
    MPI_Request send_req = MPI_REQUEST_NULL;

    if (k_right_remainder && !left_neighbor_has_different_successor) {
        // We do not reduce any summands on our own, we simply pass them to the successor
        assert(k_successor_rank >= 0);
        assert(size == k_right_remainder);

        MPI_Isend(&accumulation_buffer[accumulation_buffer_offset_pre_k], size, MPI_DOUBLE, k_successor_rank, MESSAGEBUFFER_MPI_TAG, comm, &send_req);
        return; // We are done here

    } else if (k_right_remainder > 0 && !is_last_rank) {
        // Sum & send right remainder
        assert(k_successor_rank >= 0);
        double acc = std::accumulate(&accumulation_buffer[accumulation_buffer_offset_pre_k + size - k_right_remainder], &accumulation_buffer[accumulation_buffer_offset_pre_k + size], 0.0);
        MPI_Isend(&acc, 1, MPI_DOUBLE, k_successor_rank, MESSAGEBUFFER_MPI_TAG, comm, &send_req);
    }

    // Start receive requests for the left remainder
    // TODO: ask someone with more MPI experience if this is really necessary
    // or if the sent values will be cached on the destination machine
    // regardless
    double left_remainder_accumulator;
    uint64_t left_remainder_running_index = 0;

    for (int i = 0U; i < k_predecessor_ranks.size(); ++i) {
        const auto other_rank = k_predecessor_ranks[i];
        if (i == 0) {
            assert((k_regions[other_rank].size > 0) || (other_rank == start_indices.begin()->second) || (regions.at(other_rank).globalStartIndex % k == 0));
            MPI_Irecv(&left_remainder_accumulator, 1, MPI_DOUBLE, other_rank, MESSAGEBUFFER_MPI_TAG, comm, &k_recv_reqs[i]);
        } else {
            assert(k_regions[other_rank].size == 0);
            const auto elements_to_receive = regions.at(other_rank).size; // We receive all numbers the other rank holds

            MPI_Irecv(&accumulation_buffer[left_remainder_running_index], elements_to_receive, MPI_DOUBLE, other_rank, MESSAGEBUFFER_MPI_TAG, comm, &k_recv_reqs[i]);
            left_remainder_running_index += elements_to_receive;
            assert(left_remainder_running_index < k);
        }
    }

    // Sum local k-tuples that do not overlap with PE-boundaries
    const bool has_left_remainder = (k_left_remainder > 0);
    uint64_t target_idx = has_left_remainder ? 1U : 0U;
    for (uint64_t i = k_left_remainder; i + k - 1 < size; i += k) {
        accumulation_buffer[accumulation_buffer_offset_post_k + target_idx++] = std::accumulate(&accumulation_buffer[accumulation_buffer_offset_pre_k + i], &accumulation_buffer[accumulation_buffer_offset_pre_k + i + k], 0.0);
    }

    // On the last rank manually sum right remainder since it can not be sent anywhere.
    if (k_right_remainder > 0 && is_last_rank) {
        accumulation_buffer[accumulation_buffer_offset_post_k + target_idx++] = std::accumulate(
                                                            &accumulation_buffer[accumulation_buffer_offset_pre_k + size - k_right_remainder],
                                                            &accumulation_buffer[accumulation_buffer_offset_pre_k + size],
                                                            0.0);
    }

    // Make sure the send request has gone through before waiting on received messages
    if (send_req != MPI_REQUEST_NULL) {
        MPI_Wait(&send_req, nullptr);
    }

    // Sum received values from left remainder
    if (has_left_remainder) {
        left_remainder_running_index = 0;
        MPI_Wait(&k_recv_reqs[0], nullptr);

        // TODO: if possible, join this loop with the MPI_Irecv loop above
        for (int i = 1U; i < k_predecessor_ranks.size(); ++i) {
            const auto other_rank = k_predecessor_ranks[i];
            const auto elements_to_sum = regions.at(other_rank).size; // We receive all numbers the other rank holds
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



    assert(target_idx == k_size);
}

/* Sum all numbers. Will return the total sum on rank 0
    */
double BinaryTreeSummation::accumulate(void) {
    if (k != 1 && size > 0) {
        linear_sum_k();
    }

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


const void BinaryTreeSummation::printStats() const {
    message_buffer.printStats();
}

const int BinaryTreeSummation::get_rank() const {
    return rank;
}

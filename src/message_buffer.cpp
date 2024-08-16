#include <cassert>
#include <cstdio>
#include <message_buffer.hpp>

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

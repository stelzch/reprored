#pragma once

#include <cstdint>
#include <mpi.h>
#include <array>
#include <vector>
#include <map>

using std::array;
using std::vector;
using std::map;

const uint8_t MAX_MESSAGE_LENGTH = 4;
const int MESSAGEBUFFER_MPI_TAG = 1;

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

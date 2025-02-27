#pragma once

enum class ReduceType {
    REDUCE,
    REDUCE_BCAST,
    ALLREDUCE
};

class Summation {
public:
  virtual double *getBuffer() = 0;
  virtual double accumulate() = 0;
  virtual ~Summation();
};

#pragma once

class Summation {
public:
  virtual double *getBuffer() = 0;
  virtual double accumulate() = 0;
  virtual ~Summation();
};

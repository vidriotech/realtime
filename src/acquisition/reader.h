#ifndef RTS_2_SRC_ACQUISITION_READER_H_
#define RTS_2_SRC_ACQUISITION_READER_H_

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>

#include "../probe/probe.h"

template<class T>
class Reader {
 public:
  explicit Reader(Probe &probe)
      : probe_(probe) {};
  virtual unsigned AcquireFrames(unsigned long frame_offset, int n_frames, T *buf) = 0;

 protected:
  Probe probe_;

  virtual void Open() = 0;
  virtual void Close() = 0;
};

#endif //RTS_2_SRC_ACQUISITION_READER_H_

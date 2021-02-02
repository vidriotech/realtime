#ifndef RTS_2_SRC_ACQUISITION_READER_H_
#define RTS_2_SRC_ACQUISITION_READER_H_

#include <iostream>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>

#include "../probe/probe.h"

template<class T>
class Reader {
 public:
  explicit Reader(Probe &probe)
      : probe_(probe) {};
  virtual uint32_t AcquireFrames(uint64_t frame_offset, uint32_t n_frames,
                                 T *buf) = 0;

  virtual void Open() = 0;
  virtual void Close() = 0;

 protected:
  Probe probe_;
};

#endif //RTS_2_SRC_ACQUISITION_READER_H_

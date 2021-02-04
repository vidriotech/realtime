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

  [[maybe_unused]] virtual uint32_t
  AcquireFrames(std::shared_ptr<T[]> buf, uint64_t frame_offset,
                uint32_t n_frames) = 0;

  [[maybe_unused]] virtual void Open() = 0;
  [[maybe_unused]] virtual void Close() = 0;

 protected:
  Probe probe_;
};

#endif //RTS_2_SRC_ACQUISITION_READER_H_

#ifndef RTS_2_SRC_ACQUISITION_SOCKET_READER_H_
#define RTS_2_SRC_ACQUISITION_SOCKET_READER_H_

#include "reader.h"

template<class T>
class SocketReader : public Reader<T> {
 public:
  explicit SocketReader(Probe &probe)
      : Reader<T>(probe) {};
  unsigned
  AcquireFrames(uint32_t frame_offset, int n_frames,
                T *buf);

 protected:
  void Open();
  void Close();
};

#endif //RTS_2_SRC_ACQUISITION_SOCKET_READER_H_

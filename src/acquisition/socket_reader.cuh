#ifndef RTS_2_SRC_ACQUISITION_SOCKET_READER_H_
#define RTS_2_SRC_ACQUISITION_SOCKET_READER_H_

#include "reader.cuh"

template<class T>
class SocketReader : public Reader<T> {
 public:
  explicit SocketReader(Probe &probe)
      : Reader<T>(probe) {};
  uint32_t
  AcquireFrames(std::vector<T> &buf, uint64_t frame_offset, uint32_t n_frames);

 protected:
  void Open();
  void Close();
};

#endif //RTS_2_SRC_ACQUISITION_SOCKET_READER_H_

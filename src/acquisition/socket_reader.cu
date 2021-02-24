#include "socket_reader.cuh"

template<class T>
uint32_t
SocketReader<T>::AcquireFrames(std::vector<T> &buf,
                               uint64_t frame_offset,
                               uint32_t n_frames) {
  return 0;
};

template<class T>
void SocketReader<T>::Open() {};

template<class T>
void SocketReader<T>::Close() {};

template
class SocketReader<short>;

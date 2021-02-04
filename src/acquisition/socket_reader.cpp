#include "socket_reader.h"

template<class T>
unsigned
SocketReader<T>::AcquireFrames(T *buf, uint32_t frame_offset, int n_frames) {
  return 0;
};

template<class T>
void SocketReader<T>::Open() {};

template<class T>
void SocketReader<T>::Close() {};

template class SocketReader<short>;

#include "socket_reader.h"

template<class T>
unsigned
SocketReader<T>::AcquireFrames(uint32_t frame_offset, int n_frames,
                               T *buf) { return 0; };

template<class T>
void SocketReader<T>::Open() {};

template<class T>
void SocketReader<T>::Close() {};

template class SocketReader<short>;

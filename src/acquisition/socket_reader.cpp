#include "socket_reader.h"

template<class T>
unsigned SocketReader<T>::AcquireFrames(unsigned long frame_offset, int n_frames, T *buf)
{};

template<class T>
void SocketReader<T>::Open() {};

template<class T>
void SocketReader<T>::Close() {};

template class SocketReader<short>;

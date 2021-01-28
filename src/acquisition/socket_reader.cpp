#include "socket_reader.h"

template<class T>
void SocketReader<T>::AcquireFrames(int frame_offset, int n_frames, T *buf) {};

template<class T>
void SocketReader<T>::Open() {};

template<class T>
void SocketReader<T>::Close() {};

template class SocketReader<short>;

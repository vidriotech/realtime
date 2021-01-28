#include "pipeline.h"
#include "../acquisition/file_reader.h"
#include "../acquisition/socket_reader.h"

template class Pipeline<FileReader<short>>;
template class Pipeline<SocketReader<short>>;


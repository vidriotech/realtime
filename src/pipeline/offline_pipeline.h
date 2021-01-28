#ifndef RTS_2_SRC_PIPELINE_OFFLINE_PIPELINE_H_
#define RTS_2_SRC_PIPELINE_OFFLINE_PIPELINE_H_

#include "pipeline.h"
#include "../acquisition/file_reader.h"

template<class T>
class OfflinePipeline : public Pipeline<T> {
 public:
  OfflinePipeline(Params &params, Probe &probe)
      : Pipeline<T>(params, probe), reader_(probe) {};

  void Run();

  // setters
  void set_filename(std::string filename) { reader_.set_filename(filename); };

 private:
  FileReader<T> reader_;
};

#endif //RTS_2_SRC_PIPELINE_OFFLINE_PIPELINE_H_

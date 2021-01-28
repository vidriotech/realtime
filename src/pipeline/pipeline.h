#ifndef RTS_2_SRC_PIPELINE_H_
#define RTS_2_SRC_PIPELINE_H_

#include "../params/params.h"
#include "../acquisition/reader.h"

template<class R>
class Pipeline {
 public:
  explicit Pipeline(Params &params, Probe &probe)
      : params_(params), reader_(probe) {};

 private:
  Params params_;
  R reader_;
};

#endif //RTS_2_SRC_PIPELINE_H_

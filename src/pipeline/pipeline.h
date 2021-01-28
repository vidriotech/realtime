#ifndef RTS_2_SRC_PIPELINE_H_
#define RTS_2_SRC_PIPELINE_H_

#include "../params/params.h"

template<class T>
class Pipeline {
 public:
  explicit Pipeline(Params &params);

 private:
  Params params_;
};

#endif //RTS_2_SRC_PIPELINE_H_

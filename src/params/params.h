#ifndef RTS_2_SRC_PARAMS_PARAMS_H_
#define RTS_2_SRC_PARAMS_PARAMS_H_

#include "acquire_params.h"
#include "threshold_params.h"

class Params {
 public:
  AcquireParams acquire;
  ThresholdParams threshold;
};

#endif //RTS_2_SRC_PARAMS_PARAMS_H_

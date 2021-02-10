#ifndef RTS_2_SRC_PARAMS_PARAMS_H_
#define RTS_2_SRC_PARAMS_PARAMS_H_

#include "acquire_params.h"
#include "device_params.h"
#include "detect_params.h"

class Params {
 public:
  AcquireParams acquire;
  DeviceParams device;
  DetectParams detect;
};

#endif //RTS_2_SRC_PARAMS_PARAMS_H_

#ifndef RTS_2_SRC_PARAMS_PARAMS_H_
#define RTS_2_SRC_PARAMS_PARAMS_H_

#include "acquire_params.h"
#include "device_params.h"
#include "detect_params.h"
#include "extract_params.h"

class Params {
 public:
  AcquireParams acquire;
  DeviceParams device;
  DetectParams detect;
  ExtractParams extract;
};

#endif //RTS_2_SRC_PARAMS_PARAMS_H_

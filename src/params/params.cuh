#ifndef RTS_2_SRC_PARAMS_PARAMS_H_
#define RTS_2_SRC_PARAMS_PARAMS_H_

#include "acquire_params.cuh"
#include "device_params.cuh"
#include "detect_params.cuh"
#include "extract_params.cuh"
#include "classify_params.cuh"

class Params {
 public:
  AcquireParams acquire;
  DeviceParams device;
  DetectParams detect;
  ExtractParams extract;
  ClassifyParams classify;
};

#endif //RTS_2_SRC_PARAMS_PARAMS_H_

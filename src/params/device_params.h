#ifndef RTS_2_SRC_PARAMS_DEVICE_PARAMS_H_
#define RTS_2_SRC_PARAMS_DEVICE_PARAMS_H_

class DeviceParams {
 public:
  bool gpu_enabled = true;
  uint32_t n_threads = 1024;

  uint32_t n_blocks(uint32_t sz) { return (sz + n_threads - 1) / n_threads; }
};

#endif //RTS_2_SRC_PARAMS_DEVICE_PARAMS_H_

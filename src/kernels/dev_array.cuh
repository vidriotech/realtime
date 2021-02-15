#ifndef RTS_2_SRC_KERNELS_DEV_ARRAY_CUH_
#define RTS_2_SRC_KERNELS_DEV_ARRAY_CUH_

#include <algorithm>
#include <stdexcept>
#include <cuda_runtime.h>

template<class T>
class dev_array {
 public:
  explicit dev_array()
      : start_(nullptr), stop_(nullptr) {};
  explicit dev_array(size_t sz) { alloc(sz); };
  ~dev_array() { release(); };

  size_t Set(const T *data, size_t sz);
  size_t Get(T *data, size_t sz) const;
  void Resize(size_t sz);

  // getters
  const T *data() const { return start_; };
  T *data() { return start_; };
  size_t size() const { return stop_ - start_; }

 private:
  void alloc(size_t sz);
  void release();

  T *start_;
  T *stop_;
};
#endif //RTS_2_SRC_KERNELS_DEV_ARRAY_CUH_

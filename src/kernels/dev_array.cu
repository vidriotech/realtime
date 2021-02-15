#include "dev_array.cuh"

/**
 * @brief Copy host data to device memory.
 * @param data Host memory block to copy.
 * @param sz Size of host memory block.
 * @return Number of elements copied to device.
 */
template<class T>
size_t dev_array<T>::Set(const T *data, size_t sz) {
  sz = std::min(sz, size());

  auto res = cudaMemcpy(start_, data, sz * sizeof(T), cudaMemcpyHostToDevice);
  if ( res != cudaSuccess) {
    throw std::runtime_error("failed to copy to device memory");
  }

  return sz;
}

/**
 * @brief Copy device data to host memory.
 * @param data Device memory block to copy.
 * @param sz Size of device memory block.
 * @return Number of elements copied to host.
 */
template<class T>
size_t dev_array<T>::Get(T *data, size_t sz) const {
  sz = std::min(sz, size());

  auto res = cudaMemcpy(data, start_, sz * sizeof(T), cudaMemcpyDeviceToHost);
  if ( res != cudaSuccess) {
    throw std::runtime_error("failed to copy to host memory");
  }

  return sz;
}

/**
 * @brief Resize the device array.
 * @param sz Number of elements desired in new device array.
 */
template<class T>
void dev_array<T>::Resize(size_t sz) {
  if (start_ == nullptr) {
    alloc(sz);
    return;
  } else if (size() == sz) {
    return;
  }

  release();
  alloc(sz);
}

/**
 * @brief Allocate device array.
 * @param sz Number of elements desired in new device array.
 */
template<class T>
void dev_array<T>::alloc(size_t sz) {
  if (cudaMalloc(&start_, sz * sizeof(T)) != cudaSuccess) {
    start_ = stop_ = nullptr;
    throw std::runtime_error("failed to allocate device memory");
  }

  stop_ = start_ + sz;
}

/**
 * @brief Free device array.
 */
template<class T>
void dev_array<T>::release() {
  if (start_ == nullptr) {
    return;
  }

  cudaFree(start_);
  start_ = stop_ = nullptr;
}

template
class dev_array<short>;

template
class dev_array<float>;
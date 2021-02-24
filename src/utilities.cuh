#ifndef RTS_2_SRC_UTILITIES_H_
#define RTS_2_SRC_UTILITIES_H_

#include <algorithm>
#include <cstring>
#include <map>
#include <vector>

#include <thrust/sort.h>

#include "kernels/operators.cuh"

namespace utilities {

float median(thrust::device_vector<float> &data, bool is_sorted);

template<class T>
std::vector<std::vector<T>> part_nearby(std::vector<T> &arr, uint64_t thresh);

template<class T>
long argmax(std::vector<T> vec);

template<class T>
std::vector<uint64_t> argsort(const std::vector<T> &vec);
}

#endif //RTS_2_SRC_UTILITIES_H_

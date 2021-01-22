#ifndef RTS_2_SRC_UTILITIES_H_
#define RTS_2_SRC_UTILITIES_H_

#include <cstring>
#include <algorithm>
#include <vector>

namespace utilities {

template<class T>
double median(std::vector<T> data, bool is_sorted) {
  if (data.size() == 0) {
    return 0;
  }

  if (!is_sorted) {
    std::sort(data.begin(), data.end());
  }

  auto n = data.size();
  double med;

  if (n % 2 == 0) {
    med = (data[n / 2 - 1] + data[n / 2]) / 2.0;
  } else {
    med = data[n / 2];
  }

  return med;
}
}

#endif //RTS_2_SRC_UTILITIES_H_

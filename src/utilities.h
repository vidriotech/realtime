#ifndef RTS_2_SRC_UTILITIES_H_
#define RTS_2_SRC_UTILITIES_H_

#include <algorithm>
#include <cstring>
#include <map>
#include <vector>

namespace utilities {

template<class T>
double median(std::vector<T> &data, bool is_sorted) {
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

template<class T>
std::vector<std::vector<T>> part_nearby(std::vector<T> &arr, uint64_t thresh) {
  std::map<uint64_t, std::vector<uint64_t>> collections;
  collections[arr.at(0)] = std::vector<uint64_t>();
  collections[arr.at(0)].push_back(arr.at(0));

  auto i = 0;
  while (i < arr.size()) {
    auto prev_i = i;

    for (auto j = i + 1; j < arr.size(); ++j) {
      if (arr.at(j) - arr.at(j - 1) < thresh) {
        collections[arr.at(i)].push_back(arr.at(j));
      } else {
        i = j;
        collections[arr.at(i)] = std::vector<uint64_t>();
        collections[arr.at(i)].push_back(arr.at(i));
      }
    }

    if (i == prev_i) {
      ++i;
      if (i < arr.size() - 1) {
        collections[arr.at(i)] = std::vector<uint64_t>();
        collections[arr.at(i)].push_back(arr.at(i));
      }
    }
  }

  std::vector<std::vector<T>> values;
  for (auto it = collections.begin(); it != collections.end(); ++it) {
    values.push_back(it->second);
  }

  return values;
}

template<class T>
long argmax(std::vector<T> vec) {
  return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

template<class T>
std::vector<uint64_t> argsort(const std::vector<T> &vec) {
  std::vector<uint64_t> as(vec.size());
  for (auto i = 0; i < vec.size(); ++i)
    as.at(i) = i;

  std::sort(as.begin(), as.end(), [&](uint64_t i, uint64_t j) {
    return vec.at(i) <= vec.at(j);
  });

  return as;
}
}

#endif //RTS_2_SRC_UTILITIES_H_

#include "utilities.cuh"

float utilities::median(thrust::device_vector<float> &data, bool is_sorted) {
  if (data.size() == 0) {
    return std::numeric_limits<float>::infinity();
  }

  float med;
  auto n = data.size();

  if (!is_sorted) {
    thrust::sort(data.begin(), data.end());
  }

  if (n % 2 == 0) {
    med = (data[n / 2 - 1] + data[n / 2]) / 2.0;
  } else {
    med = data[n / 2];
  }

  return med;
}

template<class T>
std::vector<std::vector<T>> utilities::part_nearby(std::vector<T> &arr,
                                                   uint64_t thresh) {
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

template
std::vector<std::vector<uint64_t>>
utilities::part_nearby(std::vector<uint64_t> &arr, uint64_t thresh);


template<class T>
long utilities::argmax(std::vector<T> vec) {
  return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

template
long utilities::argmax(std::vector<uint64_t> vec);

template<class T>
std::vector<uint64_t> utilities::argsort(const std::vector<T> &vec) {
  std::vector<uint64_t> as(vec.size());
  for (auto i = 0; i < vec.size(); ++i)
    as.at(i) = i;

  std::sort(as.begin(), as.end(), [&](uint64_t i, uint64_t j) {
    return vec.at(i) <= vec.at(j);
  });

  return as;
}

template
std::vector<uint64_t> utilities::argsort(const std::vector<uint32_t> &vec);
template
std::vector<uint64_t> utilities::argsort(const std::vector<uint64_t> &vec);
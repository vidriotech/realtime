#include "gtest/gtest.h"

#include <chrono>
#include <random>
#include <vector>

#include "../src/utilities.cuh"

TEST(UtilitiesTest, Argsort) {
  std::vector<uint64_t> vec(100);

  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(-1000,1000);
  for (auto & i : vec) {
    i = distribution(generator);
  }

  auto as = utilities::argsort(vec);
  std::vector<uint64_t> sorted_vec(vec);

  for (auto i = 0; i < sorted_vec.size(); ++i) {
    sorted_vec.at(i) = vec.at(as.at(i));
  }

  EXPECT_TRUE(std::is_sorted(sorted_vec.begin(), sorted_vec.end()));
}
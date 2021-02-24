#include "gtest/gtest.h"

#include <exception>
#include "../src/structures/distance_matrix.cuh"

TEST(DistanceMatrixTest, TestDiagonalAlwaysZero) {
  const auto N = 1000;
  DistanceMatrix<double> dm(N);

  for (int i = 0; i < N; i++)
    EXPECT_EQ(0.0, dm.at(i, i));
}

TEST(DistanceMatrixTest, TestTransposeEntriesAsExpected) {
  DistanceMatrix<float> dm(2);

  dm.set_at(0, 1, 1.0);
  EXPECT_EQ(1.0, dm.at(0, 1));
  EXPECT_EQ(1.0, dm.at(1, 0));
}

TEST(DistanceMatrixTest, TestSetAllGetAll) {
  const auto N = 1000;
  auto k = 0;

  DistanceMatrix<float> dm(N);
  for (auto i = 0; i < N; i++) {
    for (auto j = i + 1; j < N; j++) {
      dm.set_at(i, j, k++);
    }
  }

  k = 0;
  for (auto i = 0; i < N; i++) {
    for (auto j = i + 1; j < N; j++) {
      EXPECT_EQ(k, dm.at(i, j));
      EXPECT_EQ(k++, dm.at(j, i));
    }
  }
}

TEST(DistanceMatrixTest, TestSetDiagonalFails) {
  const auto N = 2;

  DistanceMatrix<float> dm(N);
  EXPECT_THROW(dm.set_at(0, 0, 1), std::domain_error);
  EXPECT_THROW(dm.set_at(1, 1, 1), std::domain_error);
}

TEST(DistanceMatrixTest, SetOutOfRangeFails) {
  const auto N = 1;

  DistanceMatrix<float> dm(N);
  EXPECT_THROW(dm.set_at(0, N, 1), std::out_of_range);
}

#include "gtest/gtest.h"

#include <memory>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

TEST(UniquePtrTest, TestCudaMemcpy) {
  auto n = 100000;

  std::unique_ptr<int[]> foo(new int[n]);
  for (auto i = 0; i < n; ++i) {
    foo[i] = i;
  }

  int *bar;
  cudaMallocManaged(&bar, n * sizeof(int));
  cudaMemcpy(bar, foo.get(), n * sizeof(int), cudaMemcpyHostToDevice);

  for (auto i = 0; i < n; ++i) {
    foo[i] = 0;
  }

  // establish preconditions for the test
  for (auto i = 0; i < n; ++i) {
    EXPECT_EQ(0, foo[i]);
  }

  // copy back to unique_ptr
  cudaMemcpy(foo.get(), bar, n * sizeof(int), cudaMemcpyDeviceToHost);

  // establish preconditions for the test
  for (auto i = 0; i < n; ++i) {
    EXPECT_EQ(i, foo[i]);
  }

  cudaFree(bar);
}

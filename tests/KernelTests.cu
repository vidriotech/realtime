#include "gtest/gtest.h"
#include "../kernels/ChanAdd.cuh"

TEST(KernelTestSuite, TestSqAdd) {
    float *x, *y;

    auto N = 1<<20;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));;

    for (int i=0; i<N; i++) {
        x[i] = 1.0;
        y[i] = -1.0;
    }

    int block_size = 256;
    int nblocks = (N + block_size - 1)/block_size;

    sq_add<<<nblocks, block_size>>>(N, x, y);

    cudaDeviceSynchronize();

    for (int i=0; i<N; i++) {
        EXPECT_EQ(2.0, y[i]);
    }

    cudaFree(x);
    cudaFree(y);
}

TEST(KernelTestSuite, TestSqDiff) {
    float *x, *y;

    auto N = 1<<20;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));;

    for (int i=0; i<N; i++) {
        x[i] = 1.0;
        y[i] = -1.0;
    }

    int block_size = 256;
    int nblocks = (N + block_size - 1)/block_size;

    sq_diff<<<nblocks, block_size>>>(N, x, y);

    cudaDeviceSynchronize();

    for (int i=0; i<N; i++) {
        EXPECT_EQ(4.0, y[i]);
    }

    cudaFree(x);
    cudaFree(y);
}
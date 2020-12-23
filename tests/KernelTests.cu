#include "gtest/gtest.h"
#include "../kernels/ChanAdd.cuh"
#include "../kernels/Filters.cuh"

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

TEST(KernelTestSuite, TestNdiff2I16) {
    auto nchans = 64;

    short *data, *filtered;
    cudaMallocManaged(&data, 4 * nchans * sizeof(short));
    cudaMallocManaged(&filtered, 4 * nchans * sizeof(short));

    /*
     * channel values: 1 1 2 2 -> (-1 * 1) + (-2 * 1) + (2 * 2) + (1 * 2) = 3
     */
    for (auto i = 0; i < 4 * nchans; i++) {
        if (i < 2 * nchans) {
            data[i] = 1;
        } else {
            data[i] = 2;
        }

        filtered[i] = 0;
    }

    auto nthreads = 256;
    auto nblocks = (4 * nchans + nthreads - 1) / nthreads;

    ndiff2_i16<<<nblocks, nthreads>>>(4 * nchans, data, filtered, nchans);
    cudaDeviceSynchronize();

    /*
     * filtered values at indices 0, 2, and 3 get 0, while channel value at index 1 gets 3
     */
    for (auto i = 0; i < 4 * nchans; i++) {
        EXPECT_EQ((i >= nchans && i < 2 * nchans) ? 3 : 0, filtered[i]);
    }

    // cleanup
    cudaFree(data);
    cudaFree(filtered);
}

TEST(KernelTestSuite, TestNdiff2F32) {
    auto nchans = 64;

    float *data, *filtered;
    cudaMallocManaged(&data, 4 * nchans * sizeof(float));
    cudaMallocManaged(&filtered, 4 * nchans * sizeof(float));

    /*
     * channel values: 1 1 2 2 -> (-1 * 1) + (-2 * 1) + (2 * 2) + (1 * 2) = 3
     */
    for (auto i = 0; i < 4 * nchans; i++) {
        if (i < 2 * nchans) {
            data[i] = 1.0f;
        } else {
            data[i] = 2.0f;
        }

        filtered[i] = 0;
    }

    auto nthreads = 256;
    auto nblocks = (4 * nchans + nthreads - 1) / nthreads;

    ndiff2_f32<<<nblocks, nthreads>>>(4 * nchans, data, filtered, nchans);
    cudaDeviceSynchronize();

    /*
     * filtered values at indices 0, 2, and 3 get 0, while channel value at index 1 gets 3
     */
    for (auto i = 0; i < 4 * nchans; i++) {
        EXPECT_EQ((i >= nchans && i < 2 * nchans) ? 3.0f : 0.0f, filtered[i]);
    }

    // cleanup
    cudaFree(data);
    cudaFree(filtered);
}
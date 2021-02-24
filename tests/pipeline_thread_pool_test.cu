#include "gtest/gtest.h"
#include "./test_utilities/test_utilities.cuh"

#include "../src/params/params.cuh"
#include "../src/probe/probe.cuh"
#include "../src/pipeline/pipeline_thread_pool.cuh"

/*
 * GIVEN
 */
TEST(PipelineThreadPoolTest, InitialState) {
  Params params;
  Probe probe = probe_from_env();
  PipelineThreadPool<short> pool(params, probe, 8);
}

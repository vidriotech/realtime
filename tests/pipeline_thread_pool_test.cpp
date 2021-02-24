#include "gtest/gtest.h"
#include "./test_utilities/test_utilities.h"

#include "../src/params/params.h"
#include "../src/probe/probe.h"
#include "../src/pipeline/pipeline_thread_pool.cuh"

/*
 * GIVEN
 */
TEST(PipelineThreadPoolTest, InitialState) {
  Params params;
  Probe probe = probe_from_env();
  PipelineThreadPool<short> pool(params, probe, 8);
}

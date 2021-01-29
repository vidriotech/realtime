#include "gtest/gtest.h"
#include "./test_utilities/test_utilities.h"

#include <cmath>

#include "../src/params/params.h"
#include "../src/probe/probe.h"
#include "../src/acquisition/file_reader.h"
#include "../src/pipeline/offline_pipeline.h"

/*
 * GIVEN a Params `params` and a Probe `probe`
 * DO construct a Pipeline `pipeline` of FileReader AND
 * TEST THAT the number of samples in the buffer of `pipeline` is as expected.
 */
TEST(OfflinePipelineTest, InitialState) {
  Params params;
  Probe probe = probe_from_env();
  OfflinePipeline<short> pipeline(params, probe);

  EXPECT_EQ(std::ceil(probe.sample_rate() * params.acquire.n_seconds),
            pipeline.n_frames());
}

TEST(OfflinePiplineTest, Run) {
  Params params;
  Probe probe = probe_from_env();
  OfflinePipeline<short> pipeline(params, probe);

  pipeline.Run();
}
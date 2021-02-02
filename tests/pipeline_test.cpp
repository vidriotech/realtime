#include "gtest/gtest.h"
#include "./test_utilities/test_utilities.h"

#include <cmath>

#include "../src/params/params.h"
#include "../src/probe/probe.h"
#include "../src/acquisition/file_reader.h"
#include "../src/pipeline/pipeline.h"

/*
 * GIVEN a Params `params` and a Probe `probe`
 * DO construct a Pipeline `pipeline` of FileReader AND
 * TEST THAT the number of samples in the buffer of `pipeline` is as expected.
 */
TEST(PipelineTest, InitialState) {
  Params params;
  Probe probe = probe_from_env();
  Pipeline<short> pipeline(params, probe);

  EXPECT_EQ(std::ceil(probe.sample_rate() * params.acquire.n_seconds),
            pipeline.n_frames_buf());
}

TEST(PipelineTest, Process) {
  Params params;
  auto probe = probe_from_env();
  auto filename = get_env_var("TEST_FILE");

  FileReader<short> reader(filename, probe);
  auto n_frames = (uint32_t) std::ceil(probe.sample_rate());
  std::shared_ptr<short[]> buffer(new short[n_frames * probe.n_total()]);
  reader.AcquireFrames(0, n_frames, buffer.get());

  Pipeline<short> pipeline(params, probe);
}
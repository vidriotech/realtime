#include "gtest/gtest.h"
#include "./test_utilities/test_utilities.h"

#include <cmath>

#include "../src/params/params.h"
#include "../src/acquisition/file_reader.h"
#include "../src/pipeline/pipeline.h"

/*
 * GIVEN a Params `params` and a Probe `probe`
 * DO construct a Pipeline `pipeline` of FileReader AND
 * TEST THAT the frame offset is 0; AND
 *           the number of samples in the data of `pipeline` is reported as
 *           equal to the the ceiling of the number of seconds as specified in
 *           `params` times the number of frames per second (the sample rate).
 */
TEST(PipelineTest, InitialState) {
  Params params;
  Probe probe = probe_from_env();
  Pipeline<short> pipeline(params, probe);

  EXPECT_EQ(0, pipeline.buffer().size());
  EXPECT_EQ(0, pipeline.frame_offset());
  EXPECT_EQ(0, pipeline.n_frames_buf());
}

/*
 * GIVEN a Pipeline `pipeline`, a buffer size `buffer_size`, and a frame offset
 *           `frame_offset_`
 * DO update the pipeline with an uninitialized shared_ptr AND
 * TEST THAT the buffer returned is a nullptr; AND
 *           the frame offset is reported correctly; AND
 *           the data size is reported as 0.
 */
TEST(PipelineTest, UpdateNullptr) {
  Params params;
  Probe probe = probe_from_env();
  Pipeline<short> pipeline(params, probe);

  // perform the update
  std::vector<short> buf;
  uint64_t frame_offset = 4294967296;
  pipeline.Update(buf, frame_offset);

  EXPECT_EQ(0, pipeline.buffer().size());
  EXPECT_EQ(frame_offset, pipeline.frame_offset());
  EXPECT_EQ(0, pipeline.n_frames_buf());
}

TEST(PipelineTest, Process) {
  Params params;
  auto probe = probe_from_env();
  auto filename = get_env_var("TEST_FILE");

  FileReader<short> reader(filename, probe);
  auto frame_offset = 0;
  auto n_frames = (uint32_t) std::ceil(probe.sample_rate());

  std::vector<short> buffer(n_frames * probe.n_total());
  reader.AcquireFrames(buffer, frame_offset, n_frames);

  Pipeline<short> pipeline(params, probe);
  pipeline.Update(buffer, frame_offset);
}

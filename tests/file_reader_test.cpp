#include "gtest/gtest.h"

#include <string>

#include "../src/acquisition/file_reader.h"
#include "./test_utilities/test_utilities.h"

template<class T>
FileReader<T> make_file_reader() {
  auto filename = get_env_var("TEST_FILE");
  auto probe = probe_from_env();

  return FileReader<T>(filename, probe);
}

/*
 * GIVEN a file name `filename` and Probe `probe_`
 * DO construct a FileReader AND
 * TEST THAT the filename getter returns the correct filename; AND
 *           the correct number of frames is computed and returned.
 */
TEST(FileReaderTest, InitialState) {
  auto filename = get_env_var("TEST_FILE");
  auto n_frames = std::stoi(get_env_var("TEST_NFRAMES"));
  auto reader = make_file_reader<short>();

  EXPECT_EQ(filename, reader.filename());
  EXPECT_EQ(n_frames, reader.n_frames());
}

/*
 * GIVEN a FileReader `reader`
 * DO acquire 5 frames' worth of data from the beginning of the file AND
 * TEST THAT the number of frames reported as acquired is 5; AND
 *           the data so acquired is equal to the data as read directly.
 */
TEST(FileReaderTest, AcquireFrames) {
  auto reader = make_file_reader<short>();

  auto n_channels = std::stoi(get_env_var("TEST_NCHANNELS"));
  auto n_frames = std::min(5, (int) reader.n_frames()); // try to use 5 frames
  auto n_samples = n_frames * n_channels;

  std::vector<short> framebuf(n_samples);
  std::shared_ptr<short[]> filebuf(new short[n_samples]);

  std::ifstream fp;
  fp.open(reader.filename());
  fp.read((char *) filebuf.get(), sizeof(short) * n_samples);
  fp.close();

  EXPECT_EQ(n_frames, reader.AcquireFrames(framebuf, 0, n_frames));

  for (auto i = 0; i < n_samples; i++)
    EXPECT_EQ(filebuf[i], framebuf.at(i));
}

/*
* GIVEN a FileReader `reader`
* DO try to acquire 5 frames' worth of data when 4 frames out from the end of
 *   the file AND
* TEST THAT the number of frames reported as acquired is 4; AND
 *          the data so acquired is equal to the data as read directly from
 *          the end of the file.
*/
TEST(FileReaderTest, AcquireFramesEOF) {
  auto reader = make_file_reader<short>();

  auto n_channels = std::stoi(get_env_var("TEST_NCHANNELS"));
  auto n_frames_desired = std::min(5, (int) reader.n_frames());
  auto n_samples_desired = n_frames_desired * n_channels;

  auto n_frames_expected = n_frames_desired - 1;
  auto n_samples_expected = n_frames_expected * n_channels;

  std::vector<short> framebuf(n_samples_desired);
  std::shared_ptr<short[]> filebuf(new short[n_samples_expected]);

  // acquire frames from the end of the file
  std::ifstream fp;
  fp.open(reader.filename());
  fp.seekg(-n_samples_expected * sizeof(short), std::ios::end);
  fp.read((char *) filebuf.get(), sizeof(short) * n_samples_expected);
  fp.close();

  // acquire using the Reader method
  auto frame_offset = reader.n_frames() - n_frames_desired + 1;
  ASSERT_EQ(n_frames_expected,
            reader.AcquireFrames(framebuf, frame_offset, n_frames_desired));

  for (auto i = 0; i < n_samples_expected; i++)
    EXPECT_EQ(filebuf[i], framebuf.at(i));
}
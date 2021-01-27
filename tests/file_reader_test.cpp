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
 * DO acquire 5 frames' worth of data_ from the beginning of the file AND
 * TEST THAT the data_ so acquired is equal to the data_ as read directly.
 */
TEST(FileReaderTest, AcquireFrames) {
  auto reader = make_file_reader<short>();

  auto n_channels = std::stoi(get_env_var("TEST_NCHANNELS"));
  auto n_frames = std::stoi(get_env_var("TEST_NFRAMES"));
  n_frames = std::min(5, n_frames); // try to use 5 frames' worth
  auto n_samples = n_frames * n_channels;

  auto *framebuf = new short[n_samples];
  auto *filebuf = new short[n_samples];

  std::ifstream fp;
  fp.open(reader.filename());
  fp.read((char *) filebuf, sizeof(short) * n_samples);
  fp.close();

  reader.AcquireFrames(0, n_frames, framebuf);

  for (auto i = 0; i < n_samples; i++)
    EXPECT_EQ(filebuf[i], framebuf[i]);

  delete[] framebuf;
  delete[] filebuf;
}

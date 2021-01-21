#include "gtest/gtest.h"

#include <string>

#include "../src/acquisition/file_reader.h"
#include "utilities/TestProbe.h"
#include "utilities/TestException.h"

std::string get_env_var(std::string const &key) {
  char *val = getenv(key.c_str());
  if (val == nullptr) {
    throw TestException(key + " is not defined.");
  }
  return std::string(val);
}

template<class T>
FileReader<T> make_file_reader() {
  auto filename = get_env_var("TEST_FILE");
  auto n_channels = std::stoi(get_env_var("TEST_NCHANNELS"));
  auto n_active = std::stoi(get_env_var("TEST_NACTIVE"));
  auto n_groups = std::stoi(get_env_var("TEST_NGROUPS"));
  auto srate_hz = std::stod(get_env_var("TEST_SRATE_HZ"));

  auto probe = make_probe(n_channels, n_active, n_groups, srate_hz);
  return FileReader<T>(filename, probe);
}

/*
 * GIVEN a file name `filename` and Probe `probe`
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
 * TEST THAT the data so acquired is equal to the data as read directly.
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

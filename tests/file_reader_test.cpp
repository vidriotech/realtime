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

// test that, given an existing file, a FileReader will initialize ok
TEST(FileReaderTest, InitOK) {
  auto reader = make_file_reader<short>();
}

// test that, given a binary file, interpreted as a recording with Probe probe,
// the correct number of frames is reported
TEST(FileReaderTest, NframesOK) {
  auto srate_hz = std::stod(get_env_var("TEST_SRATE_HZ"));
  auto reader = make_file_reader<short>();

  EXPECT_EQ(unsigned (srate_hz *60 * 10), reader.n_frames());
}

// test that, given a binary file, AcquireFrames returns the proper data
TEST(FileReaderTest, AcquireFramesOK) {
  auto reader = make_file_reader<short>();

  auto n_channels = std::stoi(get_env_var("TEST_NCHANNELS"));
  auto *framebuf = new short[n_channels * 5]; // 5 frames' worth
  auto *filebuf = new short[n_channels * 5]; // 5 frames' worth

  std::ifstream fp;
  fp.open(reader.filename());
  fp.read((char *) filebuf, sizeof(short) * n_channels * 5);
  fp.close();

  reader.AcquireFrames(0, 5 * n_channels, framebuf);

  for (auto i = 0; i < 5 * n_channels; i++)
    EXPECT_EQ(filebuf[i], framebuf[i]);

  delete[] framebuf;
  delete[] filebuf;
}

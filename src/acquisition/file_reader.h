#ifndef RTS_2_FILE_READER_H
#define RTS_2_FILE_READER_H

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include "../probe/probe.h"

/**
 * @brief A class for reading data_ from a flat binary file.
 * @tparam T The type of data_ stored in the underlying file.
 */
template<class T>
class FileReader {
 public:
  explicit FileReader(std::string &filename, Probe &probe);
  FileReader(FileReader &other)
      : filename_(other.filename_), probe(other.probe), fsize(other.fsize) {};
  ~FileReader() { Close(); };

  void AcquireFrames(int frame_offset, int n_frames, T *buf);

  // getters
  /**
   * @brief Get the path to the underlying file.
   * @return The path to the underlying file.
   */
  std::string filename() const { return filename_; };
  /**
   * @brief Calculate and return the number of frames in the file.
   * @return The number of complete frames in the file.
   */
  unsigned long n_frames() const;
 private:
  std::string filename_;
  Probe probe;
  std::ifstream fp;

  unsigned long long fsize;

  void Open();
  void Close();
};

#endif //RTS_2_FILE_READER_H

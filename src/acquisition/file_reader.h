#ifndef RTS_2_FILE_READER_H
#define RTS_2_FILE_READER_H

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>

#include "reader.h"

/**
 * @brief A class for reading data_ from a flat binary file.
 * @tparam T The type of data_ stored in the underlying file.
 */
template<class T>
class FileReader : public Reader<T> {
 public:
  explicit FileReader(Probe &probe)
      : Reader<T>(probe), fsize(0) {};
  explicit FileReader(std::string &filename, Probe &probe);
  FileReader(FileReader &other)
      : Reader<T>(other.probe_),
        filename_(other.filename_),
        fsize(other.fsize) {};
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

 protected:
  void Open();
  void Close();

 private:
  std::string filename_;
  std::ifstream fp;

  unsigned long long fsize;
};

#endif //RTS_2_FILE_READER_H

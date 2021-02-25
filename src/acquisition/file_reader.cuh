#ifndef RTS_2_FILE_READER_H
#define RTS_2_FILE_READER_H

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>

#include "reader.cuh"

/**
 * @brief A class for reading samples_ from a flat binary file.
 * @tparam T The type of samples_ stored in the underlying file.
 */
template<class T>
class FileReader : public Reader<T> {
 public:
  FileReader(std::string &filename, Probe &probe);
  FileReader(FileReader &other)
      : Reader<T>(other.probe_),
        filename_(other.filename_),
        file_size_(other.file_size_) {};
  ~FileReader() { Close(); };

  uint32_t
  AcquireFrames(std::vector<T> &buf, uint64_t frame_offset,
                uint32_t n_frames);

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
  uint64_t n_frames() const;

  // setters
  void set_filename(std::string &filename);

  void Open();
  void Close();

 private:
  std::string &filename_;
  std::ifstream fp;
  uint64_t file_size_ = 0;

  void ComputeFileSize();
};

#endif //RTS_2_FILE_READER_H

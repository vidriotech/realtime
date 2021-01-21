//
// Created by alan on 12/21/20.
//

#ifndef RTS_2_FILE_READER_H
#define RTS_2_FILE_READER_H

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include "../probe.h"

/**
 * @brief A class for reading data from a flat binary file.
 * @tparam T The type of data stored in the underlying file.
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

template<class T>
FileReader<T>::FileReader(std::string &filename, Probe &probe)
    : filename_(filename), probe(probe) {
  Open();

  // seek to the end to get the size in bytes
  fp.seekg(0, std::ios::end);
  fsize = fp.tellg();

  Close();
}

/**
 * @brief Acquire data from the file, so many frames at a time.
 * @tparam T The type of data stored in the underlying file.
 * @param frame_offset Number of frames after the beginning to start acquiring.
 * @param n_frames Number of frames to acquire.
 * @param buf Buffer where the acquired data will be stored.
 */
template<class T>
void FileReader<T>::AcquireFrames(int frame_offset, int n_frames, T *buf) {
  Open(); // no-op if already Open

  auto n_channels = probe.n_total();
  auto n_samples = n_frames * n_channels;

  fp.seekg(frame_offset * n_channels * sizeof(T), std::ios::beg);
  fp.read((char *) buf, sizeof(T) * n_samples);
}

/**
 * @brief Open the underlying file for reading.
 * @tparam T The type of data stored in the underlying file.
 */
template<class T>
void FileReader<T>::Open() {
  if (!fp.is_open())
    fp.open(filename_, std::ios::in | std::ios::binary);
}

/**
 * @brief Close the underlying file.
 * @tparam T The type of data stored in the underlying file.
 */
template<class T>
void FileReader<T>::Close() {
  if (fp.is_open())
    fp.close();
}

/**
 * @brief Compute and return the number of frames in the underlying data file.
 * @tparam T The type of data stored in the underlying file.
 * @return The number of frames in the underlying data file.
 */
template<class T>
unsigned long FileReader<T>::n_frames() const {
  return fsize / (probe.n_total() * sizeof(T));
}

#endif //RTS_2_FILE_READER_H
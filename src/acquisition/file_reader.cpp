#include "file_reader.h"
#include "reader.h"

template<class T>
FileReader<T>::FileReader(std::string &filename, Probe &probe)
    : Reader<T>(probe), file_size_(0) {
  set_filename(filename);
}

/**
 * @brief Acquire data_ from the file, so many frames at a time.
 * @tparam T The type of data_ stored in the underlying file.
 * @param frame_offset Number of frames after the beginning to start acquiring.
 * @param n_frames Number of frames to acquire.
 * @param buf Buffer where the acquired data_ will be stored.
 * @return The number of frames read.
 */
template<class T>
uint32_t
FileReader<T>::AcquireFrames(std::vector<T> &buf,
                             uint64_t frame_offset,
                             uint32_t n_frames) {
  if (n_frames == 0) {
    return 0;
  }

  Open(); // no-op if already Open
  auto n_channels = this->probe_.n_total();
  auto n_samples = n_frames * n_channels;

  if (buf.size() != n_samples) {
    buf.resize(n_samples);
  }

  auto nb = sizeof(T);
  auto fpos = frame_offset * n_channels * nb;
  auto n_bytes = nb * n_samples < file_size_ - fpos ? nb * n_samples : file_size_ - fpos;

  fp.seekg(fpos, std::ios::beg);
  fp.read((char *) buf.data(), n_bytes);

  return n_bytes / (nb * this->probe_.n_total());
}

/**
 * @brief Open the underlying file for reading.
 * @tparam T The type of data_ stored in the underlying file.
 */
template<class T>
void FileReader<T>::Open() {
  if (!fp.is_open())
    fp.open(filename_, std::ios::in | std::ios::binary);
}

/**
 * @brief Close the underlying file.
 * @tparam T The type of data_ stored in the underlying file.
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
uint64_t FileReader<T>::n_frames() const {
  return file_size_ / (Reader<T>::probe_.n_total() * sizeof(T));
}

/**
 * @brief Set the path for the underlying file.
 * @tparam T The type of data stored in the underlying file.
 * @param filename
 */
template<class T>
void FileReader<T>::set_filename(std::string &filename) {
  filename_ = filename;

  FileReader<T>::Open();

  // seek to the end to get the size in bytes
  fp.seekg(0, std::ios::end);
  file_size_ = fp.tellg();

  FileReader<T>::Close();
}

template
class FileReader<short>;

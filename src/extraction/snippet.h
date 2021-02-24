#ifndef RTS_2_SNIPPET_H
#define RTS_2_SNIPPET_H

#include <iterator>
#include <limits>
#include <memory>
#include <vector>

#include <thrust/host_vector.h>

class Snippet {
 public:
  Snippet(std::vector<float> buf, uint32_t n_frames)
      : data_(std::move(buf)), n_frames_(n_frames) {};

  float SqDist(const Snippet &other);

  // indexed getters
  [[nodiscard]] float at(uint32_t chan, uint32_t frame) const;

  // unindexed getters
  [[nodiscard]] uint32_t center_channel() const { return channel_ids_.at(0); };
  [[nodiscard]] uint32_t size() const { return data_.size(); };
  [[nodiscard]] uint32_t n_chans() const { return data_.size() / n_frames_; };
  [[nodiscard]] uint32_t n_frames() const { return n_frames_; };
  [[nodiscard]] std::vector<float> data() const { return data_; };

  // setters
  void assign(typename std::vector<float>::iterator begin, uint32_t size);
  void assign(typename thrust::host_vector<float>::iterator begin,
              uint32_t size);
  void set_frame_offset(uint64_t offset) { frame_offset_ = offset; };
  void set_channel_ids(std::vector<uint32_t> &ids);

 private:
  uint32_t n_frames_ = 0; /*!< number of frames in the snippet */

  uint64_t frame_offset_ = 0; /*!< global offset of first sample in snippet */
  std::vector<uint32_t> channel_ids_; /*!< ids of channels in snippets */

  std::vector<float> data_; /*!< data buffer, stored in row major order */
};

#endif //RTS_2_SNIPPET_H

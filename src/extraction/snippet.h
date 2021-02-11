#ifndef RTS_2_SNIPPET_H
#define RTS_2_SNIPPET_H

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

template<class T>
class Snippet {
 public:
  Snippet(std::vector<T> buf, uint32_t n_chans, uint32_t n_frames);

  double SqDist(const Snippet<T> &other);

  // indexed getters
  T at(uint32_t chan, uint32_t frame) const;

  // unindexed getters

  [[nodiscard]] uint32_t n_chans() const { return n_chans_; };
  [[nodiscard]] uint32_t n_frames() const { return n_frames_; };

  // setters
  void set_frame_offset(uint64_t offset) { frame_offset_ = offset; };
  void set_channel_ids(std::vector<uint32_t> &ids);

 private:
  uint32_t n_chans_; /*!< number of channels in the snippet */
  uint32_t n_frames_; /*!< number of frames in the snippet */

  uint64_t frame_offset_ = 0; /*!< global offset of first sample in snippet */
  std::vector<uint32_t> channel_ids_; /*!< ids of channels in snippets */

  std::vector<T> data; /*!< data data, stored in row major order */
};

#endif //RTS_2_SNIPPET_H

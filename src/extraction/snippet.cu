#include "snippet.cuh"

/**
 * @brief Compute the squared Euclidean distance between this snippet and
 * another.
 * @param other Another Snippet to compare.
 * @return The squared Euclidean distance between `this` and `other`.
 *
 * Snippets in different spaces (i.e., channel or frame counts don't match)
 * are defined to be infinitely far apart.
 */
float Snippet::SqDist(const Snippet &other) {
  if (other.size() != size() || other.n_frames_ != n_frames_) {
    return std::numeric_limits<float>::infinity();
  }

  float d = 0.0f;

  for (auto i = 0; i < data_.size(); ++i) {
    auto diff = data_.at(i) - other.data_.at(i);
    d += diff * diff;
  }

  return d;
}

/**
 * @brief Return the snippet value at the (relative) channel `chan` and frame
 * `frame`.
 * @param chan The channel offset with respect to the first channel in the
 * snippet (i.e., the center_ channel in the event).
 * @param frame The frame offset with respect to the first frame in the snippet.
 * @return The value at (`chan`, `frame`).
 */
float Snippet::at(uint32_t chan, uint32_t frame) const {
  return data_.at(chan * n_frames_ + frame);
}

/**
 * @brief Set the channel ids of this snippet.
 * @param ids Vector of channel ids.
 */
void Snippet::set_channel_ids(std::vector<uint32_t> &ids) {
  if (ids.size() != n_chans()) {
    return;
  }

  channel_ids_.assign(ids.begin(), ids.end());
}

void Snippet::assign(typename std::vector<float>::iterator begin,
                     uint32_t size) {
  data_.assign(begin, begin + size);
}

void Snippet::assign(typename thrust::host_vector<float>::iterator begin,
                     uint32_t size) {
  data_.assign(begin, begin + size);
}

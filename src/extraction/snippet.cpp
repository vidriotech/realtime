#include "snippet.h"


template<class T>
Snippet<T>::Snippet(std::vector<T> buf, uint32_t n_chans, uint32_t n_frames)
    : data(buf), n_chans_(n_chans), n_frames_(n_frames) {
  // resize data buffer to match up with our expectations of size
  if (data.size() != n_chans * n_frames) {
    data.resize(n_chans * n_frames);
  }
}

/**
 * @brief Compute the squared Euclidean distance between this snippet and
 * another.
 * @param other Another Snippet to compare.
 * @return The squared Euclidean distance between `this` and `other`.
 *
 * Snippets in different spaces (i.e., channel or frame counts don't match)
 * are defined to be infinitely far apart.
 */
template<class T>
double Snippet<T>::SqDist(const Snippet<T> &other) {
  if (other.n_frames_ != n_frames_ || other.n_chans_ != n_chans_) {
    return std::numeric_limits<double>::infinity();
  }

  double d = 0.0;

  for (auto i = 0; i < data.size(); ++i) {
    d += pow(data.at(i) - other.data.at(i), 2);
  }

  return d;
}

/**
 * @brief Return the snippet value at the (relative) channel `chan` and frame
 * `frame`.
 * @param chan The channel offset with respect to the first channel in the
 * snippet (i.e., the center channel in the event).
 * @param frame The frame offset with respect to the first frame in the snippet.
 * @return The value at (`chan`, `frame`).
 */
template<class T>
T Snippet<T>::at(uint32_t chan, uint32_t frame) const {
  return data.at(frame * n_chans_ + chan);
}

template
class Snippet<short>;

template
class Snippet<float>;
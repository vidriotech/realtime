#ifndef RTS_2_SRC_PARAMS_EXTRACT_PARAMS_H_
#define RTS_2_SRC_PARAMS_EXTRACT_PARAMS_H_

#include <cmath>

class ExtractParams {
 public:
  // snippet extraction
  uint8_t n_sites = 5; /*!< number of sites to extract in a snippet */
  float before_peak_ms = 0.25; /*!< number of ms before peak to extract */
  float after_peak_ms = 0.75; /*!< number of ms before peak to extract */

  // feature extraction
  uint8_t n_pcs = 5; /*!< number of principal components to extract */

  /**
   * @brief Compute the number of frames necessary to extract
   * `before_peak_ms` and `after_peak_ms` worth of samples before and after a
   * peak event.
   * @param sample_rate Sampling rate, in Hz.
   * @return A count of frames to extract.
   */
  uint32_t n_frames(double sample_rate) {
    auto srate_ms = sample_rate / 1000;
    auto nf = std::ceil(srate_ms * (before_peak_ms + after_peak_ms)) + 1;
    return (uint32_t) nf;
  };

  uint32_t n_frames_before(double sample_rate) {
    auto srate_ms = sample_rate / 1000;
    return (uint32_t) std::ceil(srate_ms * before_peak_ms);
  }

  uint32_t n_frames_after(double sample_rate) {
    return n_frames(sample_rate) - n_frames_before(sample_rate) - 1;
  }
};

#endif //RTS_2_SRC_PARAMS_EXTRACT_PARAMS_H_

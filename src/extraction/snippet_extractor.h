#ifndef RTS_2_SNIPPETEXTRACTOR_H
#define RTS_2_SNIPPETEXTRACTOR_H

#include <vector>

#include "../params/params.h"
#include "../probe/probe.h"
#include "./snippet.h"

template<class T>
class SnippetExtractor {
 public:
  explicit SnippetExtractor(Params &params, Probe &probe)
      : params_(params), probe_(probe) {};

  void
  Update(std::vector<T> &samples, std::vector<uint8_t> &crossings,
         uint64_t frame_offset);
  std::vector<Snippet> ExtractSnippets();

  // getters
  [[nodiscard]] uint64_t frame_offset() const { return frame_offset_; };
  uint32_t n_frames() { return samples_.size() / probe_.n_total(); };

  // setters
  void set_frame_offset(uint64_t offset) { frame_offset_ = offset; };

 private:
  std::vector<T> samples_;
  std::vector<uint8_t> crossings_;
  Probe probe_;
  Params params_;

  uint64_t frame_offset_ = 0; /*!< global frame offset */
};

#endif //RTS_2_SNIPPETEXTRACTOR_H

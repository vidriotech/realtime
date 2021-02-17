#ifndef RTS_2_SRC_EXTRACTION_EXTRACTOR_CUH_
#define RTS_2_SRC_EXTRACTION_EXTRACTOR_CUH_

#include "../params/params.h"
#include "../probe/probe.h"
#include "snippet.h"
#include "snippet_extractor.h"
#include "feature_extractor.cuh"

template<class T>
class Extractor {
 public:
  Extractor(Params &params, Probe &probe)
    : params_(params), probe_(probe), snippet_extractor_(params, probe) {};

  void
  Update(std::vector<T> &samples, std::vector<uint8_t> &crossings,
         uint64_t frame_offset);

  void MakeSnippets();
  void ExtractFeatures();

  // getters
  std::vector<Snippet<T>> snippets() const { return snippets_; };

 private:
  std::vector<T> samples_;
  std::vector<uint8_t> crossings_;
  uint64_t frame_offset_ = 0;

  Params params_;
  Probe probe_;

  SnippetExtractor<T> snippet_extractor_;
  std::vector<Snippet<T>> snippets_;
};

#endif //RTS_2_SRC_EXTRACTION_EXTRACTOR_CUH_

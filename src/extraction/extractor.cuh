#ifndef RTS_2_SRC_EXTRACTION_EXTRACTOR_CUH_
#define RTS_2_SRC_EXTRACTION_EXTRACTOR_CUH_

#include <thrust/host_vector.h>

#include "../params/params.cuh"
#include "../probe/probe.cuh"
#include "snippet.cuh"
#include "snippet_extractor.cuh"
#include "feature_extractor.cuh"

template<class T>
class Extractor {
 public:
  Extractor(Params &params, Probe &probe)
      : params_(params), probe_(probe), snippet_extractor_(params, probe) {};

  void
  Update(thrust::host_vector<T> &samples,
         thrust::host_vector<uint8_t> &crossings,
         uint64_t frame_offset);

  void MakeSnippets();
  void ExtractFeatures();

  // getters
  std::vector<Snippet> snippets() const { return snippets_; };

 private:
  thrust::host_vector<T> samples_;
  thrust::host_vector<uint8_t> crossings_;
  uint64_t frame_offset_ = 0;

  Params params_;
  Probe probe_;

  SnippetExtractor<T> snippet_extractor_;
  std::vector<Snippet> snippets_;
};

#endif //RTS_2_SRC_EXTRACTION_EXTRACTOR_CUH_

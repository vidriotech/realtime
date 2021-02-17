#ifndef RTS_2_SRC_EXTRACTION_FEATURE_EXTRACTOR_CUH_
#define RTS_2_SRC_EXTRACTION_FEATURE_EXTRACTOR_CUH_

#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../params/params.h"
#include "../probe/probe.h"
#include "snippet.h"

template<class T>
class FeatureExtractor {
 public:
  FeatureExtractor(Params &params, Probe &probe)
      : params_(params), probe_(probe) {};

  void Update(std::vector<Snippet<T>> &snippets);

 private:
  Params params_;
  Probe probe_;

  thrust::host_vector<T> host_snippets_;
  thrust::device_vector<T> device_snippets_;
};

#endif //RTS_2_SRC_EXTRACTION_FEATURE_EXTRACTOR_CUH_

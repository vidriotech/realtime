#ifndef RTS_2_SRC_EXTRACTION_FEATURE_EXTRACTOR_CUH_
#define RTS_2_SRC_EXTRACTION_FEATURE_EXTRACTOR_CUH_

#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../kernels/kernels.cuh"
#include "../params/params.h"
#include "../probe/probe.h"
#include "snippet.h"

template<class T>
class FeatureExtractor {
 public:
  FeatureExtractor(Params &params, Probe &probe)
      : params_(params), probe_(probe) {};

  void Update(std::vector<Snippet<T>> &snippets);
  void ComputeCovarianceMatrix();
  void ProjectSnippets();

 private:
  Params params_;
  Probe probe_;

  uint32_t n_feats = 0;

  std::vector<Snippet<T>> snippets_;
  thrust::device_vector<float> dev_snippets_;
  thrust::device_vector<float> features_;

  void CenterSnippets();
};

#endif //RTS_2_SRC_EXTRACTION_FEATURE_EXTRACTOR_CUH_

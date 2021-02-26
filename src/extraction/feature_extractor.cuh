#ifndef RTS_2_SRC_EXTRACTION_FEATURE_EXTRACTOR_CUH_
#define RTS_2_SRC_EXTRACTION_FEATURE_EXTRACTOR_CUH_

#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../kernels/kernels.cuh"
#include "../params/params.cuh"
#include "../probe/probe.cuh"
#include "snippet.cuh"

template<class T>
class FeatureExtractor {
 public:
  FeatureExtractor(Params &params, Probe &probe)
      : params_(params), probe_(probe) {};
  ~FeatureExtractor();

  void Update(std::vector<Snippet> &snippets);
  void ComputeCovarianceMatrix();
  void ProjectSnippets();

  // getters
  uint32_t n_obs() const { return snippets_.size(); };

 private:
  Params params_;
  Probe probe_;

  uint32_t n_feats_ = 0;

  std::vector<Snippet> snippets_;
  float *cu_snippets_ = nullptr;
  float *cu_features_ = nullptr;
  thrust::device_vector<float> features_;

  void CenterSnippets();
};

#endif //RTS_2_SRC_EXTRACTION_FEATURE_EXTRACTOR_CUH_

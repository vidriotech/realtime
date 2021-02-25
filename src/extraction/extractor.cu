#include "extractor.cuh"

template<class T>
void
Extractor<T>::Update(thrust::host_vector<T> &samples,
                     thrust::host_vector<uint8_t> &crossings,
                     uint64_t frame_offset) {
  samples_ = samples;
  crossings_ = crossings;
  frame_offset_ = frame_offset;
}

/**
 * @brief
 * @tparam T
 */
template<class T>
void Extractor<T>::MakeSnippets() {
  snippet_extractor_.Update(samples_, crossings_, frame_offset_);
  snippets_ = snippet_extractor_.ExtractSnippets();
}

template<class T>
void Extractor<T>::ExtractFeatures() {
  if (snippets_.empty()) {
    return;
  }

  FeatureExtractor<T> feature_extractor(params_, probe_);
  feature_extractor.Update(snippets_);
  feature_extractor.ComputeCovarianceMatrix();
  feature_extractor.ProjectSnippets();
}

template
class Extractor<short>;

template
class Extractor<float>;
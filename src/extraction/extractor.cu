#include "extractor.cuh"

template<class T>
void
Extractor<T>::Update(std::vector<T> &samples, std::vector<uint8_t> &crossings,
                     uint64_t frame_offset) {
  samples_ = std::move(samples);
  crossings_ = std::move(crossings);
  frame_offset_ = frame_offset;
}

/**
 * @brief
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

  feature_extractor_.Update(snippets_);
  feature_extractor_.ComputeCovarianceMatrix();
  feature_extractor_.ProjectSnippets();
}

template
class Extractor<short>;

template
class Extractor<float>;
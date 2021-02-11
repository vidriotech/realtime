#include "extractor.h"

template<class T>
void Extractor<T>::Update(std::vector<T> &samples,
                          std::vector<uint8_t> &crossings,
                          uint64_t frame_offset) {
  samples_.assign(samples.begin(), samples.end());
  crossings_.assign(crossings.begin(), crossings.end());
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

}

template
class Extractor<short>;

template
class Extractor<float>;
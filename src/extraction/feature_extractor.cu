#include "feature_extractor.cuh"

template<class T>
void FeatureExtractor<T>::Update(std::vector<Snippet<T>> &snippets) {
  if (snippets.empty()) {
    return;
  }

  auto snippet_size = snippets.at(0).size();
  auto n_snippets = snippets.size();

//  host_snippets_.resize(snippet_size * n_snippets);

  auto k = 0;
  for (auto i = 0; i < n_snippets; ++i) {
    auto snip = snippets.at(i).data();
    host_snippets_.insert(host_snippets_.end(), snip.begin(), snip.end());
  }

  device_snippets_ = host_snippets_;
}

template
class FeatureExtractor<short>;

template
class FeatureExtractor<float>;
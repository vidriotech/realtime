#include "snippet_extractor.cuh"

/**
 * @brief Update the samples, crossings, and frame offset.
 * @param samples New data data.
 * @param crossings Threshold crossing information.
 * @param frame_offset New frame offset.
 */
template<class T>
void
SnippetExtractor<T>::Update(std::vector<T> &samples,
                            std::vector<uint8_t> &crossings,
                            uint64_t frame_offset) {
  samples_ = std::move(samples);
  crossings_ = std::move(crossings);

  frame_offset_ = frame_offset;
}

/**
 * @brief Extract snippets from the underlying data.
 * @return Vector of snippets.
 */
template<class T>
std::vector<Snippet> SnippetExtractor<T>::ExtractSnippets() {
  std::vector<Snippet> snippets;

  auto n_sites_snippet = params_.extract.n_sites;
  auto n_frames_snippet = params_.extract.n_frames(probe_.sample_rate());

  if (n_frames() >= n_frames_snippet && crossings_.size() == samples_.size()) {

    auto n_before = params_.extract.n_frames_before(probe_.sample_rate());
    auto n_after = params_.extract.n_frames_after(probe_.sample_rate());

    std::vector<float> snippet_buf;
    for (auto site_idx = 0; site_idx < probe_.n_active(); ++site_idx) {
      auto chan_idx = probe_.chan_index(site_idx);
      auto neighbors = probe_.NearestNeighbors(site_idx, n_sites_snippet);

      // check all crossings on this channel
      for (auto frame = n_before; frame < n_frames() - n_after; ++frame) {
        auto k = frame * probe_.n_total() + chan_idx;
        if (!crossings_.at(k)) {
          continue;
        }

        // found a crossing -- extract snippet in row-major order
        for (auto &neighbor : neighbors) {
          auto neighbor_chan_idx = probe_.chan_index(neighbor);
          for (auto f = frame - n_before; f < frame + n_after + 1; ++f) {
            k = f * probe_.n_total() + neighbor_chan_idx;
            snippet_buf.push_back(samples_.at(k));
          }
        }

        Snippet snippet(snippet_buf, n_frames_snippet);
        snippet_buf.clear();

        snippet.set_channel_ids(neighbors);
        // TODO: extract spike time, not just frame offset
        // e.g., frame_offset_ + k
        snippet.set_frame_offset(frame_offset_);

        snippets.push_back(snippet);
      }
    }
  }

  return snippets;
}

template
class SnippetExtractor<short>;

template
class SnippetExtractor<float>;
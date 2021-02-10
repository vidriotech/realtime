#ifndef RTS_2_PROBE_H
#define RTS_2_PROBE_H

#include <algorithm>
#include <map>
#include <vector>

#include "../structures/distance_matrix.h"

struct ChannelGroup {
  /*!< indices of the channels in this group */
  std::vector<unsigned> channels;

  /*!< unique (across the entire probe_) labels of channels in this group. */
  std::vector<unsigned> site_labels;

  /*!< x coordinates of sites in this group, in microns. */
  std::vector<double> x_coords;
  /*!< y coordinates of sites in this group, in microns. */
  std::vector<double> y_coords;

  [[nodiscard]] unsigned n_channels() const { return channels.size(); }
};

struct ProbeConfig {
  /*!< the total number of channels currently recorded. */
  unsigned n_total;
  /*!< physical or logical groups of channels on the probe_. */
  std::map<unsigned, ChannelGroup> channel_groups;

  /*!< number of samples collected per channel per second. */
  double srate_hz;

  /*!< spatial extent (radius in microns) of templates that will be
   * considered for the probe_ */
  double spatial_extent;

  unsigned n_active() {
    unsigned n_act = 0;
    for (const auto &channel_group : channel_groups) {
      n_act += channel_group.second.n_channels();
    }
    return n_act;
  }
};

class Probe {
 public:
  explicit Probe(ProbeConfig cfg);

  void MakeDistanceMatrix();
  std::vector<uint32_t>
  NearestNeighbors(uint32_t site_idx, uint32_t n_neighbors);

  // unindexed getters
  /**
   * @brief Get the *total* number of channels on the probe_.
   * @return The total number of channels on the probe_.
   */
  [[nodiscard]] unsigned n_total() const { return n_total_; };
  /**
   * @brief Get the number of *active* channels on the probe_.
   * @return The number of active channels on the probe_.
   */
  [[nodiscard]] unsigned n_active() const { return chan_indices_.size(); };
  /**
   * @brief Get the sample rate, in Hz.
   * @return The sample rate, in Hz.
   */
  [[nodiscard]] double sample_rate() const { return srate_hz_; };

  // indexed getters
  [[nodiscard]] bool is_active(uint32_t i) const;
  [[nodiscard]] unsigned chan_index(uint32_t site_idx) const;
  [[nodiscard]] unsigned site_index(uint32_t chan_idx) const;
  [[nodiscard]] unsigned label_at(uint32_t i) const;
  [[nodiscard]] unsigned group_at(uint32_t i) const;
  [[nodiscard]] double x_at(uint32_t i) const;
  [[nodiscard]] double y_at(uint32_t j) const;

  float dist_between(uint32_t i, uint32_t j);

 private:
  unsigned n_total_ = 0;  // the TOTAL number of channels on the probe_
  // the number of neighbors to consider for each active channel
  unsigned n_neigh_ = 0;
  // the number of samples taken per channel per second
  double srate_hz_ = 0.0;

  // row indices of active sites in the data matrix
  std::vector<unsigned> chan_indices_;
  // (unique) label of each site in the probe mapping
  std::vector<unsigned> site_labels;
  // channel group ID of each active site
  std::vector<unsigned> chan_grps;
  // x coordinates of sites on the probe, in microns
  std::vector<double> x_coords;
  // y coordinates of sites on the probe, in microns
  std::vector<double> y_coords;

  // entries are true if the channel is active (size: n_total_)
  std::vector<bool> is_active_;

  // true iff the distance matrix has been built
  bool dist_mat_complete = false;
  // distances between channels, in microns
  DistanceMatrix<float> site_dists;

  // rearrange site labels, x_coords/y_coords in order of channels
  void SortChannels();
  // ensure channel indices/site labels are unique
  void EnsureUnique();
  // find inactive channels, populate is_active_
  void FindInactive();
};

#endif //RTS_2_PROBE_H

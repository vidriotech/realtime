#ifndef RTS_2_PROBE_H
#define RTS_2_PROBE_H

#include <algorithm>
#include <map>
#include <vector>

#include "../structures/distance_matrix.h"

struct ChannelGroup {
  /*!< indices of the channels in this group */
  std::vector<unsigned> channels;

  /*!< unique (across the entire probe) labels of channels in this group. */
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
  /*!< physical or logical groups of channels on the probe. */
  std::map<unsigned, ChannelGroup> channel_groups;

  /*!< number of samples collected per channel per second. */
  double srate_hz;

  /*!< spatial extent (radius in microns) of templates that will be
   * considered for the probe */
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

  void make_distance_matrix();

  // unindexed getters
  /**
   * @brief Get the *total* number of channels on the probe.
   * @return The total number of channels on the probe.
   */
  [[nodiscard]] unsigned n_total() const { return n_total_; };
  /**
   * @brief Get the number of *active* channels on the probe.
   * @return The number of active channels on the probe.
   */
  [[nodiscard]] unsigned n_active() const { return chan_idx.size(); };
  /**
   * @brief Get the sample rate, in Hz.
   * @return The sample rate, in Hz.
   */
  [[nodiscard]] double sample_rate() const { return srate_hz_; };

  // indexed getters
  [[nodiscard]] bool is_active(unsigned i) const;

  unsigned index_at(unsigned i);
  unsigned label_at(unsigned i);
  unsigned group_at(unsigned i);
  double x_at(unsigned i);
  double y_at(unsigned j);

  float dist_between(unsigned i, unsigned j);

 private:
  unsigned n_total_ = 0;  // the TOTAL number of channels on the probe
  // the number of neighbors to consider for each active channel
  unsigned n_neigh_ = 0;
  // the number of samples taken per channel per second
  double srate_hz_ = 0.0;

  // row indices of active sites in the data_ matrix
  std::vector<unsigned> chan_idx;
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
  DistanceMatrix<float> channel_distances;

  // rearrange site labels, x_coords/y_coords in order of channels
  void sort_channels();
  // ensure channel indices/site labels are unique
  void ensure_unique();
  // find inactive channels, populate is_active_
  void find_inactive();
};

#endif //RTS_2_PROBE_H

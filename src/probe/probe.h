#ifndef RTS_2_PROBE_H
#define RTS_2_PROBE_H

#include <algorithm>
#include <map>
#include <vector>

#include "../structures/distance_matrix.h"

struct ChannelGroup {
  // indices of the channels in this group
  std::vector<unsigned> channels;

  // unique (across the entire probe) labels of channels in this group
  std::vector<unsigned> site_labels;

  // x coordinates of sites in this group, in microns
  std::vector<double> x_coords;
  // y coordinates of sites in this group, in microns
  std::vector<double> y_coords;

  [[nodiscard]] unsigned n_channels() const {
    return channels.size();
  }
};

struct ProbeConfig {
  // the total number of channels currently recorded
  unsigned n_total;
  // physical or logical groups of channels on the probe
  std::map<unsigned, ChannelGroup> channel_groups;

  // number of samples collected per channel per second
  double srate_hz;

  // default spatial extent (in microns) of the templates that will be considered for the probe
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
 private:
  unsigned _n_total = 0;  // the TOTAL number of channels on the probe
  // the number of neighbors to consider for each active channel
  unsigned _n_neigh = 0;
  // the number of samples taken per channel per second
  double _srate_hz = 0.0;

  // row indices of active sites in the data matrix
  std::vector<unsigned> chan_idx;
  // (unique) label of each site in the probe mapping
  std::vector<unsigned> site_labels;
  // channel group ID of each active site
  std::vector<unsigned> chan_grps;
  // x coordinates of sites on the probe, in microns
  std::vector<double> x_coords;
  // y coordinates of sites on the probe, in microns
  std::vector<double> y_coords;

  // entries are true if the channel is active (size: _n_total)
  std::vector<bool> is_active;

  // true iff the distance matrix has been built
  bool dist_mat_complete = false;
  // distances between channels, in microns
  DistanceMatrix<float> channel_distances;

  // rearrange site labels, x_coords/y_coords in order of channels
  void sort_channels();
  // ensure channel indices/site labels are unique
  void ensure_unique();
  // find inactive channels, populate is_active
  void find_inactive();

 public:
  explicit Probe(ProbeConfig cfg);

  // getter for _n_total, the total number of channels on this Probe
  [[nodiscard]] unsigned n_total() const;
  // getter for _n_active, the number of active channels on this Probe
  [[nodiscard]] unsigned n_active() const;
  // getter for _srate_hz, the number of samples per channel per second
  [[nodiscard]] double sample_rate() const;

  unsigned index_at(unsigned i);
  unsigned label_at(unsigned i);
  unsigned group_at(unsigned i);
  double x_at(unsigned i);
  double y_at(unsigned j);

  float dist_between(unsigned i, unsigned j);
  void make_distance_matrix();
};

#endif //RTS_2_PROBE_H

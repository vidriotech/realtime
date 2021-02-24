#include "test_probe.cuh"

ProbeConfig make_probe_config(unsigned n_channels,
                              unsigned n_active,
                              unsigned n_groups,
                              double srate_hz) {
  // don't check that n_total >= n_active for test purposes
  if (n_groups > n_active) {
    throw std::domain_error(
        "Number of groups cannot exceed number of active sites.");
  } else if (n_active % n_groups != 0) {
    throw std::domain_error(
        "Number of groups must evenly divide number of active sites.");
  }

  ProbeConfig cfg;
  cfg.n_total = n_channels;
  cfg.spatial_extent = 50.0;
  cfg.srate_hz = srate_hz;

  // divide n_active evenly into n_groups
  auto chans_per_group = n_active / n_groups;
  auto k = 0;
  double x = 0.0, y = 0.0;

  for (auto i = 0; i < n_groups; i++) {
    ChannelGroup grp = ChannelGroup{
        std::vector<unsigned>(chans_per_group), // channels
        std::vector<unsigned>(chans_per_group), // site_labels
        std::vector<double>(chans_per_group), // x_coords
        std::vector<double>(chans_per_group), // y_coords
    };

    for (auto j = 0; j < chans_per_group; j++) {
      grp.site_labels.at(j) = k + 1;
      grp.channels.at(j) = k++;
      grp.x_coords.at(j) = x;
      grp.y_coords.at(j) = y;

      if (j % 2 == 1) {
        x += 25.0;
      } else {
        y += 20.0;
      }
    }

    cfg.channel_groups.insert(std::pair<unsigned, ChannelGroup>(i, grp));
  }

  return cfg;
}

Probe make_probe(unsigned n_channels,
                 unsigned n_active,
                 unsigned n_groups,
                 double srate_hz) {
  return Probe(make_probe_config(n_channels, n_active, n_groups, srate_hz));
}
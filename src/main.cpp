#include <cmath>
#include <iostream>
#include <thread>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "probe.h"
#include "acquisition/file_reader.h"
#include "structures/median_tree.h"

std::string get_env_var(std::string const &key) {
  char *val = getenv(key.c_str());
  return val == nullptr ? std::string("") : std::string(val);
}

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

  for (auto i = 0; i < n_groups; ++i) {
    ChannelGroup grp = ChannelGroup{
        std::vector<unsigned>(chans_per_group), // channels
        std::vector<unsigned>(chans_per_group), // site_labels
        std::vector<double>(chans_per_group), // x_coords
        std::vector<double>(chans_per_group), // y_coords
    };

    for (auto j = 0; j < chans_per_group; ++j) {
      grp.site_labels[j] = k + 1;
      grp.channels[j] = k++;
      grp.x_coords[j] = x;
      grp.y_coords[j] = y;

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

int main() {
  auto filename = get_env_var("TEST_FILE");
  auto n_channels = std::stoi(get_env_var("TEST_NCHANNELS"));
  auto n_active = std::stoi(get_env_var("TEST_NACTIVE"));
  auto n_groups = std::stoi(get_env_var("TEST_NGROUPS"));
  auto srate_hz = std::stod(get_env_var("TEST_SRATE_HZ"));

  auto probe = make_probe(n_channels, n_active, n_groups, srate_hz);
  FileReader<short> reader(filename, probe);

  auto n_frames = (int) std::ceil(srate_hz);
  auto buf = new short[n_frames * n_channels];

  reader.AcquireFrames(0, n_frames, buf);
  auto trees = new MedianTree<short>[n_channels];

  for (auto i = 0; i < n_frames; i++) {
    auto j = 30;
//    for (auto j = 0; j < n_channels; j++) {
      auto k = i * n_channels + j;
      std::cout << std::endl;
      std::cout << k << " " << buf[k] << std::endl;
      trees[j].Insert(buf[k]);
//    }
  }

//    cudaMallocManaged(&buf, (int)ceil(srate_hz) * n_channels);

//    cudaFree(buf);

  delete[] buf;
  delete[] trees;
}

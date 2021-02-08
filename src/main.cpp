#include <cmath>
#include <iostream>
#include <thread>

#include "params/params.h"
#include "probe/probe.h"
#include "acquisition/file_reader.h"
#include "structures/median_tree.h"
#include "pipeline/pipeline_thread_pool.h"

std::string get_env_var(std::string const &key) {
  char *val = getenv(key.c_str());
  return val == nullptr ? std::string("") : std::string(val);
}

ProbeConfig make_probe_config(uint32_t n_channels,
                              uint32_t n_active,
                              uint32_t n_groups,
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
        std::vector<uint32_t>(chans_per_group), // channels
        std::vector<uint32_t>(chans_per_group), // site_labels
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

    cfg.channel_groups.insert(std::pair<uint32_t, ChannelGroup>(i, grp));
  }

  return cfg;
}

Probe make_probe(uint32_t n_channels,
                 uint32_t n_active,
                 uint32_t n_groups,
                 double srate_hz) {
  return Probe(make_probe_config(n_channels, n_active, n_groups, srate_hz));
}

int main() {
  auto filename = get_env_var("TEST_FILE");
  auto n_channels = std::stoi(get_env_var("TEST_NCHANNELS"));
  auto n_active = std::stoi(get_env_var("TEST_NACTIVE"));
  auto n_groups = std::stoi(get_env_var("TEST_NGROUPS"));
  auto srate_hz = std::stod(get_env_var("TEST_SRATE_HZ"));

  Params params;
  auto probe = make_probe(n_channels, n_active, n_groups, srate_hz);
  FileReader<short> reader(filename, probe);

  auto n_frames_buf = (uint64_t) std::ceil(params.acquire.n_seconds * srate_hz);
  auto n_samples_buf = n_frames_buf * n_channels;

  std::shared_ptr<short[]> buf(new short[n_samples_buf]);
//  std::shared_ptr<short[]> shared_buf(new short[n_samples_buf]);

  // set up thread pool
  auto n_threads = 1;
//  auto n_threads = std::max((uint32_t) 1,
//                            std::thread::hardware_concurrency() / 2);
  PipelineThreadPool<short> pool(params, probe, n_threads);

  // start acquiring!
  auto tic = std::chrono::high_resolution_clock::now();
  for (uint64_t frame_offset = 0; frame_offset < reader.n_frames();
       frame_offset += n_frames_buf) {
    reader.AcquireFrames(buf, frame_offset, n_frames_buf);
//    std::memcpy(shared_buf.get(), buf.get(), n_samples_buf * sizeof(short));
    const std::shared_ptr<short[]>& shared_buf(buf);

    pool.BlockEnqueueData(shared_buf, n_samples_buf, frame_offset);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // finish up
  pool.StopWaiting();
  while (pool.is_working()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // gather stats
  auto toc = std::chrono::high_resolution_clock::now();
  auto proc_dur =
      std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
  auto rec_dur = reader.n_frames() / probe.sample_rate() * 1000;
  auto ratio{proc_dur.count() / rec_dur};

  std::cout << "processing " << rec_dur << " ms took " << proc_dur.count() <<
            " ms (" << ratio << " of record time)" << std::endl;
}

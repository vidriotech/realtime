#ifndef RTS_2_TESTPROBE_H
#define RTS_2_TESTPROBE_H

#include <stdexcept>
#include "../Probe.h"


ProbeConfig test_probeconfig(size_t n_channels, size_t n_active, size_t n_groups, double srate_hz)
{
    // don't check that n_total >= n_active for test purposes
    if (n_groups > n_active) {
        throw std::domain_error("Number of groups cannot exceed number of active sites.");
    }
    else if (n_active % n_groups != 0) {
        throw std::domain_error("Number of groups must evenly divide number of active sites.");
    }

    ProbeConfig cfg;
    cfg.n_total = n_channels;
    cfg.spatial_extent = 50.0;
    cfg.srate_hz = srate_hz;

    // divide n_active evenly into n_groups
    auto chans_per_group = n_active / n_groups;
    size_t k = 0;
    double x = 0.0, y = 0.0;

    for (size_t i = 0; i < n_groups; i++) {
        ChannelGroup grp = ChannelGroup{
            std::vector<size_t>(chans_per_group), // channels
            std::vector<size_t>(chans_per_group), // site_labels
            std::vector<double>(chans_per_group), // x_coords
            std::vector<double>(chans_per_group), // y_coords
        };

        for (size_t j = 0; j < chans_per_group; j++) {
            grp.site_labels[j] = k + 1;
            grp.channels[j] = k++;
            grp.x_coords[j] = x;
            grp.y_coords[j] = y;

            if (j % 2 == 1) {
                x += 25.0;
            }
            else {
                y += 20.0;
            }
        }

        cfg.channel_groups.insert(std::pair<size_t, ChannelGroup>(i, grp));
    }

    return cfg;
}

#endif //RTS_2_TESTPROBE_H

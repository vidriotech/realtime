#ifndef RTS_2_TESTPROBE_H
#define RTS_2_TESTPROBE_H

#include <stdexcept>
#include "../../src/probe/probe.cuh"

ProbeConfig make_probe_config(unsigned n_channels,
                              unsigned n_active,
                              unsigned n_groups,
                              double srate_hz);
Probe make_probe(unsigned n_channels,
                 unsigned n_active,
                 unsigned n_groups,
                 double srate_hz);

#endif //RTS_2_TESTPROBE_H

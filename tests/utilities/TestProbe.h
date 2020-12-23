#ifndef RTS_2_TESTPROBE_H
#define RTS_2_TESTPROBE_H

#include <stdexcept>
#include "../../src/Probe.h"

ProbeConfig make_probeconfig(size_t, size_t, size_t, double);
Probe make_probe(size_t, size_t, size_t, double);

#endif //RTS_2_TESTPROBE_H

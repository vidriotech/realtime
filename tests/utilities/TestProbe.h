#ifndef RTS_2_TESTPROBE_H
#define RTS_2_TESTPROBE_H

#include <stdexcept>
#include "../../src/probe/probe.h"

ProbeConfig make_probe_config(unsigned, unsigned, unsigned, double);
Probe make_probe(unsigned, unsigned, unsigned, double);

#endif //RTS_2_TESTPROBE_H

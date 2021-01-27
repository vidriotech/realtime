#ifndef RTS_2_TESTS_TEST_UTILS_TEST_UTILITIES_H_
#define RTS_2_TESTS_TEST_UTILS_TEST_UTILITIES_H_

#include <string>

#include "test_exception.h"
#include "test_probe.h"

std::string get_env_var(std::string const &key);
Probe probe_from_env();

#endif //RTS_2_TESTS_TEST_UTILS_TEST_UTILITIES_H_

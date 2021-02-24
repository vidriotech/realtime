#include "./test_utilities.cuh"

std::string get_env_var(std::string const &key) {
  char *val = getenv(key.c_str());
  if (val == nullptr) {
    throw TestException(key + " is not defined.");
  }
  return std::string(val);
}

Probe probe_from_env() {
  auto n_channels = std::stoi(get_env_var("TEST_NCHANNELS"));
  auto n_active = std::stoi(get_env_var("TEST_NACTIVE"));
  auto n_groups = std::stoi(get_env_var("TEST_NGROUPS"));
  auto srate_hz = std::stod(get_env_var("TEST_SRATE_HZ"));

  return make_probe(n_channels, n_active, n_groups, srate_hz);
}
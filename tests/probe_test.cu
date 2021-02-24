#include "gtest/gtest.h"

#include <stdexcept>
#include "../src/probe/probe.cuh"
#include "./test_utilities/test_utilities.cuh"

TEST(ProbeTestSuite, TestInitOK) {
  unsigned n_tot = 385, n_active = 384, n_groups = 4;
  double srate_hz = 30000.0;

  ProbeConfig cfg = make_probe_config(n_tot, n_active, n_groups, srate_hz);
  Probe prb(cfg);

  EXPECT_EQ(n_tot, prb.n_total());
  EXPECT_EQ(n_active, prb.n_active());
}

TEST(ProbeTestSuite, TestInitMoreActiveThanTotalFails) {
  unsigned n_tot = 385, n_active = 388, n_groups = 4; // 388 > 385
  double srate_hz = 30000.0;

  ProbeConfig cfg = make_probe_config(n_tot, n_active, n_groups, srate_hz);

  EXPECT_THROW(Probe prb(cfg), std::domain_error);
}

TEST(ProbeTestSuite, TestMakeDistMatrixOK) {
  unsigned n_tot = 4, n_active = 4, n_groups = 2;
  double srate_hz = 30000.0;

  Probe prb = make_probe(n_tot, n_active, n_groups, srate_hz);
  prb.MakeDistanceMatrix();

  EXPECT_EQ(0.0, prb.dist_between(0, 0));
  EXPECT_NEAR(20.0, prb.dist_between(0, 1), 1e-12);
  EXPECT_NEAR(32.0156, prb.dist_between(0, 2), 1e-4);
  EXPECT_NEAR(47.1699, prb.dist_between(0, 3), 1e-4);
  EXPECT_EQ(0.0, prb.dist_between(1, 1));
  EXPECT_NEAR(25.0, prb.dist_between(1, 2), 1e-12);
  EXPECT_NEAR(32.0156, prb.dist_between(1, 3), 1e-4);
  EXPECT_EQ(0.0, prb.dist_between(2, 2));
  EXPECT_NEAR(20.0, prb.dist_between(2, 3), 1e-12);
  EXPECT_EQ(0.0, prb.dist_between(3, 3));
}

/*
 * GIVEN a Probe probe
 * TEST THAT each active channel is actually reported as active.
 */
TEST(ProbeTestSuite, IsActive) {
  auto n_channels = std::stoi(get_env_var("TEST_NCHANNELS"));
  auto n_active = std::stoi(get_env_var("TEST_NACTIVE"));
  auto probe = probe_from_env();

  for (auto i = 0; i < n_active; ++i) {
    EXPECT_TRUE(probe.is_active(i));
  }

  for (auto i = n_active; i < n_channels; i++) {
    EXPECT_FALSE(probe.is_active((i)));
  }
}

/*
 * GIVEN a Probe `probe`, a site index `site_idx`, and a positive integer `n`
 * TEST THAT the distances to its reported n nearest neighbors are sorted; AND
 *           the smallest distance to a site *not* a nearest neighbor is
 *           at least as large as the largest distance of the nearest
 *           neighbors.
 */
TEST(ProbeTestSuite, NearestNeighbors) {
  auto probe = probe_from_env();

  ASSERT_GT(probe.n_active(), 0);
  auto site_idx = std::max((uint32_t) 0, probe.n_active() - 1);
  auto n_neighbors = 10;

  // compute nearest neighbors
  probe.MakeDistanceMatrix();
  auto nearest_neighbors = probe.NearestNeighbors(site_idx, n_neighbors);

  // test that distances given are sorted
  std::vector<float> dists(n_neighbors);
  for (auto i = 0; i < n_neighbors; ++i) {
    dists.at(i) = probe.dist_between(site_idx, nearest_neighbors.at(i));
  }

  EXPECT_TRUE(std::is_sorted(dists.begin(), dists.end()));

  // test that the min distance of sites not in nearest neighbors is no
  // smaller than the largest distance in nearest neighbors
  auto largest_distance = *(dists.end());
  std::vector<uint32_t> non_neighbors;
  for (auto i = 0; i < probe.n_active(); ++i) {
    auto it = std::find(nearest_neighbors.begin(), nearest_neighbors.end(), i);

    // value not found in nearest neighbors; test it
    if (it == nearest_neighbors.end()) {
      EXPECT_GE(probe.dist_between(site_idx, i), largest_distance);
    }
  }
}

/*
 *
 */
TEST(ProbeTestSuite, SiteIndex) {
  auto probe = probe_from_env();

  auto site_idx = 0;
  for (auto chan = 0; chan < probe.n_total(); ++chan) {
    if (probe.is_active(chan)) {
      EXPECT_EQ(site_idx++, probe.site_index(chan));
    }
  }
}
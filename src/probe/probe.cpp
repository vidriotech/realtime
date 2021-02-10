#include "probe.h"

Probe::Probe(ProbeConfig cfg)
    : chan_indices_(cfg.n_active()),
      site_labels(cfg.n_active()),
      chan_grps(cfg.n_active()),
      x_coords(cfg.n_active()),
      y_coords(cfg.n_active()),
      is_active_(cfg.n_total),
      site_dists(cfg.n_active()) {
  n_total_ = cfg.n_total;
  if (n_total_ < cfg.n_active()) {
    throw std::domain_error(
        "Number of active channels cannot exceed total number of channels.");
  }

  if (cfg.srate_hz <= 0.0) {
    throw std::domain_error("Sample rate must be positive.");
  }
  srate_hz_ = cfg.srate_hz;

  unsigned k = 0;  // nested for loop go brr
  for (const auto &channel_group : cfg.channel_groups) {
    const ChannelGroup grp = channel_group.second;

    for (auto j = 0; j < grp.n_channels(); ++j) {
      chan_indices_.at(k) = grp.channels.at(j);
      site_labels.at(k) = grp.site_labels.at(j);
      x_coords.at(k) = grp.x_coords.at(j);
      y_coords.at(k) = grp.y_coords.at(j);

      chan_grps[k++] = channel_group.first;
    }
  }

  SortChannels();
  EnsureUnique();
  FindInactive();
}

/**
 * @brief Create the matrix of distances between channels on the probe.
 */
void Probe::MakeDistanceMatrix() {
  if (dist_mat_complete || site_dists.n_cols() != n_active())
    return;

  for (unsigned i = 0; i < n_active(); ++i) {
    for (unsigned j = i + 1; j < n_active(); ++j) {
      auto dx = x_coords.at(i) - x_coords.at(j),
          dy = y_coords.at(i) - y_coords.at(j);
      site_dists.set_at(i, j, (float) std::hypot(dx, dy));
    }
  }

  dist_mat_complete = true;
}

/**
 * @brief
 * @param site_idx
 * @param n_neighbors
 * @return Site indices of nearest neighbors to `site_idx` (including
 * `site_idx`).
 */
std::vector<uint32_t> Probe::NearestNeighbors(uint32_t site_idx,
                                              uint32_t n_neighbors) {
  MakeDistanceMatrix();
  return site_dists.closest(site_idx, n_neighbors);
}

/**
 * @brief Get the channel index value of the site at `site_idx`.
 * @param site_idx Index of the site.
 * @return The channel index value of the site at `site_idx`.
 */
unsigned Probe::chan_index(unsigned site_idx) const {
  if (site_idx > n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  return chan_indices_.at(site_idx);
}

/**
  * @brief Get the site index value of the channel at `chan_idx`.
  * @param chan_idx Index of the channel.
  * @return The site index value of the channel at `chan_idx`.
  */
unsigned Probe::site_index(uint32_t chan_idx) const {
  auto idx = 0;
  for (auto it = is_active_.begin(); it < is_active_.begin() + chan_idx; ++it) {
    idx += *it;
  }

  return idx;
}

/**
 * @brief Get the label of the ith site.
 * @param i Index of the site.
 * @return The ith site label.
 */
unsigned Probe::label_at(unsigned i) const {
  if (i > n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  return site_labels.at(i);
}

/**
 * @brief Get the channel group label of the ith site.
 * @param i Index of the site.
 * @return The channel group label of the ith site.
 */
unsigned Probe::group_at(unsigned i) const {
  if (i > n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  return chan_grps.at(i);
}

/**
 * @brief Returns the x coordinate of the ith site.
 * @param i Index of the site.
 * @return The x coordinate of the ith site.
 */
double Probe::x_at(unsigned i) const {
  if (i > n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  return x_coords.at(i);
}

/**
 * @brief Returns the y coordinate of the ith site.
 * @param i Index of the site.
 * @return The y coordinate of the ith site.
 */
double Probe::y_at(unsigned i) const {
  if (i > n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  return y_coords.at(i);
}

/**
 * @brief Get the distance between the ith and jth sites.
 * @param i Site label.
 * @param j Site label.
 * @return Distance between site i and site j.
 */
float Probe::dist_between(uint32_t i, uint32_t j) {
  if (i > n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  if (!dist_mat_complete) {
    MakeDistanceMatrix();
  }

  return site_dists.at(i, j);
}

/**
 * @brief Sort site-indexed values by corresponding channel in ascending order.
 */
void Probe::SortChannels() {
  if (n_active() == 0)
    return;

  // get indices that would sort chan_indices_
  auto as = utilities::argsort(chan_indices_);
  if (std::is_sorted(as.begin(), as.end())) {  // nothing to do!
    return;
  }

  std::vector<unsigned> tmp_buf_s(n_active());
  std::vector<double> tmp_buf_d(n_active());

  // reorder chan_indices_, x_coords
  for (auto i = 0; i < as.size(); ++i) {
    tmp_buf_s.at(i) = chan_indices_.at(as.at(i));
    tmp_buf_d.at(i) = x_coords.at(as.at(i));
  }
  chan_indices_.assign(tmp_buf_s.begin(), tmp_buf_s.end());
  x_coords.assign(tmp_buf_d.begin(), tmp_buf_d.end());

  // reorder site_labels, y_coords
  for (auto i = 0; i < as.size(); ++i) {
    tmp_buf_s.at(i) = site_labels[as.at(i)];
    tmp_buf_d.at(i) = y_coords[as.at(i)];
  }
  site_labels.assign(tmp_buf_s.begin(), tmp_buf_s.end());
  y_coords.assign(tmp_buf_d.begin(), tmp_buf_d.end());

  // reorder chan_grps
  for (auto i = 0; i < as.size(); ++i) {
    tmp_buf_s.at(i) = chan_grps[as.at(i)];
  }
  chan_grps.assign(tmp_buf_s.begin(), tmp_buf_s.end());
}

/**
* @brief Check that channel indices and site labels are unique, throwing an
 * error if this is not the case.
*/
void Probe::EnsureUnique() {
  // ensure all channel indices are unique
  unsigned ct;
  for (auto it = chan_indices_.begin(); it != chan_indices_.end(); ++it) {
    ct = std::count(it, chan_indices_.end(), *(it));
    if (ct > 1) {
      throw std::domain_error("Channel indices are not unique.");
    }
  }

  for (auto it = site_labels.begin(); it != site_labels.end(); ++it) {
    ct = std::count(it, site_labels.end(), *(it));
    if (ct > 1) {
      throw std::domain_error("Site labels are not unique.");
    }
  }
}

/**
* @brief Find inactive channels and set their bits to 0 in is_active_.
*
* Assumes that chan_indices_ is sorted.
*/
void Probe::FindInactive() {
  for (auto i = 0; i < n_total_; ++i) {
    is_active_.at(i) = std::binary_search(chan_indices_.begin(),
                                          chan_indices_.end(), i);
  }
}

/**
 * @brief Determine whether a channel represents an active site.
 * @param i Channel index.
 * @return True iff the channel is an active site.
 */
bool Probe::is_active(unsigned int i) const {
  return is_active_.at(i);
}

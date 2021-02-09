#include "probe.h"

Probe::Probe(ProbeConfig cfg)
    : chan_idx(cfg.n_active()),
      site_labels(cfg.n_active()),
      chan_grps(cfg.n_active()),
      x_coords(cfg.n_active()),
      y_coords(cfg.n_active()),
      is_active_(cfg.n_total),
      channel_distances(cfg.n_active()) {
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
      chan_idx.at(k) = grp.channels.at(j);
      site_labels.at(k) = grp.site_labels.at(j);
      x_coords.at(k) = grp.x_coords.at(j);
      y_coords.at(k) = grp.y_coords.at(j);

      chan_grps[k++] = channel_group.first;
    }
  }

  sort_channels();
  ensure_unique();
  find_inactive();
}

 /**
  * @brief Get the channel index value of the ith site.
  * @param i Index of the site.
  * @return The channel index value of the ith site.
  */
unsigned Probe::index_at(unsigned i) {
  if (i > n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  return chan_idx.at(i);
}

/**
 * @brief Get the label of the ith site.
 * @param i Index of the site.
 * @return The ith site label.
 */
unsigned Probe::label_at(unsigned i) {
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
unsigned Probe::group_at(unsigned i) {
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
double Probe::x_at(unsigned i) {
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
double Probe::y_at(unsigned i) {
  if (i > n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  return y_coords.at(i);
}

/**
 * Returns the Euclidean distance between the ith channel and the jth channel.
 */
float Probe::dist_between(unsigned i, unsigned j) {
  if (i > n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  if (!dist_mat_complete) {
    make_distance_matrix();
  }

  return channel_distances.at(i, j);
}

void Probe::make_distance_matrix() {
  if (dist_mat_complete || channel_distances.n_cols() != n_active())
    return;

  for (unsigned i = 0; i < n_active(); ++i) {
    for (unsigned j = i + 1; j < n_active(); ++j) {
      auto dx = x_coords.at(i) - x_coords.at(j), dy = y_coords.at(i) - y_coords.at(j);
      channel_distances.set_at(i, j, (float) std::hypot(dx, dy));
    }
  }

  dist_mat_complete = true;
}

/**
 * Sort channel-indexed values (channel indices, site labels, channel group ID, x/y coords) by channel,
 * ascending.
 */
void Probe::sort_channels() {
  if (n_active() == 0)
    return;

  // get indices that would sort chan_idx
  std::vector<unsigned> argsort(n_active());
  for (unsigned i = 0; i < n_active(); ++i) {
    argsort.at(i) = i;
  }

  std::sort(argsort.begin(), argsort.end(),
            [&](unsigned i, unsigned j) { return chan_idx.at(i) < chan_idx.at(j); });

  if (std::is_sorted(argsort.begin(), argsort.end())) {  // nothing to do!
    return;
  }

  std::vector<unsigned> tmp_buf_s(n_active());
  std::vector<double> tmp_buf_d(n_active());

  // reorder chan_idx, x_coords
  for (unsigned i = 0; i < argsort.size(); ++i) {
    tmp_buf_s.at(i) = chan_idx[argsort.at(i)];
    tmp_buf_d.at(i) = x_coords[argsort.at(i)];
  }
  chan_idx.assign(tmp_buf_s.begin(), tmp_buf_s.end());
  x_coords.assign(tmp_buf_d.begin(), tmp_buf_d.end());

  // reorder site_labels, y_coords
  for (unsigned i = 0; i < argsort.size(); ++i) {
    tmp_buf_s.at(i) = site_labels[argsort.at(i)];
    tmp_buf_d.at(i) = y_coords[argsort.at(i)];
  }
  site_labels.assign(tmp_buf_s.begin(), tmp_buf_s.end());
  y_coords.assign(tmp_buf_d.begin(), tmp_buf_d.end());

  // reorder chan_grps
  for (unsigned i = 0; i < argsort.size(); ++i) {
    tmp_buf_s.at(i) = chan_grps[argsort.at(i)];
  }
  chan_grps.assign(tmp_buf_s.begin(), tmp_buf_s.end());
}

/**
* Check that channel indices and site labels are unique, throwing an error if this is not the case.
*/
void Probe::ensure_unique() {
  // ensure all channel indices are unique
  unsigned ct;
  for (auto it = chan_idx.begin(); it != chan_idx.end(); ++it) {
    ct = std::count(it, chan_idx.end(), *(it));
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
* Assumes that chan_idx is sorted.
*/
void Probe::find_inactive() {
  for (unsigned i = 0; i < n_total_; ++i) {
    if (std::binary_search(chan_idx.begin(),
                           chan_idx.end(),
                           i)) {  // channel not found
      is_active_.at(i) = true;
    } else {
      is_active_.at(i) = false;
    }
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

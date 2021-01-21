#include "probe.h"

Probe::Probe(ProbeConfig cfg)
    : chan_idx(cfg.n_active()),
      site_labels(cfg.n_active()),
      chan_grps(cfg.n_active()),
      x_coords(cfg.n_active()),
      y_coords(cfg.n_active()),
      is_active(cfg.n_total),
      channel_distances(cfg.n_active()) {
  _n_total = cfg.n_total;
  if (_n_total < cfg.n_active()) {
    throw std::domain_error(
        "Number of active channels cannot exceed total number of channels.");
  }

  if (cfg.srate_hz <= 0.0) {
    throw std::domain_error("Sample rate must be positive.");
  }
  this->_srate_hz = cfg.srate_hz;

  unsigned k = 0;  // nested for loop go brr
  for (const auto &channel_group : cfg.channel_groups) {
    const ChannelGroup grp = channel_group.second;

    for (unsigned j = 0; j < grp.n_channels(); ++j) {
      chan_idx[k] = grp.channels[j];
      site_labels[k] = grp.site_labels[j];
      x_coords[k] = grp.x_coords[j];
      y_coords[k] = grp.y_coords[j];

      chan_grps[k++] = channel_group.first;
    }
  }

  this->sort_channels();
  this->ensure_unique();
  this->find_inactive();
}

unsigned Probe::n_total() const {
  return this->_n_total;
}

unsigned Probe::n_active() const {
  return chan_idx.size();
}

double Probe::sample_rate() const {
  return this->_srate_hz;
}

/**
 * Returns the ith channel index value.
 *
 * @ param i Index into the chan_idx array.
 */
unsigned Probe::index_at(unsigned i) {
  if (i > this->n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  return this->chan_idx[i];
}

/**
 * Returns the ith site label.
 *
 * @ param i Index into the site_labels array.
 */
unsigned Probe::label_at(unsigned i) {
  if (i > this->n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  return this->site_labels[i];
}

/**
 * Returns the channel group label of the ith channel.
 *
 * @ param i Index into the chan_grps array.
 */
unsigned Probe::group_at(unsigned i) {
  if (i > this->n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  return this->chan_grps[i];
}

/**
 * Returns the ith x coordinate.
 *
 * @ param i Index into the x_coords array.
 */
double Probe::x_at(unsigned i) {
  if (i > this->n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  return this->x_coords[i];
}

/**
 * Returns the ith y coordinate.
 *
 * @ param i Index into the y_coords array.
 */
double Probe::y_at(unsigned i) {
  if (i > this->n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  return this->y_coords[i];
}

/**
 * Returns the Euclidean distance between the ith channel and the jth channel.
 */
float Probe::dist_between(unsigned i, unsigned j) {
  if (i > this->n_active()) {
    throw std::length_error("Index exceeds array dimensions.");
  }

  if (!this->dist_mat_complete) {
    this->make_distance_matrix();
  }

  return this->channel_distances.get_at(i, j);
}

void Probe::make_distance_matrix() {
  if (dist_mat_complete || this->channel_distances.n_cols() != this->n_active())
    return;

  for (unsigned i = 0; i < this->n_active(); ++i) {
    for (unsigned j = i + 1; j < this->n_active(); ++j) {
      auto dx = x_coords[i] - x_coords[j], dy = y_coords[i] - y_coords[j];
      this->channel_distances.set_at(i, j, (float) std::hypot(dx, dy));
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
    argsort[i] = i;
  }

  std::sort(argsort.begin(), argsort.end(),
            [&](unsigned i, unsigned j) { return chan_idx[i] < chan_idx[j]; });

  if (std::is_sorted(argsort.begin(), argsort.end())) {  // nothing to do!
    return;
  }

  std::vector<unsigned> tmp_buf_s(n_active());
  std::vector<double> tmp_buf_d(n_active());

  // reorder chan_idx, x_coords
  for (unsigned i = 0; i < argsort.size(); ++i) {
    tmp_buf_s[i] = chan_idx[argsort[i]];
    tmp_buf_d[i] = x_coords[argsort[i]];
  }
  this->chan_idx.assign(tmp_buf_s.begin(), tmp_buf_s.end());
  this->x_coords.assign(tmp_buf_d.begin(), tmp_buf_d.end());

  // reorder site_labels, y_coords
  for (unsigned i = 0; i < argsort.size(); ++i) {
    tmp_buf_s[i] = site_labels[argsort[i]];
    tmp_buf_d[i] = y_coords[argsort[i]];
  }
  this->site_labels.assign(tmp_buf_s.begin(), tmp_buf_s.end());
  this->y_coords.assign(tmp_buf_d.begin(), tmp_buf_d.end());

  // reorder chan_grps
  for (unsigned i = 0; i < argsort.size(); ++i) {
    tmp_buf_s[i] = chan_grps[argsort[i]];
  }
  this->chan_grps.assign(tmp_buf_s.begin(), tmp_buf_s.end());
}

/**
* Check that channel indices and site labels are unique, throwing an error if this is not the case.
*/
void Probe::ensure_unique() {
  // ensure all channel indices are unique
  unsigned ct;
  for (std::vector<unsigned>::iterator it = chan_idx.begin();
       it != chan_idx.end(); ++it) {
    ct = std::count(it, chan_idx.end(), *(it));
    if (ct > 1) {
      throw std::domain_error("Channel indices are not unique.");
    }
  }

  for (std::vector<unsigned>::iterator it = site_labels.begin();
       it != site_labels.end(); ++it) {
    ct = std::count(it, site_labels.end(), *(it));
    if (ct > 1) {
      throw std::domain_error("Site labels are not unique.");
    }
  }
}

/**
* \brief Find inactive channels and set their bits to 0 in is_active.
*
* Assumes that chan_idx is sorted.
*/
void Probe::find_inactive() {
  for (unsigned i = 0; i < this->_n_total; ++i) {
    if (std::binary_search(chan_idx.begin(),
                           chan_idx.end(),
                           i)) {  // channel not found
      is_active[i] = true;
    } else {
      is_active[i] = false;
    }
  }
}

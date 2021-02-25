#include "feature_extractor.cuh"

/**
 * @brief
 * @param snippets
 */
template<class T>
void FeatureExtractor<T>::Update(std::vector<Snippet> &snippets) {
  if (snippets.empty()) {
    n_feats_ = 0;
    return;
  }

  snippets_ = snippets;
  n_feats_ = snippets_.at(0).size();

  // concatenate snippet values
  thrust::host_vector<float> host_snippets_;
  for (auto i = 0; i < n_obs(); ++i) {
    auto snip = snippets_.at(i).data();
    host_snippets_.insert(host_snippets_.end(), snip.begin(), snip.end());
  }

  // copy host data to device
  dev_snippets_ = host_snippets_;
}

/**
 * @brief Compute the covariance matrix of snippet features.
 */
template<class T>
void FeatureExtractor<T>::ComputeCovarianceMatrix() {
  if (n_feats_ == 0) {
    return;
  }

//  CenterSnippets();
  features_.resize(n_feats_ * n_feats_);

  float *snip_ptr = thrust::raw_pointer_cast(dev_snippets_.data());
  float *cov_ptr = thrust::raw_pointer_cast(features_.data());

  CovMatrixArgs args{dev_snippets_.size() / n_feats_,
                     n_feats_, snip_ptr, cov_ptr};
  make_cov_matrix(args);
}

template<class T>
void FeatureExtractor<T>::ProjectSnippets() {
  auto n_pcs = params_.extract.n_pcs;

  MakePVArgs pv_args{n_feats_, n_pcs, features_};
  make_principal_vectors(pv_args);

  ProjectOntoPVsArgs project_args{n_pcs,
                                  n_feats_,
                                  n_obs(),
                                  features_,
                                  dev_snippets_};
  project_onto_pvs(project_args);

  // projected snippets live in features_
  thrust::host_vector<float> host_features_ = features_;

//  host_features_.assign(features_.begin(), features_.begin() + n_pcs * n_obs);
  features_.clear();

  // copy projections back to snippets
  thrust::host_vector<float>::iterator it = host_features_.begin();
  for (auto i = 0; i < n_obs(); ++i) {
    snippets_.at(i).assign(it, n_pcs);
    it += n_pcs;
  }
}

/**
 * @brief Compute the mean of each feature and subtract it, centering the new
 * means at 0.
 */
template<class T>
void FeatureExtractor<T>::CenterSnippets() {
  if (n_feats_ == 0) {
    return;
  }

  CenterFeaturesArgs args{dev_snippets_.size() / n_feats_,
                          n_feats_, dev_snippets_};
  center_features(args);

  std::vector<float> feats;
  feats.assign(dev_snippets_.begin(), dev_snippets_.end());
}

template
class FeatureExtractor<short>;

template
class FeatureExtractor<float>;

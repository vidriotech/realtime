#include "feature_extractor.cuh"

template<class T>
FeatureExtractor<T>::~FeatureExtractor() {
  if (cu_snippets_ != nullptr) {
    cudaFree(cu_snippets_);
    cu_snippets_ = nullptr;
  }

  if (cu_features_ != nullptr) {
    cudaFree(cu_features_);
    cu_features_ = nullptr;
  }
}

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

  snippets_ = std::move(snippets);
  n_feats_ = snippets_.at(0).size();
  auto n_samples = snippets_.size() * n_feats_;

  // free old device memory if necessary and allocate enough to accommodate
  // new snippet data
  if (cu_snippets_ != nullptr)
    cudaFree(cu_snippets_);
  cudaMallocManaged(&cu_snippets_, n_samples * sizeof(float));

  // copy snippet data to device
  std::vector<float> snippet_data(n_samples);
  for (auto i = 0; i < snippets_.size(); ++i) {
    auto snip = snippets_.at(i).data();
    auto offset = i * n_feats_;
    auto it = snippet_data.begin() + offset;
    std::copy(snip.begin(), snip.end(), it);
  }
  cudaMemcpy(cu_snippets_, snippet_data.data(), n_samples * sizeof(float),
             cudaMemcpyHostToDevice);
  snippet_data.clear();
}

/**
 * @brief Compute the covariance matrix of snippet features.
 */
template<class T>
void FeatureExtractor<T>::ComputeCovarianceMatrix() {
  if (n_feats_ == 0) {
    return;
  }

  cudaError_t stat;

  CenterSnippets();
  if (cu_features_ != nullptr)
    cudaFree(cu_features_);
  stat = cudaMallocManaged(&cu_features_,
                           n_feats_ * n_feats_ * sizeof(float));

  CovMatrixArgs args{snippets_.size(), n_feats_,
                     cu_snippets_, cu_features_};
  make_cov_matrix(args);
}

template<class T>
void FeatureExtractor<T>::ProjectSnippets() {
  auto n_pcs = params_.extract.n_pcs;

  MakePVArgs pv_args{n_feats_, n_pcs, cu_features_};
  make_principal_vectors(pv_args);

  ProjectOntoPVsArgs project_args{n_pcs,
                                  n_feats_,
                                  n_obs(),
                                  cu_features_,
                                  cu_snippets_};
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

//  auto n_samples = n_feats_ * snippets_.size();
//  CenterFeaturesArgs args{snippets_.size(), n_feats_, cu_features_};
//  center_features(args);
//
//  std::vector<float> feats(n_samples);
//  cudaMemcpy(feats.data(), cu_features_, n_samples * sizeof(float),
//             cudaMemcpyDeviceToHost);
//  feats.assign(dev_snippets_.begin(), dev_snippets_.end());
}

template
class FeatureExtractor<short>;

template
class FeatureExtractor<float>;

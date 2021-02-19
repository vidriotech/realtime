#ifndef RTS_2_SRC_KERNELS_PCA_CUH_
#define RTS_2_SRC_KERNELS_PCA_CUH_

#include <cublas_v2.h>

struct CovMatrixArgs {
  int n_obs;
  int n_feats;
  float *features;
  float *cov_matrix;
};

void cov_matrix(CovMatrixArgs &args);

#endif //RTS_2_SRC_KERNELS_PCA_CUH_

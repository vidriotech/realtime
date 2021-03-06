#ifndef RTS_SRC_CLUSTERING_DP_CLUSTERING_H_
#define RTS_SRC_CLUSTERING_DP_CLUSTERING_H_

#include <vector>

#include "../extraction/snippet.cuh"

template<class T>
class DPClustering {
 public:
  explicit DPClustering(std::vector<Snippet> &snippets)
      : snippets_(snippets) {};

 private:
  std::vector<Snippet> &snippets_;
};

#endif //RTS_SRC_CLUSTERING_DP_CLUSTERING_H_

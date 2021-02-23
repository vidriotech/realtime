#ifndef RTS_SRC_CLUSTERING_DP_CLUSTERING_H_
#define RTS_SRC_CLUSTERING_DP_CLUSTERING_H_

#include <vector>

#include "../extraction/snippet.h"

template<class T>
class DPClustering {
 public:
  explicit DPClustering(std::vector<Snippet<T>> &snippets)
      : snippets_(snippets) {};

 private:
  std::vector<Snippet<T>> &snippets_;
};

#endif //RTS_SRC_CLUSTERING_DP_CLUSTERING_H_

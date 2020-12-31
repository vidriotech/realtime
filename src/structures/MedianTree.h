#ifndef RTS_2_MEDIANTREE_H
#define RTS_2_MEDIANTREE_H

#include "MedianTreeNode.h"

template <class T>
class MedianTree {
public:
    explicit MedianTree(T a, T b)
        : lte(a <= b ? a : b), gt(a > b ? a : b){};

private:
    MedianTreeNode<T> lte;
    MedianTreeNode<T> gt;
};


#endif //RTS_2_MEDIANTREE_H

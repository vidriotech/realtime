#ifndef RTS_2_MEDIANTREE_H
#define RTS_2_MEDIANTREE_H

#include "MedianTreeNode.h"

#define MIN(a, b) ((a) <= (b) ? (a) : (b))

#define SHIFT_LTR -1
#define SHIFT_RTL 1

template <class T>
class MedianTree {
public:
    explicit MedianTree(T a, T b)
        : left(new MedianTreeNode<T>(MIN(a, b))),
          right(new MedianTreeNode<T>(MAX(a, b))) {

        right_min = right->value();
        left_max = left->value();
    };
    float median();

    void insert(T val);
    short remove(T val);

    // getters
    unsigned height();
    int balance();
    unsigned n_elements();

private:
    std::unique_ptr<MedianTreeNode<T>> left;
    std::unique_ptr<MedianTreeNode<T>> right;

    T right_min; // the min value of the greater-than subtree
    T left_max; // the max value of the less-than-or-equal subtree

    void shift_value(short dir);
};

template<class T>
float MedianTree<T>::median() {
    float med;

    if (right->n_elements() > left->n_elements()) {
        med = right_min;
    } else if (left->n_elements() > right->n_elements()) {
        med = left_max;
    } else {
        med = (right_min + left_max) / 2.0f;
    }

    return med;
}

template<class T>
unsigned MedianTree<T>::height() {
    return 1 + MAX(left->height(), right->height());
}

template<class T>
int MedianTree<T>::balance() {
    return left->n_elements() - right->n_elements();
}

template<class T>
unsigned MedianTree<T>::n_elements() {
    return left->n_elements() + right->n_elements();
}

template<class T>
void MedianTree<T>::insert(T val) {
    if (val <= median()) {
        left->insert(val);
        left_max = MAX(left_max, val);

        if (balance() > AVL_LEFT_HEAVY) {
            shift_value(SHIFT_LTR);
        }
    } else {
        right->insert(val);
        right_min = MIN(right_min, val);

        if (balance() < -AVL_RIGHT_HEAVY) {
            shift_value(SHIFT_RTL);
        }
    }
}

template<class T>
short MedianTree<T>::remove(T val) {
    short res;

    if (val <= median()) {
        res = left->remove(val);

        if (balance() < AVL_RIGHT_HEAVY) {
            shift_value(SHIFT_RTL);
        }
    } else {
        res = right->remove(val);

        if (balance() > AVL_LEFT_HEAVY) {
            shift_value(SHIFT_LTR);
        }
    }

    return res;
}

template<class T>
void MedianTree<T>::shift_value(short dir) {
    if (dir == SHIFT_LTR) {
        if (left->value() == left_max) {
            std::unique_ptr<MedianTreeNode<T>> left_child(nullptr);
            std::unique_ptr<MedianTreeNode<T>> right_child(nullptr);

            left_child.swap(left->left);
            right_child.swap(left->right);

            left.swap(left_child);
            left->insert_node(std::move(right_child));

            left_child.reset(); // free up old left pointer
        } else {
            left->remove(left_max);
        }

        right->insert(left_max);
    } else if (dir == SHIFT_RTL) {
        if (right->value() == right_min) {
            std::unique_ptr<MedianTreeNode<T>> left_child(nullptr);
            std::unique_ptr<MedianTreeNode<T>> right_child(nullptr);

            left_child.swap(right->left);
            right_child.swap(right->right);

            right.swap(right_child);
            right->insert_node(std::move(left_child));

            right_child.reset(); // free up old right pointer
        } else {
            right->remove(right_min);
        }

        left->insert(right_min);
    }

    left_max = left->max();
    right_min = right->min();
}

#endif //RTS_2_MEDIANTREE_H

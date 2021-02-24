#ifndef RTS_2_MEDIAN_TREE_NODE_H
#define RTS_2_MEDIAN_TREE_NODE_H

#include <algorithm>
#include <memory>
#include <utility>

template <class T>
class MedianTreeNode {
public:
    explicit MedianTreeNode(T val) : data(val), ht(1), n(1) {};

    // Insert and remove elements/subtrees
    short Insert(T val, bool rotate = true);
    short
    InsertSubtree(std::shared_ptr<MedianTreeNode<T>> node, bool rotate = true);
    short Remove(T val, bool rotate = true);
    std::shared_ptr<MedianTreeNode<T>> DetachSubtree(short child);

    // rotate child elements
    void RotateChildren(bool recursive = false);

    // update counts and heights of subtrees
    void UpdatePopulation();

    // getters
    [[nodiscard]] short balance();
    /**
     * @brief Get the value stored in this node.
     * @return The value stored in this node.
     */
    [[nodiscard]] T value() const { return data; };
    /**
     * @brief Get the number of elements in the subtree rooted at this node.
     * @return The number of elements in the subtree rooted at this node.
     */
    [[nodiscard]] unsigned count() const { return n; };
    /**
     * @brief Get the height of the subtree rooted at this node.
     * @return The height of the subtree rooted at this node.
     */
    [[nodiscard]] unsigned short height() const { return ht; };
    /**
     * @brief Get a pointer to the left subtree of this node.
     * @return Pointer to the left subtree of this node.
     */
    [[nodiscard]] std::shared_ptr<MedianTreeNode<T>>
    left() const { return lt; };
    /**
     * @brief Get a pointer to the right subtree of this node.
     * @return Pointer to the right subtree of this node.
     */
    [[nodiscard]] std::shared_ptr<MedianTreeNode<T>>
    right() const { return rt; };
    /**
     * @brief Return the maximum value in the subtree rooted at this node.
     * @return The maximum value in this subtree.
     */
    [[nodiscard]] T max() const { return rt == nullptr ? data : rt->max(); };
    /**
     * @brief Return the minimum value in the subtree rooted at this node.
     * @return The minimum value in this subtree.
     */
    [[nodiscard]] T min() const { return lt == nullptr ? data : lt->min(); };
private:
    T data; /*!< The samples_ in this node.  */
    std::shared_ptr<MedianTreeNode<T>> lt; /*!< Left child. */
    std::shared_ptr<MedianTreeNode<T>> rt; /*!< Right node. */

    unsigned short n; /*!< Number of elements in the subtree. */
    unsigned short ht; /*!< The height of the subtree rooted at this node. */

    void
    RemoveChild(std::shared_ptr<MedianTreeNode<T>> child, bool rotate = true);

    std::shared_ptr<MedianTreeNode<T>>
    RotateChild(std::shared_ptr<MedianTreeNode<T>> child);
    std::shared_ptr<MedianTreeNode<T>>
    LLRotate(std::shared_ptr<MedianTreeNode<T>> child);
    std::shared_ptr<MedianTreeNode<T>>
    LRRotate(std::shared_ptr<MedianTreeNode<T>> child);
    std::shared_ptr<MedianTreeNode<T>>
    RLRotate(std::shared_ptr<MedianTreeNode<T>> child);
    std::shared_ptr<MedianTreeNode<T>>
    RRRotate(std::shared_ptr<MedianTreeNode<T>> child);

    void UpdateCount();
    void UpdateHeight();
};

/**
 * @brief Insert a value into the subtree rooted at this node.
 * @tparam T The type of the value in this node.
 * @param val The value to Insert in the left or right subtree.
 * @return The updated height of the subtree rooted at this node.
 */
template<class T>
short MedianTreeNode<T>::Insert(T val, bool rotate) {
    if (val <= data) {
        if (lt == nullptr) {
            lt.reset(new MedianTreeNode<T>(val));
        } else {
            lt->Insert(val, rotate);
        }
    } else {
        if (rt == nullptr) {
            rt.reset(new MedianTreeNode<T>(val));
        } else {
            rt->Insert(val, rotate);
        }
    }

    UpdatePopulation();

    if (rotate) {
        RotateChildren(false);
    }

    return ht;
}

/**
 * @brief Insert a subtree into the subtree rooted at this node.
 * @tparam T The type of samples_ stored in the nodes of this subtree.
 * @param node Root of the subtree to Insert into the tree rooted at this node.
 * @return The updated height of the subtree rooted at this node.
 */
template<class T>
short
MedianTreeNode<T>::InsertSubtree(std::shared_ptr<MedianTreeNode<T>> node,
                                 bool rotate) {
    if (node == nullptr)
        return ht;

    if (node->value() <= data) {
        if (lt == nullptr) {
            lt.swap(node);
        } else {
            lt->InsertSubtree(node, rotate);
        }
    } else {
        if (rt == nullptr) {
            rt.swap(node);
        } else {
            rt->InsertSubtree(node, rotate);
        }
    }

    UpdatePopulation();

    if (rotate) {
        RotateChildren(false);
    }

    return ht;
}

/**
 * @brief Remove a value from the subtree rooted at this node.
 * @tparam T The type of the value in this node.
 * @param val The value to remove from the subtree rooted at this node.
 * @return 0 if value successfully found and removed, 1 otherwise.
 */
template<class T>
short MedianTreeNode<T>::Remove(T val, bool rotate) {
    short res = 1;

    if (val <= data && lt != nullptr) {
        if (lt->value() == val) {
            RemoveChild(std::move(lt), rotate);
            res = 0;
        } else {
            res = lt->Remove(val, rotate);
        }
    } else if (val > data && rt != nullptr) {
        if (rt->value() == val) {
            RemoveChild(std::move(rt), rotate);
            res = 0;
        } else {
            res = rt->Remove(val, rotate);
        }
    }

    UpdatePopulation();

    if (rotate) {
        RotateChildren(false);
    }

    return res;
}

/**
 * @brief Detach and return the left or right subtree.
 * @tparam T The type of samples_ stored in the nodes of the subtree.
 * @param child -1 for right child, 1 for left child
 * @return Pointer to the (former) subtree.
 */
template<class T>
std::shared_ptr<MedianTreeNode<T>>
MedianTreeNode<T>::DetachSubtree(short child) {
    std::shared_ptr<MedianTreeNode<T>> res;

    if (child == -1) { // right child
        res.swap(rt);
    } else if (child == 1) { // left child
        res.swap(lt);
    }

    UpdatePopulation();

    return res;
}

/**
 * @brief Perform tree rotations on left and right child elements, if necessary.
 * @tparam T The type of the value stored in the elements of the subtree.
 */
template<class T>
void MedianTreeNode<T>::RotateChildren(bool recursive) {
    if (recursive) {
        if (lt != nullptr)
            lt->RotateChildren(recursive);

        if (rt != nullptr)
            rt->RotateChildren(recursive);
    }

    lt = RotateChild(std::move(lt));
    rt = RotateChild(std::move(rt));

    UpdatePopulation();
}

/**
 * @brief Update element count and subtree height.
 * @tparam T T The type of samples_ stored in the nodes of this subtree.
 */
template<class T>
void MedianTreeNode<T>::UpdatePopulation() {
    UpdateCount();
    UpdateHeight();
}

/**
 * @brief Remove an immediate child of this node.
 *
 * Only the value in the node is removed. Subtrees of `child` will be
 * reinserted back into this subtree. We do NOT check here that `child` is
 * actually a child of this node. This is the responsibility of the caller.
 *
 * @tparam T The type of samples_ stored in the nodes of this subtree.
 * @param child Pointer to the child to remove.
 */
template<class T>
void MedianTreeNode<T>::RemoveChild(std::shared_ptr<MedianTreeNode<T>> child,
                                    bool rotate) {
    auto left_child = child->left();
    auto right_child = child->right();

    child.reset();
    InsertSubtree(left_child, rotate);
    InsertSubtree(right_child, rotate);
}

/**
 * @brief Perform an LL, LR, RL, or RR rotation on a child node if it needs one.
 * @tparam T The type of samples_ stored in the nodes of this subtree.
 * @param child Pointer to the child to rotate.
 * @return Root of the subtree to be inserted into child's former place.
 */
template<class T>
std::shared_ptr<MedianTreeNode<T>>
MedianTreeNode<T>::RotateChild(std::shared_ptr<MedianTreeNode<T>> child) {
    if (child == nullptr) {
        return child;
    }

    if (child->balance() >= 2) {
        if (child->left()->balance() >= 1) {
            child = LLRotate(std::move(child));
        } else if (child->left()->balance() <= -1) {
            child = LRRotate(std::move(child));
        }
    } else if (child->balance() <= -2) {
        if (child->right()->balance() >= 1) {
            child = RLRotate(std::move(child));
        }
        else if (child->right() != nullptr && child->right()->balance() <= -1) {
            child = RRRotate(std::move(child));
        }
    } else {
        return child;
    }

    if (child->left() != nullptr) {
        child->left()->UpdatePopulation();
    }

    if (child->right() != nullptr) {
        child->right()->UpdatePopulation();
    }

    child->UpdatePopulation();

    return child;
}

/**
 * @brief Perform an LL rotation on a child node.
 * @tparam T The type of samples_ stored in the nodes of this subtree.
 * @param child The child needing a rotation.
 * @return Root of the subtree to be inserted into child's former place.
 */
template<class T>
std::shared_ptr<MedianTreeNode<T>>
MedianTreeNode<T>::LLRotate(std::shared_ptr<MedianTreeNode<T>> child) {
    std::shared_ptr<MedianTreeNode<T>> tmp;

    tmp.swap(child->lt);
    tmp->rt.swap(child->lt);
    tmp->rt.swap(child);
    tmp.swap(child);

    return child;
}

/**
 * @brief Perform an LR rotation on a child node.
 * @tparam T The type of samples_ stored in the nodes of this subtree.
 * @param child The child needing a rotation.
 * @return Root of the subtree to be inserted into child's former place.
 */
template<class T>
std::shared_ptr<MedianTreeNode<T>>
MedianTreeNode<T>::LRRotate(std::shared_ptr<MedianTreeNode<T>> child) {
    std::shared_ptr<MedianTreeNode<T>> tmp;

    tmp.swap(child->lt);
    tmp->rt->rt.swap(child->lt);
    tmp->rt->rt.swap(child);
    tmp->rt.swap(child);
    tmp->rt.swap(child->lt);
    tmp.swap(child->lt);

    return child;
}

/**
 * @brief Perform an RL rotation on a child node.
 * @tparam T The type of samples_ stored in the nodes of this subtree.
 * @param child The child needing a rotation.
 * @return Root of the subtree to be inserted into child's former place.
 */
template<class T>
std::shared_ptr<MedianTreeNode<T>>
MedianTreeNode<T>::RLRotate(std::shared_ptr<MedianTreeNode<T>> child) {
    std::shared_ptr<MedianTreeNode<T>> tmp;

    tmp.swap(child->rt);
    tmp->lt->lt.swap(child->rt);
    tmp->lt->lt.swap(child);
    tmp->lt.swap(child);
    tmp->lt.swap(child->rt);
    tmp.swap(child->rt);

    return child;
}

/**
 * @brief Perform an RR rotation on a child node.
 * @tparam T The type of samples_ stored in the nodes of this subtree.
 * @param child The child needing a rotation.
 * @return Root of the subtree to be inserted into child's former place.
 */
template<class T>
std::shared_ptr<MedianTreeNode<T>>
MedianTreeNode<T>::RRRotate(std::shared_ptr<MedianTreeNode<T>> child) {
    std::shared_ptr<MedianTreeNode<T>> tmp;

    tmp.swap(child->rt);
    tmp->lt.swap(child->rt);
    tmp->lt.swap(child);
    tmp.swap(child);

    return child;
}

/**
 * @brief Update the height of the subtree rooted at this node.
 * @tparam T The type of samples_ stored in the nodes of this subtree.
 */
template<class T>
void MedianTreeNode<T>::UpdateHeight() {
    auto left_height = lt == nullptr ? 0 : lt->height();
    auto right_height = rt == nullptr ? 0 : rt->height();

    ht = 1 + std::max(left_height, right_height);
}

/**
 * @brief Update the number of elements in the subtree rooted at this node.
 * @tparam T The type of samples_ stored in the nodes of this subtree.
 */
template<class T>
void MedianTreeNode<T>::UpdateCount() {
    auto left_count = lt == nullptr ? 0 : lt->count();
    auto right_count = rt == nullptr ? 0 : rt->count();

    n = 1 + left_count + right_count;
}

/**
 * @brief Compute the balance between the heights of the left and right
 * subtrees.
 *
 * Balance factor is defined to be the difference between the height of the
 * left subtree and that of the right subtree. A tree is said to be
 * "left-heavy" if the subtree has a larger height than the right subtree.
 * "Right-heavy" is analogously defined. In AVL trees, a tree's balance,
 * together with that of its subtrees, is used to determine when to rotate.
 *
 * @tparam T The type of samples_ stored in the nodes of this subtree.
 * @return The balance factor.
 */
template<class T>
short MedianTreeNode<T>::balance() {
    auto left_height = lt == nullptr ? 0 : lt->height();
    auto right_height = rt == nullptr ? 0 : rt->height();

    return left_height - right_height;
}

#endif //RTS_2_MEDIAN_TREE_NODE_H

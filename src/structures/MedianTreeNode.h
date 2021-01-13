#ifndef RTS_2_MEDIANTREENODE_H
#define RTS_2_MEDIANTREENODE_H

#include <algorithm>
#include <memory>
#include <utility>

template <class T>
class MedianTreeNode {
public:
    MedianTreeNode(T val) : data(val), ht(1), n(1) {};

    short insert(T val);
    short insert_subtree(std::shared_ptr<MedianTreeNode<T>> node);
    short remove(T val);

    // getters
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
    [[nodiscard]] short balance();
    /**
     * @brief Get a pointer to the left subtree of this node.
     * @return Pointer to the left subtree of this node.
     */
    [[nodiscard]] std::shared_ptr<MedianTreeNode<T>> left() const { return lt; };
    /**
     * @brief Get a pointer to the right subtree of this node.
     * @return Pointer to the right subtree of this node.
     */
    [[nodiscard]] std::shared_ptr<MedianTreeNode<T>> right() const { return rt; };
private:
    T data; /*!< The data in this node.  */
    std::shared_ptr<MedianTreeNode<T>> lt; /*!< Left child. */
    std::shared_ptr<MedianTreeNode<T>> rt; /*!< Right node. */

    unsigned short n; /*!< Number of elements in the subtree. */
    unsigned short ht; /*!< The height of the subtree rooted at this node. */

    void remove_child(std::shared_ptr<MedianTreeNode<T>> child);
    void update_height();
    void update_count();
};

/**
 * @brief Insert a value into the subtree rooted at this node.
 *
 * @tparam T The type of the value in this node.
 * @param val The value to insert in the left or right subtree.
 * @return The updated height of the subtree rooted at this node.
 */
template<class T>
short MedianTreeNode<T>::insert(T val) {
    if (val <= data) {
        if (lt == nullptr) {
            lt.reset(new MedianTreeNode<T>(val));
        } else {
            lt->insert(val);
        }
    } else {
        if (rt == nullptr) {
            rt.reset(new MedianTreeNode<T>(val));
        } else {
            rt->insert(val);
        }
    }

    update_count();
    update_height();
    return ht;
}

/**
 * @brief Insert a subtree into the subtree rooted at this node.
 *
 * @tparam T The type of data stored in the nodes of this subtree.
 * @param node Root of the subtree to insert into the tree rooted at this node.
 * @return The updated height of the subtree rooted at this node.
 */
template<class T>
short MedianTreeNode<T>::insert_subtree(std::shared_ptr<MedianTreeNode<T>> node) {
    if (node == nullptr)
        return ht;

    auto ct = node->count();
    if (node->value() <= data) {
        if (lt == nullptr) {
            lt.swap(node);
        } else {
            lt->insert_subtree(node);
        }
    } else {
        if (rt == nullptr) {
            rt.swap(node);
        } else {
            rt->insert_subtree(node);
        }
    }

    update_count();
    update_height();
    return ht;
}

/**
 * @brief Remove a value from the subtree rooted at this node.
 *
 * @tparam T The type of the value in this node.
 * @param val The value to remove the subtree rooted at this node.
 * @return 0 if value successfully found and removed, 1 otherwise.
 */
template<class T>
short MedianTreeNode<T>::remove(T val) {
    short res = 1;

    if (val <= data && lt != nullptr) {
        if (lt->value() == val) {
            remove_child(std::move(lt));
            res = 0;
        } else {
            res = lt->remove(val);
        }
    } else if (val > data && rt != nullptr) {
        if (rt->value() == val) {
            remove_child(std::move(rt));
            res = 0;
        } else {
            res = rt->remove(val);
        }
    }

    update_count();
    update_height();
    return res;
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
 * @tparam T The type of data stored in the nodes of this subtree.
 * @return The balance factor.
 */
template<class T>
short MedianTreeNode<T>::balance() {
    auto left_height = lt == nullptr ? 0 : lt->height();
    auto right_height = rt == nullptr ? 0 : rt->height();

    return left_height - right_height;
}

/**
 * @brief Remove an immediate child of this node.
 *
 * Only the value in the node is removed. Subtrees of `child` will be
 * reinserted back into this subtree. We do NOT check here that `child` is
 * actually a child of this node. This is the responsibility of the caller.
 *
 * @tparam T The type of data stored in the nodes of this subtree.
 * @param child Pointer to the child to remove.
 */
template<class T>
void MedianTreeNode<T>::remove_child(std::shared_ptr<MedianTreeNode<T>> child) {
    auto left_child = child->left();
    auto right_child = child->right();

    child.reset();
    insert_subtree(left_child);
    insert_subtree(right_child);
}

/**
 * @brief Update the height of the subtree rooted at this node.
 * @tparam T The type of data stored in the nodes of this subtree.
 */
template<class T>
void MedianTreeNode<T>::update_height() {
    auto left_height = lt == nullptr ? 0 : lt->height();
    auto right_height = rt == nullptr ? 0 : rt->height();

    ht = 1 + std::max(left_height, right_height);
}

/**
 * @brief Update the number of elements in the subtree rooted at this node.
 * @tparam T The type of data stored in the nodes of this subtree.
 */
template<class T>
void MedianTreeNode<T>::update_count() {
    auto left_count = lt == nullptr ? 0 : lt->count();
    auto right_count = rt == nullptr ? 0 : rt->count();

    n = 1 + left_count + right_count;
}

#endif //RTS_2_MEDIANTREENODE_H

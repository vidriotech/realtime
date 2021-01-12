#ifndef RTS_2_MEDIANTREENODE_H
#define RTS_2_MEDIANTREENODE_H

#include <algorithm>
#include <memory>
#include <utility>

template <class T>
class MedianTreeNode {
public:
    MedianTreeNode(T val) : data(val), ht(1), bal(0) {};

    short insert(T val);

    // getters
    /**
     * @return The value stored in this node.
     */
    [[nodiscard]] T value() const { return data; };
    /**
     * @return The height of the subtree rooted at this node.
     */
    [[nodiscard]] unsigned short height() const { return ht; };
    /**
     * @return The height of the subtree rooted at this node.
     */
    [[nodiscard]] short balance() const { return bal; };
    /**
     * @return Pointer to the left subtree of this node.
     */
    [[nodiscard]] std::shared_ptr<MedianTreeNode<T>> left() const { return lt; };
    /**
     * @return Pointer to the right subtree of this node.
     */
    [[nodiscard]] std::shared_ptr<MedianTreeNode<T>> right() const { return rt; };
private:
    T data; /*!< The data in this node.  */
    std::shared_ptr<MedianTreeNode<T>> lt; /*!< Left child. */
    std::shared_ptr<MedianTreeNode<T>> rt; /*!< Right node. */

    unsigned short ht; /*!< The height of the subtree rooted at this node. */
    short bal; /*!< This subtree's balance factor. */
};

/**
 * Insert a value into the subtree rooted at this node.
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

    // update tree height
    auto left_height = lt == nullptr ? 0 : lt->height();
    auto right_height = rt == nullptr ? 0 : rt->height();
    ht = 1 + std::max(left_height, right_height);

    // update tree balance
    bal = left_height - right_height;

    return ht;
}


#endif //RTS_2_MEDIANTREENODE_H

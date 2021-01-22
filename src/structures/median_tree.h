#ifndef RTS_2_MEDIAN_TREE_H
#define RTS_2_MEDIAN_TREE_H

#include <algorithm>
#include <memory>
#include <utility>
//#include <iostream>

#include "median_tree_node.h"

template<class T>
class MedianTree {
 public:
  MedianTree() = default;;
  explicit MedianTree(T a)
      : lt(new MedianTreeNode<T>(a)), left_max(a) {};
  MedianTree(T a, T b);

  // Insert and remove elements
  void Insert(T val, bool balance = true, bool rotate = true);
  short Remove(T val, short subtree = 0);

  // rotate and rebalance subtrees
  void BalanceElements();
  void RotateSubtrees(bool recursive = false);

  // getters
  /**
   * @brief Get a pointer to the left, or smaller, subtree.
   * @return Pointer to the left subtree.
   */
  std::shared_ptr<MedianTreeNode<T>> left() const { return lt; };
  /**
   * @brief Get a pointer to the right, or larger, subtree.
   * @return Pointer to the right subtree.
   */
  std::shared_ptr<MedianTreeNode<T>> right() const { return rt; };
  /**
   * @brief Get the number of elements in the left subtree.
   * @return The number of elements in the left subtree.
   */
  [[nodiscard]] int
  count_left() const { return lt == nullptr ? 0 : lt->count(); };
  /**
   * @brief Get the number of elements in the right subtree.
   * @return The number of elements in the right subtree.
   */
  [[nodiscard]] int
  count_right() const { return rt == nullptr ? 0 : rt->count(); };
  /**
   * @brief Get the number of elements in the tree.
   * @return The sum of the number of elements in both subtrees.
   */
  [[nodiscard]] unsigned
  count() const { return count_left() + count_right(); };
  [[nodiscard]] unsigned short height();
  [[nodiscard]] short balance();
  [[nodiscard]] short el_balance();
  [[nodiscard]] float median() const;
 private:
  std::shared_ptr<MedianTreeNode<T>> lt; /*!< Left subtree. */
  std::shared_ptr<MedianTreeNode<T>> rt; /*!< Right subtree. */

  T left_max = 0; /*!< Maximum value of the left subtree. */
  T right_min = 0; /*!< Minimum value of the right subtree. */

  std::shared_ptr<MedianTreeNode<T>> RemoveRoot(std::shared_ptr<MedianTreeNode<T>> root);

  void ShiftLTR();
  void ShiftRTL();

  std::shared_ptr<MedianTreeNode<T>>
  RotateSubtree(std::shared_ptr<MedianTreeNode<T>> subtree);
  std::shared_ptr<MedianTreeNode<T>>
  LLRotate(std::shared_ptr<MedianTreeNode<T>> subtree);
  std::shared_ptr<MedianTreeNode<T>>
  LRRotate(std::shared_ptr<MedianTreeNode<T>> subtree);
  std::shared_ptr<MedianTreeNode<T>>
  RLRotate(std::shared_ptr<MedianTreeNode<T>> subtree);
  std::shared_ptr<MedianTreeNode<T>>
  RRRotate(std::shared_ptr<MedianTreeNode<T>> subtree);
};

/**
 * @brief Construct a MedianTree with two elements, one for each subtree.
 * @tparam T The type of values stored in this tree.
 * @param a A value to store in the tree.
 * @param b Another value to store in the tree.
 */
template<class T>
MedianTree<T>::MedianTree(T a, T b) {
  auto min_val = std::min(a, b);
  auto max_val = std::max(a, b);

  lt.reset(new MedianTreeNode<T>(min_val));
  left_max = min_val;

  rt.reset(new MedianTreeNode<T>(max_val));
  right_min = max_val;
}

/**
 * @brief Insert a value into this tree.
 * @tparam T The type of data_ stored in the nodes of this tree.
 * @param val The value to insert.
 */
template<class T>
void MedianTree<T>::Insert(T val, bool balance, bool rotate) {
  if (val <= median()) {
    if (lt == nullptr) {
      lt.reset(new MedianTreeNode<T>(val));
    } else {
      lt->Insert(val, rotate);
    }

    left_max = count_left() == 1 ? val : std::max(left_max, val);
  } else {
    if (rt == nullptr) {
      rt.reset(new MedianTreeNode<T>(val));
    } else {
      rt->Insert(val, rotate);
    }

    right_min = count_right() == 1 ? val : std::min(right_min, val);
  }

  // rebalance elements if necessary
  if (balance) {
    BalanceElements();
  }

  // perform rotations if necessary
  if (rotate) {
    RotateSubtrees();
  }
}

/**
 * @brief Remove a value from this tree.
 * @tparam T The type of the data_ stored in the nodes of this tree.
 * @param val The value to remove from this tree.
 * @param subtree Force remove from left subtree if 1, right subtree if -1,
 *                or detect if 0.
 * @return 0 if value successfully found and removed, 1 otherwise.
 */
template<class T>
short MedianTree<T>::Remove(T val, short subtree) {
  short res = 1;

  if (lt != nullptr && (subtree == 1 || (subtree == 0 && val <= left_max))) {
    if (lt->value() == val) {
      lt = RemoveRoot(std::move(lt));
      res = 0;
    } else { // search for the value in the left subtree
      res = lt->Remove(val, true);
    }

    // update right_min if we need to
    if (res == 0 && val == left_max) {
      left_max = lt == nullptr ? 0 : lt->max();
    }
  } else if (rt != nullptr &&
      (subtree == -1 || (subtree == 0 && val >= right_min))) {
    if (rt->value() == val) {
      // actually remove the value at the root node
      rt = RemoveRoot(std::move(rt));
      res = 0;
    } else { // search for the value in the right subtree
      res = rt->Remove(val, true);
    }

    // update right_min if we need to
    if (res == 0 && val == right_min) {
      right_min = rt == nullptr ? 0 : rt->min();
    }
  }

  return res;
}

/**
 * @brief Perform tree rotations on left and right subtrees, if necessary.
 * @tparam T The type of the value stored in the elements of each subtree.
 */
template<class T>
void MedianTree<T>::RotateSubtrees(bool recursive) {
  if (recursive) {
    if (lt != nullptr)
      lt->RotateChildren(recursive);

    if (rt != nullptr)
      rt->RotateChildren(recursive);
  }

  lt = RotateSubtree(std::move(lt));
  rt = RotateSubtree(std::move(rt));
}

/**
 * @brief Shift elements between subtrees to maintain a difference of no more
 *        than a single element.
 * @tparam T The type of data_ stored in the nodes of this tree.
 */
template<class T>
void MedianTree<T>::BalanceElements() {
  if (-2 < el_balance() && el_balance() < 2) { // tree is already balanced
    return;
  }

  if (el_balance() < -1) {
    ShiftRTL();
  }

  if (el_balance() > 1) {
    ShiftLTR();
  }
}

/**
 * @brief Compute and return the height of this tree.
 * @tparam T The type of data_ stored in the nodes of this tree.
 * @return The height of this tree.
 */
template<class T>
unsigned short MedianTree<T>::height() {
  auto left_height = lt == nullptr ? 0 : lt->height();
  auto right_height = rt == nullptr ? 0 : rt->height();

  return 1 + std::max(left_height, right_height);
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
 * @tparam T The type of data_ stored in the nodes of this tree.
 * @return The balance factor.
 */
template<class T>
short MedianTree<T>::balance() {
  auto left_height = lt == nullptr ? 0 : lt->height();
  auto right_height = rt == nullptr ? 0 : rt->height();

  return left_height - right_height;
}

/**
 * @brief Compute the balance between the counts of the left and right subtrees.
 * @tparam T The type of data_ stored in the nodes of this tree.
 * @return The difference between the left tree's count and the right tree's
 * count.
 */
template<class T>
short MedianTree<T>::el_balance() {
  auto left_count = lt == nullptr ? 0 : lt->count();
  auto right_count = rt == nullptr ? 0 : rt->count();

  return left_count - right_count;
}

/**
 * @brief Compute and return the median of all values in this tree.
 * @tparam T The type of data_ stored in the nodes of this tree.
 * @return The median of all values.
 */
template<class T>
float MedianTree<T>::median() const {
  float med;

  if (count_left() == count_right()) {
    med = (left_max + right_min) / 2.0f;
  } else if (count_left() > count_right()) {
    med = left_max;
  } else { // count_right() > count_left()
    med = right_min;
  }

  return med;
}

/**
 * @brief Remove the root node of one of the subtrees.
 *
 * Only the root node is removed. Subtrees of `root` will be reinserted back
 * into
 *
 * @tparam T The type of data_ stored in the nodes of this tree.
 * @param root The node to remove.
 */
template<class T>
std::shared_ptr<MedianTreeNode<T>>
MedianTree<T>::RemoveRoot(std::shared_ptr<MedianTreeNode<T>> root) {
  auto left_child = root->left();
  auto right_child = root->right();

  if (left_child != nullptr) {
    root.swap(left_child);
    left_child.reset();

    root->InsertSubtree(right_child, false);
  } else {
    root.swap(right_child);
    right_child.reset();
  }

  return root;
}

/**
 * @brief Perform an LL, LR, RL, or RR rotation on a subtree if it needs one.
 * @tparam T The type of data_ stored in the subtree.
 * @param subtree Pointer to the subtree to rotate.
 * @return Root of the new subtree to be inserted into the old subtree's place.
 */
template<class T>
std::shared_ptr<MedianTreeNode<T>>
MedianTree<T>::RotateSubtree(std::shared_ptr<MedianTreeNode<T>> subtree) {
  if (subtree == nullptr) {
    return subtree;
  }

  if (subtree->balance() >= 2) {
    if (subtree->left()->balance() >= 1) {
      subtree = LLRotate(std::move(subtree));
    } else if (subtree->left()->balance() <= -1) {
      subtree = LRRotate(std::move(subtree));
    }
  } else if (subtree->balance() <= -2) {
    if (subtree->right()->balance() >= 1) {
      subtree = RLRotate(std::move(subtree));
    } else if (subtree->right() != nullptr
        && subtree->right()->balance() <= -1) {
      subtree = RRRotate(std::move(subtree));
    }
  } else {
    return subtree;
  }

  if (subtree->left() != nullptr) {
    subtree->left()->UpdatePopulation();
  }

  if (subtree->right() != nullptr) {
    subtree->right()->UpdatePopulation();
  }

  subtree->UpdatePopulation();

  return subtree;
}

/**
 * @brief Shift the largest value in the left subtree to the right subtree.
 * @tparam T The type of data_ stored in the nodes of this tree.
 */
template<class T>
void MedianTree<T>::ShiftLTR() {
  auto max_val = left_max;
  auto res = Remove(max_val, 1); // resets left_max for us
  if (res == 1) // failure
    return;

  if (rt == nullptr) {
    rt.reset(new MedianTreeNode<T>(max_val));
    right_min = max_val;
  } else {
    rt->Insert(max_val, true);
    right_min = std::min(max_val, right_min);
  }
}

/**
 * @brief Shift the smallest value in the right subtree to the left subtree.
 * @tparam T T The type of data_ stored in the nodes of this tree.
 */
template<class T>
void MedianTree<T>::ShiftRTL() {
  auto min_val = right_min;
  auto res = Remove(min_val, -1); // resets right_min for us
  if (res == 1) // failure
    return;

  if (lt == nullptr) {
    lt.reset(new MedianTreeNode<T>(min_val));
    left_max = min_val;
  } else {
    lt->Insert(min_val, true);
    left_max = std::max(min_val, left_max);
  }
}

/**
 * @brief Perform an LL rotation on a subtree node.
 * @tparam T The type of data_ stored in the nodes of this subtree.
 * @param subtree The subtree needing a rotation.
 * @return Root of the new subtree to be inserted into the old subtree's place.
 */
template<class T>
std::shared_ptr<MedianTreeNode<T>>
MedianTree<T>::LLRotate(std::shared_ptr<MedianTreeNode<T>> subtree) {
  std::shared_ptr<MedianTreeNode<T>> tmp;

  auto left_child = subtree->DetachSubtree(1);
  auto right_child = subtree->DetachSubtree(-1);

  subtree->InsertSubtree(left_child->DetachSubtree(-1)); // inserts on the left
  subtree->InsertSubtree(right_child); // inserts on the right

  tmp.swap(subtree);
  left_child->InsertSubtree(tmp);

  subtree.swap(left_child);

  return subtree;
}

/**
 * @brief Perform an LR rotation on a subtree node.
 * @tparam T The type of data_ stored in the nodes of this subtree.
 * @param subtree The subtree needing a rotation.
 * @return Root of the new subtree to be inserted into the old subtree's place.
 */
template<class T>
std::shared_ptr<MedianTreeNode<T>>
MedianTree<T>::LRRotate(std::shared_ptr<MedianTreeNode<T>> subtree) {
  auto left_child = subtree->DetachSubtree(1);
  auto tmp = left_child->DetachSubtree(-1);

  subtree->InsertSubtree(tmp->DetachSubtree(-1)); // inserts on the left
  left_child->InsertSubtree(tmp->DetachSubtree(1)); // inserts on the right

  tmp->InsertSubtree(left_child); // inserts on the left
  tmp->InsertSubtree(subtree); // inserts on the right

  subtree.swap(tmp);

  return subtree;
}

/**
 * @brief Perform an RL rotation on a subtree.
 * @tparam T The type of data_ stored in the nodes of this subtree.
 * @param subtree The subtree needing a rotation.
 * @return Root of the new subtree to be inserted into the old subtree's place.
 */
template<class T>
std::shared_ptr<MedianTreeNode<T>>
MedianTree<T>::RLRotate(std::shared_ptr<MedianTreeNode<T>> subtree) {
  auto right_child = subtree->DetachSubtree(-1);
  auto tmp = right_child->DetachSubtree(1);

  subtree->InsertSubtree(tmp->DetachSubtree(1)); // inserts on the right
  right_child->InsertSubtree(tmp->DetachSubtree(-1)); // inserts on the left

  tmp->InsertSubtree(right_child); // inserts on the right
  tmp->InsertSubtree(subtree); // inserts on the left

  subtree.swap(tmp);

  return subtree;
}

/**
 * @brief Perform an RR rotation on a subtree.
 * @tparam T The type of data_ stored in the nodes of this subtree.
 * @param subtree The subtree needing a rotation.
 * @return Root of the new subtree to be inserted into the old subtree's place.
 */
template<class T>
std::shared_ptr<MedianTreeNode<T>>
MedianTree<T>::RRRotate(std::shared_ptr<MedianTreeNode<T>> subtree) {
  std::shared_ptr<MedianTreeNode<T>> tmp;

  auto left_child = subtree->DetachSubtree(1);
  auto right_child = subtree->DetachSubtree(-1);

  // inserts on the right
  subtree->InsertSubtree(right_child->DetachSubtree(1));
  subtree->InsertSubtree(left_child); // inserts on the left

  tmp.swap(subtree);
  right_child->InsertSubtree(tmp);

  subtree.swap(right_child);

  return subtree;
}

#endif //RTS_2_MEDIAN_TREE_H

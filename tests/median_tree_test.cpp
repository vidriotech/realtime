#include "gtest/gtest.h"

#include "../src/structures/median_tree.h"

/*
 * DO construct a MedianTree `tree` with no arguments; AND
 * TEST THAT `tree` has a null pointer for its left subtree; AND
 *           the number of elements in its left subtree is 0; AND
 *           `tree` has a null pointer for its right subtree; AND
 *           the number of elements in its right subtree is 0; AND
 *           `tree` has 0 elements; AND
 *           the height of `tree` is 1; AND
 *           the height balance factor of `tree` is 0; AND
 *           the element-wise balance factor of `tree` is 0; AND
 *           the median of `tree` is 0.
 */
TEST(MedianTreeTest, InitialStateNoArgs) {
  MedianTree<short> tree;

  EXPECT_EQ(nullptr, tree.left());
  EXPECT_EQ(nullptr, tree.right());

  // check tree count, height, and balance factor
  EXPECT_EQ(0, tree.count());
  EXPECT_EQ(1, tree.height());
  EXPECT_EQ(0, tree.balance());

  // check element-wise balance and median
  EXPECT_EQ(0, tree.el_balance());
  EXPECT_EQ(0, tree.median());
}

/*
 * GIVEN an initial value v
 * DO construct a MedianTree `tree` with v as sole argument; AND
 * TEST THAT `tree` has 1 element in its left tree; AND
 *           the value at the root of the left subtree of `tree` is v; AND
 *           `tree` has a null pointer for its right tree; AND
 *           `tree` has 1 element; AND
 *           the height of `tree` is 2; AND
 *           the height balance factor of `tree` is 1; AND
 *           the element-wise balance factor of `tree` is 1; AND
 *           the median of `tree` is v.
 */
TEST(MedianTreeTest, InitialStateSingleArg) {
  short v = -1;
  MedianTree<short> tree(v);

  EXPECT_EQ(1, tree.count_left());
  ASSERT_NE(nullptr, tree.left());
  EXPECT_EQ(v, tree.left()->value());
  EXPECT_EQ(nullptr, tree.right());

  // check tree count, height, and balance factor
  EXPECT_EQ(1, tree.count());
  EXPECT_EQ(2, tree.height());
  EXPECT_EQ(1, tree.balance());

  // check element-wise balance and median
  EXPECT_EQ(1, tree.el_balance());
  EXPECT_EQ(v, tree.median());
}

/*
 * GIVEN two initial values u and v
 * DO construct a MedianTree `tree` with arguments u and v; AND
 * TEST THAT `tree` has 1 element in its left subtree; AND
 *           the value at the root of the left subtree is v; AND
 *           `tree` has 1 element in its right subtree; AND
 *           the value at the root of the right subtree is u; AND
 *           `tree` has 2 elements; AND
 *           the height of `tree` is 2; AND
 *           the height balance factor of `tree` is 0; AND
 *           the element-wise balance factor of `tree` is 0; AND
 *           the median of `tree` is the mean of u and v.
 */
TEST(MedianTreeTest, InitialStateTwoArgs) {
  short u = 0;
  auto v = u - 1;
  MedianTree<short> tree(u, v);

  EXPECT_EQ(1, tree.count_left());
  ASSERT_NE(nullptr, tree.left());
  EXPECT_EQ(v, tree.left()->value());

  EXPECT_EQ(1, tree.count_right());
  ASSERT_NE(nullptr, tree.right());
  EXPECT_EQ(u, tree.right()->value());

  // check tree count, height, and balance factor
  EXPECT_EQ(2, tree.count());
  EXPECT_EQ(2, tree.height());
  EXPECT_EQ(0, tree.balance());

  // check element-wise balance and median
  EXPECT_EQ(0, tree.el_balance());
  EXPECT_EQ((u + v) / 2.0, tree.median());
}

/*
 * GIVEN an empty MedianTree `tree` and a value v
 * DO insert v into `tree` AND
 * TEST THAT `tree` has 1 element in its left subtree; AND
 *           the value at the root of the left subtree of `tree` is v; AND
 *           `tree` has a null pointer for its right tree; AND
 *           `tree` has 1 element; AND
 *           the height of `tree` is 2; AND
 *           the height balance factor of `tree` is 1; AND
 *           the element-wise balance factor of `tree` is 1; AND
 *           the median of `tree` is v.
 */
TEST(MedianTreeTest, InsertIntoEmpty) {
  short v = -1;
  MedianTree<short> tree;

  // establish preconditions for the test
  EXPECT_EQ(0, tree.count());
  EXPECT_EQ(1, tree.height());
  EXPECT_EQ(0, tree.balance());
  EXPECT_EQ(0, tree.el_balance());
  EXPECT_EQ(0, tree.median());
  EXPECT_EQ(nullptr, tree.left());

  // perform the insert
  tree.Insert(v, false, false);

  // ensure that v is in the left subtree
  ASSERT_NE(nullptr, tree.left());
  EXPECT_EQ(v, tree.left()->value());

  EXPECT_EQ(1, tree.count_left());
  EXPECT_EQ(nullptr, tree.right());

  // check tree count, height, and balance factor
  EXPECT_EQ(1, tree.count());
  EXPECT_EQ(2, tree.height());
  EXPECT_EQ(1, tree.balance());

  // check element-wise balance and median
  EXPECT_EQ(1, tree.el_balance());
  EXPECT_EQ(v, tree.median());
}

/*
 * GIVEN a MedianTreeNode with a single value v
 * DO insert a value u > v AND
 * TEST THAT `tree` has 1 element in its right subtree; AND
 *           the value at the root of the right subtree is u; AND
 *           `tree` has 2 elements; AND
 *           the height of `tree` is 2; AND
 *           the height balance factor of `tree` is 0; AND
 *           the element-wise balance factor of `tree` is 0; AND
 *           the median of `tree` is the mean of u and v.
 */
TEST(MedianTreeTest, InsertLargerValue) {
  short v = -1;
  MedianTree<short> tree(v);

  // establish preconditions for the test
  EXPECT_EQ(1, tree.count());
  EXPECT_EQ(2, tree.height());
  EXPECT_EQ(1, tree.balance());
  EXPECT_EQ(1, tree.el_balance());
  EXPECT_EQ(v, tree.median());
  ASSERT_NE(nullptr, tree.left());
  EXPECT_EQ(v, tree.left()->value());
  EXPECT_EQ(nullptr, tree.right());

  // perform the insert
  auto u = v + 1;
  tree.Insert(u, false, false);

  // ensure that u is in the right subtree
  ASSERT_NE(nullptr, tree.right());
  EXPECT_EQ(u, tree.right()->value());
  EXPECT_EQ(1, tree.count_right());

  // check tree count, height, and balance factor
  EXPECT_EQ(2, tree.count());
  EXPECT_EQ(2, tree.height());
  EXPECT_EQ(0, tree.balance());

  // check element-wise balance and median
  EXPECT_EQ(0, tree.el_balance());
  EXPECT_EQ((u + v) / 2.0, tree.median());
}

/*
 * GIVEN a MedianTree `tree` with 3 elements u < v < w with the left subtree
 *       root containing u and the right subtree root containing v
 * DO remove v AND
 * TEST THAT the return value of the remove call is 0 (success); AND
 *           the value at the root of the left subtree is u; AND
 *           the right subtree remains unchanged; AND
 *           there are two elements in `tree`; AND
 *           the height of `tree` is 2; AND
 *           the element-wise balance of `tree` is 0; AND
 *           the median of `tree` is the mean of u and w.
 */
TEST(MedianTreeTest, RemoveSubtreeRoot) {
  short u = -1;
  auto v = u + 1;
  auto w = v + 1;

  MedianTree<short> tree(v);
  tree.Insert(w, false, false);
  tree.Insert(u, false, false);

  // establish preconditions for the test
  EXPECT_EQ(3, tree.count());
  EXPECT_EQ(3, tree.height());
  EXPECT_EQ(1, tree.balance());
  EXPECT_EQ(1, tree.el_balance());
  EXPECT_EQ(v, tree.median());

  ASSERT_NE(nullptr, tree.left());
  EXPECT_EQ(v, tree.left()->value());
  EXPECT_EQ(u, tree.left()->min());
  ASSERT_NE(nullptr, tree.right());
  EXPECT_EQ(w, tree.right()->value());

  // perform the remove
  ASSERT_EQ(0, tree.Remove(v, 0));

  // ensure that u now resides in tree.left()
  ASSERT_NE(nullptr, tree.left());
  EXPECT_EQ(u, tree.left()->value());

  // ensure that tree.right() is unchanged
  ASSERT_NE(nullptr, tree.right());
  EXPECT_EQ(w, tree.right()->value());

  // check tree count, height, and balance factor
  EXPECT_EQ(2, tree.count());
  EXPECT_EQ(2, tree.height());
  EXPECT_EQ(0, tree.balance());

  // check element-wise balance and median
  EXPECT_EQ(0, tree.el_balance());
  EXPECT_EQ((u + w) / 2.0, tree.median());
}

/*
 * GIVEN a MedianTree `tree` with 2 elements u < v in the left subtree and 1
 *       element w > v in the right subtree
 * DO remove u AND
 * TEST THAT the return value of the remove call is 0 (success); AND
 *           the value at the root of the left subtree is still v; AND
 *           the right subtree remains unchanged; AND
 *           there are two elements in `tree`; AND
 *           the height of `tree` is 2; AND
 *           the element-wise balance of `tree` is 0; AND
 *           the median of `tree` is the mean of v and w.
 */
TEST(MedianTreeTest, RemoveSubtreeDescendant) {
  short u = -1;
  auto v = u + 1;
  auto w = v + 1;

  MedianTree<short> tree(v);
  tree.Insert(w, false, false);
  tree.Insert(u, false, false);

  // establish preconditions for the test
  EXPECT_EQ(3, tree.count());
  EXPECT_EQ(3, tree.height());
  EXPECT_EQ(1, tree.balance());
  EXPECT_EQ(1, tree.el_balance());
  EXPECT_EQ(v, tree.median());

  ASSERT_NE(nullptr, tree.left());
  EXPECT_EQ(v, tree.left()->value());
  EXPECT_EQ(u, tree.left()->min());
  ASSERT_NE(nullptr, tree.right());
  EXPECT_EQ(w, tree.right()->value());

  // perform the remove
  ASSERT_EQ(0, tree.Remove(u, 0));

  // ensure that u now resides in tree.left()
  ASSERT_NE(nullptr, tree.left());
  EXPECT_EQ(v, tree.left()->value());

  // ensure that tree.right() is unchanged
  ASSERT_NE(nullptr, tree.right());
  EXPECT_EQ(w, tree.right()->value());

  // check tree count, height, and balance factor
  EXPECT_EQ(2, tree.count());
  EXPECT_EQ(2, tree.height());
  EXPECT_EQ(0, tree.balance());

  // check element-wise balance and median
  EXPECT_EQ(0, tree.el_balance());
  EXPECT_EQ((v + w) / 2.0, tree.median());
}

/*
 * GIVEN a MedianTree `tree` with 3 elements t < u < v in the left subtree
 *       and 1 element w > v in the right subtree
 * DO rebalance the tree: shift the largest element of the left subtree to the
 *    right subtree AND
 * TEST THAT the number of elements in `tree` remains 4; AND
 *           the element-wise balance of `tree` is 0; AND
 *           t is the minimum of the left subtree; AND
 *           u is the maximum of the left subtree; AND
 *           v is the minimum of the right subtree; AND
 *           w is the maximum of the right subtree; AND
 *           the median of `tree` is mean of u and v.
 */
TEST(MedianTreeTest, BalanceElementsLTR) {
  short t = -1;
  auto u = t + 1;
  auto v = u + 1;
  auto w = v + 1;

  MedianTree<short> tree(v);
  tree.Insert(w, false, false);
  tree.Insert(u, false, false);
  tree.Insert(t, false, false);

  // establish preconditions for the test
  EXPECT_EQ(4, tree.count());
  EXPECT_EQ(2, tree.el_balance());
  // equivalently:
  // EXPECT_EQ(3, tree.count_left());
  // EXPECT_EQ(1, tree.count_right());

  ASSERT_NE(nullptr, tree.left());
  EXPECT_EQ(t, tree.left()->min());
  EXPECT_EQ(v, tree.left()->max());

  ASSERT_NE(nullptr, tree.right());
  EXPECT_EQ(w, tree.right()->min());
  EXPECT_EQ(w, tree.right()->max());

  // perform the rebalance
  tree.BalanceElements();

  EXPECT_EQ(4, tree.count());
  EXPECT_EQ(0, tree.el_balance());

  ASSERT_NE(nullptr, tree.left());
  EXPECT_EQ(t, tree.left()->min());
  EXPECT_EQ(u, tree.left()->max());

  ASSERT_NE(nullptr, tree.right());
  EXPECT_EQ(v, tree.right()->min());
  EXPECT_EQ(w, tree.right()->max());

  EXPECT_EQ((u + v) / 2.0, tree.median());
}

/*
 * GIVEN a MedianTree `tree` with 3 elements u < v < w in the right subtree
 *       and 1 element t < u in the left subtree
 * DO rebalance the tree: shift the smallest element of the right subtree to the
 *    left subtree AND
 * TEST THAT the number of elements in `tree` remains 4; AND
 *           the element-wise balance of `tree` is 0; AND
 *           t is the minimum of the left subtree; AND
 *           u is the maximum of the left subtree; AND
 *           v is the minimum of the right subtree; AND
 *           w is the maximum of the right subtree; AND
 *           the median of `tree` is mean of u and v.
 */
TEST(MedianTreeTest, BalanceElementsRTL) {
  short t = -1;
  auto u = t + 1;
  auto v = u + 1;
  auto w = v + 1;

  MedianTree<short> tree(t);
  tree.Insert(u, false, false);
  tree.Insert(v, false, false);
  tree.Insert(w, false, false);

  // establish preconditions for the test
  EXPECT_EQ(4, tree.count());
  EXPECT_EQ(-2, tree.el_balance());
  // equivalently:
  // EXPECT_EQ(3, tree.count_left());
  // EXPECT_EQ(1, tree.count_right());

  ASSERT_NE(nullptr, tree.left());
  EXPECT_EQ(t, tree.left()->min());
  EXPECT_EQ(t, tree.left()->max());

  ASSERT_NE(nullptr, tree.right());
  EXPECT_EQ(u, tree.right()->min());
  EXPECT_EQ(w, tree.right()->max());

  // perform the rebalance
  tree.BalanceElements();

  EXPECT_EQ(4, tree.count());
  EXPECT_EQ(0, tree.el_balance());

  ASSERT_NE(nullptr, tree.left());
  EXPECT_EQ(t, tree.left()->min());
  EXPECT_EQ(u, tree.left()->max());

  ASSERT_NE(nullptr, tree.right());
  EXPECT_EQ(v, tree.right()->min());
  EXPECT_EQ(w, tree.right()->max());

  EXPECT_EQ((u + v) / 2.0, tree.median());
}

/*
 * GIVEN a MedianTree `tree` whose left child `node` requires an LL rotation
 * DO perform an LL rotation AND
 * TEST THAT the balance factor of `node` is 0; AND
 *           `node` has the correct number of children; AND
 *           both the left and right children of `node` are balanced.
 */
TEST(MedianTreeTest, LLRotation) {
  MedianTree<short> tree(6);
  auto node = tree.left();
  ASSERT_NE(nullptr, node);

  node->Insert(4, false);
  node->Insert(7, false);
  node->Insert(2, false);
  node->Insert(5, false);
  node->Insert(1, false);
  node->Insert(3, false);

  // establish preconditions for the test
  EXPECT_EQ(4, node->height());
  EXPECT_EQ(7, node->count());
  EXPECT_EQ(2, node->balance());
  ASSERT_NE(nullptr, node->left());
  EXPECT_EQ(1, node->left()->balance());

  // perform the rotation
  tree.RotateSubtrees(false);

  // node should no longer be tree's left child
  ASSERT_NE(node, tree.left());

  // reassign node symbol to be tree's new left child, which should be 4
  node = tree.left();
  ASSERT_NE(nullptr, node);
  EXPECT_EQ(4, node->value());

  // check height, count, balance of node
  EXPECT_EQ(3, node->height());
  EXPECT_EQ(7, node->count());
  EXPECT_EQ(0, node->balance());

  // check value, height, and balance of both children of `node`
  ASSERT_NE(nullptr, node->left());
  EXPECT_EQ(2, node->left()->value());
  EXPECT_EQ(2, node->left()->height());
  EXPECT_EQ(3, node->left()->count());
  EXPECT_EQ(0, node->left()->balance());

  ASSERT_NE(nullptr, node->right());
  EXPECT_EQ(6, node->right()->value());
  EXPECT_EQ(2, node->right()->height());
  EXPECT_EQ(3, node->right()->count());
  EXPECT_EQ(0, node->right()->balance());
}

/*
 * GIVEN a MedianTree `tree` whose left child `node` requires an LR rotation
 * DO perform an LR rotation AND
 * TEST THAT the balance factor of `node` is 0; AND
 *           `node` has the correct number of children; AND
 *           both the left and right children of `node` are balanced.
 */
TEST(MedianTreeTest, LRRotation) {
  MedianTree<short> tree(6);
  auto node = tree.left();
  ASSERT_NE(nullptr, node);

  node->Insert(2, false);
  node->Insert(7, false);
  node->Insert(1, false);
  node->Insert(4, false);
  node->Insert(3, false);
  node->Insert(5, false);

  // establish preconditions for the test
  EXPECT_EQ(4, node->height());
  EXPECT_EQ(7, node->count());
  EXPECT_EQ(2, node->balance());
  ASSERT_NE(nullptr, node->left());
  EXPECT_EQ(-1, node->left()->balance());

  // perform the rotation
  tree.RotateSubtrees(false);

  // node should no longer be tree's left child
  ASSERT_NE(node, tree.left());

  // reassign node symbol to be base's new left child, which should be 4
  node = tree.left();
  ASSERT_NE(nullptr, node);
  EXPECT_EQ(4, node->value());

  // check height, count, balance of node
  EXPECT_EQ(3, node->height());
  EXPECT_EQ(7, node->count());
  EXPECT_EQ(0, node->balance());

  // check height and balance of both children of `node`
  ASSERT_NE(nullptr, node->left());
  EXPECT_EQ(2, node->left()->value());
  EXPECT_EQ(2, node->left()->height());
  EXPECT_EQ(3, node->left()->count());
  EXPECT_EQ(0, node->left()->balance());

  ASSERT_NE(nullptr, node->right());
  EXPECT_EQ(6, node->right()->value());
  EXPECT_EQ(2, node->right()->height());
  EXPECT_EQ(3, node->right()->count());
  EXPECT_EQ(0, node->right()->balance());
}

/*
 * GIVEN a MedianTree `tree` whose left child `node` requires an RL rotation
 * DO perform an RL rotation AND
 * TEST THAT the balance factor of `node` is 0; AND
 *           `node` has the correct number of children; AND
 *           both the left and right children of `node` are balanced.
 */
TEST(MedianTreeTest, RLRotation) {
  MedianTree<short> tree(2);
  auto node = tree.left();
  ASSERT_NE(nullptr, node);

  node->Insert(1, false);
  node->Insert(6, false);
  node->Insert(4, false);
  node->Insert(7, false);
  node->Insert(3, false);
  node->Insert(5, false);

  // establish preconditions for the test
  EXPECT_EQ(4, node->height());
  EXPECT_EQ(7, node->count());
  EXPECT_EQ(-2, node->balance());
  ASSERT_NE(nullptr, node->right());
  EXPECT_EQ(1, node->right()->balance());

  // perform the rotation
  tree.RotateSubtrees(false);

  // node should no longer be tree's left child
  ASSERT_NE(node, tree.left());

  // reassign node symbol to be tree's new left child, which should be 4
  node = tree.left();
  ASSERT_NE(nullptr, node);
  EXPECT_EQ(4, node->value());

  // check height, count, balance of node
  EXPECT_EQ(3, node->height());
  EXPECT_EQ(7, node->count());
  EXPECT_EQ(0, node->balance());

  // check height and balance of both children of `node`
  ASSERT_NE(nullptr, node->left());
  EXPECT_EQ(2, node->left()->value());
  EXPECT_EQ(2, node->left()->height());
  EXPECT_EQ(3, node->left()->count());
  EXPECT_EQ(0, node->left()->balance());

  ASSERT_NE(nullptr, node->right());
  EXPECT_EQ(6, node->right()->value());
  EXPECT_EQ(2, node->right()->height());
  EXPECT_EQ(3, node->right()->count());
  EXPECT_EQ(0, node->right()->balance());
}

/*
 * GIVEN a MedianTreeNode `tree` whose left child `node` requires an RR rotation
 * DO perform an RR rotation AND
 * TEST THAT the balance factor of `node` is 0; AND
 *           `node` has the correct number of children; AND
 *           both the left and right children of `node` are balanced.
 */
TEST(MedianTreeTest, RRRotation) {
  MedianTree<short> tree(2);
  auto node = tree.left();
  ASSERT_NE(nullptr, node);

  node->Insert(1, false);
  node->Insert(4, false);
  node->Insert(3, false);
  node->Insert(6, false);
  node->Insert(5, false);
  node->Insert(7, false);

  // establish preconditions for the test
  EXPECT_EQ(4, node->height());
  EXPECT_EQ(7, node->count());
  EXPECT_EQ(-2, node->balance());
  ASSERT_NE(nullptr, node->right());
  EXPECT_EQ(-1, node->right()->balance());

  // perform the rotation
  tree.RotateSubtrees(false);

  // node should no longer be base's left child
  ASSERT_NE(node, tree.left());

  // reassign node symbol to be base's new left child, which should be 4
  node = tree.left();
  ASSERT_NE(nullptr, node);
  EXPECT_EQ(4, node->value());

  // check height, count, balance of node
  EXPECT_EQ(3, node->height());
  EXPECT_EQ(7, node->count());
  EXPECT_EQ(0, node->balance());

  // check height and balance of both children of `node`
  ASSERT_NE(nullptr, node->left());
  EXPECT_EQ(2, node->left()->value());
  EXPECT_EQ(2, node->left()->height());
  EXPECT_EQ(3, node->left()->count());
  EXPECT_EQ(0, node->left()->balance());

  ASSERT_NE(nullptr, node->right());
  EXPECT_EQ(6, node->right()->value());
  EXPECT_EQ(2, node->right()->height());
  EXPECT_EQ(3, node->right()->count());
  EXPECT_EQ(0, node->right()->balance());
}

/*
 * GIVEN an empty MedianTree `tree` and values u < v < w < x < y < z
 * DO insert u, v, w, x, y, z, in that order, maintaining element-wise balance,
 * AND
 * TEST THAT the element-wise balance never exceeds 1 in absolute value; AND
 *           the median value updates as appropriate.
 */
TEST(MedianTreeTest, AutoBalanceOnInsert) {
  MedianTree<short> tree;

  short u = 0;
  auto v = u + 1;
  auto w = v + 1;
  auto x = w + 1;
  auto y = x + 1;
  auto z = y + 1;

  // establish preconditions for the test
  EXPECT_EQ(0, tree.balance());
  EXPECT_EQ(0, tree.el_balance());

  // perform the inserts
  tree.Insert(u, true, false);
  EXPECT_EQ(1, tree.balance());
  EXPECT_EQ(1, tree.el_balance());
  EXPECT_EQ(u, tree.median());

  tree.Insert(v, true, false);
  EXPECT_EQ(0, tree.balance());
  EXPECT_EQ(0, tree.el_balance());
  EXPECT_EQ((u + v) / 2.0, tree.median());

  tree.Insert(w, true, false);
  EXPECT_EQ(-1, tree.balance());
  EXPECT_EQ(-1, tree.el_balance());
  EXPECT_EQ(v, tree.median());

  tree.Insert(x, true, false);
  EXPECT_EQ(0, tree.balance());
  EXPECT_EQ(0, tree.el_balance());
  EXPECT_EQ((v + w) / 2.0, tree.median());

  tree.Insert(y, true, false);
  EXPECT_EQ(-1, tree.balance());
  EXPECT_EQ(-1, tree.el_balance());
  EXPECT_EQ(w, tree.median());

  tree.Insert(z, true, false);
  EXPECT_EQ(0, tree.balance());
  EXPECT_EQ(0, tree.el_balance());
  EXPECT_EQ((w + x) / 2.0, tree.median());
}

/*
 * GIVEN an empty MedianTree `tree` and values u < v < w < x < y < z
 * DO insert u, v, w, x, y, z, in that order, maintaining both element-wise
 *    and rotational balance, AND
 * TEST THAT the element-wise balance never exceeds 1 in absolute value; AND
 *           the subtree balance never exceeds 1 in absolute value; AND
 *           the balances of the subtrees themselves never exceed 1 in
 *           absolute value; AND
 *           the median value updates as appropriate.
 */
TEST(MedianTreeTest, AutoBalanceAndRotateOnInsert) {
  MedianTree<short> tree;

  short u = 0;
  auto v = u + 1;
  auto w = v + 1;
  auto x = w + 1;
  auto y = x + 1;
  auto z = y + 1;

  // establish preconditions for the test
  EXPECT_EQ(0, tree.balance());
  EXPECT_EQ(0, tree.el_balance());

  // perform the inserts
  tree.Insert(u, true, true);
  EXPECT_GE(1, tree.balance()); // 1 >= balance
  EXPECT_LE(-1, tree.balance()); // -1 <= balance
  EXPECT_GE(1, tree.el_balance()); // 1 >= el_balance
  EXPECT_LE(-1, tree.el_balance()); // -1 <= el_balance
  ASSERT_NE(nullptr, tree.left());
  EXPECT_GE(1, tree.left()->balance()); // 1 >= balance
  EXPECT_LE(-1, tree.left()->balance()); // -1 <= balance
  EXPECT_EQ(u, tree.median());

  tree.Insert(v, true, true);
  EXPECT_GE(1, tree.balance()); // 1 >= balance
  EXPECT_LE(-1, tree.balance()); // -1 <= balance
  EXPECT_GE(1, tree.el_balance()); // 1 >= el_balance
  EXPECT_LE(-1, tree.el_balance()); // -1 <= el_balance
  ASSERT_NE(nullptr, tree.left());
  EXPECT_GE(1, tree.left()->balance()); // 1 >= balance
  EXPECT_LE(-1, tree.left()->balance()); // -1 <= balance
  ASSERT_NE(nullptr, tree.right());
  EXPECT_GE(1, tree.right()->balance()); // 1 >= balance
  EXPECT_LE(-1, tree.right()->balance()); // -1 <= balance
  EXPECT_EQ((u + v) / 2.0, tree.median());

  tree.Insert(w, true, true);
  EXPECT_GE(1, tree.balance()); // 1 >= balance
  EXPECT_LE(-1, tree.balance()); // -1 <= balance
  EXPECT_GE(1, tree.el_balance()); // 1 >= el_balance
  EXPECT_LE(-1, tree.el_balance()); // -1 <= el_balance
  ASSERT_NE(nullptr, tree.left());
  EXPECT_GE(1, tree.left()->balance()); // 1 >= balance
  EXPECT_LE(-1, tree.left()->balance()); // -1 <= balance
  ASSERT_NE(nullptr, tree.right());
  EXPECT_GE(1, tree.right()->balance()); // 1 >= balance
  EXPECT_LE(-1, tree.right()->balance()); // -1 <= balance
  EXPECT_EQ(v, tree.median());

  tree.Insert(x, true, true);
  EXPECT_GE(1, tree.balance()); // 1 >= balance
  EXPECT_LE(-1, tree.balance()); // -1 <= balance
  EXPECT_GE(1, tree.el_balance()); // 1 >= el_balance
  EXPECT_LE(-1, tree.el_balance()); // -1 <= el_balance
  ASSERT_NE(nullptr, tree.left());
  EXPECT_GE(1, tree.left()->balance()); // 1 >= balance
  EXPECT_LE(-1, tree.left()->balance()); // -1 <= balance
  ASSERT_NE(nullptr, tree.right());
  EXPECT_GE(1, tree.right()->balance()); // 1 >= balance
  EXPECT_LE(-1, tree.right()->balance()); // -1 <= balance
  EXPECT_EQ((v + w) / 2.0, tree.median());

  tree.Insert(y, true, true);
  EXPECT_GE(1, tree.balance()); // 1 >= balance
  EXPECT_LE(-1, tree.balance()); // -1 <= balance
  EXPECT_GE(1, tree.el_balance()); // 1 >= el_balance
  EXPECT_LE(-1, tree.el_balance()); // -1 <= el_balance
  ASSERT_NE(nullptr, tree.left());
  EXPECT_GE(1, tree.left()->balance()); // 1 >= balance
  EXPECT_LE(-1, tree.left()->balance()); // -1 <= balance
  ASSERT_NE(nullptr, tree.right());
  EXPECT_GE(1, tree.right()->balance()); // 1 >= balance
  EXPECT_LE(-1, tree.right()->balance()); // -1 <= balance
  EXPECT_EQ(w, tree.median());

  tree.Insert(z, true, true);
  EXPECT_GE(1, tree.balance()); // 1 >= balance
  EXPECT_LE(-1, tree.balance()); // -1 <= balance
  EXPECT_GE(1, tree.el_balance()); // 1 >= el_balance
  EXPECT_LE(-1, tree.el_balance()); // -1 <= el_balance
  ASSERT_NE(nullptr, tree.left());
  EXPECT_GE(1, tree.left()->balance()); // 1 >= balance
  EXPECT_LE(-1, tree.left()->balance()); // -1 <= balance
  ASSERT_NE(nullptr, tree.right());
  EXPECT_GE(1, tree.right()->balance()); // 1 >= balance
  EXPECT_LE(-1, tree.right()->balance()); // -1 <= balance
  EXPECT_EQ((w + x) / 2.0, tree.median());
}
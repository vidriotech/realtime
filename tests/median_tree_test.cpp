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
TEST(MedianTreeTests, InitialStateNoArgs)
{
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
TEST(MedianTreeTests, InitialStateSingleArg)
{
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
TEST(MedianTreeTests, InitialStateTwoArgs)
{
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
TEST(MedianTreeTests, InsertIntoEmpty)
{
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
TEST(MedianTreeTests, InsertLargerValue)
{
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
TEST(MedianTreeTests, RemoveSubtreeRoot)
{
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
    ASSERT_EQ(0, tree.Remove(v));

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
TEST(MedianTreeTests, RemoveSubtreeDescendant)
{
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
    ASSERT_EQ(0, tree.Remove(u));

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
TEST(MedianTreeTests, BalanceElementsLTR)
{
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
TEST(MedianTreeTests, BalanceElementsRTL)
{
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
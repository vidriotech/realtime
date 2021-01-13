#include "gtest/gtest.h"

#include "../src/structures/median_tree.h"

/*
 * DO construct a MedianTree `tree` with no arguments; AND
 * TEST THAT `tree` has null pointers for both left and right children; AND
 *           `tree` has 0 elements; AND
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
 * TEST THAT `tree` has 1 element in its left tree; AND
 *           the value at the root of the left subtree is v; AND
 *           the value at the root of the right subtree is u; AND
 *           `tree` has 2 elements; AND
 *           `tree` has a height of 2; AND
 *           `tree` is balanced; AND
 *           the median of `tree` is v.
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
 * TEST THAT `tree` has 1 element in its left tree; AND
 *           the value at the root of the left subtree of `tree` is v; AND
 *           `tree` has a null pointer for its right tree; AND
 *           `tree` has 1 element; AND
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
    EXPECT_EQ(nullptr, tree.left());

    // perform the insert
    tree.Insert(v);

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
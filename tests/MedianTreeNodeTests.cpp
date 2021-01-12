#include "gtest/gtest.h"

#include "../src/structures/MedianTreeNode.h"

/**
 * GIVEN a value v
 * DO construct a MedianTreeNode from v AND
 * TEST THAT the constructor does not crash.
 */
TEST(MedianTreeNodeTests, InitOK)
{
    short v = 0;
    MedianTreeNode<short> node(v);
}

/**
 * GIVEN a MedianTreeNode containing a value v
 * TEST THAT the value() method returns v.
 */
TEST(MedianTreeNodeTests, ValueReturned)
{
    short v = 0;
    MedianTreeNode<short> node(v);

    EXPECT_EQ(v, node.value());
}

/**
 * GIVEN a newly-constructed MedianTreeNode
 * TEST THAT its left and right children are null.
 */
TEST(MedianTreeNodeTests, ChildrenAreNull)
{
    short v = 0;
    MedianTreeNode<short> node(v);

    EXPECT_EQ(nullptr, node.left());
    EXPECT_EQ(nullptr, node.right());
}

/**
 * GIVEN a MedianTreeNode node with value v and no children
 * DO insert a value u smaller than v AND
 * TEST THAT the left child of node has value u;
 *           the height of the subtree rooted at node is 2; AND
 *           the balance factor of the subtree is 1, i.e., left-heavy.
 */
TEST(MedianTreeNodeTests, InsertChildLeft)
{
    short v = 0;
    MedianTreeNode<short> node(v);

    // establish preconditions for the test
    EXPECT_EQ(nullptr, node.left());
    EXPECT_EQ(nullptr, node.right());

    // perform the insert
    auto u = node.value() - 1;
    EXPECT_EQ(2, node.insert(u)); // success

    // ensure that u is node's left child
    EXPECT_NE(nullptr, node.left());
    EXPECT_EQ(u, node.left()->value());
    EXPECT_EQ(nullptr, node.right());

    // check subtree height and balance factor
    EXPECT_EQ(2, node.height());
    EXPECT_EQ(1, node.balance());
}

/**
 * GIVEN a MedianTreeNode node with value v and no children
 * DO insert a value u larger than v AND
 * TEST THAT the right child of node has value u;
 *           the height of the subtree rooted at node is 2; AND
 *           the balance factor of the subtree is -1, i.e., right-heavy.
 */
TEST(MedianTreeNodeTests, InsertChildRight)
{
    short v = 0;
    MedianTreeNode<short> node(v);

    // establish preconditions for the test
    EXPECT_EQ(nullptr, node.left());
    EXPECT_EQ(nullptr, node.right());

    // perform the insert
    auto u = node.value() + 1;
    EXPECT_EQ(2, node.insert(u)); // success

    // ensure that u is node's right child
    EXPECT_EQ(nullptr, node.left());
    EXPECT_NE(nullptr, node.right());
    EXPECT_EQ(u, node.right()->value());

    // check subtree height and balance factor
    EXPECT_EQ(2, node.height());
    EXPECT_EQ(-1, node.balance());
}
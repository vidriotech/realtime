#include "gtest/gtest.h"

#include "../src/structures/median_tree_node.h"

/*
 * GIVEN a value v
 * DO construct a MedianTreeNode node from v AND
 * TEST THAT the value of `node` is v; AND
 *           the left and right children of `node` are null; AND
 *           the number of elements in the subtree rooted at `node` is 1; AND
 *           the height of the subtree rooted at `node` is 1; AND
 *           the balance factor of the subtree rooted at `node` is 0.
 */
TEST(MedianTreeNodeTests, InitialState)
{
    short v = 0;
    MedianTreeNode<short> node(v);

    EXPECT_EQ(v, node.value());
    EXPECT_EQ(nullptr, node.left());
    EXPECT_EQ(nullptr, node.right());
    EXPECT_EQ(1, node.count());
    EXPECT_EQ(1, node.height());
    EXPECT_EQ(0, node.balance());
}

/*
 * GIVEN a MedianTreeNode node with value v and no children
 * DO Insert a value u smaller than v AND
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

    // perform the Insert
    auto u = node.value() - 1;
    EXPECT_EQ(2, node.Insert(u)); // success

    // ensure that u is node's left child
    ASSERT_NE(nullptr, node.left());
    EXPECT_EQ(u, node.left()->value());
    EXPECT_EQ(nullptr, node.right());

    // check subtree count, height, and balance factor
    EXPECT_EQ(2, node.count());
    EXPECT_EQ(2, node.height());
    EXPECT_EQ(1, node.balance());
}

/*
 * GIVEN a MedianTreeNode node with value v and no children
 * DO Insert a value u larger than v AND
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

    // perform the Insert
    auto u = node.value() + 1;
    EXPECT_EQ(2, node.Insert(u)); // success

    // ensure that u is node's right child
    EXPECT_EQ(nullptr, node.left());
    ASSERT_NE(nullptr, node.right());
    EXPECT_EQ(u, node.right()->value());

    // check subtree count, height, and balance factor
    EXPECT_EQ(2, node.count());
    EXPECT_EQ(2, node.height());
    EXPECT_EQ(-1, node.balance());
}

/*
 * GIVEN a MedianTreeNode node with value v and left child with value u < v
 *       and no children
 * DO Insert a value t < u AND
 * TEST THAT the left child of the left child of `node` has value t; AND
 *           the height of the subtree rooted at `node` is 3; AND
 *           the balance of the subtree rooted at `node` is 2.
 */
TEST(MedianTreeNodeTests, InsertGrandchildLeft)
{
    short v = 0;
    MedianTreeNode<short> node(v);

    // establish preconditions for the test
    auto u = v - 1;
    node.Insert(u);

    EXPECT_EQ(2, node.count());
    EXPECT_EQ(2, node.height());
    EXPECT_EQ(1, node.balance());
    ASSERT_NE(nullptr, node.left());
    EXPECT_EQ(u, node.left()->value());

    // perform the Insert
    auto t = u - 1;
    EXPECT_EQ(3, node.Insert(t));

    // ensure that t is the left child of node's left child
    ASSERT_NE(nullptr, node.left()->left());
    EXPECT_EQ(t, node.left()->left()->value());
    EXPECT_EQ(nullptr, node.left()->right());

    // check subtree count, height, and balance factor
    EXPECT_EQ(3, node.count());
    EXPECT_EQ(3, node.height());
    EXPECT_EQ(2, node.balance());
}

/*
 * GIVEN a MedianTreeNode `node` of height 2, balance 1
 * DO Insert a subtree of height 2 on the right hand side of `node` AND
 * TEST THAT `node` has height 3; AND
 *           `node` has balance -1.
 */
TEST(MedianTreeNodeTests, InsertSubtreeAsChild)
{
    short v = 0;
    MedianTreeNode<short> node(v);

    // establish preconditions for the test
    auto u = v - 1;
    node.Insert(u);

    EXPECT_EQ(2, node.count());
    EXPECT_EQ(2, node.height());
    EXPECT_EQ(1, node.balance());
    ASSERT_NE(nullptr, node.left());
    EXPECT_EQ(u, node.left()->value());
    EXPECT_EQ(nullptr, node.right());

    // create new subtree with 3 elements, height 2
    auto t = v + 2;
    std::shared_ptr<MedianTreeNode<short>> node2(new MedianTreeNode<short>(t));
    node2->Insert(t - 1);
    node2->Insert(t + 1);
    EXPECT_EQ(3, node2->count());
    EXPECT_EQ(0, node2->balance());
    EXPECT_EQ(2, node2->height());

    // perform the Insert
    node.InsertSubtree(node2);

    // ensure that t is in node's right child
    ASSERT_NE(nullptr, node.right());
    EXPECT_EQ(t, node.right()->value());

    // check subtree count, height, and balance factor
    EXPECT_EQ(5, node.count());
    EXPECT_EQ(3, node.height());
    EXPECT_EQ(-1, node.balance());
}

/*
 * GIVEN a MedianTreeNode `node` of height 3, balance 2
 * DO Insert a subtree of height 2 on the right hand side of the left child
 *    of `node` AND
 * TEST THAT `node` has height 4; AND
 *           `node` has balance 3.
 */
TEST(MedianTreeNodeTests, InsertSubtreeAsGrandchild)
{
    short v = 0;
    MedianTreeNode<short> node(v);

    // establish preconditions for the test
    auto u = v - 2;
    node.Insert(u);
    node.Insert(u - 1);

    EXPECT_EQ(3, node.count());
    EXPECT_EQ(3, node.height());
    EXPECT_EQ(2, node.balance());
    ASSERT_NE(nullptr, node.left());
    EXPECT_EQ(u, node.left()->value());
    EXPECT_EQ(u - 1, node.left()->left()->value());
    EXPECT_EQ(nullptr, node.right());

    // create new subtree with 3 elements, height 2
    auto t = v - 1;
    std::shared_ptr<MedianTreeNode<short>> node2(new MedianTreeNode<short>(t));
    node2->Insert(t - 1);
    node2->Insert(t + 1);
    EXPECT_EQ(3, node2->count());
    EXPECT_EQ(0, node2->balance());
    EXPECT_EQ(2, node2->height());

    // perform the Insert
    node.InsertSubtree(node2);

    // ensure that t is in node's left child's right child
    EXPECT_EQ(nullptr, node.right());

    // check subtree count, height, and balance factor
    EXPECT_EQ(6, node.count());
    EXPECT_EQ(4, node.height());
    EXPECT_EQ(3, node.balance());
}

/*
 * GIVEN a MedianTreeNode `node` with value v and left child `child` with
 *       value u < v and no children
 * DO remove u from `node` AND
 * TEST THAT `node` has height 1; AND
 *           `node` has balance 0; AND
 *           `node` has a null pointer for its left child.
 */
TEST(MedianTreeNodeTests, RemoveLeftChildWithNoChildren)
{
    short v = 0;
    MedianTreeNode<short> node(v);

    // establish preconditions for the test
    auto u = v - 1;
    node.Insert(u);

    EXPECT_EQ(2, node.count());
    EXPECT_EQ(2, node.height());
    EXPECT_EQ(1, node.balance());
    ASSERT_NE(nullptr, node.left());
    EXPECT_EQ(u, node.left()->value());

    // perform the remove
    EXPECT_EQ(0, node.Remove(u));

    // ensure that left child is now null
    EXPECT_EQ(nullptr, node.left());

    // check subtree count, height, and balance factor
    EXPECT_EQ(1, node.count());
    EXPECT_EQ(1, node.height());
    EXPECT_EQ(0, node.balance());
}

/*
 * GIVEN a MedianTreeNode `node` with value v and right child `child` with
 *       value u > v and no children
 * DO remove u from `node` AND
 * TEST THAT `node` has height 1; AND
 *           `node` has balance 0; AND
 *           `node` has a null pointer for its right child.
 */
TEST(MedianTreeNodeTests, RemoveRightChildWithNoChildren)
{
    short v = 0;
    MedianTreeNode<short> node(v);

    // establish preconditions for the test
    auto u = v + 1;
    node.Insert(u);

    EXPECT_EQ(2, node.count());
    EXPECT_EQ(2, node.height());
    EXPECT_EQ(-1, node.balance());
    ASSERT_NE(nullptr, node.right());
    EXPECT_EQ(u, node.right()->value());

    // perform the remove
    EXPECT_EQ(0, node.Remove(u));

    // ensure that left child is now null
    EXPECT_EQ(nullptr, node.right());

    // check subtree count, height, and balance factor
    EXPECT_EQ(1, node.count());
    EXPECT_EQ(1, node.height());
    EXPECT_EQ(0, node.balance());
}

/*
 * GIVEN a MedianTreeNode `node` with value v, child `child` with value u < v,
 *       and grandchild `grandchild` with value t
 * DO remove u from the subtree rooted at `node` AND
 * TEST THAT `node` has height 2; AND
 *           `node` has balance 1; AND
 *           the left child of `node` now has t instead of u for a value.
 */
TEST(MedianTreeNodeTests, RemoveChildWithChildren)
{
    short v = 0;
    MedianTreeNode<short> node(v);

    // establish preconditions for the test
    auto u = v - 1;
    auto t = u - 1;
    node.Insert(u);
    node.Insert(t);

    EXPECT_EQ(3, node.count());
    EXPECT_EQ(3, node.height());
    EXPECT_EQ(2, node.balance());
    ASSERT_NE(nullptr, node.left());
    EXPECT_EQ(u, node.left()->value());

    // perform the remove
    EXPECT_EQ(0, node.Remove(u));

    // ensure that left child is not null
    ASSERT_NE(nullptr, node.left());
    EXPECT_EQ(t, node.left()->value());

    // check subtree count, height, and balance factor
    EXPECT_EQ(2, node.count());
    EXPECT_EQ(2, node.height());
    EXPECT_EQ(1, node.balance());
}


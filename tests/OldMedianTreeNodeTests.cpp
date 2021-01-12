#include "gtest/gtest.h"

#include "../src/structures/OldMedianTreeNode.h"

TEST(MedianTreeNodeTestSuite, InitOK)
{
    short test_data = 1;
    auto node = OldMedianTreeNode<short>(test_data);

    EXPECT_EQ(test_data, node.value());
    EXPECT_EQ(1, node.n_elements());
    EXPECT_EQ(1, node.height());
    EXPECT_EQ(0, node.balance());
}

TEST(MedianTreeNodeTestSuite, InsertValueOK)
{
    short test_data = 1, test_data_left = 0, test_data_right = 2;
    auto node = OldMedianTreeNode<short>(test_data);

    EXPECT_EQ(1, node.height());

    node.insert(test_data_left);
    EXPECT_EQ(2, node.n_elements());
    EXPECT_EQ(2, node.height());
    EXPECT_EQ(AVL_LEFT_HEAVY, node.balance());

    node.insert(test_data_right);
    EXPECT_EQ(3, node.n_elements());
    EXPECT_EQ(2, node.height()); // still height of 2
    EXPECT_EQ(AVL_BALANCED, node.balance());
}

TEST(MedianTreeNodeTestSuite, UpsetBalance)
{
    short test_data = 0;
    auto node = OldMedianTreeNode<short>(test_data);

    EXPECT_EQ(0, node.balance());
    EXPECT_EQ(1, node.n_elements());

    node.insert(test_data + 1);
    EXPECT_EQ(2, node.n_elements());
    EXPECT_EQ(AVL_RIGHT_HEAVY, node.balance());

    node.insert(test_data + 2);
    EXPECT_EQ(3, node.n_elements());
    EXPECT_EQ(2 * AVL_RIGHT_HEAVY, node.balance());
}

TEST(MedianTreeNodeTestSuite, RemoveChildOK)
{
    short test_data = 0, test_child = 1;

    auto node = OldMedianTreeNode<short>(test_data);
    EXPECT_EQ(1, node.n_elements());

    node.insert(test_child);
    EXPECT_EQ(2, node.n_elements());
    EXPECT_EQ(2, node.height());
    EXPECT_EQ(AVL_RIGHT_HEAVY, node.balance());

    EXPECT_EQ(0, node.remove(test_child));
    EXPECT_EQ(1, node.n_elements());
    EXPECT_EQ(1, node.height());
}

TEST(MedianTreeNodeTestSuite, RemoveArbitraryDescendentOK)
{
    /* construct a height-4, perfectly balanced, full tree, looks like
     *
     *         7
     *     3        8
     *  1   5      9  11
     * 0 2 4 6   8 10 12 14
     */

    auto node = OldMedianTreeNode<int>(7);
    node.insert(3); // left child
    node.insert(11); // right child
    node.insert(1); // left child of left child
    node.insert(5); // right child of left child
    node.insert(9); // left child of right child
    node.insert(13); // right child of right child
    node.insert(0); // left child of left child of left child
    node.insert(2); // right child of left child of left child
    node.insert(4); // left child of right child of left child
    node.insert(6); // right child of right child of left child
    node.insert(8); // left child of left child of right child
    node.insert(10); // right child of left child of right child
    node.insert(12); // left child of right child of right child
    node.insert(14); // right child of right child of right child

    EXPECT_EQ(15, node.n_elements());
    EXPECT_EQ(4, node.height());

    // remove the 1 node
    EXPECT_EQ(0, node.remove(1)); // success
    EXPECT_EQ(14, node.n_elements());
    EXPECT_EQ(4, node.height());
}

TEST(MedianTreeNodeTestSuite, AttemptRemoveMissingValue)
{
    short test_data = 0;
    auto node = OldMedianTreeNode<short>(test_data);

    EXPECT_EQ(-1, node.remove(1));
}

TEST(MedianTreeNodeTestSuite, PopValueOK)
{
    short test_data = 0, test_child = 1;

    auto node = OldMedianTreeNode<short>(test_data);
    EXPECT_EQ(1, node.n_elements());

    node.insert(test_child);
    EXPECT_EQ(2, node.n_elements());

    auto nodeptr = node.pop(test_child);
    EXPECT_EQ(1, nodeptr->n_elements());
    EXPECT_EQ(test_child, nodeptr->value());
}

TEST(MedianTreeNodeTestSuite, Extrema)
{
    short test_data = 0;
    auto node = OldMedianTreeNode<short>(test_data);
    EXPECT_EQ(test_data, node.max());
    EXPECT_EQ(test_data, node.min());

    node.insert(test_data - 1);
    EXPECT_EQ(test_data, node.max());
    EXPECT_EQ(test_data - 1, node.min());

    node.insert(test_data - 2);
    EXPECT_EQ(test_data, node.max());
    EXPECT_EQ(test_data - 2, node.min());

    node.insert(test_data + 1);
    EXPECT_EQ(test_data + 1, node.max());
    EXPECT_EQ(test_data - 2, node.min());

    node.insert(test_data + 2);
    EXPECT_EQ(test_data + 2, node.max());
    EXPECT_EQ(test_data - 2, node.min());
}

#include "gtest/gtest.h"

#include "../src/structures/MedianTreeNode.h"

TEST(MedianTreeNodeTestSuite, InitOK)
{
    short test_data = 1;
    auto node = MedianTreeNode<short>(test_data);

    EXPECT_EQ(test_data, node.value());
    EXPECT_EQ(1, node.n_elements());
    EXPECT_EQ(1, node.height());
    EXPECT_EQ(0, node.balance());
}

TEST(MedianTreeNodeTestSuite, InsertValueOK)
{
    short test_data = 1, test_data_left = 0, test_data_right = 2;
    auto node = MedianTreeNode<short>(test_data);

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
    auto node = MedianTreeNode<short>(test_data);

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

    auto node = MedianTreeNode<short>(test_data);
    EXPECT_EQ(1, node.n_elements());

    node.insert(test_child);
    EXPECT_EQ(2, node.n_elements());
    EXPECT_EQ(2, node.height());
    EXPECT_EQ(AVL_RIGHT_HEAVY, node.balance());

    EXPECT_EQ(0, node.remove(test_child));
    EXPECT_EQ(1, node.n_elements());
    EXPECT_EQ(1, node.height());
}

TEST(MedianTreeNodeTestSuite, AttemptRemoveMissingValue)
{
    short test_data = 0;
    auto node = MedianTreeNode<short>(test_data);

    EXPECT_EQ(-1, node.remove(1));
}

TEST(MedianTreeNodeTestSuite, PopValueOK)
{
    short test_data = 0, test_child = 1;

    auto node = MedianTreeNode<short>(test_data);
    EXPECT_EQ(1, node.n_elements());

    node.insert(test_child);
    EXPECT_EQ(2, node.n_elements());

    auto nodeptr = node.pop(test_child);
    EXPECT_EQ(1, nodeptr->n_elements());
    EXPECT_EQ(test_child, nodeptr->value());
}
#include "gtest/gtest.h"

#include <cmath>
#include "../src/structures/OldMedianTree.h"

TEST(MedianTreeTestSuite, Init0ArgsOK)
{
    OldMedianTree<short> tree;

    EXPECT_EQ(0, tree.n_elements());
    EXPECT_TRUE(std::isnan(tree.median()));
}

TEST(MedianTreeTestSuite, Init2ArgsOK)
{
    short left = 0, right = 1;
    OldMedianTree<short> tree(left, right);

    EXPECT_EQ(2, tree.n_elements());
    EXPECT_EQ((left + right) / 2.0f, tree.median());
}

TEST(MedianTreeTestSuite, TestInsertOK)
{
    short left = 0, right = 1;
    OldMedianTree<short> tree(right, left);

    tree.insert(left - 1);
    EXPECT_EQ(3, tree.n_elements());
    EXPECT_EQ(left, tree.median());
}

TEST(MedianTreeTestSuite, TestMaintainBalance)
{
    short left = 1, right = 2;
    OldMedianTree<short> tree(left, right); // 1] [2
    EXPECT_EQ(AVL_BALANCED, tree.balance());
    EXPECT_EQ((left + right) / 2.0f, tree.median());

    tree.insert(left - 1); // 0, 1] [2
    EXPECT_EQ(AVL_LEFT_HEAVY, tree.balance());
    EXPECT_EQ(left, tree.median());

    tree.insert(left - 2); // -1, 0] [1, 2
    EXPECT_EQ(AVL_BALANCED, tree.balance());
    EXPECT_EQ((left - 1 + left) / 2.0f, tree.median());

    tree.insert(right + 1); // -1, 0] [1, 2, 3
    EXPECT_EQ(AVL_RIGHT_HEAVY, tree.balance());
    EXPECT_EQ(left, tree.median());

    tree.insert(right + 2); // -1, 0, 1] [2, 3, 4
    EXPECT_EQ(AVL_BALANCED, tree.balance());
    EXPECT_EQ((left + right) / 2.0f, tree.median());

    tree.insert(right + 3); // -1, 0, 1] [2 3, 4, 5
    EXPECT_EQ(AVL_RIGHT_HEAVY, tree.balance());
    EXPECT_EQ(right, tree.median());

    tree.insert(right + 4); // -1, 0, 1, 2] [3, 4, 5, 6
    EXPECT_EQ(AVL_BALANCED, tree.balance());
    EXPECT_EQ((right + right + 1) / 2.0f, tree.median());
}
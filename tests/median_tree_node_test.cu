#include "gtest/gtest.h"

#include "../src/structures/median_tree_node.cuh"

/*
 * GIVEN a value v
 * DO construct a MedianTreeNode `node` from v AND
 * TEST THAT the value of `node` is v; AND
 *           the left and right children of `node` are null; AND
 *           the number of elements in the subtree rooted at `node` is 1; AND
 *           the height of the subtree rooted at `node` is 1; AND
 *           the balance factor of the subtree rooted at `node` is 0.
 */
TEST(MedianTreeNodeTests, InitialState) {
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
 * GIVEN a MedianTreeNode `node` with value v and no children
 * DO Insert a value u smaller than v AND
 * TEST THAT the left child of node has value u;
 *           the height of the subtree rooted at node is 2; AND
 *           the balance factor of the subtree is 1, i.e., left-heavy.
 */
TEST(MedianTreeNodeTests, InsertChildLeft) {
  short v = 0;
  MedianTreeNode<short> node(v);

  // establish preconditions for the test
  EXPECT_EQ(nullptr, node.left());
  EXPECT_EQ(nullptr, node.right());

  // perform the Insert
  auto u = node.value() - 1;
  EXPECT_EQ(2, node.Insert(u, false)); // success

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
 * GIVEN a MedianTreeNode `node` with value v and no children
 * DO Insert a value u larger than v AND
 * TEST THAT the right child of node has value u;
 *           the height of the subtree rooted at node is 2; AND
 *           the balance factor of the subtree is -1, i.e., right-heavy.
 */
TEST(MedianTreeNodeTests, InsertChildRight) {
  short v = 0;
  MedianTreeNode<short> node(v);

  // establish preconditions for the test
  EXPECT_EQ(nullptr, node.left());
  EXPECT_EQ(nullptr, node.right());

  // perform the Insert
  auto u = node.value() + 1;
  EXPECT_EQ(2, node.Insert(u, false)); // success

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
 * GIVEN a MedianTreeNode `node` with value v and left child with value u < v
 *       and no children
 * DO Insert a value t < u AND
 * TEST THAT the left child of the left child of `node` has value t; AND
 *           the height of the subtree rooted at `node` is 3; AND
 *           the balance of the subtree rooted at `node` is 2.
 */
TEST(MedianTreeNodeTests, InsertGrandchildLeft) {
  short v = 0;
  MedianTreeNode<short> node(v);

  // establish preconditions for the test
  auto u = v - 1;
  node.Insert(u, false);

  EXPECT_EQ(2, node.count());
  EXPECT_EQ(2, node.height());
  EXPECT_EQ(1, node.balance());
  ASSERT_NE(nullptr, node.left());
  EXPECT_EQ(u, node.left()->value());

  // perform the Insert
  auto t = u - 1;
  EXPECT_EQ(3, node.Insert(t, false));

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
TEST(MedianTreeNodeTests, InsertSubtreeAsChild) {
  short v = 0;
  MedianTreeNode<short> node(v);

  // establish preconditions for the test
  auto u = v - 1;
  node.Insert(u, false);

  EXPECT_EQ(2, node.count());
  EXPECT_EQ(2, node.height());
  EXPECT_EQ(1, node.balance());
  ASSERT_NE(nullptr, node.left());
  EXPECT_EQ(u, node.left()->value());
  EXPECT_EQ(nullptr, node.right());

  // create new subtree with 3 elements, height 2
  auto t = v + 2;
  std::shared_ptr<MedianTreeNode<short>> node2(new MedianTreeNode<short>(t));
  node2->Insert(t - 1, false);
  node2->Insert(t + 1, false);
  EXPECT_EQ(3, node2->count());
  EXPECT_EQ(0, node2->balance());
  EXPECT_EQ(2, node2->height());

  // perform the Insert
  node.InsertSubtree(node2, false);

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
TEST(MedianTreeNodeTests, InsertSubtreeAsGrandchild) {
  short v = 0;
  MedianTreeNode<short> node(v);

  // establish preconditions for the test
  auto u = v - 2;
  node.Insert(u, false);
  node.Insert(u - 1, false);

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
  node2->Insert(t - 1, false);
  node2->Insert(t + 1, false);
  EXPECT_EQ(3, node2->count());
  EXPECT_EQ(0, node2->balance());
  EXPECT_EQ(2, node2->height());

  // perform the Insert
  node.InsertSubtree(node2, false);

  // ensure that t is in node's left child's right child
  EXPECT_EQ(nullptr, node.right());

  // check subtree count, height, and balance factor
  EXPECT_EQ(6, node.count());
  EXPECT_EQ(4, node.height());
  EXPECT_EQ(3, node.balance());
}

/*
 * GIVEN a MedianTreeNode `node` with three values t < u < v
 * DO query the max value M of `node` AND
 * TEST THAT M = v.
 */
TEST(MedianTreeNodeTests, MaxValue) {
  short u = 0;
  auto t = u - 1;
  auto v = u + 1;

  MedianTreeNode<short> node(u);
  node.Insert(t, false);
  node.Insert(v, false);

  EXPECT_EQ(v, node.max());
}

/*
 * GIVEN a MedianTreeNode `node` with three values t < u < v
 * DO query the min value M of `node` AND
 * TEST THAT M = t.
 */
TEST(MedianTreeNodeTests, MinValue) {
  short u = 0;
  auto t = u - 1;
  auto v = u + 1;

  MedianTreeNode<short> node(u);
  node.Insert(t, false);
  node.Insert(v, false);

  EXPECT_EQ(t, node.min());
}

/*
 * GIVEN a MedianTreeNode `node` with value v and left child `child` with
 *       value u < v and no children
 * DO remove u from `node` AND
 * TEST THAT `node` has height 1; AND
 *           `node` has balance 0; AND
 *           `node` has a null pointer for its left child.
 */
TEST(MedianTreeNodeTests, RemoveLeftChildWithNoChildren) {
  short v = 0;
  MedianTreeNode<short> node(v);

  // establish preconditions for the test
  auto u = v - 1;
  node.Insert(u, false);

  EXPECT_EQ(2, node.count());
  EXPECT_EQ(2, node.height());
  EXPECT_EQ(1, node.balance());
  ASSERT_NE(nullptr, node.left());
  EXPECT_EQ(u, node.left()->value());

  // perform the remove
  EXPECT_EQ(0, node.Remove(u, false));

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
TEST(MedianTreeNodeTests, RemoveRightChildWithNoChildren) {
  short v = 0;
  MedianTreeNode<short> node(v);

  // establish preconditions for the test
  auto u = v + 1;
  node.Insert(u, false);

  EXPECT_EQ(2, node.count());
  EXPECT_EQ(2, node.height());
  EXPECT_EQ(-1, node.balance());
  ASSERT_NE(nullptr, node.right());
  EXPECT_EQ(u, node.right()->value());

  // perform the remove
  EXPECT_EQ(0, node.Remove(u, false));

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
TEST(MedianTreeNodeTests, RemoveChildWithChildren) {
  short v = 0;
  MedianTreeNode<short> node(v);

  // establish preconditions for the test
  auto u = v - 1;
  auto t = u - 1;
  node.Insert(u, false);
  node.Insert(t, false);

  EXPECT_EQ(3, node.count());
  EXPECT_EQ(3, node.height());
  EXPECT_EQ(2, node.balance());
  ASSERT_NE(nullptr, node.left());
  EXPECT_EQ(u, node.left()->value());

  // perform the remove
  EXPECT_EQ(0, node.Remove(u, false));

  // ensure that left child is not null
  ASSERT_NE(nullptr, node.left());
  EXPECT_EQ(t, node.left()->value());

  // check subtree count, height, and balance factor
  EXPECT_EQ(2, node.count());
  EXPECT_EQ(2, node.height());
  EXPECT_EQ(1, node.balance());
}

/*
 * GIVEN a MedianTreeNode `node` with a left subtree `subtree`
 * DO extract `subtree` AND
 * TEST THAT `node` has only 1 element; AND
 *           `subtree`
 */
TEST(MedianTreeNodeTests, DetachSubtree) {
  short v = 0;
  auto u = v - 1;

  MedianTreeNode<short> node(v);
  node.Insert(u);

  // establish preconditions for the test
  EXPECT_EQ(2, node.count());
  EXPECT_EQ(2, node.height());
  EXPECT_EQ(1, node.balance());
  ASSERT_NE(nullptr, node.left());
  EXPECT_EQ(u, node.left()->value());

  // perform the detach
  auto subtree = node.DetachSubtree(1);

  // check that `subtree` has been detached from `node`
  EXPECT_EQ(1, node.count());
  EXPECT_EQ(1, node.height());
  EXPECT_EQ(0, node.balance());
  EXPECT_EQ(nullptr, node.left());

  // check that `subtree` is what we expect
  ASSERT_NE(nullptr, subtree);
  EXPECT_EQ(1, subtree->count());
  EXPECT_EQ(1, subtree->height());
  EXPECT_EQ(0, subtree->balance());
  EXPECT_EQ(u, subtree->value());
}

/*
 * GIVEN a MedianTreeNode `node` with a balance factor of 2 and whose own left
 *       child has a balance factor of 1, i.e., is in need of an LL rotation
 * DO perform an LL rotation AND
 * TEST THAT the balance factor of `node` is 0; AND
 *           `node` has the correct number of children; AND
 *           both the left and right children of `node` are balanced.
 */
TEST(MedianTreeNodeTests, LLRotation) {
  std::shared_ptr<MedianTreeNode<short>> node(new MedianTreeNode<short>(6));
  node->Insert(4, false);
  node->Insert(7, false);
  node->Insert(2, false);
  node->Insert(5, false);
  node->Insert(1, false);
  node->Insert(3, false);

  // hook node into another, higher node in order to do the rotate
  MedianTreeNode<short> base(10);
  base.InsertSubtree(node, false);

  // establish preconditions for the test
  EXPECT_EQ(4, node->height());
  EXPECT_EQ(7, node->count());
  EXPECT_EQ(2, node->balance());
  ASSERT_NE(nullptr, node->left());
  EXPECT_EQ(1, node->left()->balance());

  // perform the rotation
  base.RotateChildren(false);

  // node should no longer be base's left child
  ASSERT_NE(node, base.left());

  // reassign node symbol to be base's new left child, which should be 4
  node = base.left();
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
 * GIVEN a MedianTreeNode `node` with a balance factor of 2 and whose own left
 *       child has a balance factor of -1, i.e., is in need of an LR rotation
 * DO perform an LR rotation AND
 * TEST THAT the balance factor of `node` is 0; AND
 *           `node` has the correct number of children; AND
 *           both the left and right children of `node` are balanced.
 */
TEST(MedianTreeNodeTests, LRRotation) {
  std::shared_ptr<MedianTreeNode<short>> node(new MedianTreeNode<short>(6));
  node->Insert(2, false);
  node->Insert(7, false);
  node->Insert(1, false);
  node->Insert(4, false);
  node->Insert(3, false);
  node->Insert(5, false);

  // hook node into another, higher node in order to do the rotate
  MedianTreeNode<short> base(0);
  base.InsertSubtree(node, false);

  // establish preconditions for the test
  EXPECT_EQ(4, node->height());
  EXPECT_EQ(7, node->count());
  EXPECT_EQ(2, node->balance());
  ASSERT_NE(nullptr, node->left());
  EXPECT_EQ(-1, node->left()->balance());

  // perform the rotation
  base.RotateChildren(false);

  // node should no longer be base's right child
  ASSERT_NE(node, base.right());

  // reassign node symbol to be base's new right child, which should be 4
  node = base.right();
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
 * GIVEN a MedianTreeNode `node` with a balance factor of -2 and whose own right
 *       child has a balance factor of 1, i.e., is in need of an RL rotation
 * DO perform an RL rotation AND
 * TEST THAT the balance factor of `node` is 0; AND
 *           `node` has the correct number of children; AND
 *           both the left and right children of `node` are balanced.
 */
TEST(MedianTreeNodeTests, RLRotation) {
  std::shared_ptr<MedianTreeNode<short>> node(new MedianTreeNode<short>(2));
  node->Insert(1, false);
  node->Insert(6, false);
  node->Insert(4, false);
  node->Insert(7, false);
  node->Insert(3, false);
  node->Insert(5, false);

  // hook node into another, higher node in order to do the rotate
  MedianTreeNode<short> base(10);
  base.InsertSubtree(node, false);

  // establish preconditions for the test
  EXPECT_EQ(4, node->height());
  EXPECT_EQ(7, node->count());
  EXPECT_EQ(-2, node->balance());
  ASSERT_NE(nullptr, node->right());
  EXPECT_EQ(1, node->right()->balance());

  // perform the rotation
  base.RotateChildren(false);

  // node should no longer be base's left child
  ASSERT_NE(node, base.left());

  // reassign node symbol to be base's new left child, which should be 4
  node = base.left();
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
 * GIVEN a MedianTreeNode `node` with a balance factor of -2 and whose own right
 *       child has a balance factor of -1, i.e., is in need of an RR rotation
 * DO perform an RR rotation AND
 * TEST THAT the balance factor of `node` is 0; AND
 *           `node` has the correct number of children; AND
 *           both the left and right children of `node` are balanced.
 */
TEST(MedianTreeNodeTests, RRRotation) {
  std::shared_ptr<MedianTreeNode<short>> node(new MedianTreeNode<short>(2));
  node->Insert(1, false);
  node->Insert(4, false);
  node->Insert(3, false);
  node->Insert(6, false);
  node->Insert(5, false);
  node->Insert(7, false);

  // hook node into another, higher node in order to do the rotate
  MedianTreeNode<short> base(0);
  base.InsertSubtree(node, false);

  // establish preconditions for the test
  EXPECT_EQ(4, node->height());
  EXPECT_EQ(7, node->count());
  EXPECT_EQ(-2, node->balance());
  ASSERT_NE(nullptr, node->right());
  EXPECT_EQ(-1, node->right()->balance());

  // perform the rotation
  base.RotateChildren(false);

  // node should no longer be base's right child
  ASSERT_NE(node, base.right());

  // reassign node symbol to be base's new right child, which should be 4
  node = base.right();
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
 * GIVEN a MedianTreeNode `node` with value u and no children
 * DO insert values v < w < x < y < z (with v > u), in that order, AND
 * TEST THAT the balance of the tree containing `node` never exceeds 1 in
 *           absolute value; AND
 *           the height of the tree containing `node` never exceeds 3.
 */
TEST(MedianTreeNodeTests, RotateOnInsert) {
  short u = 0;
  auto v = u + 1;
  auto w = v + 1;
  auto x = w + 1;
  auto y = x + 1;
  auto z = y + 1;

  std::shared_ptr<MedianTreeNode<short>> node(new MedianTreeNode<short>(u));

  // hook node into another, higher node in order to do the rotates
  MedianTreeNode<short> base(u - 1); // all insertions go to the right
  base.InsertSubtree(node);

  // establish preconditions for the test
  ASSERT_NE(nullptr, base.right());
  EXPECT_GE(1, base.right()->balance()); // 1 >= balance
  EXPECT_LE(-1, base.right()->balance()); // -1 <= balance
  EXPECT_GE(3, base.right()->height()); // 3 >= height

  // perform the inserts
  base.Insert(v); // insert defaults to rotate on insert
  EXPECT_GE(1, base.right()->balance()); // 1 >= balance
  EXPECT_LE(-1, base.right()->balance()); // -1 <= balance
  EXPECT_GE(3, base.right()->height()); // 3 >= height

  base.Insert(w);
  EXPECT_GE(1, base.right()->balance()); // 1 >= balance
  EXPECT_LE(-1, base.right()->balance()); // -1 <= balance
  EXPECT_GE(3, base.right()->height()); // 3 >= height

  base.Insert(x);
  EXPECT_GE(1, base.right()->balance()); // 1 >= balance
  EXPECT_LE(-1, base.right()->balance()); // -1 <= balance
  EXPECT_GE(3, base.right()->height()); // 3 >= height

  base.Insert(y);
  EXPECT_GE(1, base.right()->balance()); // 1 >= balance
  EXPECT_LE(-1, base.right()->balance()); // -1 <= balance
  EXPECT_GE(3, base.right()->height()); // 3 >= height

  base.Insert(z);
  EXPECT_GE(1, base.right()->balance()); // 1 >= balance
  EXPECT_LE(-1, base.right()->balance()); // -1 <= balance
  EXPECT_GE(3, base.right()->height()); // 3 >= height
}

/*
 * GIVEN a MedianTreeNode `node` with 6 values u < v < w < x < y < z, inserted
 *       in that order without rotation
 * DO perform a recursive rotation AND
 * TEST THAT the subtree containing `node` is balanced; AND
 *           the count of elements remains unchanged.
 */
TEST(MedianTreeNodeTests, RecursiveRotate) {
  short u = 1;
  auto v = u + 1;
  auto w = v + 1;
  auto x = w + 1;
  auto y = x + 1;
  auto z = y + 1;

  std::shared_ptr<MedianTreeNode<short>> node(new MedianTreeNode<short>(u));
  node->Insert(v, false);
  node->Insert(w, false);
  node->Insert(x, false);
  node->Insert(y, false);
  node->Insert(z, false);

  // hook node into another, higher node in order to do the rotate
  MedianTreeNode<short> base(z + 1);
  base.InsertSubtree(node, false);

  // establish preconditions for the test
  ASSERT_NE(nullptr, base.left());
  EXPECT_EQ(-5, base.left()->balance());

  // perform the rotation
  base.RotateChildren(true);

  // node no longer lives at base.left(), so reassign node
  ASSERT_NE(nullptr, base.left());
  EXPECT_NE(node, base.left());
  node = base.left();

  // ensure the node is balanced
  EXPECT_EQ(0, node->balance());
  EXPECT_EQ(6, node->count());
}
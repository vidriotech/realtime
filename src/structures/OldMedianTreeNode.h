#ifndef RTS_2_OLDMEDIANTREENODE_H
#define RTS_2_OLDMEDIANTREENODE_H

#include <memory>
#include <utility>

#define AVL_LEFT_HEAVY 1
#define AVL_BALANCED 0
#define AVL_RIGHT_HEAVY -1

#define NODE_LEFT 1
#define NODE_RIGHT -1

#define MAX(a, b) ((a) < (b) ? (b) : (a))

template <class T>
class OldMedianTreeNode {
public:
    explicit OldMedianTreeNode(T val)
        : data(val), left(nullptr), right(nullptr), n(1) {};

    std::unique_ptr<OldMedianTreeNode<T>> left;
    std::unique_ptr<OldMedianTreeNode<T>> right;

    void insert(T val);
    void insert_node(std::unique_ptr<OldMedianTreeNode<T>> node);
    short remove(T val);
    std::unique_ptr<OldMedianTreeNode<T>> pop(T val);

    // getters
    T value() const { return data; }
    unsigned height() const;
    int balance() const;
    unsigned n_elements() const { return n; };

    T max() const;
    T min() const;
private:
    T data;
    unsigned n;

    short remove_child(short id);
    std::unique_ptr<OldMedianTreeNode<T>> pop_child(short id);
};

template<class T>
void OldMedianTreeNode<T>::insert_node(std::unique_ptr<OldMedianTreeNode<T>> node) {
    if (node == nullptr) {
        return;
    }

    auto val = node->value();
    auto ht = node->height();
    if (val <= data && left == nullptr) {
        left.swap(node);
    } else if (val <= data) {
        left->insert_node(std::move(node));
    } else if (right == nullptr) { // val > data
        right.swap(node);
    } else { // val > data && right != nullptr
        right->insert_node(std::move(node));
    }

    n += ht;
}

template<class T>
void OldMedianTreeNode<T>::insert(T val) {
    std::unique_ptr<OldMedianTreeNode<T>> node(new OldMedianTreeNode<T>(val));
    insert_node(std::move(node));
}

template<class T>
short OldMedianTreeNode<T>::remove_child(short id) {
    std::unique_ptr<OldMedianTreeNode<T>> node = pop_child(id);
    short res = node == nullptr ? -1 : 0;

    node.reset(nullptr);

    return res;
}

template<class T>
short OldMedianTreeNode<T>::remove(T val) {
    short res = -1;

    if (val <= data && left != nullptr && left->value() == val) {
        res = remove_child(NODE_LEFT);
    } else if (val <= data && left != nullptr) {
        res = left->remove(val);
        if (res == 0)
            n--;
    } else if (val > data && right != nullptr && right->value() == val) {
        res = remove_child(NODE_RIGHT);
    } else if (val > data && right != nullptr) {
        res = right->remove(val);
        if (res == 0)
            n--;
    }

    return res;
}

template<class T>
std::unique_ptr<OldMedianTreeNode<T>> OldMedianTreeNode<T>::pop(T val) {
    std::unique_ptr<OldMedianTreeNode<T>> node;

    if (val <= data && left != nullptr && left->value() == val) {
        node = pop_child(NODE_LEFT);
    } else if (val <= data && left != nullptr) {
        node = left->pop(val);
        if (node != nullptr)
            n--;
    } else if (val > data && right != nullptr && right->value() == val) {
        node = pop_child(NODE_RIGHT);
    } else if (val > data && right != nullptr) {
        node = right->pop(val);
        if (node != nullptr)
            n--;
    } else {
        node.reset(nullptr);
    }

    return std::move(node);
}

template<class T>
std::unique_ptr<OldMedianTreeNode<T>> OldMedianTreeNode<T>::pop_child(short id) {
    std::unique_ptr<OldMedianTreeNode<T>> res(nullptr);
    std::unique_ptr<OldMedianTreeNode<T>> left_child(nullptr);
    std::unique_ptr<OldMedianTreeNode<T>> right_child(nullptr);

    if (id == NODE_LEFT) {
        left_child.swap(left->left);
        right_child.swap(left->right);
        res.swap(left);
    } else if (id == NODE_RIGHT) {
        left_child.swap(right->left);
        right_child.swap(right->right);
        res.swap(right);
    } else {
        return res;
    }

    n--;
    this->insert_node(std::move(left_child));
    this->insert_node(std::move(right_child));

    return std::move(res);
}

template<class T>
int OldMedianTreeNode<T>::balance() const {
    int balance;

    if (left == nullptr && right == nullptr) {
        balance = 0;
    } else if (left == nullptr) {
        balance = -(right->height());
    } else if (right == nullptr) {
        balance = left->height();
    } else {
        balance = left->height() - right->height();
    }

    return balance;
}

template<class T>
unsigned OldMedianTreeNode<T>::height() const {
    unsigned height = 1;

    if (left != nullptr && right != nullptr) {
        height += MAX(left->height(), right->height());
    } else if (left != nullptr) {
        height += left->height();
    } else if (right != nullptr) {
        height += right->height();
    }

    return height;
}

template<class T>
T OldMedianTreeNode<T>::max() const {
    T val;

    if (n_elements() == 1 || right == nullptr) {
        val = data;
    } else {
        val = right->max();
    }

    return val;
}

template<class T>
T OldMedianTreeNode<T>::min() const {
    T val;

    if (n_elements() == 1 || left == nullptr) {
        val = data;
    } else {
        val = left->min();
    }

    return val;
}

#endif //RTS_2_OLDMEDIANTREENODE_H

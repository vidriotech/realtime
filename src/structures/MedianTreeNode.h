#ifndef RTS_2_MEDIANTREENODE_H
#define RTS_2_MEDIANTREENODE_H

#include <memory>
#include <utility>

#define AVL_LEFT_HEAVY 1
#define AVL_BALANCED 0
#define AVL_RIGHT_HEAVY -1

#define NODE_LEFT 1
#define NODE_RIGHT -1

#define MAX(a, b) ((a) < (b) ? (b) : (a))

template <class T>
class MedianTreeNode {
public:
    explicit MedianTreeNode(T val)
        : data(val), left(nullptr), right(nullptr), n(1) {};

    void insert(T val);
    short remove(T val);
    std::unique_ptr<MedianTreeNode<T>> pop(T val);

    // getters
    T value() { return data; }
    unsigned height();
    int balance();
    unsigned n_elements() { return n; };
private:
    T data;
    std::unique_ptr<MedianTreeNode<T>> left;
    std::unique_ptr<MedianTreeNode<T>> right;
    unsigned n;

    void insert_node(std::unique_ptr<MedianTreeNode<T>> node);
    short remove_child(short id);
    std::unique_ptr<MedianTreeNode<T>> pop_child(short id);
};

template<class T>
void MedianTreeNode<T>::insert_node(std::unique_ptr<MedianTreeNode<T>> node) {
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
void MedianTreeNode<T>::insert(T val) {
    std::unique_ptr<MedianTreeNode<T>> node(new MedianTreeNode<T>(val));
    insert_node(std::move(node));
}

template<class T>
short MedianTreeNode<T>::remove_child(short id) {
    std::unique_ptr<MedianTreeNode<T>> node = pop_child(id);
    short res = node == nullptr ? -1 : 0;

    node.reset(nullptr);

    return res;
}

template<class T>
short MedianTreeNode<T>::remove(T val) {
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
std::unique_ptr<MedianTreeNode<T>> MedianTreeNode<T>::pop(T val) {
    std::unique_ptr<MedianTreeNode<T>> node;

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
std::unique_ptr<MedianTreeNode<T>> MedianTreeNode<T>::pop_child(short id) {
    std::unique_ptr<MedianTreeNode<T>> res(nullptr);
    std::unique_ptr<MedianTreeNode<T>> left_child(nullptr);
    std::unique_ptr<MedianTreeNode<T>> right_child(nullptr);

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
int MedianTreeNode<T>::balance() {
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
unsigned MedianTreeNode<T>::height() {
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

#endif //RTS_2_MEDIANTREENODE_H

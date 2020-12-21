#ifndef RTS_2_DISTANCEMATRIX_H
#define RTS_2_DISTANCEMATRIX_H

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

// T: the type of value being stored
// N: the number of elements whose distances are being stored
template <class T>
class DistanceMatrix
{
private:
    size_t N{};
    std::vector<T> data;

    size_t index_at(size_t i, size_t j);

public:
    explicit DistanceMatrix(size_t N);

    size_t n_cols();  // column count (also row count)

    T get_at(size_t i, size_t j);  // get the element at the (i,j) index
    void set_at(size_t i, size_t j, T val);  // set the element at the (i,j) index
};

template <class T>
DistanceMatrix<T>::DistanceMatrix(size_t N)
    : data(N * (N - 1) / 2) {
    this->N = N;
}

template <class T>
size_t DistanceMatrix<T>::index_at(size_t i, size_t j)
{
    // private and accessed only from get_at, so we can skip a bounds check
    if (j < i) {
        return index_at(j, i);
    }

    // the index in the data array of the (i, j) element
    return i * N - (i + 1) * (i + 2) / 2 + j;
}

template <class T>
size_t DistanceMatrix<T>::n_cols()
{
    return N;
}

template <class T>
T DistanceMatrix<T>::get_at(size_t i, size_t j)
{
    if (i >= N || j >= N) {
        throw std::out_of_range("Index is out of bounds for this size matrix.");
    }
    if (i == j) {
        return (T)0;
    }

    auto idx = index_at(i, j);
    return data[idx];
}

template <class T>
void DistanceMatrix<T>::set_at(size_t i, size_t j, T val)
{
    if (i >= N || j >= N) {
        throw std::out_of_range("Index is out of bounds for this size matrix.");
    }
    if (i == j) {
        throw std::domain_error("Setting a diagonal element is forbidden.");
    }

    auto idx = index_at(i, j);
    data[idx] = val;
}

#endif //RTS_2_DISTANCEMATRIX_H

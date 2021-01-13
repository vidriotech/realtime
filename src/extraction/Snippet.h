#ifndef RTS_2_SNIPPET_H
#define RTS_2_SNIPPET_H

#include <cmath>
#include <vector>

template <class T, unsigned M, unsigned N>
class Snippet {
public:
    Snippet();
    Snippet(T *buf);
    T get(unsigned i, unsigned j);
    double sq_dist(Snippet<T, M, N> &other);

protected:
    T at(unsigned);

private:
    std::vector<T> data;
};

template<class T, unsigned M, unsigned N>
Snippet<T, M, N>::Snippet()
    : data(M * N) {}

template<class T, unsigned int M, unsigned int N>
Snippet<T, M, N>::Snippet(T *buf)
    : data(buf) {}

template <class T, unsigned M, unsigned N>
T Snippet<T, M, N>::get(unsigned i, unsigned j)
{
    return data.at(j * N + i);
}

template<class T, unsigned int M, unsigned int N>
double Snippet<T, M, N>::sq_dist(Snippet<T, M, N> &other) {
    double d = 0.0;
    for (auto i = 0; i < data.capacity(); ++i) {
        d += pow(data.at(i) - other.at(i), 2);
    }

    return d;
}

template<class T, unsigned int M, unsigned int N>
T Snippet<T, M, N>::at(unsigned int k) {
    return data.at(k);
}

#endif //RTS_2_SNIPPET_H

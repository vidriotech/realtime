#include "pipeline.h"

template<class T>
Pipeline<T>::Pipeline(Params &params)
    : params_(params) {

}

template
class Pipeline<short>;

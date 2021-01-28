#include "pipeline.h"

template<class T>
Pipeline<T>::Pipeline(Params &params, Probe &probe)
    : params_(params), probe_(probe) {
  buf_.reset(new T[n_frames()]);
}

template<class T>
unsigned Pipeline<T>::n_frames() const {
  return (unsigned) std::ceil(probe_.sample_rate() * params_.acquire.n_seconds);
}

template
class Pipeline<short>;
